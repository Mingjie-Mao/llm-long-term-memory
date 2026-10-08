import hashlib
import json
import re
import sys
from pathlib import Path

repo = Path.cwd()
sys.path[:0] = [str(repo / 'src'), str(repo / 'tools')]
from analysis_io import sha256_file
from grounded_request_preview import PreviewClient
from grounded_operand_replay import restore_ledger
from llm_long_term_memory import cli
from llm_long_term_memory.config import Settings
from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.evaluation.datasets.longmemeval import load
from llm_long_term_memory.runtime.grounded_answering import SessionEvidenceLedger
from llm_long_term_memory.store import external_session_id

out = repo / 'results/analysis/grounded-reader-v15.public-source-audit.json'
assert not out.exists()
metadata_path = repo / 'results/analysis/grounded-reader-v11.public-source-metadata.json'
metadata = json.loads(metadata_path.read_text())
data_path = repo / 'data/longmemeval_s_cleaned.json'
assert sha256_file(data_path) == metadata['file']['lfs']['oid']
assert data_path.stat().st_size == metadata['file']['size']
preview_path = repo / 'results/analysis/grounded-reader-v15.request-preview-r2.json'
preview = json.loads(preview_path.read_text())
instances = {i.question_id: i for i in load('s', Settings().data_dir)}
client = PreviewClient()
_, _, runner, _, _ = cli._build('two_stage_raw_primary_grounded_v15', 'configs/fallback.yaml', 'train150', client_override=client, read_only_store=True)
raw_count = memory_count = 0
source_pools = {}
permutations = []
try:
    for row in preview['rows']:
        qid = row['question_id']
        instance = instances[qid]
        start = len(client.requests)
        answer = runner.answer_request(AnswerRequest(instance.question, instance.question_date, instance.store_namespace))
        actual = client.requests[start]
        expected = row['reader_request']
        if actual != expected:
            assert {k: v for k, v in actual.items() if k != 'prompt'} == {k: v for k, v in expected.items() if k != 'prompt'}
            normalize = lambda text: re.sub(r'\[E\d+\]', '[SOURCE]', text)
            assert normalize(actual['prompt']) == normalize(expected['prompt'])
            permutations.append({'question_id': qid, 'only_evidence_handle_assignment_differs': True})
        audit = answer.notes['grounded_calls'][0]['evidence']
        ledger = restore_ledger(runner.store, instance.store_namespace, audit, ledger_type=SessionEvidenceLedger)
        context = row['reader_request']['prompt'].split('\n\nQuestion date: ', 1)[0]
        assert re.sub(r'\[E\d+\]', '[SOURCE]', ledger.render()) == re.sub(r'\[E\d+\]', '[SOURCE]', context)
        memories = {m.id: m for m in runner.store.iter_all(instance.store_namespace)}
        originals = {(s.session_id, n): t for s in instance.sessions for n, t in enumerate(s.turns)}
        for source in ledger.sources:
            session = runner.store.get_session(source.session_id)
            assert session and session.user_id == instance.store_namespace
            if source.kind == 'raw':
                original = originals[(external_session_id(source.session_id), source.turn_index)]
                assert source.text == original.content and source.role == original.role
                raw_count += 1
            else:
                memory = memories[source.source_id]
                assert memory.user_id == instance.store_namespace
                assert memory.content == source.text
                original = originals[(external_session_id(memory.source_session_id), memory.source_turn_index)]
                turn = {t.turn_index: t for t in runner.store.turns_for_session(memory.source_session_id)}[memory.source_turn_index]
                assert original.content == turn.content and original.role == turn.role
                memory_count += 1
        source_pools[qid] = {'canonical_source_pool_sha256': hashlib.sha256(json.dumps(sorted([{k: v for k, v in item.items() if k != 'id'} for item in audit['sources']], key=lambda item: (item['kind'], item['source_id'])), sort_keys=True).encode()).hexdigest(), 'context_sha256': audit['context_sha256'], 'sources': len(ledger.sources), 'bridge': answer.notes.get('conversation_bridge_turns', [])}
finally:
    runner.store.close()
result = {'status': 'PASS', 'questions': len(preview['rows']), 'public_dataset_url': preview['dataset_origin'], 'public_metadata_path': str(metadata_path.relative_to(repo)), 'metadata_sha256': sha256_file(metadata_path), 'local_and_public_dataset_sha256': sha256_file(data_path), 'preview_sha256': sha256_file(preview_path), 'generator_sha256': sha256_file(Path(__file__)), 'raw_source_bodies_exactly_match_public_dataset': raw_count, 'derived_memories_public_origin_and_tenant_verified': memory_count, 'local_source_label_permutations': permutations, 'source_pools': source_pools, 'provider_calls': 0, 'other_user_data': False}
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'source_pools'}))
