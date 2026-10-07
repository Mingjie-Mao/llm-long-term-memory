import hashlib
import json
import sys
from pathlib import Path

repo = Path.cwd()
sys.path[:0] = [str(repo / 'src'), str(repo / 'tools')]
from analysis_io import rows_by_question, sha256_file
from grounded_reader_report import summarize
from grounded_operand_replay import restore_ledger
from llm_long_term_memory.config import Settings
from llm_long_term_memory.conversation import AnswerRequest
from llm_long_term_memory.evaluation.datasets.longmemeval import load
from llm_long_term_memory.runtime.grounded_answering import GroundedAnswererV15, SessionEvidenceLedger, FocusSessionEvidenceLedger
from llm_long_term_memory.store import SQLiteMemoryStore

stem = 'grounded-reader-v15'
answers_path = repo / f'results/raw/{stem}.answers.jsonl'
grades_path = repo / f'results/raw/{stem}.grades.jsonl'
inventory_path = repo / f'results/analysis/{stem}.execution.json'
usage_path = repo / f'results/raw/{stem}.usage.json'
proof_path = repo / f'results/analysis/{stem}.public-source-audit.json'
answers = rows_by_question(answers_path)
proof = json.loads(proof_path.read_text())
identity = json.loads(inventory_path.read_text())
assert proof['status'] == 'PASS'
for rel, sha in identity['files'].items():
    assert sha256_file(repo / rel) == sha, rel
expected = set()
for cohort in ['reasoning-errors', 'correct-sample10']:
    expected.update(json.loads((repo / f'results/manifests/train150-raw-v1-{cohort}.json').read_text())['question_ids'])
assert set(answers) == expected
instances = {i.question_id: i for i in load('s', Settings().data_dir)}
engine = object.__new__(GroundedAnswererV15)
store = SQLiteMemoryStore(repo / 'stores/train150.db', read_only=True)
results = []
def pool_sha(audit):
    values = sorted([{k: v for k, v in item.items() if k != 'id'} for item in audit['sources']], key=lambda item: (item['kind'], item['source_id']))
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
try:
    for qid, row in answers.items():
        instance = instances[qid]
        request = AnswerRequest(instance.question, instance.question_date, instance.store_namespace)
        calls = row['answer']['notes']['grounded_calls']
        assert row['inventory_sha256'] == sha256_file(inventory_path)
        assert pool_sha(calls[0]['evidence']) == proof['source_pools'][qid]['canonical_source_pool_sha256'], qid
        previous_focus = ()
        last_calc = last_verdict = None
        for saved in calls:
            assert pool_sha(saved['evidence']) == pool_sha(calls[0]['evidence']), qid
            def ledger_type(*, sources, max_tokens):
                return (FocusSessionEvidenceLedger(sources=sources, max_tokens=max_tokens, focus_ids=previous_focus) if previous_focus else SessionEvidenceLedger(sources=sources, max_tokens=max_tokens))
            ledger = restore_ledger(store, instance.store_namespace, saved['evidence'], ledger_type=ledger_type)
            last_verdict = engine.parse_verdict(saved['provider_response'], ledger, request)
            last_calc = engine.validate_verdict(last_verdict, ledger, request)
            assert last_calc.computed == saved['computed'], qid
            assert last_calc.detail == saved['calculation'], qid
            previous_focus = tuple(saved.get('calculation', {}).get('focus_sources', []))
        expected_answer = (last_calc.answer if last_calc.computed else last_verdict.answer.strip()) if engine.can_answer(last_verdict, last_calc) else 'I do not know.'
        assert expected_answer == row['answer']['text'], qid
        results.append({'question_id': qid, 'tenant_raw_hashes_fixed_pool_and_final_replay': True, 'mechanism': last_calc.detail.get('mechanism'), 'cause': last_calc.detail.get('cause'), 'computed': last_calc.computed})
        if qid == 'f9e8c073':
            assert last_calc.detail.get('mechanism') == 'scoped_quantity_disclosure'
            assert {c['period'] for c in last_calc.detail['quantity_citations']} == {'last year', 'period unspecified'}
            assert 'combined total cannot be determined' in row['answer']['text']
        if qid == 'gpt4_2f8be40d':
            assert not engine.can_answer(last_verdict, last_calc)
            assert last_calc.detail.get('cause')
finally:
    store.close()
report = summarize(answers_path, grades_path, inventory_path, usage_path)
unsupported = {'f9e8c073', 'gpt4_2f8be40d'}
supported = [r for r in report['rows'] if r['question_id'] not in unsupported]
assert len(supported) == 17
rules = {'original_grading_complete_for_all19': report['grading_complete_for_saved_readers'], 'all17_original_judge_correct': all(r['correct'] is True for r in supported), 'two_predeclared_uncertainties_faithfully_preserved': True, 'all19_tenant_sources_context_hash_fixed_pool_and_final_code_replay': True}
out = repo / f'results/analysis/{stem}.acceptance.json'
if out.exists():
    raise SystemExit('refusing overwrite')
result = {'pass': all(rules.values()), 'rules': rules, 'experiment_class': 'exposed train150 mechanism acceptance; not overall accuracy or unseen final', 'supported_correct': sum(r['correct'] is True for r in supported), 'supported_total': 17, 'original19_correct': sum(r['correct'] is True for r in report['rows']), 'source_rows': results, 'inputs': report['inputs'] | {str(proof_path.relative_to(repo)): sha256_file(proof_path), 'results/prereg-raw-primary-grounded-v12-acceptance.md': sha256_file(repo / 'results/prereg-raw-primary-grounded-v12-acceptance.md')}, 'generator_sha256': sha256_file(Path(__file__)), 'generator_source': Path(__file__).read_text(), 'provider_calls': 0}
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['source_rows', 'generator_source', 'inputs']}))
