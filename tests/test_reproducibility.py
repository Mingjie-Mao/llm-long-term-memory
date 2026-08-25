from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from llm_long_term_memory.config import ExperimentConfig
from llm_long_term_memory.evaluation.reproducibility import (
    FreezeError,
    FreezeRequest,
    capture_system,
    differences,
    formal_protocol_paths,
    logical_sqlite_fingerprint,
)
from llm_long_term_memory.ingest import fingerprint
from llm_long_term_memory.ingest.pipeline import resolved_sessions_per_request
from llm_long_term_memory.llm.usage import CallRecord, UsageTracker
from llm_long_term_memory.store import Memory, Session, SQLiteMemoryStore, Turn, scoped_session_id


def build_freezable_system(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "llm_long_term_memory").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "configs").mkdir()
    (repo / "results" / "manifests").mkdir(parents=True)
    data_dir = repo / "data"
    store_dir = repo / "stores"
    data_dir.mkdir()
    store_dir.mkdir()
    (repo / "src" / "llm_long_term_memory" / "answer.py").write_text(
        "PROMPT = 'one'\n", encoding="utf-8"
    )
    (repo / "scripts" / "run.py").write_text("print('run')\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    config = repo / "configs" / "v2.yaml"
    config.write_text("{}\n", encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(config)

    manifest = repo / "results" / "manifests" / "dev100.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "dev100",
                "variant": "s",
                "seed": 0,
                "question_ids": ["private-question"],
            }
        ),
        encoding="utf-8",
    )
    dataset = [
        {
            "question_id": "private-question",
            "question_type": "single-session-user",
            "question": "sealed question",
            "answer": "sealed answer",
            "question_date": "2026-01-02",
            "haystack_session_ids": ["source-session"],
            "haystack_dates": ["2026-01-01"],
            "haystack_sessions": [
                [{"role": "user", "content": "sealed source", "has_answer": True}]
            ],
            "answer_session_ids": ["source-session"],
        }
    ]
    (data_dir / "longmemeval_s_cleaned.json").write_text(json.dumps(dataset), encoding="utf-8")

    internal = scoped_session_id("private-question", "source-session")
    store = SQLiteMemoryStore(store_dir / "dev100.db")
    store.initialize()
    store.add_session(
        Session(
            id=internal,
            user_id="private-question",
            started_at=datetime(2026, 1, 1),
            turns=[
                Turn(
                    id=f"{internal}:0",
                    session_id=internal,
                    turn_index=0,
                    role="user",
                    content="sealed source",
                    ts=datetime(2026, 1, 1),
                )
            ],
        )
    )
    store.add_memories(
        [
            Memory(
                id="memory-one",
                user_id="private-question",
                type="semantic",
                content="sealed memory",
                token_count=2,
                source_session_id=internal,
            )
        ]
    )
    store.set_meta("extractor_version", "fixture-v1")
    batch_size = resolved_sessions_per_request(
        cfg.ingest.sessions_per_request,
        cfg.quota.tpm,
    )
    store.set_meta(
        "ingest_fingerprint",
        fingerprint.dumps(fingerprint.from_config(cfg, sessions_per_request=batch_size)),
    )
    store.close()
    (store_dir / "dev100-ingest.json").write_text(
        json.dumps(
            {
                "done_sessions": ["private-question:source-session"],
                "blocked_sessions": [],
                "memories_written": 1,
            }
        ),
        encoding="utf-8",
    )
    (store_dir / "dev100-index.ids.json").write_text(json.dumps(["memory-one"]), encoding="utf-8")
    with (store_dir / "dev100-index.npy").open("wb") as handle:
        np.save(handle, np.ones((1, cfg.models.embedding_dim), dtype=np.float32))
    UsageTracker(records=[CallRecord("extractor", "extractor-test", 10, 2, 1.0)]).save(
        repo / "results" / "raw" / "dev100.ingest.usage.json"
    )
    request = FreezeRequest(
        repo=repo,
        config_path=config,
        manifest_path=manifest,
        data_dir=data_dir,
        store_dir=store_dir,
        results_dir=repo / "results",
        variants=("two_stage_fallback", "two_stage_coherent"),
        store_name="dev100",
        pre_ingest_store_name=None,
    )
    return request


def test_capture_hashes_dirty_source_config_data_and_logical_store(tmp_path):
    request = build_freezable_system(tmp_path)

    frozen = capture_system(request)

    assert frozen["git_head_for_reference_only"] is None
    assert frozen["data"]["manifest"]["questions"] == 1
    assert frozen["store"]["terminal"] == {"expected": 1, "done": 1, "blocked": 0}
    assert frozen["store"]["database"]["counts"]["memories"] == 1
    assert frozen["store"]["index"]["ids"] == 1
    assert frozen["store"]["usage"]["requests"] == 1
    serialized = json.dumps(frozen)
    assert "sealed question" not in serialized
    assert "sealed answer" not in serialized
    assert "private-question" not in serialized


def test_changed_source_is_detected_without_a_commit(tmp_path):
    request = build_freezable_system(tmp_path)
    frozen = capture_system(request)
    source = request.repo / "src" / "llm_long_term_memory" / "answer.py"
    source.write_text("PROMPT = 'two'\n", encoding="utf-8")

    moved = differences(frozen, capture_system(request))

    assert "source.tree_sha256: changed" in moved
    assert any("source.files.src/llm_long_term_memory/answer.py.sha256" in item for item in moved)


def test_protocol_artifact_changes_are_detected(tmp_path):
    request = build_freezable_system(tmp_path)
    protocol = request.repo / "results" / "prereg-context-shape.md"
    protocol.write_text("fixed rule\n", encoding="utf-8")
    request = replace(request, protocol_paths=(protocol,))
    frozen = capture_system(request)
    protocol.write_text("changed rule\n", encoding="utf-8")

    moved = differences(frozen, capture_system(request))

    assert "protocol.tree_sha256: changed" in moved
    assert any("protocol.files.results/prereg-context-shape.md.sha256" in item for item in moved)


def test_formal_test_protocol_binds_dev_decision_and_aggregate(tmp_path):
    repo = tmp_path / "repo"

    paths = formal_protocol_paths(repo, "test100")

    assert paths[-2:] == (
        repo / "results" / "validation" / "dev100-aggregate.json",
        repo / "results" / "validation" / "dev100-decision.json",
    )
    assert repo / "results" / "analysis" / "train150-context-selection.final.json" in paths
    assert len([path for path in paths if "train150-context-grid-final" in str(path)]) == 7


def test_logical_database_hash_changes_when_a_value_changes(tmp_path):
    request = build_freezable_system(tmp_path)
    db = request.store_dir / "dev100.db"
    before = logical_sqlite_fingerprint(db)
    connection = sqlite3.connect(db)
    connection.execute("UPDATE memories SET content='changed' WHERE id='memory-one'")
    connection.commit()
    connection.close()

    after = logical_sqlite_fingerprint(db)

    assert before["logical_sha256"] != after["logical_sha256"]
    assert before["counts"] == after["counts"], "counts alone would have missed this change"


def test_incomplete_checkpoint_cannot_be_frozen(tmp_path):
    request = build_freezable_system(tmp_path)
    (request.store_dir / "dev100-ingest.json").write_text(
        json.dumps({"done_sessions": [], "blocked_sessions": []}), encoding="utf-8"
    )

    with pytest.raises(FreezeError, match="not exactly terminal"):
        capture_system(request)


def test_index_and_database_memory_counts_must_match(tmp_path):
    request = build_freezable_system(tmp_path)
    (request.store_dir / "dev100-index.ids.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FreezeError, match="vector/database memory mismatch"):
        capture_system(request)


def test_equal_index_count_with_wrong_memory_id_cannot_be_frozen(tmp_path):
    request = build_freezable_system(tmp_path)
    (request.store_dir / "dev100-index.ids.json").write_text(
        json.dumps(["different-memory"]), encoding="utf-8"
    )

    with pytest.raises(FreezeError, match="ids do not match"):
        capture_system(request)


def test_wrong_vector_dimension_cannot_be_frozen(tmp_path):
    request = build_freezable_system(tmp_path)
    with (request.store_dir / "dev100-index.npy").open("wb") as handle:
        np.save(handle, np.ones((1, 3), dtype=np.float32))

    with pytest.raises(FreezeError, match="vector dimension mismatch"):
        capture_system(request)


def test_store_from_a_different_extractor_config_cannot_be_frozen(tmp_path):
    request = build_freezable_system(tmp_path)
    config = ExperimentConfig.from_yaml(request.config_path)
    config.ingest.dedupe_similarity_threshold = 0.1
    config.to_yaml(request.config_path)

    with pytest.raises(FreezeError, match="ingest fingerprint mismatch"):
        capture_system(request)


def test_ingest_usage_summary_must_match_its_call_records(tmp_path):
    request = build_freezable_system(tmp_path)
    path = request.results_dir / "raw" / "dev100.ingest.usage.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"]["total_requests"] = 999
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FreezeError, match="summary disagrees"):
        capture_system(request)


def test_ingest_usage_changes_are_detected_by_the_freeze(tmp_path):
    request = build_freezable_system(tmp_path)
    frozen = capture_system(request)
    path = request.results_dir / "raw" / "dev100.ingest.usage.json"
    UsageTracker(records=[CallRecord("extractor", "m", 1, 1, 1.0)]).save(path, merge=True)

    moved = differences(frozen, capture_system(request))

    assert "store.usage.sha256: changed" in moved
    assert "store.usage.requests: changed" in moved


def test_pre_ingest_freeze_records_planned_store_without_hashing_a_partial_store(tmp_path):
    complete = build_freezable_system(tmp_path)
    request = replace(complete, store_name=None, pre_ingest_store_name="dev100")

    system = capture_system(request)

    assert system["pre_ingest_store_name"] == "dev100"
    assert system["store"] is None


def test_freeze_cannot_be_pre_and_post_ingest_at_once(tmp_path):
    request = replace(build_freezable_system(tmp_path), pre_ingest_store_name="dev100")

    with pytest.raises(FreezeError, match="both pre-ingest and post-ingest"):
        capture_system(request)


def test_reference_only_fields_are_recorded_but_not_enforced():
    """`git_head_for_reference_only` must not invalidate a freeze.

    Every file that affects a result is hashed individually in `source.files`, so the
    commit sha adds nothing to the guarantee. Comparing it made *any* commit — even
    one touching only `docs/` — break a pre-ingest freeze, which meant a freeze
    captured against an uncommitted tree could only be honoured by never committing
    the work it froze.
    """
    frozen = {"git_head_for_reference_only": "abc", "source": {"tree_sha256": "t"}}
    current = {"git_head_for_reference_only": "def", "source": {"tree_sha256": "t"}}
    assert differences(frozen, current) == []

    moved = differences(frozen, {**current, "source": {"tree_sha256": "other"}})
    assert moved == ["source.tree_sha256: changed"]
