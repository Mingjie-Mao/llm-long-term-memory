"""Exact judge-request reuse with auditable, unanimous saved verdicts.

No answer generation is cached. False verdicts are reused too; disagreement is a
cache miss. Every hit verifies the original reader and grade row identities.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from analysis_io import rows_by_question
from grade_saved_recovering import validate

from llm_long_term_memory.evaluation.judge import JUDGE_SYSTEM, Judge, JudgeResult, Verdict


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def request_key(judge, kwargs):
    request = {
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "system": JUDGE_SYSTEM,
        "prompt": Judge._prompt_for(
            kwargs["question"],
            kwargs["gold"],
            kwargs["hypothesis"],
            kwargs.get("is_abstention", False),
            kwargs.get("question_type"),
        ),
        "schema": Verdict.model_json_schema(),
        "temperature": 0.0,
        "thinking": judge.thinking,
    }
    return row_hash(request)


def grade_kwargs(instance, reader):
    return {
        "question": instance.question,
        "gold": instance.answer,
        "hypothesis": reader["answer"]["text"],
        "is_abstention": instance.is_abstention,
        "question_type": instance.question_type,
    }


class ExactRequestJudge:
    def __init__(self, judge):
        self.judge = judge
        self.prompt_version = judge.prompt_version
        self.entries = {}
        self.hits = 0
        self.misses = 0

    def add(self, instance, reader, grade, reader_path, grade_path):
        validate(reader, grade, instance.question_id)
        if grade["judge_prompt_version"] != self.prompt_version:
            raise ValueError("cache judge prompt changed")
        key = request_key(self.judge, grade_kwargs(instance, reader))
        self.entries.setdefault(key, []).append(
            {
                "question_id": instance.question_id,
                "reader_path": str(Path(reader_path).resolve()),
                "grade_path": str(Path(grade_path).resolve()),
                "reader_row_sha256": row_hash(reader),
                "grade_row_sha256": row_hash(grade),
                "result": grade["verdict"],
            }
        )

    def grade(self, **kwargs):
        key = request_key(self.judge, kwargs)
        witnesses = self.entries.get(key, [])
        if not witnesses or len({w["result"]["correct"] for w in witnesses}) != 1:
            self.misses += 1
            return self.judge.grade(**kwargs)
        for witness in witnesses:
            qid = witness["question_id"]
            reader = rows_by_question(Path(witness["reader_path"]))[qid]
            grade = rows_by_question(Path(witness["grade_path"]))[qid]
            validate(reader, grade, qid)
            if (
                row_hash(reader) != witness["reader_row_sha256"]
                or row_hash(grade) != witness["grade_row_sha256"]
            ):
                raise ValueError("cache evidence changed")
        result = witnesses[0]["result"]
        self.hits += 1
        return JudgeResult(
            correct=result["correct"],
            reason=result["reason"],
            input_tokens=0,
            output_tokens=0,
            api_latency_ms=0,
            score=result.get("score"),
            details={
                "judge_cache": {
                    "fresh_provider_call": False,
                    "request_sha256": key,
                    "witnesses": [{k: v for k, v in w.items() if k != "result"} for w in witnesses],
                    "original_results": [dict(w["result"]) for w in witnesses],
                }
            },
        )

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "request_keys": len(self.entries)}
