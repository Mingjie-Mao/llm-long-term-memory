"""Exact logical-request replay for unchanged cases; never a fresh model answer."""

from __future__ import annotations

import hashlib
import json

from llm_long_term_memory.llm.client import Completion


class ReplayMismatch(ValueError):
    pass


def normalized_request(kwargs):
    schema = kwargs.get("schema")
    return {k: v for k, v in kwargs.items() if k != "schema"} | {
        "schema": schema.model_json_schema() if schema else None
    }


class SavedReplyClient:
    def __init__(self, delegate, reader):
        self.delegate = delegate
        self.receipts = reader["answer"]["notes"]["actual_answer_calls"]
        self.position = 0

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def generate(self, **kwargs):
        if self.position >= len(self.receipts):
            raise ReplayMismatch("extra generation requested")
        receipt = self.receipts[self.position]
        if normalized_request(kwargs) != receipt["request"]:
            raise ReplayMismatch("logical generation request changed")
        self.position += 1
        return Completion(receipt["response"], kwargs["model"], 0, 0, 0, 0, 0)

    def verify_output(self, answer, reader):
        if self.position != len(self.receipts) or answer.text != reader["answer"]["text"]:
            raise ReplayMismatch("output or generation count changed")


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def grade_request_sha(instance, hypothesis):
    return row_sha(
        {
            "question": instance.question,
            "gold": instance.answer,
            "hypothesis": hypothesis,
            "is_abstention": instance.is_abstention,
            "question_type": instance.question_type,
        }
    )


def cached_grade(instance, reader, original_reader, original_grade, identity, mode="original"):
    if original_grade["reader_row_sha256"] != row_sha(original_reader):
        raise ValueError("historical grade not bound to reader")
    if reader["answer"]["text"] != original_reader["answer"]["text"]:
        raise ValueError("changed hypothesis cannot reuse grade")
    proof = reader["answer"]["notes"].get("exact_request_replay")
    if not proof or proof["source_reader_sha256"] != row_sha(original_reader):
        raise ValueError("grade reuse requires exact request replay proof")
    if proof["judge_request_sha256"][mode] != grade_request_sha(instance, reader["answer"]["text"]):
        raise ValueError("question/gold/type/abstention changed")
    if proof["judge_prompt_version"] != original_grade["judge_prompt_version"]:
        raise ValueError("judge prompt changed")
    if original_grade["inventory_sha256"] != original_reader["inventory_sha256"]:
        raise ValueError("historical inventory changed")
    if (
        original_grade["question_id"] != instance.question_id
        or type(original_grade["verdict"]["correct"]) is not bool
    ):
        raise ValueError("foreign or untyped historical grade")
    return {
        "question_id": instance.question_id,
        "cohort": reader["cohort"],
        "inventory_sha256": identity,
        "judge_prompt_version": original_grade["judge_prompt_version"],
        "reader_row_sha256": row_sha(reader),
        "verdict": original_grade["verdict"],
        "exact_grade_replay": {
            "source_grade_sha256": row_sha(original_grade),
            "source_reader_sha256": row_sha(original_reader),
            "scope": "same question/gold/hypothesis/judge; historical verdict, zero new calls",
        },
    }
