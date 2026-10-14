import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from resume_original_test100_server import amendment_profile, verify_previous_binding


def test_profile_uses_original_manifest_and_store_and_restores_on_failure(tmp_path):
    original_instances = [SimpleNamespace(question_id="original-q")]
    profile = SimpleNamespace(
        STEM="original-test100",
        ANALYSIS=tmp_path,
        experiment=SimpleNamespace(instances=lambda: original_instances),
    )

    def old_lookup(split):
        return ["foreign dev instance"]

    amendment = SimpleNamespace(
        STEM="dev100",
        ANALYSIS=Path("old"),
        AMENDMENT=Path("old.md"),
        INVENTORY=Path("old.json"),
        LEDGER=Path("old-ledger.json"),
        instances_for=old_lookup,
    )
    calls = []

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return "unused reader, unchanged judge"

    cli = SimpleNamespace(_build=build)
    with pytest.raises(RuntimeError, match="stop"), amendment_profile(amendment, profile, cli):
        assert amendment.instances_for("dev100") == original_instances
        assert amendment.STEM == "original-test100"
        assert tmp_path / "original-test100.judge-amendment-v1.json" == amendment.LEDGER
        cli._build("same-variant", "same-config", "dev100", read_only_store=True)
        cli._build("same-variant", "same-config", "test100", read_only_store=True)
        raise RuntimeError("stop")
    assert [args[2] for args, _ in calls] == ["test100", "test100"]
    assert all(kwargs == {"read_only_store": True} for _, kwargs in calls)
    assert amendment.STEM == "dev100"
    assert amendment.instances_for is old_lookup
    assert cli._build is build


@pytest.mark.parametrize(
    "field", ["original_recovery_sha256", "reader_sha256", "amendment_inventory_sha256"]
)
def test_restarting_extension_cannot_change_saved_answer_or_reset_original_state(field):
    expected = {
        "arm": "control",
        "repeat": 3,
        "question_id": "q",
        "original_recovery_sha256": "original-state",
        "reader_sha256": "original-answer",
        "amendment_inventory_sha256": "fixed-policy",
    }
    verify_previous_binding([expected], expected)
    with pytest.raises(ValueError, match="changed across extension"):
        verify_previous_binding([expected], {**expected, field: "different"})
    verify_previous_binding([{**expected, "question_id": "other", field: "different"}], expected)
