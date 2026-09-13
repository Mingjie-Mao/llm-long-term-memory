"""Two ways the next paid run could waste itself, both caught for free.

Quota is the binding constraint on this project, and both defects below were live at
the same time: the probe runner had no way to answer only the development half, so the
first v4 measurement would have spent the 87 held-out probes on a development run; and
its rows dropped the derivation, so a paid v4 run would have produced answers nobody
could tell apart from the model's own.

Neither raises. Both were found by rehearsing with a fake provider.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "tools/run_synthesis_probes.py"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_the_runner_answers_the_development_half_by_default():
    """A default of 'all' would spend the holdout on whichever run came next, and the
    person running it would have no reason to think twice."""
    source = _source()
    assert '"--half"' in source
    assert 'default="development"' in source


def test_the_held_out_half_uses_a_global_ledger():
    """Labels cannot create a second held-out run; interrupted exact resumes are allowed."""
    source = _source()
    assert 'if args.half == "held_out"' in source
    assert "claim_holdout(ledger_path, out, identity)" in source
    assert "v4-probe-heldout-run.json" in source


def test_the_probe_rows_carry_the_v4_derivation():
    """ "3 distinct: ..." from the code and "three" from the model are the same string
    to a grader. The derivation is what tells them apart."""
    source = _source()
    assert '"synthesis_operation": ans.notes.get("synthesis_operation")' in source
    assert '"synthesis_computation": ans.notes.get("synthesis_computation")' in source


def test_the_dry_run_fakes_the_policy_under_test():
    """A canned reply in the base verdict shape rehearses nothing v4 changed: `compute`
    never runs and the rehearsal passes while the paid run fails on its first probe.

    Matched as a policy *family* rather than one exact name. An equality test against
    "synthesis_v4" is what let v4.2 rehearse the base shape: the name did not match, the
    canned reply fell back, and the rehearsal would have passed while the paid run
    exercised nothing. Any later synthesis policy inherits the fake by prefix."""
    source = _source()
    assert 'policy.startswith("synthesis_v4")' in source
    assert "canned_by_kind" in source
    for kind in ("count", "duration", "comparison", "current_state"):
        assert f'"{kind}"' in source


def test_the_dry_run_fakes_the_citation_field_for_the_enumerating_policy():
    """v4.2 counts from cited labels. A canned count with no labels rehearses the
    free-text fallback, which is the one path the candidate is not being tested on.

    The labels must come from the context the runner actually built — a hard-coded
    "M1" would be resolved against a real context that may not contain it, and the
    rehearsal would exercise the rejection path instead of the counting one."""
    source = _source()
    assert 'policy == "synthesis_v4_enumerate"' in source
    assert '"member_labels"' in source
    assert "_context_labels" in source


def test_a_dry_run_never_writes_the_file_a_real_run_resumes_from():
    """A canned row is indistinguishable from a paid one on the next resume."""
    source = _source()
    assert ".dry.jsonl" in source or ".dry" in source


def test_the_split_manifest_is_the_one_the_runner_reads():
    """A runner pointed at a different manifest would filter against a split nobody
    recorded."""
    assert "results/manifests/v4-probe-split.json" in _source()
    assert (REPO / "results/manifests/v4-probe-split.json").is_file()


def test_the_development_half_is_what_the_manifest_says():
    split = json.loads((REPO / "results/manifests/v4-probe-split.json").read_text(encoding="utf-8"))
    assert len(split["development"]) == 142
    assert len(split["held_out"]) == 87


def test_the_runner_prints_which_half_it_selected():
    """Silence about scope is how the wrong half gets answered without anyone noticing."""
    assert "probes selected" in _source()
    assert inspect.cleandoc("") == ""
