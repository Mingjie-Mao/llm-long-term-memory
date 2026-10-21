"""Fail fast when the buildless public site drifts or loses basic semantics."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
EXPECTED_FILES = {
    "404.html",
    "demo-run.mp4",
    "demo-poster.jpg",
    "recording.json",
    "app.js",
    "content.js",
    "index.html",
    "playground.html",
    "release.json",
    "release.schema.json",
    "styles.css",
}


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def main() -> int:
    # The whole deploy root is uploaded, including subdirectories. Counting only
    # top-level regular files lets an accidental backup/ or stores/ directory pass.
    entries = list(SITE.iterdir())
    files = {path.name for path in entries}
    assert files == EXPECTED_FILES, f"unexpected deploy surface: {sorted(files)}"
    assert all(path.is_file() and not path.is_symlink() for path in entries), (
        "deploy entries must be regular files, not directories or symlinks"
    )

    index = (SITE / "index.html").read_text(encoding="utf-8")
    parser = SiteParser()
    parser.feed(index)

    elements = parser.elements
    ids = [attrs["id"] for _, attrs in elements if attrs.get("id")]
    assert len(ids) == len(set(ids)), "duplicate id in index.html"
    assert sum(tag == "h1" for tag, _ in elements) == 1, "the page needs exactly one h1"
    assert any(tag == "main" for tag, _ in elements), "missing main landmark"

    labels = {attrs.get("for") for tag, attrs in elements if tag == "label" and attrs.get("for")}
    fields = {
        attrs["id"]
        for tag, attrs in elements
        if tag in {"input", "textarea", "select"} and attrs.get("id")
    }
    assert fields <= labels, f"fields without associated labels: {sorted(fields - labels)}"

    for tag, attrs in elements:
        if "data-zh" in attrs:
            assert "data-en" in attrs, f"{tag} has Chinese copy without English copy"
        if "data-placeholder-zh" in attrs:
            assert "data-placeholder-en" in attrs, f"{tag} has an unpaired translated placeholder"
        if "data-aria-zh" in attrs:
            assert "data-aria-en" in attrs, f"{tag} has an unpaired translated accessible name"
        if "data-requires-live" in attrs:
            assert "disabled" in attrs, f"{tag} live control is enabled before session issuance"

    status = next(attrs for _, attrs in elements if attrs.get("id") == "statusText")
    assert status.get("aria-live") == "polite", "engine status changes are not announced"
    assert "560 tests" not in index and "six endpoints" not in index
    assert "hard-deleted after 60 minutes" not in index, (
        "site over-promises restart-durable retention that main does not provide"
    )
    assert "not a production retention guarantee" in index
    assert any(attrs.get("id") == "releaseMeta" for _, attrs in elements)
    assert any(attrs.get("id") == "releaseCommit" for _, attrs in elements)
    for section in ("architecture-title", "result-title", "highlights-title"):
        assert any(tag == "h2" and attrs.get("id") == section for tag, attrs in elements), (
            f"missing public overview section {section}"
        )
    for table in ("regressionRows", "historicalRows"):
        assert any(tag == "tbody" and attrs.get("id") == table for tag, attrs in elements), (
            f"missing separate comparison table {table}"
        )
    for _, attrs in elements:
        if (attrs.get("href") or "").startswith("#"):
            assert attrs["href"][1:] in ids, f"broken section link: {attrs['href']}"
    assert any((attrs.get("href") or "").startswith("./release.json") for _, attrs in elements), (
        "release manifest is not linked from the evidence block"
    )

    live_buttons = [attrs for _, attrs in elements if "data-live" in attrs]
    assert live_buttons, "no live scenarios"
    assert all(attrs.get("aria-pressed") == "false" for attrs in live_buttons), (
        "live scenario selection has no initial accessible state"
    )

    assert "two-stage LLM extraction" in index and "t4 answers" in index
    assert "never a rule-based substitute" in index, "missing honest unavailable-model state"
    assert any(attrs.get("id") == "modelInputs" for _, attrs in elements)
    assert any(attrs.get("id") == "sources" for _, attrs in elements)
    # The headline is one sentence, and the sentence that qualifies it must be present.
    assert any(attrs.get("id") == "headline" for _, attrs in elements), "no headline result"
    assert any(attrs.get("id") == "footnote" for _, attrs in elements), (
        "the headline result has no qualifying footnote"
    )

    for _, attrs in elements:
        asset = attrs.get("href") or attrs.get("src") or attrs.get("poster")
        if asset and asset.startswith("./") and not asset.startswith("./#"):
            target = asset.removeprefix("./").split("?", 1)[0].split("#", 1)[0]
            assert (SITE / target).exists(), f"missing local asset: {asset}"

    for identity in ("historyDetails", "selectionDetails", "promptDetails"):
        attrs = next(
            attrs for tag, attrs in elements if tag == "details" and attrs.get("id") == identity
        )
        assert "open" not in attrs, f"advanced inspector panel {identity} must start collapsed"
    video = next(attrs for tag, attrs in elements if tag == "video")
    assert "controls" in video and "autoplay" not in video and video.get("preload") == "none"
    assert "recorded run" in index.lower() and "not current service status" in index
    import hashlib

    recording = json.loads((SITE / "recording.json").read_text(encoding="utf-8"))
    assert recording["evidence_class"] == "functional-demo-recording"
    assert recording["transcript"] and recording["usage"]["calls"] > 0
    assert recording["usage"]["failed_calls"] >= 0
    for name, digest in recording["assets"].items():
        assert (
            name in EXPECTED_FILES
            and hashlib.sha256((SITE / name).read_bytes()).hexdigest() == digest
        ), f"recording asset differs from its provenance: {name}"
    assert 0 < (SITE / "demo-run.mp4").stat().st_size < 10_000_000

    release = json.loads((SITE / "release.json").read_text(encoding="utf-8"))
    schema = json.loads((SITE / "release.schema.json").read_text(encoding="utf-8"))
    assert release["$schema"] == "./release.schema.json"
    assert schema["properties"]["schema_version"]["const"] == release["schema_version"]
    assert release["schema_version"] == 2
    assert release["release"], "release manifest has no public release id"
    engineering = release["engineering"]
    assert engineering["tests"] > 0
    # Schema 1 counted one case per tracked document inside `tests`, so the figure rose
    # when documents were committed and said nothing about the code. They are separate
    # from schema 2 on, and a page that showed one under the other's label would be
    # restating the same defect in a new place.
    assert engineering["document_checks"] >= 0
    # A figure the public repository cannot reproduce is not publishable. The digest is
    # what lets a reader check that the tree these numbers came from is the tree in front
    # of them — and unlike a commit SHA it survives an amend, a squash or a rebase.
    assert re.fullmatch(r"[0-9a-f]{64}", engineering["measured_tree_sha256"] or ""), (
        "the release manifest has no usable measured-tree digest"
    )
    reference = engineering.get("source_commit_for_reference")
    assert reference is None or re.fullmatch(r"[0-9a-f]{7,40}", reference)
    assert "source_commit" not in engineering, (
        "a bare source_commit invites reliance on a SHA a history rewrite can orphan"
    )

    result = release["result"]
    for key in ("accuracy", "naive_rag_accuracy"):
        assert 0 <= result[key] <= 100, f"{key} is not a percentage"
    assert result["baseline_run_on_heldout"] is False
    # Both arms must come from the same run on the same questions. A headline comparing
    # this system's one-shot result against a baseline measured elsewhere would be the
    # easiest number on the page to get wrong.
    assert result["one_shot"] is True
    assert result["questions"] > 0
    assert result["median_context_tokens"] >= 0
    assert result["naive_rag_median_context_tokens"] >= 0
    # Manifest and schema must agree on the required shape, so a field the page reads
    # cannot quietly disappear.
    for key in schema["properties"]["result"]["required"]:
        assert key in result, f"release manifest is missing required result field {key!r}"

    regression = release.get("regression")
    if regression is not None:
        for key in schema["properties"]["regression"]["required"]:
            assert key in regression, f"missing regression field {key!r}"
        assert regression["evidence_class"] == "exposed-regression"
        assert regression["gate_passed"] is True
        assert regression["reader_rows"] == regression["grade_rows"] == 600
        assert regression["questions"] == 100 and regression["repeats_per_arm"] == 3
        assert any(attrs.get("id") == "historicalResult" for _, attrs in elements)
        for arm in ("control", "candidate"):
            values = regression[f"{arm}_correct"]
            assert len(values) == 3 and all(type(n) is int and 0 <= n <= 100 for n in values)
            assert abs(sum(values) / 3 - regression[f"{arm}_accuracy"]) < 1e-8
        assert (
            abs(
                regression["candidate_accuracy"]
                - regression["control_accuracy"]
                - regression["mean_difference_pp"]
            )
            < 1e-8
        )

    app = (SITE / "app.js").read_text(encoding="utf-8")
    assert "renderMetrics();" in app, "language changes do not re-render manifest copy"
    # No measured figure may be typed into the page. Everything shown comes from the
    # manifest, which records the commit it was measured at.
    for figure in (
        "72%",
        "574",
        "65%",
        "12,763",
        "12763",
        "109,059",
        "86%",
        "68.67",
        "17.33",
        "2313",
        "2,313",
        "89%",
        "476,864",
        "2,465,965",
    ):
        assert figure not in app, f"measured figure {figure!r} hard-coded outside release.json"
        assert figure not in index, f"measured figure {figure!r} hard-coded in index.html"
    # Whether the gap is significant is a claim, so it is driven by the manifest rather
    # than written once and left behind by a future release that is significant.
    assert "significant_vs_naive_rag" in app, (
        "the significance footnote is not driven by the release manifest"
    )
    assert "full_context_accuracy" in app and "vs_naive_rag_p_value" in app, (
        "the comparison or its p-value is missing from the public result"
    )
    assert "median_context_tokens" not in app, (
        "the page must not compare context estimates with provider input tokens"
    )
    for field in (
        "engineering.tests",
        "engineering.line_coverage",
        "engineering.ci_platforms",
        "engineering.verified_at",
        "engineering.document_checks",
        "regression.control_correct",
        "regression.candidate_correct",
    ):
        assert field in app, f"public overview does not read manifest field {field}"
    scenario_body = re.search(
        r"async function runLiveScenario\(name\) \{(?P<body>.*?)\n\}", app, re.DOTALL
    )
    assert scenario_body, "runLiveScenario is missing"
    body = scenario_body.group("body")
    assert body.index("await freshSession();") < body.index("await advanceScenario();")
    assert '"/demo/messages"' in app and '"/demo/answer"' in app
    assert '"/demo/facts"' not in app and "PATTERNS" not in app, "UI bypasses LLM extraction"
    assert 'result.mode !== "llm"' in app and 'result.answer_profile !== "t4"' in app
    assert "health.llm_enabled" in app, "UI enables generation without a live backend"
    assert "session_id: conversationId()" in app
    # A sleeping backend is waking, not down, and its HTML error pages are not JSON.
    assert 'status("waking"' in app, "connecting backend is shown as down"
    assert 'get("content-type")' in app, "non-JSON backend responses reach the visitor raw"
    # A refresh must return to the same namespace, and deletion must forget it.
    assert "sessionStorage.setItem" in app and "sessionStorage.removeItem" in app
    # Dated conversations are the session time sent to the extractor, and as_of reads
    # stored intervals; neither may be computed in the page.
    assert "observed_on" in app and "/demo/as_of" in app, "conversation dates or as_of missing"
    assert "recovered_turns" in app and "not_selected" in app, "answer evidence is not shown"
    # Only an overloaded provider is retried, and only through the idempotent write key.
    assert 'retryOverloaded(() => api("POST", "/demo/messages", pendingWrite))' in app
    assert "input.prompt" in app and "pre.textContent" in app, "model input is not shown safely"
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    narrow = re.search(r"@media \(max-width: 920px\) \{(?P<body>.*?)\n\}", css, re.DOTALL)
    assert narrow, "no narrow-viewport rules"
    # The two panes must stack rather than shrink. A memory row that wraps mid-key stops
    # reading as a key/value pair, which is the one thing the state panel exists to show.
    assert ".split { grid-template-columns: 1fr; }" in narrow.group("body"), (
        "narrow layout does not stack the conversation and state panes"
    )
    # A retired fact has to look retired, or the page shows a list and claims a timeline.
    assert ".row.superseded" in css, "superseded facts are not visually distinguished"

    content = (SITE / "content.js").read_text(encoding="utf-8")
    assert "SCENARIOS" in content and all(
        name in content for name in ("recall:", "update:", "temporal:")
    )
    assert "PATTERNS" not in content and "predicate:" not in content and "object:" not in content
    assert "zh:" in content and "en:" in content
    print(f"public site OK: {len(files)} deploy files, {len(fields)} labelled fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
