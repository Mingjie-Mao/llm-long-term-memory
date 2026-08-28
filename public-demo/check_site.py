"""Fail fast when the buildless public site drifts or loses basic semantics."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE / "site"
EXPECTED_FILES = {
    "app.js",
    "content.js",
    "index.html",
    "playground.html",
    "release.json",
    "styles.css",
}


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def main() -> int:
    files = {path.name for path in SITE.iterdir() if path.is_file()}
    assert files == EXPECTED_FILES, f"unexpected deploy surface: {sorted(files)}"

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
        if "data-requires-live" in attrs:
            assert "disabled" in attrs, f"{tag} live control is enabled before session issuance"

    status = next(attrs for _, attrs in elements if attrs.get("id") == "statusText")
    assert status.get("aria-live") == "polite", "engine status changes are not announced"
    assert "560 tests" not in index and "six endpoints" not in index

    for _, attrs in elements:
        asset = attrs.get("href") or attrs.get("src")
        if asset and asset.startswith("./") and not asset.startswith("./#"):
            target = asset.removeprefix("./").split("?", 1)[0].split("#", 1)[0]
            assert (SITE / target).exists(), f"missing local asset: {asset}"

    release = json.loads((SITE / "release.json").read_text(encoding="utf-8"))
    assert release["schema_version"] == 1
    assert release["engineering"]["tests"] > 0
    # A figure the public repository cannot reproduce is not publishable. The commit is
    # what lets a reader re-run the suite and get the number the page shows.
    assert re.fullmatch(r"[0-9a-f]{7,40}", release["engineering"]["source_commit"] or "")
    assert 0 <= release["heldout"]["accuracy"] <= 100
    assert release["heldout"]["baseline_run_on_heldout"] is False

    content = (SITE / "content.js").read_text(encoding="utf-8")
    for scenario in ("changed", "detail", "multi"):
        assert f"  {scenario}:" in content, f"missing bilingual scenario: {scenario}"
    assert "zh:" in content and "en:" in content

    # The live pane claims to show what the engine was sent. The engine is sent English,
    # so a translated bubble would misrepresent the request; Chinese belongs in `gloss`.
    sent = re.findall(r'^\s*text:\s*"([^"]*)"', content, re.MULTILINE)
    assert sent, "no live conversation payloads found"
    cjk = [line for line in sent if re.search(r"[\u4e00-\u9fff]", line)]
    assert not cjk, f"live conversation shows text the engine never receives: {cjk}"
    assert "gloss:" in content, "Chinese readers lose the conversation without a gloss"

    print(f"public site OK: {len(files)} deploy files, {len(fields)} labelled fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
