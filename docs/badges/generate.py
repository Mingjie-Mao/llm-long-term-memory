"""Rebuild the README language buttons.

    python3 docs/badges/generate.py

Four files, because a two-button switch needs each button in both states: the language
you are reading is filled, the one you can switch to is grey. `README.md` uses
`lang-en-active` + `lang-zh-idle`; `README.zh-CN.md` uses the other pair.

Drawn here rather than fetched from a badge service for the same reason the architecture
atlas is: the repository renders its own images. A shields.io URL would put the top line
of the README behind a third-party request, and the button would break the day that
service changes its API or goes down.

No dependencies. Deterministic: running this twice writes identical bytes.
"""

from __future__ import annotations

import html
from pathlib import Path

HERE = Path(__file__).resolve().parent

ACTIVE = "#2563EB"
IDLE = "#555555"
LABEL = "#FFFFFF"
HEIGHT = 28
PAD = 14
# Advance widths at the 11px used below, measured against the fallback stack rather than a
# specific font: Latin capitals are close enough to uniform at this size, and CJK is square.
LATIN_ADVANCE = 8.1
CJK_ADVANCE = 13.0
TRACKING = 1.1
FONT = (
    "-apple-system, BlinkMacSystemFont, Segoe UI, PingFang SC, Hiragino Sans GB, "
    "Microsoft YaHei, Noto Sans CJK SC, Helvetica, Arial, sans-serif"
)


def width_of(text: str) -> float:
    """Rendered width, so the box fits its label without a measuring pass in a browser."""
    total = 0.0
    for character in text:
        total += CJK_ADVANCE if ord(character) > 0x2E7F else LATIN_ADVANCE
        total += TRACKING
    return total - TRACKING


def badge(text: str, *, active: bool, title: str) -> str:
    width = round(width_of(text) + PAD * 2)
    # `title` is what a screen reader announces and what GitHub shows on hover; the README
    # also carries alt text, so a reader with images off still sees both languages.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{HEIGHT}" '
        f'viewBox="0 0 {width} {HEIGHT}" role="img" aria-label="{html.escape(title)}">'
        f"<title>{html.escape(title)}</title>"
        f'<rect width="{width}" height="{HEIGHT}" fill="{ACTIVE if active else IDLE}"/>'
        f'<text x="{width / 2:.1f}" y="{HEIGHT / 2:.1f}" fill="{LABEL}" '
        f'font-family="{FONT}" font-size="11" font-weight="700" '
        f'letter-spacing="{TRACKING}" text-anchor="middle" dominant-baseline="central">'
        f"{html.escape(text)}</text>"
        f"</svg>\n"
    )


BUTTONS = {
    "lang-en-active.svg": ("ENGLISH", True, "English — the page you are reading"),
    "lang-en-idle.svg": ("ENGLISH", False, "Switch to English"),
    "lang-zh-active.svg": ("中文", True, "中文 — 当前阅读的版本"),
    "lang-zh-idle.svg": ("中文", False, "切换到中文"),
}


def main() -> int:
    for name, (text, active, title) in BUTTONS.items():
        (HERE / name).write_text(badge(text, active=active, title=title), encoding="utf-8")
        print(f"wrote docs/badges/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
