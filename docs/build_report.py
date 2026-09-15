"""Render the current Chinese report and its data-backed comparison figure.

Run with the project environment: .venv/bin/python docs/build_report.py
markdown-it-py is already installed through the project's Rich dependency.
No provider calls, browser downloads, or external assets are needed.
"""

# Chinese editorial copy intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPORT = DOCS / "PROJECT_REPORT.zh-CN.md"
OUTPUT = DOCS / "PROJECT_REPORT.zh-CN.html"
FONT = "PingFang SC, Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif"


def chart() -> str:
    aggregate = json.loads(
        (ROOT / "results/final/test100-aggregate.json").read_text(encoding="utf-8")
    )
    arms = {a["arm"]: a for a in aggregate["arms"]}
    entries = [
        ("v2", "v2 记忆 + 回退", "#246C5E"),
        ("full_context", "整段历史", "#355D8A"),
        ("naive_rag", "naive RAG", "#8291A2"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="470" '
        'viewBox="0 0 1180 470" role="img" aria-labelledby="title desc">',
        '<title id="title">v2 正式终测：正确率与回答上下文</title>',
        '<desc id="desc">100 题，每臂一次。左侧正确率使用零起点线性轴。'
        "右侧列出上下文 token 原始数值，不共用正确率坐标。"
        "记忆与 RAG 上下文为字符估算，全文为 provider 输入 token。</desc>",
        '<rect width="1180" height="470" fill="#FAFBFC"/>',
    ]

    def text(x, y, value, size=19, fill="#223448", bold=False):
        parts.append(
            f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}" fill="{fill}">{html.escape(value)}</text>'
        )

    text(32, 42, "同一组 100 题，三个取舍", 25, bold=True)
    text(32, 73, "冻结 v2 · 每臂一次 · 未包含共享抽取成本", 17, "#627184")
    text(235, 116, "回答正确率", 18, bold=True)
    text(818, 116, "中位回答上下文", 18, bold=True)
    for tick in (0, 25, 50, 75, 100):
        x = 235 + tick * 4.2
        parts.append(f'<path d="M{x} 132V332" stroke="#DCE3EB"/>')
        text(x - 9, 360, f"{tick}%", 15, "#627184")
    for index, (key, label, color) in enumerate(entries):
        arm = arms[key]
        y = 143 + index * 67
        text(32, y + 26, label, 20, color, bold=True)
        width = arm["accuracy"] * 420
        parts.append(f'<rect x="235" y="{y}" width="{width}" height="34" rx="3" fill="{color}"/>')
        text(235 + width + 12, y + 25, f"{arm['accuracy']:.0%}", 21, color, bold=True)
        text(818, y + 26, f"{arm['median_context_tokens']:,.0f} token", 24, color, bold=True)
        text(1030, y + 25, "provider" if key == "full_context" else "字符估算", 16, "#627184")
    text(32, 409, "v2 与 naive RAG 的 +7 个百分点尚不显著；整段历史更准确。", 18)
    text(32, 440, "上下文口径不同：比值只能近似比较，不直接等于账单节省。", 17, "#627184")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


CSS = """
:root{color-scheme:light;--ink:#223448;--muted:#637184;--line:#dce3e8;--accent:#246c5e}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:35px}
body{margin:0;background:#f5f4f0;color:var(--ink);font:17px/1.9 -apple-system,
BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
a{color:#286c79;text-underline-offset:4px}a:hover{color:#143f4a}
.top{padding:22px 4vw;border-bottom:1px solid var(--line);display:flex;
justify-content:space-between;
gap:20px;font:13px/1.5 ui-monospace,monospace;letter-spacing:.08em;background:#fff}
.layout{display:grid;grid-template-columns:210px minmax(0,1140px);gap:50px;max-width:1490px;
margin:50px auto;padding:0 32px}.toc{position:sticky;top:30px;align-self:start;font-size:14px}
.toc strong{display:block;color:var(--muted);font-size:12px;letter-spacing:.12em;margin-bottom:15px}
.toc a{display:block;text-decoration:none;color:var(--muted);border-left:2px solid var(--line);
padding:6px 14px}.toc a:hover{border-color:var(--accent);color:var(--accent)}
article{background:#fff;padding:55px 60px 70px;min-width:0;box-shadow:0 4px 28px #24354a05}
h1{font-size:46px;line-height:1.3;letter-spacing:-.04em;font-weight:650;margin:0 0 23px}
h2{font-size:28px;font-weight:650;line-height:1.5;margin:60px 0 22px;padding-top:24px;
border-top:1px solid var(--line)}h3{font-size:21px;font-weight:650;margin:32px 0 12px}
p{margin:0 0 21px}article>p:first-of-type{font-size:13px;color:var(--muted);letter-spacing:.05em}
strong{font-weight:650}ul,ol{padding-left:24px}li{margin:7px 0}
.figure{margin:30px -34px;padding:12px;background:#fafbfc;border:1px solid #edf0f3}
.figure-viewport{overflow:auto}.figure svg{display:block;width:100%;height:auto;min-width:680px}
.figure label{font-size:13px;color:var(--accent);cursor:pointer}.figure input{margin-right:8px}
.figure input:checked~.figure-viewport svg{width:2600px;max-width:none}
.figure figcaption{font-size:13px;color:var(--muted);
padding:9px 4px 2px}.table-wrap{overflow:auto;margin:25px 0 32px}table{border-collapse:collapse;
width:100%;font-size:14px;line-height:1.75}th{background:#edf3f1;text-align:left;font-weight:650}
td,th{padding:12px 13px;border-bottom:1px solid var(--line);vertical-align:top}
td:first-child{min-width:95px}tbody tr:nth-child(even){background:#fafbfc}
pre{overflow:auto;padding:22px;background:#edf1f4;border-left:3px solid #8296ad;font-size:14px;
line-height:1.8}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em}
:not(pre)>code{background:#f0f3f5;padding:2px 5px;border-radius:3px}blockquote{margin:25px 0;
border-left:3px solid var(--accent);padding-left:20px;color:var(--muted)}
footer{max-width:1140px;margin:0 auto 50px;padding:0 32px;font-size:13px;color:var(--muted)}
@media(max-width:1000px){.layout{display:block;margin:25px auto;padding:0 18px}.toc{position:static;
margin-bottom:25px}.toc a{display:inline-block;padding:4px 12px}.toc strong{margin-bottom:8px}
article{padding:30px 25px}.figure{margin-left:-12px;margin-right:-12px}h1{font-size:36px}}
@media(max-width:600px){body{font-size:16px}.top{font-size:11px}.layout{padding:0 10px}
article{padding:28px 18px}h1{font-size:31px}h2{font-size:24px}td,th{padding:10px}}
@media print{body{background:#fff;font-size:11pt}.top,.toc,.figure summary{display:none}
.layout{display:block;margin:0;padding:0}article{box-shadow:none;padding:0}.figure{margin:18px 0;
overflow:visible;break-inside:avoid}.figure svg{min-width:0!important;width:100%!important}
h1{font-size:28pt}h2{font-size:19pt;break-after:avoid}h3{break-after:avoid}table{font-size:9pt}}
"""


def build() -> None:
    (DOCS / "figures/benchmark-v2.svg").write_text(chart(), encoding="utf-8")
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = markdown.parse(REPORT.read_text(encoding="utf-8"))
    toc = []
    for index, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h2":
            label = tokens[index + 1].content
            anchor = "section-" + str(len(toc) + 1)
            token.attrSet("id", anchor)
            toc.append(f'<a href="#{anchor}">{html.escape(label)}</a>')
    body = markdown.renderer.render(tokens, markdown.options, {})

    def inline_svg(match):
        name, alt = match.groups()
        path = (DOCS / name).resolve()
        if not path.is_relative_to(DOCS / "figures") or path.suffix != ".svg":
            raise ValueError(f"Unexpected report image: {name}")
        svg = path.read_text(encoding="utf-8")
        prefix = path.stem.replace(".", "-")
        svg = svg.replace('id="title"', f'id="{prefix}-title"')
        svg = svg.replace('id="desc"', f'id="{prefix}-desc"')
        svg = svg.replace(
            'aria-labelledby="title desc"', f'aria-labelledby="{prefix}-title {prefix}-desc"'
        )
        return (
            f'<figure class="figure"><input type="checkbox" id="zoom-{prefix}">'
            f'<label for="zoom-{prefix}">按原尺寸展开（可横向滚动）</label>'
            f'<div class="figure-viewport">{svg}</div><figcaption>{alt}</figcaption></figure>'
        )

    body = re.sub(r'<p><img src="([^"]+\.svg)" alt="([^"]*)"\s*/?></p>', inline_svg, body)
    body = body.replace("<table>", '<div class="table-wrap"><table>')
    body = body.replace("</table>", "</table></div>")
    document = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light">'
        "<title>让对话留下可用的记忆 · 项目报告</title>"
        f"<style>{CSS}</style></head><body>"
        '<header class="top"><span>LLTM / PROJECT REPORT</span><span>2026.09.13</span></header>'
        '<main class="layout"><nav class="toc" aria-label="报告目录"><strong>阅读目录</strong>'
        + "".join(toc)
        + "</nav><article>"
        + body
        + "</article></main>"
        "<footer>图表来自仓库中的冻结结果。本文没有发起新的模型实验。"
        "可编辑正文：PROJECT_REPORT.zh-CN.md</footer></body></html>\n"
    )
    OUTPUT.write_text(document, encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    build()
