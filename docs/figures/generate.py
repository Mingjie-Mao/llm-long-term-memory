"""Rebuild the architecture atlas without importing or executing the memory engine.

SVG (six figures, with a Chinese overview variant): python3 docs/figures/generate.py
SVG + vector PDF: python3 docs/figures/generate.py --pdf
Only the optional PDF export needs reportlab. All coordinates use a shared scene,
so SVG and PDF have the same layout. The PDF is written to output/pdf/.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

from overview import build_overview

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INK = "#203247"
MUTED = "#586775"
LINE = "#8795A1"
BORDER = "#CED6DC"
BLUE = "#345E80"
TEAL = "#32766D"
AMBER = "#96662D"
BLUE_BG = "#EEF4F8"
TEAL_BG = "#EFF6F3"
AMBER_BG = "#FBF5EB"
GRAY_BG = "#F6F7F8"


class Figure:
    def __init__(self, number, name, title, subtitle, height, *, width=1200, header=True):
        self.name, self.title, self.height = name, title, height
        self.width = width
        self.description = (
            "Architecture verified against the repository on 2026-09-07. "
            "See docs/ARCHITECTURE.md for the detailed figure caption and source mapping. "
            "Solid arrows show flow; dashed arrows show conditional paths."
        )
        self.ops = []
        self.rect(0, 0, width, height, fill="#FFFFFF", stroke=None, radius=7 if header else 0)
        if header:
            self.text(40, 42, f"FIGURE {number:02d}", 15, BLUE, bold=True)
            self.text(40, 82, title, 31, serif=True)
            self.text(40, 113, subtitle, 17, MUTED)
            self.path([(40, 133), (width - 40, 133)], BORDER, 1)

    def text(
        self,
        x,
        y,
        value,
        size=18,
        color=INK,
        *,
        bold=False,
        serif=False,
        mono=False,
        anchor="start",
    ):
        self.ops.append(
            (
                "text",
                dict(
                    x=x,
                    y=y,
                    value=value,
                    size=size,
                    color=color,
                    bold=bold,
                    serif=serif,
                    mono=mono,
                    anchor=anchor,
                ),
            )
        )

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke=BORDER, radius=7, dash=False):
        self.ops.append(
            ("rect", dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, radius=radius, dash=dash))
        )

    def path(self, points, color=LINE, width=1.6, dash=False, arrow=False):
        self.ops.append(("path", dict(points=points, color=color, width=width, dash=dash)))
        if arrow:
            x, y = points[-1]
            px, py = points[-2]
            a = math.atan2(y - py, x - px)
            size = 8
            triangle = [
                (x, y),
                (
                    x - size * math.cos(a) + 3.7 * math.sin(a),
                    y - size * math.sin(a) - 3.7 * math.cos(a),
                ),
                (
                    x - size * math.cos(a) - 3.7 * math.sin(a),
                    y - size * math.sin(a) + 3.7 * math.cos(a),
                ),
            ]
            self.ops.append(("polygon", dict(points=triangle, fill=color)))

    def arrow(self, points, color=LINE, dash=False):
        self.path(points, color, 1.7, dash, arrow=True)

    def box(self, x, y, w, h, title, lines=(), *, accent=BLUE, fill=BLUE_BG, dashed=False, size=18):
        self.rect(x, y, w, h, fill=fill, stroke=accent, dash=dashed)
        self.text(x + 16, y + 29, title, 19, accent, bold=True)
        for i, line in enumerate(lines):
            self.text(x + 16, y + 57 + 24 * i, line, size)

    def label(self, x, y, value, color=MUTED):
        self.text(x, y, value, 16, color)

    def section(self, x, y, value):
        self.text(x, y, value, 17, MUTED, bold=True)

    def footer(self, *lines):
        base = self.height - 31 - 23 * (len(lines) - 1)
        self.path([(40, base - 23), (1160, base - 23)], BORDER, 1)
        for i, line in enumerate(lines):
            self.text(40, base + 23 * i, line, 16, MUTED)

    def svg(self):
        esc = html.escape
        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{esc(self.title)}</title>',
            f'<desc id="desc">{esc(self.description)}</desc>',
        ]
        for kind, a in self.ops:
            if kind == "rect":
                dash = ' stroke-dasharray="7 5"' if a["dash"] else ""
                out.append(
                    f'<rect x="{a["x"]}" y="{a["y"]}" width="{a["w"]}" height="{a["h"]}" '
                    f'rx="{a["radius"]}" fill="{a["fill"]}" stroke="{a["stroke"] or "none"}" '
                    f'stroke-width="1.3"{dash}/>'
                )
            elif kind == "text":
                family = (
                    "Georgia, Times New Roman, serif"
                    if a["serif"]
                    else "Courier New, monospace"
                    if a["mono"]
                    else "Helvetica, Arial, sans-serif"
                )
                if any("\u3400" <= char <= "\u9fff" for char in a["value"]):
                    family = "PingFang SC, Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif"
                out.append(
                    f'<text x="{a["x"]}" y="{a["y"]}" font-family="{family}" '
                    f'font-size="{a["size"]}" font-weight="{700 if a["bold"] else 400}" '
                    f'fill="{a["color"]}" text-anchor="{a["anchor"]}">{esc(a["value"])}</text>'
                )
            elif kind == "path":
                pts = " ".join(f"{x},{y}" for x, y in a["points"])
                dash = ' stroke-dasharray="7 5"' if a["dash"] else ""
                out.append(
                    f'<polyline points="{pts}" fill="none" stroke="{a["color"]}" '
                    f'stroke-width="{a["width"]}" stroke-linejoin="round"{dash}/>'
                )
            elif kind == "polygon":
                pts = " ".join(f"{x},{y}" for x, y in a["points"])
                out.append(f'<polygon points="{pts}" fill="{a["fill"]}"/>')
        return "\n".join([*out, "</svg>", ""])


def ingestion():
    f = Figure(
        2,
        "write-path",
        "From conversation sessions to versioned facts",
        (
            "The batch pipeline: extraction, source registration, deduplication, "
            "persistence, then temporal resolution."
        ),
        830,
    )
    f.section(40, 175, "A. EXTRACT  /  ONE NAMESPACE PER BATCH")
    specs = [
        (40, "1  Batch sessions", ["Skip completed work", "Check ingest fingerprint"]),
        (270, "2  Stage A", ["Fact lines + speaker", "Subject + scope"]),
        (500, "3  Stage B", ["Temporal key + object", "Update operation"]),
        (730, "4  Rules", ["Type, entities, priority", "Date from session"]),
        (960, "5  Source anchors", ["Best lexical sentence", "Turn + character span"]),
    ]
    for x, title, lines in specs:
        f.box(x, 202, 200, 126, title, lines, size=16)
    for a in [240, 470, 700, 930]:
        f.arrow([(a, 265), (a + 30, 265)])
    f.label(286, 352, "1 LLM request", BLUE)
    f.label(516, 352, "+1 if facts exist", BLUE)
    f.label(744, 352, "No LLM request")
    f.section(40, 411, "B. STORE AND UPDATE  /  AFTER EXTRACTION RETURNS")
    f.box(
        40,
        441,
        200,
        151,
        "6  Archive source",
        ["sessions + turns", "FTS5 triggers update", "Scoped session IDs"],
        accent=AMBER,
        fill=AMBER_BG,
        size=16,
    )
    f.box(
        270,
        441,
        200,
        151,
        "7  Deduplicate",
        ["Embed candidates", "Find close / keyed facts", "LLM judges neighbours"],
        size=16,
    )
    f.box(
        500,
        441,
        200,
        151,
        "8  Persist facts",
        ["INSERT memory rows", "Add normalized vectors", "Keep new / distinct facts"],
        accent=TEAL,
        fill=TEAL_BG,
        size=16,
    )
    f.box(
        730,
        441,
        200,
        151,
        "9  Resolve keys",
        ["Read active + history", "Sort by event_time", "Rewrite validity windows"],
        accent=TEAL,
        fill=TEAL_BG,
        size=16,
    )
    f.box(
        960,
        441,
        200,
        151,
        "10  Checkpoint",
        ["Save vector sidecars", "Save session progress", "Resume after quota stop"],
        accent=TEAL,
        fill=TEAL_BG,
        size=16,
    )
    f.arrow([(1060, 328), (1060, 386), (20, 386), (20, 516), (40, 516)])
    for a in [240, 470, 700, 930]:
        f.arrow([(a, 516), (a + 30, 516)])
    f.rect(40, 635, 1120, 101, fill=GRAY_BG)
    f.text(59, 664, "Cost and failure semantics", 18, bold=True)
    f.text(
        59,
        694,
        (
            "Two extraction requests when Stage A yields facts; near-duplicate "
            "adjudication can add requests."
        ),
        18,
    )
    f.text(
        59,
        720,
        "A similarity of 0.92 nominates neighbours. Only a DUPLICATE verdict drops the new fact.",
        18,
    )
    f.footer(
        (
            "Content-blocked batches retain raw turns and record their blocked "
            "status. Other failed batches remain pending."
        ),
        (
            "Temporal resolution follows memory insertion. Vector files and SQLite "
            "do not share a single atomic transaction."
        ),
    )
    return f


def storage():
    f = Figure(
        3,
        "storage",
        "A relational source of truth with vector sidecars",
        "Physical names and relationships from schema.sql, SQLiteMemoryStore, and NumpyFlatIndex.",
        900,
    )
    f.rect(40, 165, 800, 570, fill="#FFFFFF", stroke=BLUE)
    f.section(62, 198, "SQLITE  /  <store>.db  /  WAL + FOREIGN KEYS")
    f.box(64, 224, 326, 110, "sessions", ["PK id  |  user_id", "started_at  |  source"])
    f.box(464, 224, 350, 110, "turns", ["PK id  |  FK session_id", "turn_index, role, content, ts"])
    f.arrow([(390, 270), (464, 270)])
    f.label(407, 255, "1 : N")
    f.box(
        64,
        397,
        400,
        284,
        "memories",
        [
            "PK id  |  user_id  |  content",
            "subject, predicate, object",
            "source_role, scope, type",
            "event_time, valid_from, valid_to",
            "ingested_at, update_op, status",
            "superseded_by  ->  memories.id",
            "source_session_id  ->  sessions.id",
            "source_turn_index + char offsets",
            "importance, strength, token_count",
        ],
        size=17,
    )
    f.arrow([(208, 397), (208, 334)])
    f.label(222, 369, "session FK")
    f.arrow([(464, 418), (492, 418), (492, 360), (638, 360), (638, 334)], dash=True)
    f.label(507, 389, "turn/span lookup")
    f.box(
        539,
        436,
        275,
        94,
        "turns_fts",
        ["BM25 over original turns"],
        accent=AMBER,
        fill=AMBER_BG,
        size=17,
    )
    f.arrow([(784, 334), (784, 436)], AMBER)
    f.label(688, 414, "triggers", AMBER)
    f.box(
        539,
        587,
        275,
        94,
        "memories_fts",
        ["BM25 over fact content"],
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(464, 635), (539, 635)], TEAL)
    f.label(473, 614, "triggers", TEAL)
    f.label(64, 711, "Auxiliary: entities + memory_entities  |  evidence  |  meta")
    f.rect(869, 165, 291, 570, fill=TEAL_BG, stroke=TEAL)
    f.section(890, 198, "VECTOR SIDECARS")
    f.text(890, 240, "<store>-index.npy", 19, TEAL, bold=True)
    f.text(890, 275, "N x 384 float32", 19)
    f.text(890, 307, "L2-normalized vectors", 18)
    f.text(890, 339, "Exact inner-product scan", 18)
    f.path([(890, 370), (1138, 370)], BORDER, 1)
    f.text(890, 409, "<store>-index.ids.json", 18, TEAL, bold=True)
    f.text(890, 444, "Row position -> memory ID", 18)
    f.text(890, 476, "Application-managed link", 18)
    f.text(890, 508, "No SQL foreign key", 18)
    f.path([(890, 538), (1138, 538)], BORDER, 1)
    f.text(890, 576, "Same store directory", 18, bold=True)
    f.text(890, 608, "Separate save operation", 18)
    f.text(890, 640, "Status filtering at retrieval", 18)
    f.rect(40, 763, 1120, 58, fill=GRAY_BG)
    f.text(
        58,
        798,
        (
            "Source spans are heuristic, nullable anchors. The session link is a "
            "real foreign key; turn/offset links are not."
        ),
        18,
    )
    f.footer(
        "Regular supersession and forget update status. Migration cleanup and "
        "demo session deletion can hard-delete rows."
    )
    return f


def retrieval():
    f = Figure(
        4,
        "read-path",
        "Retrieve memory first; recover raw evidence on demand",
        (
            "Default memory-first route with raw_fallback enabled. Dashed paths run"
            " only when source recovery is needed."
        ),
        1010,
    )
    f.box(40, 193, 190, 125, "Question", ["user_id + query", "MiniLM embedding"], size=17)
    f.box(
        285,
        167,
        280,
        100,
        "Semantic candidates",
        ["Scan index, then scope/status", "Keep up to 50"],
        size=17,
    )
    f.box(
        285,
        294,
        280,
        100,
        "Lexical candidates",
        ["FTS5 + SQL scope/status", "Keep up to 50"],
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(230, 236), (255, 236), (255, 217), (285, 217)])
    f.arrow([(230, 281), (255, 281), (255, 344), (285, 344)], TEAL)
    f.box(
        620,
        193,
        250,
        151,
        "Union and score",
        ["Deduplicate memory IDs", "Up to 100 candidates", "Default weight: semantic 1"],
        size=17,
    )
    f.arrow([(565, 217), (620, 217)])
    f.arrow([(565, 344), (592, 344), (592, 313), (620, 313)], TEAL)
    f.box(
        915,
        193,
        245,
        151,
        "Select and render",
        ["Top-k ranked memories", "Group by scope / time", "Build answer context"],
        size=17,
    )
    f.arrow([(870, 267), (915, 267)])
    f.label(630, 373, "Other four score weights: 0")
    f.label(926, 373, "Service k=10; v2 eval k=20")
    f.arrow([(1140, 344), (1140, 424), (177, 424), (177, 480)])
    f.box(40, 480, 275, 296, "First LLM pass", ["Parse AnswerVerdict"], size=18)
    f.text(60, 590, "answer", 18, BLUE, mono=True)
    f.text(60, 661, "need_source", 18, AMBER, mono=True)
    f.text(60, 735, "no_evidence", 18, AMBER, mono=True)
    f.box(
        405,
        474,
        325,
        90,
        "Return answer",
        ["No raw retrieval or second pass"],
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(315, 581), (360, 581), (360, 519), (405, 519)], TEAL)
    f.box(
        405,
        601,
        325,
        113,
        "Source-aware recovery",
        ["Rank archive turns with BM25", "Choose local or archive-wide"],
        accent=AMBER,
        fill=AMBER_BG,
        size=17,
    )
    f.arrow([(315, 654), (405, 654)], AMBER, dash=True)
    f.box(
        405,
        751,
        325,
        90,
        "Archive-wide recovery",
        ["Search without selected anchors"],
        accent=AMBER,
        fill=AMBER_BG,
        size=17,
    )
    f.arrow([(315, 730), (362, 730), (362, 796), (405, 796)], AMBER, dash=True)
    f.box(
        840,
        601,
        320,
        113,
        "Any raw turns recovered?",
        ["At most 3 turns", "2,400 source characters"],
        accent=AMBER,
        fill=AMBER_BG,
        size=17,
    )
    f.arrow([(730, 654), (840, 654)], AMBER, dash=True)
    f.arrow([(730, 796), (785, 796), (785, 688), (840, 688)], AMBER, dash=True)
    f.box(
        840,
        474,
        320,
        90,
        "No evidence found",
        ["Return verdict text / abstain"],
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(1000, 601), (1000, 564)], AMBER, dash=True)
    f.label(1015, 586, "no", AMBER)
    f.box(
        840,
        783,
        320,
        103,
        "Second LLM pass",
        ["Answer from recovered source", "Return final prose"],
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(1000, 714), (1000, 783)], AMBER, dash=True)
    f.label(1015, 752, "yes", AMBER)
    f.footer(
        (
            "A need_source verdict can still choose archive-wide recovery if the "
            "best BM25 hit is outside local source sessions."
        ),
        (
            "Reranking, session-coherent context, packing, decay and adaptive "
            "hydration are optional branches; see the atlas captions."
        ),
        (
            "Limits shown are from the checked-in default configurations. A "
            "malformed first verdict has a separate parse-failure return."
        ),
    )
    return f


def temporal():
    f = Figure(
        5,
        "temporal",
        "Rebuild a fact timeline independently of arrival order",
        (
            "Illustrative replacement chain on one (user_id, subject, predicate) "
            "key; each successor explicitly replaces the prior value."
        ),
        770,
    )
    f.section(40, 175, "A. INGESTION ORDER  /  THE ORDER IN WHICH THE SYSTEM LEARNS")
    for x, title, lines in [
        (40, "Arrives first", ["August: Sydney"]),
        (330, "Arrives second", ["January: Canberra"]),
        (620, "Arrives third", ["March: Melbourne"]),
    ]:
        f.box(x, 199, 245, 89, title, lines)
    f.arrow([(285, 244), (330, 244)])
    f.arrow([(575, 244), (620, 244)])
    f.box(
        910,
        199,
        250,
        89,
        "Resolve the full key",
        ["Sort by event_time, then ID"],
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(865, 244), (910, 244)])
    f.section(40, 350, "B. VALIDITY ORDER  /  HALF-OPEN INTERVALS [START, END)")
    for x, date in [(355, "Jan 01"), (540, "Mar 01"), (900, "Aug 01")]:
        f.text(x, 391, date, 18, MUTED, anchor="middle")
        f.path([(x, 407), (x, 588)], BORDER, 1, dash=True)
    f.text(61, 440, "Canberra", 20, bold=True)
    f.rect(355, 418, 185, 38, fill=BLUE_BG, stroke=BLUE, radius=3)
    f.text(448, 444, "superseded", 17, BLUE, anchor="middle")
    f.text(61, 507, "Melbourne", 20, bold=True)
    f.rect(540, 485, 360, 38, fill=BLUE_BG, stroke=BLUE, radius=3)
    f.text(720, 511, "superseded", 17, BLUE, anchor="middle")
    f.text(61, 574, "Sydney", 20, bold=True)
    f.rect(900, 552, 215, 38, fill=TEAL_BG, stroke=TEAL, radius=3)
    f.text(993, 578, "active", 17, TEAL, anchor="middle")
    f.arrow([(1080, 571), (1140, 571)], TEAL)
    f.text(
        61,
        619,
        (
            "Late March evidence moves Canberra's valid_to from August back to "
            "March; the August fact stays current."
        ),
        18,
    )
    f.footer(
        (
            "Coexisting successors do not close the previous fact. Repeated values "
            "may fold into the earliest interval owner."
        ),
        (
            "Undated facts are left unresolved. event_time currently comes from the"
            " session date; ingested_at records arrival time."
        ),
        (
            "This is an explanatory example, not a benchmark result or a claim of "
            "full transaction-time history."
        ),
    )
    return f


def runtime():
    f = Figure(
        6,
        "runtime-evaluation",
        "Runtime entry points and the research evaluation boundary",
        (
            "Three adapters share components, but they do not execute the same "
            "write path or expose the same capabilities."
        ),
        1055,
    )
    f.section(40, 175, "A. CURRENT ENTRY POINTS")
    f.box(
        40,
        201,
        350,
        99,
        "CLI / Python batch pipeline",
        ["lltm ingest run", "IngestionPipeline"],
        size=18,
    )
    f.box(
        425,
        201,
        350,
        99,
        "REST + Inspector + MCP",
        ["api.app / mcp_server", "MemoryService"],
        size=18,
    )
    f.box(
        810,
        201,
        350,
        99,
        "Public playground",
        ["public-demo/site -> demo-api", "Playground"],
        accent=AMBER,
        fill=AMBER_BG,
        size=18,
    )
    f.box(
        40,
        341,
        350,
        165,
        "Full batch write path",
        [
            "TwoStageExtractor + dedup",
            "Archive + store + vectors",
            "TemporalResolver + checkpoint",
            "Uses provider for extraction",
        ],
        size=18,
    )
    f.box(
        425,
        341,
        350,
        165,
        "Service read / write path",
        [
            "Search, raw lookup, timeline",
            "Answer via MemoryRunner",
            "Lazy batch-extractor adapter",
            "Dedup + temporal resolution",
        ],
        size=18,
    )
    f.box(
        810,
        341,
        350,
        165,
        "Direct structured facts",
        [
            "Typed facts + raw turns",
            "Local embedder + resolver",
            "Signed temporary namespace",
            "No LLM extraction or answering",
        ],
        accent=AMBER,
        fill=AMBER_BG,
        size=18,
    )
    for x in [215, 600, 985]:
        f.arrow([(x, 300), (x, 341)])
    f.rect(40, 550, 1120, 72, fill=TEAL_BG, stroke=TEAL)
    f.text(60, 580, "Shared primitives", 19, TEAL, bold=True)
    f.text(
        60,
        607,
        (
            "MemoryStore / SQLite   |   NumpyFlatIndex   |   Encoder   |   "
            "HybridRetriever   |   TemporalResolver"
        ),
        19,
    )
    for x in [215, 600, 985]:
        f.arrow([(x, 506), (x, 550)], TEAL)
    f.section(40, 678, "B. EVALUATION  /  EXPLICIT RUNNERS, FROZEN INPUTS AND AUDITABLE OUTPUTS")
    specs = [
        (40, "Freeze inputs", ["Config + source hashes", "Manifests + store state"]),
        (270, "Run arms", ["Memory / full context", "Naive RAG baselines"]),
        (500, "Grade answers", ["Independent judge role", "Answer + reference"]),
        (730, "Record evidence", ["JSONL rows + usage", "Lock + resume ledger"]),
        (960, "Aggregate", ["Accuracy + cost + recall", "Paired checks + gates"]),
    ]
    for x, title, lines in specs:
        f.box(x, 712, 200, 132, title, lines, size=16)
    for x in [240, 470, 700, 930]:
        f.arrow([(x, 778), (x + 30, 778)])
    f.rect(40, 880, 1120, 72, fill=GRAY_BG)
    f.text(59, 909, "Research extensions selected by the runner", 18, bold=True)
    f.text(
        59,
        936,
        (
            "Coherent context / oracle   |   v3 evidence hydration   |   v4 "
            "arithmetic + relation scan   |   influence / packing"
        ),
        18,
    )
    f.footer(
        (
            "Current dependency: MemoryService imports MemoryRunner and Instance "
            "from evaluation/. The proposed split is not built."
        ),
        (
            "Evaluation gold labels belong to grading and explicit oracle "
            "diagnostics; ordinary retrieval does not use them."
        ),
    )
    return f


def font_name(a):
    family = "Times" if a["serif"] else "Courier" if a["mono"] else "Helvetica"
    return family + ("-Bold" if a["bold"] else "-Roman" if family == "Times" else "")


def export_pdf(figures, path):
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen.canvas import Canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(path), pageCompression=1, invariant=1)
    canvas.setTitle("Long-term memory: architecture atlas")
    canvas.setAuthor("llm-long-term-memory")
    for f in figures:
        canvas.setPageSize((f.width, f.height))
        for kind, a in f.ops:
            canvas.saveState()
            if kind == "text":
                font = font_name(a)
                width = stringWidth(a["value"], font, a["size"])
                x = a["x"] - (
                    width / 2 if a["anchor"] == "middle" else width if a["anchor"] == "end" else 0
                )
                if x < 0 or x + width > f.width:
                    raise ValueError(f"Text outside canvas in {f.name}: {a['value']}")
                canvas.setFillColor(HexColor(a["color"]))
                canvas.setFont(font, a["size"])
                canvas.drawString(x, f.height - a["y"], a["value"])
            elif kind == "rect":
                canvas.setFillColor(HexColor(a["fill"]))
                if a["stroke"]:
                    canvas.setStrokeColor(HexColor(a["stroke"]))
                canvas.setLineWidth(1.3)
                if a["dash"]:
                    canvas.setDash(7, 5)
                canvas.roundRect(
                    a["x"],
                    f.height - a["y"] - a["h"],
                    a["w"],
                    a["h"],
                    a["radius"],
                    fill=1,
                    stroke=int(bool(a["stroke"])),
                )
            else:
                p = canvas.beginPath()
                for i, (x, y) in enumerate(a["points"]):
                    (p.moveTo if i == 0 else p.lineTo)(x, f.height - y)
                if kind == "polygon":
                    p.close()
                    canvas.setFillColor(HexColor(a["fill"]))
                    canvas.drawPath(p, fill=1, stroke=0)
                else:
                    canvas.setStrokeColor(HexColor(a["color"]))
                    canvas.setLineWidth(a["width"])
                    if a["dash"]:
                        canvas.setDash(7, 5)
                    canvas.drawPath(p)
            canvas.restoreState()
        canvas.showPage()
    canvas.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf", action="store_true", help="Also export one six-page vector PDF using reportlab"
    )
    args = parser.parse_args()
    figures = [build_overview(Figure), ingestion(), storage(), retrieval(), temporal(), runtime()]
    for f in [*figures, build_overview(Figure, chinese=True)]:
        path = HERE / f"{f.name}.svg"
        path.write_text(f.svg(), encoding="utf-8")
        print(path.relative_to(ROOT))
    if args.pdf:
        path = ROOT / "output/pdf/architecture-atlas.pdf"
        export_pdf(figures, path)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
