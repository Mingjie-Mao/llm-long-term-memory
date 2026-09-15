"""Rebuild the architecture atlas without importing or executing the memory engine.

SVG (six figures, in English and Chinese): python3 docs/figures/generate.py
SVG + vector PDF: python3 docs/figures/generate.py --pdf
Only the optional PDF export needs reportlab. All coordinates use a shared scene,
so SVG and PDF have the same layout. The PDF is written to output/pdf/.
"""

# Chinese diagram labels intentionally use full-width Chinese punctuation, the same
# exemption overview.py carries. Substituting ASCII punctuation would be wrong in the
# rendered figure, which is the artifact this file exists to produce.
# ruff: noqa: RUF001

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


def ingestion(chinese=False):
    def tr(zh, en):
        return zh if chinese else en

    f = Figure(
        2,
        tr("write-path.zh-CN", "write-path"),
        tr("从对话会话到带版本的事实", "From conversation sessions to versioned facts"),
        tr(
            "批量管线：抽取、原文登记、去重、持久化，然后重建时间线。",
            "The batch pipeline: extraction, source registration, deduplication, "
            "persistence, then temporal resolution.",
        ),
        830,
    )
    f.section(
        40, 175, tr("A. 抽取  /  每批只属于一个命名空间", "A. EXTRACT  /  ONE NAMESPACE PER BATCH")
    )
    specs = [
        (
            40,
            tr("1  会话分批", "1  Batch sessions"),
            tr(
                ["跳过已完成的", "核对摄取指纹"],
                ["Skip completed work", "Check ingest fingerprint"],
            ),
        ),
        (
            270,
            tr("2  阶段 A", "2  Stage A"),
            tr(["事实句 + 说话人", "主体 + 范围"], ["Fact lines + speaker", "Subject + scope"]),
        ),
        (
            500,
            tr("3  阶段 B", "3  Stage B"),
            tr(["属性槽位 + 值", "更新操作"], ["Temporal key + object", "Update operation"]),
        ),
        (
            730,
            tr("4  规则补充", "4  Rules"),
            tr(
                ["类型、实体、优先级", "日期取自会话"],
                ["Type, entities, priority", "Date from session"],
            ),
        ),
        (
            960,
            tr("5  出处锚点", "5  Source anchors"),
            tr(
                ["词面最像的句子", "轮次 + 字符区间"],
                ["Best lexical sentence", "Turn + character span"],
            ),
        ),
    ]
    for x, title, lines in specs:
        f.box(x, 202, 200, 126, title, lines, size=16)
    for a in [240, 470, 700, 930]:
        f.arrow([(a, 265), (a + 30, 265)])
    f.label(286, 352, tr("1 次模型请求", "1 LLM request"), BLUE)
    f.label(516, 352, tr("有事实则 +1 次", "+1 if facts exist"), BLUE)
    f.label(744, 352, tr("不调模型", "No LLM request"))
    f.section(
        40,
        411,
        tr("B. 写入与更新  /  抽取返回之后", "B. STORE AND UPDATE  /  AFTER EXTRACTION RETURNS"),
    )
    f.box(
        40,
        441,
        200,
        151,
        tr("6  归档原文", "6  Archive source"),
        tr(
            ["sessions + turns", "FTS5 触发器更新", "带命名空间的会话 ID"],
            ["sessions + turns", "FTS5 triggers update", "Scoped session IDs"],
        ),
        accent=AMBER,
        fill=AMBER_BG,
        size=16,
    )
    f.box(
        270,
        441,
        200,
        151,
        tr("7  去重", "7  Deduplicate"),
        tr(
            ["候选嵌入", "找同命名空间近邻", "模型裁定"],
            ["Embed candidates", "Find close / keyed facts", "LLM judges neighbours"],
        ),
        size=16,
    )
    f.box(
        500,
        441,
        200,
        151,
        tr("8  写入事实", "8  Persist facts"),
        tr(
            ["插入 memory 行", "追加归一化向量", "保留新的 / 不同的"],
            ["INSERT memory rows", "Add normalized vectors", "Keep new / distinct facts"],
        ),
        accent=TEAL,
        fill=TEAL_BG,
        size=16,
    )
    f.box(
        730,
        441,
        200,
        151,
        tr("9  重建时间线", "9  Resolve keys"),
        tr(
            ["读取全链含历史", "按 event_time 排序", "重写有效区间"],
            ["Read active + history", "Sort by event_time", "Rewrite validity windows"],
        ),
        accent=TEAL,
        fill=TEAL_BG,
        size=16,
    )
    f.box(
        960,
        441,
        200,
        151,
        tr("10  检查点", "10  Checkpoint"),
        tr(
            ["保存向量文件", "保存会话进度", "配额停止后续跑"],
            ["Save vector sidecars", "Save session progress", "Resume after quota stop"],
        ),
        accent=TEAL,
        fill=TEAL_BG,
        size=16,
    )
    f.arrow([(1060, 328), (1060, 386), (20, 386), (20, 516), (40, 516)])
    for a in [240, 470, 700, 930]:
        f.arrow([(a, 516), (a + 30, 516)])
    f.rect(40, 635, 1120, 101, fill=GRAY_BG)
    f.text(59, 664, tr("开销与失败语义", "Cost and failure semantics"), 18, bold=True)
    f.text(
        59,
        694,
        tr(
            "阶段 A 抽到事实时共两次抽取请求；近重复裁定可能再增加若干次。",
            "Two extraction requests when Stage A yields facts; near-duplicate "
            "adjudication can add requests.",
        ),
        18,
    )
    f.text(
        59,
        720,
        tr(
            "0.92 相似度只是提名近邻；只有 DUPLICATE 判定才丢弃新事实。近邻只在同一命名空间内取。",
            "A similarity of 0.92 nominates neighbours. Only a DUPLICATE verdict "
            "drops the new fact.",
        ),
        18,
    )
    f.footer(
        tr(
            "被内容策略拒绝的批次仍保留原文并记录拒绝状态；其他失败批次保留为待处理。",
            "Content-blocked batches retain raw turns and record their blocked "
            "status. Other failed batches remain pending.",
        ),
        tr(
            "时间线重建发生在写入之后。向量文件与 SQLite 不共享同一个原子事务。",
            "Temporal resolution follows memory insertion. Vector files and SQLite "
            "do not share a single atomic transaction.",
        ),
    )
    return f


def storage(chinese=False):
    def tr(zh, en):
        return zh if chinese else en

    f = Figure(
        3,
        tr("storage.zh-CN", "storage"),
        tr("关系型真相源，外挂向量文件", "A relational source of truth with vector sidecars"),
        tr(
            "物理表名与关系取自 schema.sql、SQLiteMemoryStore 与 NumpyFlatIndex。",
            "Physical names and relationships from schema.sql, SQLiteMemoryStore, "
            "and NumpyFlatIndex.",
        ),
        900,
    )
    f.rect(40, 165, 800, 570, fill="#FFFFFF", stroke=BLUE)
    f.section(
        62,
        198,
        tr("SQLITE  /  <store>.db  /  WAL + 外键", "SQLITE  /  <store>.db  /  WAL + FOREIGN KEYS"),
    )
    f.box(
        64,
        224,
        326,
        110,
        "sessions",
        tr(
            ["PK id  |  user_id", "started_at  |  source"],
            ["PK id  |  user_id", "started_at  |  source"],
        ),
    )
    f.box(
        464,
        224,
        350,
        110,
        "turns",
        ["PK id  |  FK session_id", "turn_index, role, content, ts"],
    )
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
            tr("source_turn_index + 字符偏移", "source_turn_index + char offsets"),
            "importance, strength, token_count",
        ],
        size=17,
    )
    f.arrow([(208, 397), (208, 334)])
    f.label(222, 369, tr("会话外键", "session FK"))
    f.arrow([(464, 418), (492, 418), (492, 360), (638, 360), (638, 334)], dash=True)
    f.label(507, 389, tr("轮次 / 区间查找", "turn/span lookup"))
    f.box(
        539,
        436,
        275,
        94,
        "turns_fts",
        tr(["对原始轮次做 BM25"], ["BM25 over original turns"]),
        accent=AMBER,
        fill=AMBER_BG,
        size=17,
    )
    f.arrow([(784, 334), (784, 436)], AMBER)
    f.label(688, 414, tr("触发器", "triggers"), AMBER)
    f.box(
        539,
        587,
        275,
        94,
        "memories_fts",
        tr(["对事实内容做 BM25"], ["BM25 over fact content"]),
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(464, 635), (539, 635)], TEAL)
    f.label(473, 614, tr("触发器", "triggers"), TEAL)
    f.label(
        64,
        711,
        tr(
            "辅助表：entities + memory_entities  |  evidence  |  meta",
            "Auxiliary: entities + memory_entities  |  evidence  |  meta",
        ),
    )
    f.rect(869, 165, 291, 570, fill=TEAL_BG, stroke=TEAL)
    f.section(890, 198, tr("外挂向量文件", "VECTOR SIDECARS"))
    f.text(890, 240, "<store>-index.npy", 19, TEAL, bold=True)
    f.text(890, 275, "N x 384 float32", 19)
    f.text(890, 307, tr("L2 归一化向量", "L2-normalized vectors"), 18)
    f.text(890, 339, tr("精确内积全扫描", "Exact inner-product scan"), 18)
    f.path([(890, 370), (1138, 370)], BORDER, 1)
    f.text(890, 409, "<store>-index.ids.json", 18, TEAL, bold=True)
    f.text(890, 444, tr("行号 -> memory ID", "Row position -> memory ID"), 18)
    f.text(890, 476, tr("由应用维护的对应关系", "Application-managed link"), 18)
    f.text(890, 508, tr("不是 SQL 外键", "No SQL foreign key"), 18)
    f.path([(890, 538), (1138, 538)], BORDER, 1)
    f.text(890, 576, tr("与库同目录", "Same store directory"), 18, bold=True)
    f.text(890, 608, tr("独立的保存操作", "Separate save operation"), 18)
    f.text(890, 640, tr("状态过滤在检索时做", "Status filtering at retrieval"), 18)
    f.rect(40, 763, 1120, 58, fill=GRAY_BG)
    f.text(
        58,
        798,
        tr(
            "出处区间是启发式的、可为空的锚点。会话链接是真外键；轮次与偏移链接不是。",
            "Source spans are heuristic, nullable anchors. The session link is a "
            "real foreign key; turn/offset links are not.",
        ),
        18,
    )
    f.footer(
        tr(
            "常规 supersession 与 forget 只更新状态。迁移清理与演示会话删除会物理删除行。",
            "Regular supersession and forget update status. Migration cleanup and "
            "demo session deletion can hard-delete rows.",
        )
    )
    return f


def retrieval(chinese=False):
    def tr(zh, en):
        return zh if chinese else en

    f = Figure(
        4,
        tr("read-path.zh-CN", "read-path"),
        tr("先读记忆，按需恢复原文证据", "Retrieve memory first; recover raw evidence on demand"),
        tr(
            "默认的记忆优先路径，已启用原文回退。虚线只在需要恢复原文时执行。",
            "Default memory-first route with raw_fallback enabled. Dashed paths run"
            " only when source recovery is needed.",
        ),
        1010,
    )
    f.box(
        40,
        193,
        190,
        125,
        tr("问题", "Question"),
        tr(["user_id + 查询", "MiniLM 嵌入"], ["user_id + query", "MiniLM embedding"]),
        size=17,
    )
    f.box(
        285,
        167,
        280,
        100,
        tr("语义候选", "Semantic candidates"),
        tr(
            ["扫索引，再过滤租户/状态", "最多保留 50 条"],
            ["Scan index, then scope/status", "Keep up to 50"],
        ),
        size=17,
    )
    f.box(
        285,
        294,
        280,
        100,
        tr("词法候选", "Lexical candidates"),
        tr(
            ["FTS5 + SQL 过滤租户/状态", "最多保留 50 条"],
            ["FTS5 + SQL scope/status", "Keep up to 50"],
        ),
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
        tr("并集与打分", "Union and score"),
        tr(
            ["按 memory ID 去重", "至多 100 个候选", "默认权重：语义 1"],
            ["Deduplicate memory IDs", "Up to 100 candidates", "Default weight: semantic 1"],
        ),
        size=17,
    )
    f.arrow([(565, 217), (620, 217)])
    f.arrow([(565, 344), (592, 344), (592, 313), (620, 313)], TEAL)
    f.box(
        915,
        193,
        245,
        151,
        tr("选取与组装", "Select and render"),
        tr(
            ["取排名前 k 条记忆", "按范围 / 时间分组", "组装回答上下文"],
            ["Top-k ranked memories", "Group by scope / time", "Build answer context"],
        ),
        size=17,
    )
    f.arrow([(870, 267), (915, 267)])
    f.label(630, 373, tr("其余四路权重为 0", "Other four score weights: 0"))
    f.label(926, 373, tr("服务 k=10；v2 评测 k=20", "Service k=10; v2 eval k=20"))
    f.arrow([(1140, 344), (1140, 424), (177, 424), (177, 480)])
    f.box(
        40,
        480,
        275,
        296,
        tr("第一次模型调用", "First LLM pass"),
        tr(["解析 AnswerVerdict"], ["Parse AnswerVerdict"]),
        size=18,
    )
    f.text(60, 590, "answer", 18, BLUE, mono=True)
    f.text(60, 661, "need_source", 18, AMBER, mono=True)
    f.text(60, 735, "no_evidence", 18, AMBER, mono=True)
    f.box(
        405,
        474,
        325,
        90,
        tr("直接返回答案", "Return answer"),
        tr(["不搜原文，不做第二次调用"], ["No raw retrieval or second pass"]),
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
        tr("带来源的恢复", "Source-aware recovery"),
        tr(
            ["用 BM25 给原文轮次排名", "选本地出处或全档案"],
            ["Rank archive turns with BM25", "Choose local or archive-wide"],
        ),
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
        tr("全档案恢复", "Archive-wide recovery"),
        tr(["不带所选记忆的锚点搜索"], ["Search without selected anchors"]),
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
        tr("找到原文轮次了吗？", "Any raw turns recovered?"),
        tr(["最多 3 个轮次", "2,400 个原文字符"], ["At most 3 turns", "2,400 source characters"]),
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
        tr("没有找到证据", "No evidence found"),
        tr(["返回判定文本 / 弃答"], ["Return verdict text / abstain"]),
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(1000, 601), (1000, 564)], AMBER, dash=True)
    f.label(1015, 586, tr("否", "no"), AMBER)
    f.box(
        840,
        783,
        320,
        103,
        tr("第二次模型调用", "Second LLM pass"),
        tr(
            ["依据恢复的原文作答", "返回最终自然语言"],
            ["Answer from recovered source", "Return final prose"],
        ),
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(1000, 714), (1000, 783)], AMBER, dash=True)
    f.label(1015, 752, tr("是", "yes"), AMBER)
    f.footer(
        tr(
            "need_source 判定仍可能走全档案恢复——当最佳 BM25 命中落在本地来源会话之外时。",
            "A need_source verdict can still choose archive-wide recovery if the "
            "best BM25 hit is outside local source sessions.",
        ),
        tr(
            "重排、会话连贯上下文、预算打包、衰减与自适应补水都是可选分支，默认关闭。",
            "Reranking, session-coherent context, packing, decay and adaptive "
            "hydration are optional branches; see the atlas captions.",
        ),
        tr(
            "图中上限来自已提交的默认配置。第一次判定解析失败有单独的返回路径。",
            "Limits shown are from the checked-in default configurations. A "
            "malformed first verdict has a separate parse-failure return.",
        ),
    )
    return f


def temporal(chinese=False):
    def tr(zh, en):
        return zh if chinese else en

    f = Figure(
        5,
        tr("temporal.zh-CN", "temporal"),
        tr(
            "与到达顺序无关地重建事实时间线",
            "Rebuild a fact timeline independently of arrival order",
        ),
        tr(
            "同一个 (user_id, subject, predicate) 键上的示例替换链；每个后继都显式替换前值。",
            "Illustrative replacement chain on one (user_id, subject, predicate) "
            "key; each successor explicitly replaces the prior value.",
        ),
        770,
    )
    f.section(
        40,
        175,
        tr(
            "A. 摄取顺序  /  系统学到它们的先后",
            "A. INGESTION ORDER  /  THE ORDER IN WHICH THE SYSTEM LEARNS",
        ),
    )
    for x, title, lines in [
        (40, tr("最先到达", "Arrives first"), tr(["八月：悉尼"], ["August: Sydney"])),
        (330, tr("第二个到达", "Arrives second"), tr(["一月：堪培拉"], ["January: Canberra"])),
        (620, tr("第三个到达", "Arrives third"), tr(["三月：墨尔本"], ["March: Melbourne"])),
    ]:
        f.box(x, 199, 245, 89, title, lines)
    f.arrow([(285, 244), (330, 244)])
    f.arrow([(575, 244), (620, 244)])
    f.box(
        910,
        199,
        250,
        89,
        tr("重建整个键", "Resolve the full key"),
        tr(["按 event_time 再按 ID 排序"], ["Sort by event_time, then ID"]),
        accent=TEAL,
        fill=TEAL_BG,
        size=17,
    )
    f.arrow([(865, 244), (910, 244)])
    f.section(
        40,
        350,
        tr(
            "B. 有效期顺序  /  半开区间 [起, 止)",
            "B. VALIDITY ORDER  /  HALF-OPEN INTERVALS [START, END)",
        ),
    )
    for x, date in [
        (355, tr("1 月 1 日", "Jan 01")),
        (540, tr("3 月 1 日", "Mar 01")),
        (900, tr("8 月 1 日", "Aug 01")),
    ]:
        f.text(x, 391, date, 18, MUTED, anchor="middle")
        f.path([(x, 407), (x, 588)], BORDER, 1, dash=True)
    f.text(61, 440, tr("堪培拉", "Canberra"), 20, bold=True)
    f.rect(355, 418, 185, 38, fill=BLUE_BG, stroke=BLUE, radius=3)
    f.text(448, 444, tr("已退位", "superseded"), 17, BLUE, anchor="middle")
    f.text(61, 507, tr("墨尔本", "Melbourne"), 20, bold=True)
    f.rect(540, 485, 360, 38, fill=BLUE_BG, stroke=BLUE, radius=3)
    f.text(720, 511, tr("已退位", "superseded"), 17, BLUE, anchor="middle")
    f.text(61, 574, tr("悉尼", "Sydney"), 20, bold=True)
    f.rect(900, 552, 215, 38, fill=TEAL_BG, stroke=TEAL, radius=3)
    f.text(993, 578, tr("当前", "active"), 17, TEAL, anchor="middle")
    f.arrow([(1080, 571), (1140, 571)], TEAL)
    f.text(
        61,
        619,
        tr(
            "晚到的三月证据把堪培拉的 valid_to 从八月改回三月；八月那条仍是当前值。",
            "Late March evidence moves Canberra's valid_to from August back to "
            "March; the August fact stays current.",
        ),
        18,
    )
    f.footer(
        tr(
            "并存的后继不关闭前一条事实。重复值可能折叠到最早的区间归属者。",
            "Coexisting successors do not close the previous fact. Repeated values "
            "may fold into the earliest interval owner.",
        ),
        tr(
            "无日期的事实原样留着不解析。event_time 目前来自会话日期；ingested_at 记录到达时间。",
            "Undated facts are left unresolved. event_time currently comes from the"
            " session date; ingested_at records arrival time.",
        ),
        tr(
            "这是一个解释性示例，不是实验结果，也不表示已具备完整的事务时间历史。",
            "This is an explanatory example, not a benchmark result or a claim of "
            "full transaction-time history.",
        ),
    )
    return f


def runtime(chinese=False):
    def tr(zh, en):
        return zh if chinese else en

    f = Figure(
        6,
        tr("runtime-evaluation.zh-CN", "runtime-evaluation"),
        tr("运行入口与研究评测的边界", "Runtime entry points and the research evaluation boundary"),
        tr(
            "三个适配层共用组件，但它们并不执行同一条写入路径，也不暴露同样的能力。",
            "Three adapters share components, but they do not execute the same "
            "write path or expose the same capabilities.",
        ),
        1055,
    )
    f.section(40, 175, tr("A. 当前的运行入口", "A. CURRENT ENTRY POINTS"))
    f.box(
        40,
        201,
        350,
        99,
        tr("CLI / Python 批量管线", "CLI / Python batch pipeline"),
        ["lltm ingest run", "IngestionPipeline"],
        size=18,
    )
    f.box(
        425,
        201,
        350,
        99,
        tr("REST + 检查界面 + MCP", "REST + Inspector + MCP"),
        ["api.app / mcp_server", "MemoryService"],
        size=18,
    )
    f.box(
        810,
        201,
        350,
        99,
        tr("公开演示后端", "Public playground"),
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
        tr("完整批量写入路径", "Full batch write path"),
        tr(
            [
                "两阶段抽取 + 去重",
                "归档 + 写库 + 向量",
                "时间线重建 + 检查点",
                "抽取需要调用模型",
            ],
            [
                "TwoStageExtractor + dedup",
                "Archive + store + vectors",
                "TemporalResolver + checkpoint",
                "Uses provider for extraction",
            ],
        ),
        size=18,
    )
    f.box(
        425,
        341,
        350,
        165,
        tr("服务的读写路径", "Service read / write path"),
        tr(
            [
                "检索、原文查找、时间线",
                "经 MemoryRunner 回答",
                "按需构造单轮抽取器",
                "去重 + 时间线重建",
            ],
            [
                "Search, raw lookup, timeline",
                "Answer via MemoryRunner",
                "Lazy batch-extractor adapter",
                "Dedup + temporal resolution",
            ],
        ),
        size=18,
    )
    f.box(
        810,
        341,
        350,
        165,
        tr("直接写入结构化事实", "Direct structured facts"),
        tr(
            [
                "带类型的事实 + 原始轮次",
                "本地嵌入 + 时间线重建",
                "签名的临时命名空间",
                "没有 LLM 抽取或回答",
            ],
            [
                "Typed facts + raw turns",
                "Local embedder + resolver",
                "Signed temporary namespace",
                "No LLM extraction or answering",
            ],
        ),
        accent=AMBER,
        fill=AMBER_BG,
        size=18,
    )
    for x in [215, 600, 985]:
        f.arrow([(x, 300), (x, 341)])
    f.rect(40, 550, 1120, 72, fill=TEAL_BG, stroke=TEAL)
    f.text(60, 580, tr("共用的基础组件", "Shared primitives"), 19, TEAL, bold=True)
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
    f.section(
        40,
        678,
        tr(
            "B. 评测  /  显式的运行器、冻结的输入、可审计的产物",
            "B. EVALUATION  /  EXPLICIT RUNNERS, FROZEN INPUTS AND AUDITABLE OUTPUTS",
        ),
    )
    specs = [
        (
            40,
            tr("冻结输入", "Freeze inputs"),
            tr(
                ["配置 + 源码哈希", "清单 + 库状态"],
                ["Config + source hashes", "Manifests + store state"],
            ),
        ),
        (
            270,
            tr("跑各个臂", "Run arms"),
            tr(
                ["记忆 / 整段历史", "朴素检索基线"],
                ["Memory / full context", "Naive RAG baselines"],
            ),
        ),
        (
            500,
            tr("评分", "Grade answers"),
            tr(["独立的评分角色", "答案 + 参考"], ["Independent judge role", "Answer + reference"]),
        ),
        (
            730,
            tr("记录证据", "Record evidence"),
            tr(
                ["JSONL 逐题行 + 用量", "运行锁 + 续跑账本"],
                ["JSONL rows + usage", "Lock + resume ledger"],
            ),
        ),
        (
            960,
            tr("聚合", "Aggregate"),
            tr(
                ["正确率 + 成本 + 召回", "配对检验 + 门限"],
                ["Accuracy + cost + recall", "Paired checks + gates"],
            ),
        ),
    ]
    for x, title, lines in specs:
        f.box(x, 712, 200, 132, title, lines, size=16)
    for x in [240, 470, 700, 930]:
        f.arrow([(x, 778), (x + 30, 778)])
    f.rect(40, 880, 1120, 72, fill=GRAY_BG)
    f.text(
        59,
        909,
        tr("由运行器选择的研究扩展", "Research extensions selected by the runner"),
        18,
        bold=True,
    )
    f.text(
        59,
        936,
        tr(
            "会话连贯上下文 / oracle   |   v3 证据补水   |   "
            "v4 算术 + 关系扫描   |   影响力 / 预算打包",
            "Coherent context / oracle   |   v3 evidence hydration   |   v4 "
            "arithmetic + relation scan   |   influence / packing",
        ),
        18,
    )
    f.footer(
        tr(
            "当前依赖：MemoryService 从 evaluation/ 导入 MemoryRunner 和 Instance。"
            "拟议的拆分尚未实现。",
            "Current dependency: MemoryService imports MemoryRunner and Instance "
            "from evaluation/. The proposed split is not built.",
        ),
        tr(
            "评测金标只用于评分和显式 oracle 诊断；普通检索不使用它们。",
            "Evaluation gold labels belong to grading and explicit oracle "
            "diagnostics; ordinary retrieval does not use them.",
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
    builders = [build_overview, ingestion, storage, retrieval, temporal, runtime]
    # English is the PDF set: reportlab's built-in Helvetica has no CJK glyphs, so a
    # Chinese page would export as empty boxes. The SVGs carry both, because there the
    # font is resolved by the reader's system rather than embedded here.
    figures = [build_overview(Figure), *[b() for b in builders[1:]]]
    chinese = [build_overview(Figure, chinese=True), *[b(chinese=True) for b in builders[1:]]]
    for f in [*figures, *chinese]:
        path = HERE / f"{f.name}.svg"
        path.write_text(f.svg(), encoding="utf-8")
        print(path.relative_to(ROOT))
    if args.pdf:
        path = ROOT / "output/pdf/architecture-atlas.pdf"
        export_pdf(figures, path)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
