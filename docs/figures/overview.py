"""Bilingual, editable overview. Coordinates and icons share the atlas scene API.

Run generate.py to rebuild both SVGs; no model, network or image assets are used.
"""

# Chinese diagram labels intentionally use full-width Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import math

INK = "#182B4D"
MUTED = "#53647B"
BLUE = "#235BA7"
GREEN = "#28754B"
ORANGE = "#A75C24"
RED = "#B83B52"
PURPLE = "#654DA0"
PALE_BLUE = "#EDF5FF"
PALE_GREEN = "#EDF9F0"
PALE_ORANGE = "#FFF3E8"
PALE_RED = "#FFF0F3"
PALE_PURPLE = "#F2EEFF"


def circle(f, x, y, radius, color, *, fill=None):
    points = [
        (x + radius * math.cos(i * math.tau / 48), y + radius * math.sin(i * math.tau / 48))
        for i in range(49)
    ]
    if fill:
        f.ops.append(("polygon", dict(points=points, fill=fill)))
    f.path(points, color, 3)


def icon(f, kind, x, y, color=INK):
    """Small original line icons, all within a 56 by 56 coordinate box."""

    def path(points):
        f.path([(x + px, y + py) for px, py in points], color, 3)

    if kind == "user":
        circle(f, x + 28, y + 15, 10, color)
        path([(9, 50), (9, 40), (16, 32), (40, 32), (47, 40), (47, 50), (9, 50)])
    elif kind == "agent":
        f.rect(x + 5, y + 14, 46, 34, stroke=color, radius=8)
        path([(28, 14), (28, 5)])
        circle(f, x + 28, y + 4, 3, color)
        for px in (18, 38):
            circle(f, x + px, y + 29, 2, color, fill=color)
        path([(19, 40), (37, 40)])
    elif kind == "chat":
        path([(7, 7), (49, 7), (49, 38), (26, 38), (13, 49), (13, 38), (7, 38), (7, 7)])
        for px in (18, 28, 38):
            circle(f, x + px, y + 23, 1.5, color, fill=color)
    elif kind == "chip":
        f.rect(x + 12, y + 12, 32, 32, stroke=color, radius=7)
        for offset in (19, 28, 37):
            path([(offset, 5), (offset, 12)])
            path([(offset, 44), (offset, 51)])
            path([(5, offset), (12, offset)])
            path([(44, offset), (51, offset)])
        path([(21, 33), (21, 23), (35, 23), (35, 33), (21, 33)])
    elif kind == "database":
        for cy in (12, 28, 44):
            points = [
                (
                    x + 28 + 21 * math.cos(i * math.tau / 40),
                    y + cy + 7 * math.sin(i * math.tau / 40),
                )
                for i in range(41)
            ]
            f.path(points, color, 3)
        path([(7, 12), (7, 44)])
        path([(49, 12), (49, 44)])
    elif kind == "search":
        circle(f, x + 24, y + 23, 16, color)
        path([(36, 36), (50, 50)])
    elif kind == "list":
        for py in (14, 28, 42):
            circle(f, x + 9, y + py, 2, color, fill=color)
            path([(20, py), (49, py)])
    elif kind == "file":
        path([(12, 5), (35, 5), (46, 16), (46, 51), (12, 51), (12, 5)])
        path([(35, 5), (35, 16), (46, 16)])
        for py in (26, 34, 42):
            path([(20, py), (37, py)])


def build_overview(figure_class, *, chinese=False):
    def tr(zh, en):
        return zh if chinese else en

    f = figure_class(
        1,
        "overview.zh-CN" if chinese else "overview",
        tr("长期记忆系统 · 项目架构", "Long-term memory · System architecture"),
        "",
        1040,
        width=2600,
        header=False,
    )
    f.description = tr(
        "按 2026-09-13 项目实现绘制。对话经过抽取写入存储；问题独立进入检索。"
        "租户和状态在召回时过滤，默认仅语义信号参与排序。回答不足时按条件搜索原文，"
        "找到证据才再次调用模型。实线为主路径，虚线为条件原文读取。详见 docs/ARCHITECTURE.md。",
        "Repository implementation, 2026-09-13. Conversations enter extraction; questions "
        "bypass writing. Recall filters namespace and status before ranking. Only semantic "
        "weight is nonzero by default. Conditional raw recovery triggers a second model "
        "call only with evidence. Solid: main flow; dashed: conditional archive read. "
        "See docs/ARCHITECTURE.md for source mappings and limitations.",
    )
    f.text(40, 57, "LLTM", 32, BLUE, bold=True)
    f.text(157, 57, f.title, 32, INK, bold=True)
    f.text(
        40,
        96,
        tr(
            "从对话提取持久事实，保留来源；回答先读记忆，必要时找回原文。",
            "Extract persistent facts, retain their sources, and recover raw evidence when needed.",
        ),
        21,
        MUTED,
    )
    f.text(2560, 54, "llm-long-term-memory", 20, MUTED, anchor="end")
    f.text(
        2560,
        88,
        tr("当前实现 · 2026-09-13", "Current implementation · 2026-09-13"),
        17,
        MUTED,
        anchor="end",
    )

    def panel(x, w, title, subtitle, accent, fill):
        f.rect(x, 190, w, 465, fill=fill, stroke=accent, radius=12)
        f.text(x + 18, 228, title, 24, accent, bold=True)
        f.text(x + 18, 256, subtitle, 16, accent)
        f.path([(x + 18, 274), (x + w - 18, 274)], accent, 1)

    def card(x, y, w, h, title, lines, accent, *, kind=None, size=19):
        f.rect(x, y, w, h, fill="#FFFFFF", stroke=accent, radius=8)
        tx = x + 16
        if kind:
            icon(f, kind, x + 12, y + 12, accent)
            tx = x + 77
        f.text(tx, y + 33, title, 20, accent, bold=True)
        for index, line in enumerate(lines):
            f.text(tx, y + 61 + index * 27, line, size, INK)

    # A separate question bus makes clear that asking does not ingest the question.
    f.arrow([(375, 190), (375, 149), (1395, 149), (1395, 190)], BLUE)
    f.rect(648, 130, 474, 33, fill="#FFFFFF", stroke=None)
    f.text(
        885,
        153,
        tr("查询问题 → 直接进入检索", "Question → retrieval, without a write"),
        20,
        BLUE,
        bold=True,
        anchor="middle",
    )

    panel(40, 180, tr("用户 / Agent", "User / Agent"), "CLIENT", PURPLE, PALE_PURPLE)
    f.rect(56, 296, 148, 328, stroke=None)
    icon(f, "user", 102, 327, PURPLE)
    f.text(130, 420, tr("用户", "User"), 22, INK, bold=True, anchor="middle")
    icon(f, "agent", 102, 470, PURPLE)
    f.text(130, 561, "AI Agent", 21, INK, bold=True, anchor="middle")
    f.text(130, 590, "MCP client", 16, MUTED, anchor="middle")

    panel(260, 230, tr("1. 对话与问题", "1. Input"), "ENTRY POINTS", BLUE, PALE_BLUE)
    card(276, 296, 198, 89, "REST API", [tr("令牌 → 租户*", "Token → tenant*")], BLUE)
    card(276, 403, 198, 89, "MCP", [tr("可信本地 stdio", "Trusted local stdio")], BLUE, size=17)
    card(
        276,
        510,
        198,
        111,
        tr("批量摄取 / CLI", "Batch ingest / CLI"),
        [tr("会话与轮次", "Sessions + turns"), tr("原文与来源保留", "Retain source text")],
        BLUE,
        size=17,
    )

    panel(530, 300, tr("2. 记忆抽取", "2. Extraction"), "LLM + RULES", GREEN, PALE_GREEN)
    card(
        548,
        296,
        264,
        101,
        "Two-stage extractor",
        [
            tr("A  提取事实与主体", "A  Extract facts + subject"),
            tr("B  分配 key 与更新意图", "B  Key + update intent"),
        ],
        GREEN,
        size=18,
    )
    card(
        548,
        413,
        264,
        128,
        tr("结构化事实 + 出处", "Facts + provenance"),
        [
            "subject · predicate · object",
            tr("时间 / 范围 / 来源锚点", "Time / scope / source"),
            tr("近邻检索 + LLM 判重", "Neighbors + LLM dedup"),
        ],
        GREEN,
        size=18,
    )
    f.text(550, 580, "Embed → persist → resolve", 19, GREEN, bold=True)
    f.text(550, 612, tr("重建受影响的有效时间线", "Rebuild affected timelines"), 18, MUTED)

    panel(870, 310, tr("3. 存储层", "3. Storage"), "SQLITE + NUMPY", ORANGE, PALE_ORANGE)
    card(
        888,
        296,
        274,
        112,
        "SQLite",
        ["sessions · turns", "memories · evidence"],
        ORANGE,
        kind="database",
        size=17,
    )
    card(
        888,
        425,
        274,
        89,
        "NumpyFlatIndex",
        [tr("向量 .npy + ID 映射", "Vectors .npy + ID map")],
        ORANGE,
        size=18,
    )
    card(888, 531, 274, 89, "FTS5 / BM25", ["memories_fts · turns_fts"], ORANGE, size=18)

    panel(1220, 350, tr("4. 混合召回", "4. Hybrid recall"), "SEMANTIC + LEXICAL", RED, PALE_RED)
    card(
        1238,
        296,
        314,
        89,
        tr("语义检索", "Vector search"),
        ["Embedding + NumPy"],
        RED,
        kind="search",
        size=18,
    )
    card(
        1238,
        403,
        314,
        89,
        tr("词法检索", "Lexical search"),
        ["memories_fts + BM25"],
        RED,
        kind="file",
        size=17,
    )
    card(
        1238,
        510,
        314,
        112,
        tr("召回时过滤租户与状态", "Filter during recall"),
        [
            tr("排除 evicted；隔离 namespace", "Namespace / exclude evicted"),
            tr("temporal=true：仅 active", "temporal=true: active only"),
        ],
        RED,
        size=18,
    )

    panel(1610, 260, tr("5. 候选记忆", "5. Context"), "RANK + PACK", PURPLE, PALE_PURPLE)
    card(
        1628,
        296,
        224,
        115,
        tr("候选并集 + 排序", "Union + rank"),
        [
            tr("语义 / BM25 / 时间", "Semantic / BM25 / age"),
            tr("重要性 / 实体匹配", "Importance / entity"),
        ],
        PURPLE,
        size=17,
    )
    card(
        1628,
        428,
        224,
        112,
        "Top-K memories",
        [
            tr("服务默认 K = 10", "Service default K = 10"),
            tr("v2 评测 K = 20", "v2 evaluation K = 20"),
        ],
        PURPLE,
        size=18,
    )
    f.text(1630, 578, tr("默认：仅语义权重为 1", "Default: semantic = 1"), 18, PURPLE, bold=True)
    f.text(1630, 608, tr("其余为 0；重排可选", "Others = 0; rerank opt-in"), 17, MUTED)
    f.text(1630, 636, tr("按预算组装上下文", "Pack selected context"), 17, MUTED)

    panel(1910, 320, tr("6. 回答生成", "6. Answer"), "MEMORY-FIRST ANSWERER", GREEN, PALE_GREEN)
    icon(f, "chip", 2042, 294, GREEN)
    f.text(2070, 382, "LLM verdict", 23, GREEN, bold=True, anchor="middle")
    f.arrow([(2070, 397), (2070, 428)], GREEN)
    diamond = [(2070, 428), (2196, 491), (2070, 554), (1944, 491), (2070, 428)]
    f.ops.append(("polygon", dict(points=diamond, fill="#FFFFFF")))
    f.path(diamond, GREEN, 2)
    f.text(2070, 487, tr("记忆足够？", "Sufficient?"), 22, INK, bold=True, anchor="middle")
    f.text(2070, 515, "status = answer", 17, MUTED, anchor="middle")
    f.arrow([(2196, 491), (2270, 491)], GREEN)
    f.text(2248, 476, tr("是", "Yes"), 17, GREEN, bold=True, anchor="middle")
    f.arrow([(2070, 554), (2070, 710)], RED)
    f.rect(1944, 588, 252, 56, fill=PALE_GREEN, stroke=None)
    f.text(2070, 609, "need_source / no_evidence", 17, RED, bold=True, anchor="middle")
    f.text(
        2070,
        636,
        tr("不足 → 尝试原文回退", "Insufficient → recover source"),
        17,
        RED,
        anchor="middle",
    )

    card(
        2270,
        422,
        290,
        137,
        tr("直接回答", "Direct answer"),
        [
            tr("基于记忆生成答案", "Answer from memories"),
            tr("返回所选记忆摘要", "Selected-memory summary"),
            tr("不再搜索原文", "No archive recovery"),
        ],
        GREEN,
        size=18,
    )

    # Horizontal spine. The storage/recall boundary is a read, not a write-through.
    for start, end in [
        (220, 260),
        (490, 530),
        (830, 870),
        (1180, 1220),
        (1570, 1610),
        (1870, 1910),
    ]:
        f.arrow([(start, 391), (end, 391)], INK)
    f.text(1200, 374, tr("读", "Read"), 15, ORANGE, anchor="middle")

    # Conditional evidence recovery uses the same raw archive, not external RAG.
    f.rect(1220, 710, 1010, 237, fill=PALE_RED, stroke=RED, radius=12)
    f.text(1240, 749, tr("7. 原文回退", "7. Raw-source fallback"), 24, RED, bold=True)
    f.text(
        2208,
        747,
        tr("同一命名空间 · 启用 fallback 时", "Same namespace · when fallback is enabled"),
        17,
        RED,
        anchor="end",
    )
    card(
        1240,
        774,
        265,
        139,
        "Source-local",
        [
            tr("沿所选记忆的来源", "Selected-memory sources"),
            tr("找回原始对话轮次", "Recover original turns"),
            tr("结合 BM25 命中选择", "Chosen with BM25 hits"),
        ],
        RED,
        size=17,
    )
    f.text(1524, 850, tr("或", "or"), 18, RED, bold=True, anchor="middle")
    card(
        1543,
        774,
        285,
        139,
        "Archive-wide",
        [
            tr("搜索该用户全部原文", "Search this user's archive"),
            "turns_fts + BM25",
            tr("no_evidence 不带锚点", "no_evidence: no anchors"),
        ],
        RED,
        size=17,
    )
    card(
        1850,
        774,
        358,
        139,
        tr("找到证据才再次回答", "Second pass only with evidence"),
        [
            tr("默认最多 3 轮 / 2,400 字符", "Default: 3 turns / 2,400 characters"),
            tr("恢复原文 → 再次调用回答器", "Raw turns → answerer again"),
            tr("无命中：保留判定或返回未知", "No hits: keep verdict / unknown"),
        ],
        RED,
        size=17,
    )
    f.arrow([(2230, 843), (2270, 843)], RED)
    card(
        2270,
        774,
        290,
        139,
        tr("回答与恢复摘要", "Answer + recovery"),
        [
            tr("有证据：根据原文回答", "Evidence: answer from source"),
            tr("无证据：保留判定 / 未知", "No evidence: verdict / unknown"),
            tr("附回退来源与轮次信息", "Recovery sources + turns"),
        ],
        RED,
        size=17,
    )
    f.arrow([(1025, 655), (1025, 843), (1220, 843)], ORANGE, dash=True)
    f.text(1045, 810, tr("读取原文", "Read raw turns"), 18, ORANGE, bold=True)
    f.text(1045, 835, "turns_fts", 16, ORANGE)

    # Operational boundaries are kept adjacent to, but outside, the data flow.
    f.rect(40, 710, 930, 237, fill="#F7F9FC", stroke="#C9D3E1", radius=12)
    f.text(
        62,
        749,
        tr("实现边界与可选能力", "Implementation boundaries & opt-in capabilities"),
        23,
        INK,
        bold=True,
    )
    notes = [
        tr(
            "* REST 配置令牌后绑定租户；未配置为开放模式。MCP stdio 由本地客户端信任。",
            "* REST binds tenants when tokens are configured; otherwise open. "
            "MCP stdio is trusted locally.",
        ),
        tr(
            "时间：写入重建有效区间；召回可过滤旧状态，历史查询使用 timeline / as_of。",
            "Time: writes rebuild validity intervals; recall filters status. "
            "History uses timeline / as_of.",
        ),
        tr(
            "存储：SQLite 与向量分开保存；服务进程内串行访问，不代表跨资源原子事务。",
            "Storage: SQLite and vectors persist separately; "
            "the service lock is not a cross-store transaction.",
        ),
        tr(
            "可选：重排、衰减、合并、v3/v4 回答策略；均不作为默认主路径展示。",
            "Opt-in: reranking, decay, consolidation and v3/v4 answer policies; "
            "outside the default flow.",
        ),
        tr(
            "数据操作：REST 支持导出与在线硬删除；forget 为软删除，备份生命周期另行处理。",
            "Data: REST export + online hard erasure; forget is soft deletion. "
            "Backup lifecycle is separate.",
        ),
    ]
    for index, note in enumerate(notes):
        circle(f, 66, 784 + 32 * index, 2, BLUE, fill=BLUE)
        f.text(80, 790 + 32 * index, note, 17, MUTED)

    f.path([(40, 980), (2560, 980)], "#D4DDE8", 1)
    f.arrow([(40, 1010), (85, 1010)], INK)
    f.text(98, 1016, tr("主路径", "Main flow"), 17, MUTED)
    f.arrow([(240, 1010), (285, 1010)], ORANGE, dash=True)
    f.text(298, 1016, tr("条件原文读取", "Conditional archive read"), 17, MUTED)
    f.text(
        2560,
        1016,
        tr(
            "逻辑职责图 · 并非独立微服务 · 实现与配置说明见 docs/ARCHITECTURE.md",
            "Logical responsibilities, not microservices · Source mapping: docs/ARCHITECTURE.md",
        ),
        17,
        MUTED,
        anchor="end",
    )
    return f
