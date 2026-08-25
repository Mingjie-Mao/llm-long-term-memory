"""Human-readable, evidence-graded reporting for zero-yield extraction.

The store can prove that a session produced no memory, but not why.  This module
keeps direct observations, controlled causal evidence, and unresolved cases in
separate rows so a final report cannot silently relabel every empty session as
"model randomness".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import comb
from typing import Any


@dataclass(frozen=True, slots=True)
class BatchPilotSummary:
    rows: int
    cohort_size: int
    duplicate_keys: int
    position_paired_sessions: int
    yielded_front_zero_back: int
    zero_front_yielded_back: int
    position_mcnemar_p: float
    front_mean_memories: float
    back_mean_memories: float
    batch_sizes: dict[str, dict[str, int | float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize_batch_pilot(payload: dict[str, Any]) -> BatchPilotSummary:
    """Validate and summarize the completed randomized batch pilot."""
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("batch-position pilot has no records")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("batch-position pilot records must be objects")

    keys = [
        (
            row.get("arm"),
            row.get("arrangement"),
            row.get("batch_index"),
            row.get("session_id"),
        )
        for row in records
    ]
    duplicate_keys = len(keys) - len(set(keys))
    if duplicate_keys:
        raise ValueError(f"batch-position pilot has {duplicate_keys} duplicate record keys")

    size_rows: dict[int, list[dict[str, Any]]] = {
        size: [row for row in records if row.get("arm") == f"batchsize{size}"]
        for size in (15, 5, 1)
    }
    cohorts = {size: {str(row["session_id"]) for row in rows} for size, rows in size_rows.items()}
    if any(not cohort for cohort in cohorts.values()):
        raise ValueError("batch-position pilot is missing a batch-size arm")
    base = cohorts[15]
    if any(cohort != base for cohort in cohorts.values()):
        raise ValueError("batch-position pilot arms do not share one cohort")
    cohort_size = len(base)
    if any(len(rows) != cohort_size for rows in size_rows.values()):
        raise ValueError("a batch-size arm is incomplete or repeats a session")

    position_rows = [row for row in records if row.get("arm") == "position"]
    arrangements: dict[str, set[str]] = defaultdict(set)
    for row in position_rows:
        arrangements[str(row["arrangement"])].add(str(row["session_id"]))
    if len(arrangements) < 2 or any(ids != base for ids in arrangements.values()):
        raise ValueError("position arrangements are incomplete or use a different cohort")

    front: dict[str, list[dict[str, Any]]] = defaultdict(list)
    back: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        position = int(row["position"])
        if position <= 3:
            front[str(row["session_id"])].append(row)
        elif position >= 11:
            back[str(row["session_id"])].append(row)
    paired = sorted(set(front) & set(back))
    if not paired:
        raise ValueError("batch-position pilot has no front/back session pairs")

    wins = sum(
        1
        for session_id in paired
        if not any(int(row["n_memories"]) == 0 for row in front[session_id])
        and all(int(row["n_memories"]) == 0 for row in back[session_id])
    )
    losses = sum(
        1
        for session_id in paired
        if all(int(row["n_memories"]) == 0 for row in front[session_id])
        and not any(int(row["n_memories"]) == 0 for row in back[session_id])
    )
    front_means = [
        sum(int(row["n_memories"]) for row in front[session_id]) / len(front[session_id])
        for session_id in paired
    ]
    back_means = [
        sum(int(row["n_memories"]) for row in back[session_id]) / len(back[session_id])
        for session_id in paired
    ]

    batch_sizes: dict[str, dict[str, int | float]] = {}
    for size, rows in size_rows.items():
        zero = sum(int(row["n_memories"]) == 0 for row in rows)
        memories = sum(int(row["n_memories"]) for row in rows)
        batch_sizes[str(size)] = {
            "sessions": len(rows),
            "zero": zero,
            "zero_rate": zero / len(rows),
            "memories": memories,
            "memories_per_session": memories / len(rows),
        }

    return BatchPilotSummary(
        rows=len(records),
        cohort_size=cohort_size,
        duplicate_keys=duplicate_keys,
        position_paired_sessions=len(paired),
        yielded_front_zero_back=wins,
        zero_front_yielded_back=losses,
        position_mcnemar_p=exact_mcnemar(wins, losses),
        front_mean_memories=sum(front_means) / len(front_means),
        back_mean_memories=sum(back_means) / len(back_means),
        batch_sizes=batch_sizes,
    )


def _attribution(categories: dict[str, int], total: int) -> dict[str, int]:
    groups = {
        "confirmed_content_policy": categories.get("content_policy", 0),
        "confirmed_source_format": categories.get("source_format", 0),
        "confirmed_source_id_collision": categories.get("source_id_collision", 0),
        "confirmed_exact_input_variability": categories.get("inconsistent_repeat", 0),
        "confirmed_annotated_evidence_miss": categories.get("evidence_extraction_miss", 0),
        "near_repeat_date_difference": categories.get("date_variant_outcome_difference", 0),
        "short_no_durable_fact_candidate": categories.get("short_no_durable_fact_candidate", 0),
    }
    groups["unresolved_or_no_durable_fact"] = total - sum(groups.values())
    if groups["unresolved_or_no_durable_fact"] < 0:
        raise ValueError("zero-yield attribution groups exceed the substantive total")
    return groups


def render_zero_yield_markdown(
    audit: dict[str, Any],
    pilot: BatchPilotSummary,
    *,
    audit_source: str,
    pilot_source: str,
) -> str:
    """Render an aggregate final report without exposing individual benchmark rows."""
    if not audit.get("complete"):
        raise ValueError("refusing a final zero-yield report from an incomplete audit")
    all_zero = int(audit["all_zero_yield_sessions"])
    substantive = int(audit["substantive_zero_yield_sessions"])
    categories = {str(key): int(value) for key, value in audit.get("all_categories", {}).items()}
    if sum(categories.values()) != all_zero:
        raise ValueError("all zero-yield categories do not partition every empty session")
    substantive_categories = {
        str(key): int(value) for key, value in audit.get("categories", {}).items()
    }
    if sum(substantive_categories.values()) != substantive:
        raise ValueError("zero-yield categories do not partition the substantive sessions")
    attribution = _attribution(categories, all_zero)
    bands = audit["position_bands"]
    effect = audit["batch_position_effect"]

    rows = [
        (
            "内容审查拒绝",
            attribution["confirmed_content_policy"],
            "有明确的终止状态; 不是普通空结果",
        ),
        (
            "源数据格式问题",
            attribution["confirmed_source_format"],
            "日期、角色或空文本检查发现问题",
        ),
        (
            "数据 ID 冲突",
            attribution["confirmed_source_id_collision"],
            "同一公开 ID 实际对应不同聊天内容",
        ),
        (
            "完全相同输入出现不同结果",
            attribution["confirmed_exact_input_variability"],
            "可以直接证明模型/批次结果不稳定",
        ),
        (
            "正确答案证据被漏提取",
            attribution["confirmed_annotated_evidence_miss"],
            "标注证明这里确实有应保留的信息; 原因未必是随机",
        ),
        (
            "相同文字、不同日期, 结果不同",
            attribution["near_repeat_date_difference"],
            "日期也是模型输入, 因此只能算近重复, 不能证明随机",
        ),
        (
            "短会话, 可能没有长期信息",
            attribution["short_no_durable_fact_candidate"],
            "这是合理候选解释, 但不冒充已证明原因",
        ),
        (
            "仍无法确定, 或原聊天没有持久信息",
            attribution["unresolved_or_no_durable_fact"],
            "保持未知, 不用猜测填补证据空白",
        ),
    ]

    lines = [
        "# train150 零产出最终分析",
        "",
        "> 这份报告只给出证据能支持的结论。`没有生成记忆`不自动等于模型出错。",
        "",
        "## 总览",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 全部会话 | {int(audit['total_sessions']):,} |",
        f"| 全部零产出会话 | {all_zero:,} |",
        (
            "| 少于最低对话轮数的零产出 | "
            f"{int(audit['short_zero_yield_sessions_below_min_turns']):,} |"
        ),
        f"| 进入详细分析的实质性零产出 | {substantive:,} |",
        "",
        "## 逐类归因",
        "",
        "| 分类 | 数量 | 可以得出什么结论 |",
        "|---|---:|---|",
    ]
    lines.extend(f"| {label} | {count:,} | {meaning} |" for label, count, meaning in rows)
    lines.extend(
        [
            "",
            "## train150 中的批次位置现象 (观察性)",
            "",
            "| 位置 | 零产出 / 会话 | 零产出率 |",
            "|---|---:|---:|",
            (
                "| 前部 0-3 | "
                f"{int(bands['front_0_3']['zero']):,} / {int(bands['front_0_3']['total']):,} | "
                f"{float(bands['front_0_3']['rate']):.1%} |"
            ),
            (
                "| 后部 4-14 | "
                f"{int(bands['later_4_end']['zero']):,} / "
                f"{int(bands['later_4_end']['total']):,} | "
                f"{float(bands['later_4_end']['rate']):.1%} |"
            ),
            "",
            (
                "后部风险约为前部的 "
                f"**{float(effect['later_vs_front_risk_ratio']):.2f} 倍**。这能估计总体损失, "
                "但不能据此把某一个会话标成随机失败。"
            ),
            "",
            "## 随机批次实验 (因果证据)",
            "",
            "| 检查 | 结果 |",
            "|---|---|",
            (
                "| 同一聊天从批次前部移到后部 | "
                f"有记忆→空结果 {pilot.yielded_front_zero_back}; "
                f"空结果→有记忆 {pilot.zero_front_yielded_back}; "
                f"McNemar p={pilot.position_mcnemar_p:.4f} |"
            ),
            (
                "| 同一聊天平均记忆数 | "
                f"前部 {pilot.front_mean_memories:.1f}; 后部 {pilot.back_mean_memories:.1f} |"
            ),
        ]
    )
    for size in (15, 5, 1):
        item = pilot.batch_sizes[str(size)]
        lines.append(
            f"| 批量 {size} | 零产出 {int(item['zero'])}/{int(item['sessions'])} "
            f"({float(item['zero_rate']):.1%}); 平均 "
            f"{float(item['memories_per_session']):.1f} 条记忆/会话 |"
        )
    lines.extend(
        [
            "",
            "结论: 较大的提取批次会造成**位置相关的遗漏**, 这是受控实验支持的因果结论。",
            "至于是输出长度、枚举漂移还是模型注意力分配导致, 现有证据不能继续细分。",
            "",
            "## 可复核来源",
            "",
            f"- 完整逐会话审计: `{audit_source}`",
            f"- 随机批次实验原始记录: `{pilot_source}`",
            "",
        ]
    )
    return "\n".join(lines)
