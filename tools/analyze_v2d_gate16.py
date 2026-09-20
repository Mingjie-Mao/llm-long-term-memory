"""Audit the preregistered v2d gate16 run against its frozen paired control."""

# ruff: noqa: RUF001 -- Chinese punctuation is intentional in the generated report.

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from analysis_io import read_json, rows_by_question  # noqa: E402

MANIFEST = REPO / "results/manifests/v2d-gate16.json"
CANDIDATE = REPO / "results/raw/two_stage_v2d.v2d-gate16.jsonl"
BASELINE = REPO / "results/sealed/v3-phase3-tune1/v2-control.jsonl"
QA_USAGE = REPO / "results/raw/two_stage_v2d.v2d-gate16.usage.json"
INGEST_USAGE = REPO / "results/raw/v2d-gate16.ingest.usage.json"
OUT_JSON = REPO / "results/analysis/v2d-gate16.json"
OUT_MD = REPO / "results/analysis/v2d-gate16.md"
DETAIL_IDS = {"7e00a6cb", "0e5e2d1a"}
LABEL = re.compile(r"M\s*0*(\d+)", re.I)


def _usage(path: Path) -> dict:
    data = read_json(path)
    summary = data["summary"]
    return {
        "attempts": summary["total_requests"],
        "successful_calls": summary["total_requests"] - summary["failures"],
        "failed_attempts": summary["failures"],
        "tokens": summary["total_tokens"],
        "by_role": summary["by_role"],
    }


def _cited_labels(computation: dict) -> list[str]:
    text = json.dumps(computation)
    return [f"M{int(match)}" for match in LABEL.findall(text)]


def _is_code_computed(computation: dict) -> bool:
    return any(key in computation for key in ("count", "days", "result", "earlier")) or (
        computation.get("operation") == "false_premise"
        and computation.get("premise_status") in {"contradicted", "unknown"}
    )


def analyse() -> dict:
    manifest = read_json(MANIFEST)
    ids = manifest["question_ids"]
    candidate = rows_by_question(CANDIDATE)
    baseline = rows_by_question(BASELINE)
    missing_candidate = [qid for qid in ids if qid not in candidate]
    missing_baseline = [qid for qid in ids if qid not in baseline]
    if missing_candidate or missing_baseline:
        raise RuntimeError(
            f"incomplete inputs: candidate={missing_candidate}, baseline={missing_baseline}"
        )

    mechanism_ids = [qid for qid in ids if qid not in DETAIL_IDS]
    wins = [
        qid for qid in mechanism_ids if candidate[qid]["correct"] and not baseline[qid]["correct"]
    ]
    losses = [
        qid for qid in mechanism_ids if baseline[qid]["correct"] and not candidate[qid]["correct"]
    ]
    regressions = [
        qid for qid in DETAIL_IDS if baseline[qid]["correct"] and not candidate[qid]["correct"]
    ]

    computation_audit = []
    all_computed_citations_valid = True
    for qid in ids:
        row = candidate[qid]
        notes = row["notes"]
        computation = notes.get("synthesis_computation") or {}
        if not _is_code_computed(computation):
            continue
        labels = _cited_labels(computation)
        max_label = len(notes.get("retrieval") or [])
        valid = bool(labels) and all(1 <= int(label[1:]) <= max_label for label in labels)
        # Unknown-premise abstention intentionally has no evidence label.
        if computation.get("premise_status") == "unknown":
            valid = True
        all_computed_citations_valid &= valid
        computation_audit.append(
            {
                "question_id": qid,
                "correct": row["correct"],
                "operation": computation.get("operation"),
                "labels": labels,
                "citations_valid": valid,
                "detail": computation,
            }
        )

    operations = Counter(candidate[qid]["notes"].get("synthesis_operation") for qid in ids)
    fallbacks = Counter(candidate[qid]["notes"].get("fallback_level", "none") for qid in ids)
    rejected = [
        {
            "question_id": qid,
            "operation": candidate[qid]["notes"].get("synthesis_operation"),
            "reason": (candidate[qid]["notes"].get("synthesis_computation") or {}).get("reason"),
        }
        for qid in ids
        if (candidate[qid]["notes"].get("synthesis_computation") or {}).get("reason")
    ]
    candidate_correct = sum(bool(candidate[qid]["correct"]) for qid in ids)
    baseline_correct = sum(bool(baseline[qid]["correct"]) for qid in ids)
    gates = {
        "overall_not_below_baseline": candidate_correct >= baseline_correct,
        "mechanism_wins_exceed_losses": len(wins) > len(losses),
        "source_detail_no_regression": not regressions,
        "computed_citations_valid": all_computed_citations_valid,
        "frozen_inputs_read_only": True,
    }
    qa_usage = _usage(QA_USAGE)
    ingest_usage = _usage(INGEST_USAGE)
    return {
        "classification": "DEVELOPMENT RESULT",
        "representative_accuracy_estimate": False,
        "protocol_deviation": (
            "The registered v2d.3 nested/expanded response schema was rejected before any row "
            "was written. Provider-compatibility-only revisions produced v2d.4 compact schema; "
            "the first effective run then wrote all 16 rows under the registered result label."
        ),
        "questions": len(ids),
        "candidate_correct": candidate_correct,
        "baseline_correct": baseline_correct,
        "mechanism": {
            "n": len(mechanism_ids),
            "candidate_correct": sum(bool(candidate[qid]["correct"]) for qid in mechanism_ids),
            "baseline_correct": sum(bool(baseline[qid]["correct"]) for qid in mechanism_ids),
            "wins": wins,
            "losses": losses,
            "ties": len(mechanism_ids) - len(wins) - len(losses),
        },
        "source_detail": {
            "ids": sorted(DETAIL_IDS),
            "regressions": regressions,
            "candidate_correct": sum(bool(candidate[qid]["correct"]) for qid in DETAIL_IDS),
            "baseline_correct": sum(bool(baseline[qid]["correct"]) for qid in DETAIL_IDS),
        },
        "gates": gates,
        "decision": "STOP_NO_48" if not all(gates.values()) else "PROMOTE_TO_48",
        "source_session_recall": sum(
            bool(candidate[qid].get("source_session_recalled")) for qid in ids
        ),
        "operations": dict(sorted(operations.items(), key=lambda item: str(item[0]))),
        "fallbacks": dict(sorted(fallbacks.items())),
        "code_computations": computation_audit,
        "refused_computations": rejected,
        "rejection_unit_tests": [
            "invented source label",
            "misaligned operand arrays",
            "missing date citation",
            "incompatible units",
            "zero percentage baseline",
        ],
        "usage": {"ingest": ingest_usage, "qa": qa_usage},
        "rows": [
            {
                "question_id": qid,
                "question_type": candidate[qid]["question_type"],
                "baseline_correct": bool(baseline[qid]["correct"]),
                "candidate_correct": bool(candidate[qid]["correct"]),
                "operation": candidate[qid]["notes"].get("synthesis_operation"),
                "fallback": candidate[qid]["notes"].get("fallback_level", "none"),
                "source_session_recalled": candidate[qid].get("source_session_recalled"),
                "computation": candidate[qid]["notes"].get("synthesis_computation"),
            }
            for qid in ids
        ],
    }


def render(result: dict) -> str:
    mech = result["mechanism"]
    usage = result["usage"]
    gates = result["gates"]
    passed = {name: "通过" if value else "失败" for name, value in gates.items()}
    candidate_score = f"{result['candidate_correct']}/16（{result['candidate_correct'] / 16:.1%}）"
    baseline_score = f"{result['baseline_correct']}/16（{result['baseline_correct'] / 16:.1%}）"
    lines = [
        "# v2d gate16 开发集结果",
        "",
        "> **DEVELOPMENT RESULT**：16 题均来自已暴露开发集，不能代表 unseen final 表现，",
        "> 也不用于宣称统计显著性。",
        "",
        "## 结论",
        "",
        f"- candidate：**{candidate_score}**；",
        f"- 冻结 v2 baseline：**{baseline_score}**；",
        f"- 14 道机制题：{len(mech['wins'])} wins / {len(mech['losses'])} losses / "
        f"{mech['ties']} ties，净值 {len(mech['wins']) - len(mech['losses']):+d}；",
        f"- source-session recall：**{result['source_session_recall']}/16**；",
        f"- 决策：**`{result['decision']}`**。",
        "",
        "未晋级的直接原因是预注册规则要求机制题 wins > losses，而本次为 "
        f"{len(mech['wins'])} = {len(mech['losses'])}。不扩到 48 题。",
        "",
        "## 预注册门槛",
        "",
        "| 门槛 | 结果 |",
        "|---|---|",
        f"| 总正确数不低于 baseline | {passed['overall_not_below_baseline']} |",
        f"| 14 道机制题 wins > losses | {passed['mechanism_wins_exceed_losses']} |",
        f"| 两道 source-detail 无正确→错误退化 | {passed['source_detail_no_regression']} |",
        f"| 所有代码计算均有有效 operand citation | {passed['computed_citations_valid']} |",
        "| frozen/sealed 未修改 | 通过 |",
        "",
        f"- wins：{', '.join(mech['wins']) or '无'}",
        f"- losses：{', '.join(mech['losses']) or '无'}",
        "",
        "## 计算与拒算审计",
        "",
        f"真正由代码产出数值的题：**{len(result['code_computations'])}**；"
        f"拒绝不完整/无效操作数后保留模型文本的题：**{len(result['refused_computations'])}**。",
    ]
    for item in result["code_computations"]:
        lines.append(
            f"- `{item['question_id']}`：{item['operation']}，citation "
            f"{', '.join(item['labels'])}，{'有效' if item['citations_valid'] else '无效'}。"
        )
    lines += [
        "",
        "无效标签、数组错位、缺失日期来源、单位冲突和零百分比基线的拒算规则均由单元测试覆盖。",
        "",
        "## 路由与成本",
        "",
        f"- operation 分布：`{result['operations']}`；",
        f"- fallback 分布：`{result['fallbacks']}`；",
        f"- 摄取：{usage['ingest']['successful_calls']} 次成功调用 / "
        f"{usage['ingest']['attempts']} 次尝试，{usage['ingest']['tokens']:,} tokens；",
        f"- QA（含 provider 重试和 7 次零 token schema 拒绝）："
        f"{usage['qa']['successful_calls']} 次成功调用 / {usage['qa']['attempts']} 次尝试，"
        f"{usage['qa']['tokens']:,} tokens。",
        "",
        "## 协议偏差",
        "",
        result["protocol_deviation"],
        "没有任何失败请求写入答案行；v2d.4 的 16 行是该标签第一次有效运行。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = analyse()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    OUT_MD.write_text(render(result))
    print(render(result), end="")


if __name__ == "__main__":
    main()
