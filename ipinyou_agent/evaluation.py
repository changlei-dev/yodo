"""阶段5：评测闭环 —— 从数据集构造评测集，跑 Agent，量化根因准确率/建议合理性/工具路径冗余，
沉淀 bad-case 反哺知识库与 Prompt，形成迭代闭环。

评测维度：
  1. 根因判断准确率  ：predicted root_cause_tags vs 注入的真实标签(ground truth)
  2. 建议合理性       ：建议与领域专家预期动作的关键词覆盖率(+可选 LLM 复核)
  3. 工具路径冗余     ：非必要工具调用数 / 总调用数；最小必要工具集覆盖率
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from . import config as C
from .agent_core import run_diagnosis
from .generator import campaign_plan
from .knowledge_base import get_kb
from .llm_client import ChatLLM

# ---------------------------------------------------------------------------
# 评测集构造
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    id: str
    ad_id: int
    query: str
    gt_tags: list[str] = field(default_factory=list)          # 注入根因
    gt_action_kw: list[str] = field(default_factory=list)      # 建议关键词
    minimal_tools: list[str] = field(default_factory=list)     # 必要工具集
    difficulty: str = "medium"
    note: str = ""


CASE_LIB: dict[str, dict] = {
    "delivery_outage": dict(
        gt_tags=["delivery_outage"],
        gt_action_kw=["creative", "review", "channel", "targeting", "bid", "delivery"],
        minimal_tools=["get_campaign_metrics", "run_data_quality_check",
                       "get_campaign_events", "search_knowledge_base"],
        difficulty="hard",
        queries=[
            "Campaign AdID:2345 spend collapsed in the last 24h while the bid was not lowered; "
            "find the root cause and give fixes",
            "Campaign 2345 impressions and spend nearly hit zero in the past two days, "
            "clicks/conversions gone, but bid logs look normal; help locate and advise",
            "AdID 2345 spend dropped 85% with no bid change; suspect a creative or channel issue; "
            "verify and give troubleshooting advice",
        ],
        note="real incident: normal bidding but delivery outage (simulated creative rejection / "
             "channel failure)"),
    "conv_clock_anomaly": dict(
        gt_tags=["conv_clock_anomaly"],
        gt_action_kw=["timestamp", "attribution", "clock", "sdk", "calibrat", "recomput", "reversal"],
        minimal_tools=["get_campaign_metrics", "run_data_quality_check",
                       "search_knowledge_base"],
        difficulty="medium",
        queries=[
            "Campaign 1002 conversions dropped sharply in the last 24h while clicks and impressions "
            "look normal; investigate the cause",
            "AdID 1002 conversion timestamps look wrong; check the data quality and advise",
            "Unit 1002 clicks are fine but conversions dropped; suspect attribution or reporting-time "
            "problems; analyze it",
        ],
        note="injected fault: conversion timestamps rolled back ~1 day (time reversal, R3)"),
    "price_anomaly": dict(
        gt_tags=["price_anomaly"],
        gt_action_kw=["billing", "reconcil", "bid", "compensat", "mode", "overcharg"],
        minimal_tools=["get_campaign_metrics", "run_data_quality_check",
                       "search_knowledge_base"],
        difficulty="easy",
        queries=[
            "Some impressions of campaign 1003 billed above the bid (PayingPrice > BiddingPrice); "
            "investigate and advise",
            "AdID 1003 reconciliation shows unusually high cost; suspect a billing anomaly; "
            "check the data and advise",
            "Unit 1003 has many rows billed higher than the bid; what is the cause and how to handle it?",
        ],
        note="injected fault: PayingPrice > BiddingPrice (R2)"),
    "imp_dataloss": dict(
        gt_tags=["imp_dataloss"],
        gt_action_kw=["log", "reporting", "backfill", "gap", "pipeline", "access", "recompute"],
        minimal_tools=["get_campaign_metrics", "run_data_quality_check",
                       "get_campaign_events", "search_knowledge_base"],
        difficulty="hard",
        queries=[
            "Campaign 1004 had a spend/impression gap for some hours yesterday, then recovered; "
            "suspect data-reporting loss; locate the root cause",
            "AdID 1004 impressions were mysteriously ~30% lower for a few hours and recovered by "
            "themselves; bid normal; judge whether it is log loss",
            "Unit 1004 spend curve has a gap; suspect the reporting pipeline; give a conclusion and a plan",
        ],
        note="injected fault: random 30% impression-log loss that later recovered"),
    "ctr_stat_outlier": dict(
        gt_tags=["ctr_stat_outlier"],
        gt_action_kw=["anti-fraud", "invalid traffic", "click", "a/b", "outlier", "traffic",
                      "fingerprint"],
        minimal_tools=["get_campaign_metrics", "run_data_quality_check",
                       "search_knowledge_base"],
        difficulty="medium",
        queries=[
            "Campaign 1005 CTR jumped to nearly 3x normal today with low impression volume; judge "
            "whether it is a real lift or invalid traffic, and advise",
            "AdID 1005 had abnormal CTR spikes in a few hours; analyze data quality and click sources",
            "Unit 1005 click rate is an outlier: few impressions but many clicks; suspected click "
            "fraud; verify and advise",
        ],
        note="injected fault: hourly CTR spikes + all-day CTR lift (R4 statistical outlier)"),
    "bid_drop": dict(
        gt_tags=["bid_drop"],
        gt_action_kw=["auto-bid", "bid", "ocpx", "cost", "restore", "coefficient", "win rate"],
        minimal_tools=["get_campaign_metrics", "run_data_quality_check",
                       "search_knowledge_base"],
        difficulty="easy",
        queries=[
            "Campaign 1006 spend has been declining since last night; check whether the bid strategy changed",
            "AdID 1006 spend is falling and the average bid looks lower too; confirm the root cause and advise",
            "Unit 1006 volume shrank; suspect the auto-bid was lowered; how to recover it?",
        ],
        note="injected fault: bid lowered ~55% in the last 12h"),
    "healthy": dict(
        gt_tags=["no_anomaly"],
        gt_action_kw=["watch", "normal", "fluctuation", "compare", "no action"],
        minimal_tools=["get_campaign_metrics", "run_data_quality_check"],
        difficulty="easy",
        queries=[
            "Campaign 1001: is the overall performance in the last 24h normal? run a health check "
            "and highlight anything abnormal",
            "Campaign 1007: any abnormal metric changes? run a full data-quality pass",
            "AdID 1001: were there any anomalies yesterday? run a check-up and explain",
        ],
        note="healthy control campaign (real incident baseline)"),
    "optimization": dict(
        gt_tags=["optimization"],
        gt_action_kw=["creative", "a/b", "targeting", "bid", "landing", "remarketing", "audience"],
        minimal_tools=["get_campaign_metrics", "search_knowledge_base"],
        difficulty="easy",
        queries=[
            "Campaign 1007: how can we further optimize CTR? give actionable advice",
            "Help improve 1007's click-through rate; give an optimization plan",
            "1007 performs mediocre; how should we tune it to lift conversions?",
        ],
        note="non-incident consulting: optimization path"),
    "not_found": dict(
        gt_tags=["campaign_not_found"],
        gt_action_kw=["verify", "confirm", "adid", "ingestion", "account"],
        minimal_tools=["get_campaign_metrics"],
        difficulty="easy",
        queries=[
            "Campaign 9999: how is the recent spend? analyze it",
            "Check today's data for AdID: 88888",
        ],
        note="edge case: non-existent campaign"),
}


def build_cases(cfg: dict | None = None) -> list[EvalCase]:
    """根据当前数据集的 fault 分布构造评测集(每类取1~N个自然语言变体)。"""
    cfg = cfg or C.load_config()
    plans = campaign_plan(cfg["data"].get("fault_mode", "full"))
    by_id = {p.ad_id: p for p in plans}
    healthy_ids = [p.ad_id for p in plans if p.healthy]

    cases: list[EvalCase] = []
    idx = 0
    for kind, spec in CASE_LIB.items():
        if kind == "healthy":
            ad_ids = healthy_ids[:1] if healthy_ids else [1001]
        elif kind == "optimization":
            ad_ids = [1007]
        elif kind == "not_found":
            ad_ids = [9999]
        else:
            candidates = [p.ad_id for p in plans
                          if p.fault.kind.replace("none", "healthy") == kind or
                          _kind_of_fault(p.fault.kind) == kind]
            if not candidates:
                continue
            ad_ids = candidates[:1]
        for q in spec["queries"]:
            idx += 1
            cases.append(EvalCase(
                id=f"case-{idx:02d}-{kind}", ad_id=ad_ids[0], query=q,
                gt_tags=list(spec["gt_tags"]), gt_action_kw=list(spec["gt_action_kw"]),
                minimal_tools=list(spec["minimal_tools"]),
                difficulty=spec["difficulty"], note=spec["note"]))
    return cases


def _kind_of_fault(kind: str) -> str:
    map_ = {
        "channel_outage": "delivery_outage",
        "imp_dataloss": "imp_dataloss",
        "conv_clock_reversal": "conv_clock_anomaly",
        "price_anomaly": "price_anomaly",
        "ctr_outlier": "ctr_stat_outlier",
        "bid_drop": "bid_drop",
        "none": "healthy",
    }
    return map_.get(kind, kind)


# ---------------------------------------------------------------------------
# 评测运行
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case: EvalCase
    report: dict = field(default_factory=dict)
    used_tools: list = field(default_factory=list)
    n_calls: int = 0
    duration: float = 0.0
    mode: str = ""
    tag_hit: int = 0          # 命中的 gt tag 数
    root_precision: float = 0.0
    root_recall: float = 0.0
    root_correct: bool = False
    action_recall: float = 0.0
    action_reasonable: bool = False
    extra_calls: int = 0
    coverage: float = 0.0
    error: str = ""
    judge_comment: str = ""


def _tag_score(pred_tags: list, gt: list):
    gt_set = set(gt)
    pred_set = set(pred_tags)
    hit = len(gt_set & pred_set)
    prec = hit / len(pred_set) if pred_set else 0.0
    rec = hit / len(gt_set) if gt_set else 0.0
    return hit, prec, rec, gt_set.issubset(pred_set)


def _action_recall(recommendations: list, kw: list) -> float:
    if not kw:
        return 1.0
    text = "".join(recommendations).lower()
    hit = sum(1 for k in kw if k in text)
    return hit / len(kw)


def _tool_stats(transcript_calls: list, minimal: list) -> tuple[int, float]:
    """返回 (冗余调用数, 最小必要工具覆盖率)。"""
    min_set = set(minimal)
    seen = set()
    extra = 0
    for tool in transcript_calls:
        if tool in min_set and tool not in seen:
            seen.add(tool)
        else:
            extra += 1
    coverage = len(seen) / len(min_set) if min_set else 1.0
    return extra, round(coverage, 3)


def _llm_judge(cfg: dict, case: EvalCase, report: dict) -> tuple[bool | None, str]:
    """可选：用 LLM 复核建议是否合理/可执行。返回 (是否合理, 评语)。"""
    try:
        llm = ChatLLM(cfg)
        if not llm.available:
            return None, ""
        prompt = (
            "You are a senior ad-delivery expert. Judge whether the Agent's recommendations for the "
            "campaign anomaly are reasonable and executable.\n"
            f"Business question: {case.query}\nExpected action keywords (reference only): "
            f"{case.gt_action_kw}\n"
            f"Agent recommendations: {report.get('recommendations')}\n"
            'Output JSON only: {"reasonable": true/false, "comment": "one-line comment in English"}')
        obj = llm.chat_json([{"role": "user", "content": prompt}])
        if obj:
            return bool(obj.get("reasonable")), str(obj.get("comment", ""))
    except Exception:
        pass
    return None, ""


def evaluate(cfg: dict | None = None, mode: Optional[str] = None, max_cases: int = 0,
             use_judge: bool | None = None, shuffle: bool = False) -> dict:
    cfg = cfg or C.load_config()
    cases = build_cases(cfg)
    if max_cases and max_cases > 0:
        cases = cases[:max_cases]
    if shuffle:
        rng = np.random.default_rng(int(time.time()))
        rng.shuffle(cases)
    use_judge = cfg["eval"].get("llm_judge", False) if use_judge is None else use_judge

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        cr = CaseResult(case=case)
        try:
            res = run_diagnosis(case.query, cfg, ad_ids=[case.ad_id], mode=mode)
            cr.report = res.report
            cr.used_tools = res.used_tools
            cr.n_calls = res.n_tool_calls
            cr.mode = res.mode
            cr.duration = res.duration_sec
            cr.error = res.error
            # 评分
            hit, prec, rec, correct = _tag_score(res.report.get("root_cause_tags") or [],
                                                 case.gt_tags)
            cr.tag_hit, cr.root_precision, cr.root_recall, cr.root_correct = \
                hit, round(prec, 3), round(rec, 3), correct
            recs = res.report.get("recommendations") or []
            cr.action_recall = round(_action_recall(recs, case.gt_action_kw), 3)
            calls = [s["tool"] for s in res.transcript if s.get("tool")]
            cr.extra_calls, cr.coverage = _tool_stats(calls, case.minimal_tools)
            cr.action_reasonable = cr.action_recall >= 0.5
            if use_judge and (case.gt_tags or ["x"])[0] != "no_anomaly":
                ok, comment = _llm_judge(cfg, case, res.report)
                if ok is not None:
                    cr.action_reasonable = ok
                    cr.judge_comment = comment
        except Exception as e:  # noqa: BLE001
            cr.error = repr(e)
        print(f"[eval] {i}/{len(cases)} {case.id} ad={case.ad_id} gt={case.gt_tags} "
              f"pred={cr.report.get('root_cause_tags')} calls={cr.n_calls} "
              f"correct={cr.root_correct} action_recall={cr.action_recall} "
              f"extra={cr.extra_calls} ({cr.duration:.1f}s)")
        results.append(cr)

    summary = _summarize(results)
    meta = _persist(cfg, results, summary)
    return {"summary": summary, "results": results, "cases": cases, "meta_paths": meta}


def _summarize(results: list[CaseResult]) -> dict:
    n = len(results) or 1
    correct = sum(1 for r in results if r.root_correct)
    rec_ok = sum(1 for r in results if r.action_reasonable)
    extra = sum(r.extra_calls for r in results)
    total_calls = sum(r.n_calls for r in results)
    return {
        "total_cases": len(results),
        "root_accuracy": round(correct / n, 4),
        "root_precision_macro": round(sum(r.root_precision for r in results) / n, 4),
        "root_recall_macro": round(sum(r.root_recall for r in results) / n, 4),
        "action_reasonable_rate": round(rec_ok / n, 4),
        "action_recall_macro": round(sum(r.action_recall for r in results) / n, 4),
        "avg_tool_calls": round(total_calls / n, 3),
        "avg_extra_calls": round(extra / n, 3),
        "extra_ratio": round(extra / max(1, total_calls), 4),
        "tool_coverage_macro": round(sum(r.coverage for r in results) / n, 4),
        "by_difficulty": _groupby(results, "difficulty"),
        "by_tag": _groupby(results, "tag"),
        "failed_cases": [r.case.id for r in results
                         if not r.root_correct or not r.action_reasonable],
    }


def _groupby(results: list[CaseResult], key: str):
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in results:
        k = r.case.difficulty if key == "difficulty" else r.case.gt_tags[0]
        agg[k][1] += 1
        if r.root_correct:
            agg[k][0] += 1
    return {k: {"cases": v[1], "correct": v[0],
                "acc": round(v[0] / v[1], 3)} for k, v in sorted(agg.items())}


def _persist(cfg: dict, results: list[CaseResult], summary: dict) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    C.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = C.EVAL_DIR / f"eval_{stamp}.json"
    lines_path = C.EVAL_DIR / f"eval_cases_{stamp}.jsonl"
    md_path = C.REPORT_DIR / f"eval_report_{stamp}.md"
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    payload = []
    for r in results:
        row = {"case": asdict(r.case),
               "report": r.report, "used_tools": r.used_tools,
               "n_calls": r.n_calls, "mode": r.mode, "duration": r.duration,
               "tag_hit": r.tag_hit, "root_correct": r.root_correct,
               "root_precision": r.root_precision, "root_recall": r.root_recall,
               "action_recall": r.action_recall, "action_reasonable": r.action_reasonable,
               "extra_calls": r.extra_calls, "coverage": r.coverage,
               "error": r.error, "judge_comment": r.judge_comment}
        payload.append(row)
    json_path.write_text(json.dumps({"summary": summary, "results": payload},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    with open(lines_path, "w", encoding="utf-8") as f:
        for row in payload:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    md_path.write_text(eval_report_md(summary, payload), encoding="utf-8")
    return {"json": str(json_path), "jsonl": str(lines_path), "md": str(md_path)}


def eval_report_md(summary: dict, payload: list) -> str:
    s = summary
    lines = [
        "# Agent Evaluation Report (Iteration Loop)",
        "",
        f"- Evaluated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Cases: {s['total_cases']}",
        f"- **Root-cause accuracy**: {s['root_accuracy']:.1%} "
        f"(macro precision {s['root_precision_macro']:.2f} / recall {s['root_recall_macro']:.2f})",
        f"- **Actionable recommendations (kw recall >= 50%)**: {s['action_reasonable_rate']:.1%} "
        f"(action recall macro {s['action_recall_macro']:.2f})",
        f"- **Tool path**: avg calls {s['avg_tool_calls']}, avg redundant {s['avg_extra_calls']}, "
        f"minimal-tool coverage {s['tool_coverage_macro']:.2f}",
        f"- Failed (bad) cases: {s['failed_cases']}",
        "",
        "## Accuracy by difficulty / root cause",
        "| Group | Cases | Correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for grp_name in ("by_difficulty", "by_tag"):
        lines.append(f"**{grp_name}**")
        for k, v in s[grp_name].items():
            lines.append(f"| {k} | {v['cases']} | {v['correct']} | {v['acc']:.1%} |")
        lines.append("")
    lines += ["## Per-case details", "",
              "| Case | AdID | Difficulty | GT | Predicted | Root OK | Action recall | Calls/extra |",
              "|---|---|---|---|---|---|---|---|"]
    for p in payload:
        case = p["case"]
        lines.append(
            f"| {case['id']} | {case['ad_id']} | {case['difficulty']} | "
            f"{case['gt_tags']} | {p['report'].get('root_cause_tags')} | "
            f"{'✓' if p['root_correct'] else '✗'} | {p['action_recall']:.2f} | "
            f"{p['n_calls']}/{p['extra_calls']} |")
    return "\n".join(lines)


def absorb_bad_cases(cfg: dict | None = None, result: dict | None = None) -> int:
    """把失败(bad)case 沉淀进知识库，形成『评测->失败->补知识->重测』闭环。"""
    cfg = cfg or C.load_config()
    kb = get_kb(cfg)
    results = result.get("results", []) if result else []
    if not results:  # 直接扫描最近一次 eval json
        import glob as _glob
        files = sorted(_glob.glob(str(C.EVAL_DIR / "eval_*.json")), reverse=True)
        if files:
            data = json.loads(open(files[0], encoding="utf-8").read())
            results = [type("R", (), {k: v for k, v in p.items()})() for p in data["results"]]
    n = 0
    for r in results:
        if getattr(r, "root_correct", False) and getattr(r, "action_reasonable", False):
            continue
        case = getattr(r, "case", None)
        if isinstance(case, dict):
            case = type("C", (), {k: v for k, v in case.items()})()
        if not case:
            continue
        n += 1
        kb.add_doc({
            "doc_id": f"badcase-{case.id}",
            "title": f"[bad-case] {case.query[:40]}",
            "symptom": f"evaluation failure case {case.id}: campaign {case.ad_id}, "
                       f"ground-truth root cause {case.gt_tags}",
            "root_cause_tag": (case.gt_tags or ["no_anomaly"])[0],
            "root_cause": f"Agent concluded "
                          f"{getattr(r, 'report', {}).get('root_cause_tags')} with action recall "
                          f"{getattr(r, 'action_recall', 0):.2f}; knowledge for this scenario needs "
                          f"strengthening",
            "evidence": f"expected action keywords: {case.gt_action_kw}; Agent made "
                        f"{n_calls_of(r)} tool call(s).",
            "actions": list(case.gt_action_kw or []),
            "tags": list(case.gt_tags or []),
            "source": "eval-badcase",
        })
    return n


def n_calls_of(r) -> int:
    return getattr(r, "n_calls", getattr(r, "extra_calls", 0))
