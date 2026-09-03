"""阶段1：数据质量校验模块（Agent 可调用工具之一 run_data_quality_check 的后端）。

规则（对齐 JD 要求）：
  R1 主键完整性：有曝光/点击/转化但无对应出价记录(丢包/孤儿)；以及 bid->imp 占比突增(上报中断)
  R2 业务合理性：PayingPrice > BiddingPrice（扣费异常）
  R3 时间一致性：转化/点击时间早于曝光时间（时间戳上报 bug）
  R4 统计异常：按广告单元聚合 CTR/CPC/消耗 出现统计离群值(robust-z)

输出结构化的质量报告 JSON；由调用方写入向量知识库，供 Agent 检索。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from . import warehouse as W

ISSUE_STATUS = ["critical", "warning", "info"]


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------

def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / (mad + 1e-9)


def _fmt_dt(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def _pct(nume, deno) -> float:
    return float(nume) / float(deno) if deno else 0.0


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_quality_checks(
    cfg: dict | None = None,
    ad_id: Optional[int] = None,
    window_hours: Optional[int] = None,
    events_df: Optional[pd.DataFrame] = None,
) -> dict:
    """对事件数据执行 4 类质量校验。scope 可按广告单元/时间窗口收敛。"""
    cfg = cfg or C.load_config()
    qcfg = cfg["quality"]
    ev = events_df if events_df is not None else W.load_events()
    if ad_id is not None:
        ev = ev[ev["AdID"] == ad_id]
    if ev.empty:
        return _empty_report(ad_id)

    end_ts = ev["timestamp"].max()
    start_ts = ev["timestamp"].min()
    if window_hours:
        start_ts = max(start_ts, end_ts - pd.Timedelta(hours=window_hours))

    bid_ids = set(ev.loc[ev["LogType"] == 0, "BidID"])
    issues: list[dict] = []

    # ---------------- R1a 有下游事件但无出价记录（孤儿事件/丢包） ----------------
    child = ev[ev["LogType"].isin([1, 2, 3])]
    orphan = child[~child["BidID"].isin(bid_ids)] if bid_ids else child
    orphan_imps = int((orphan["LogType"] == 1).sum())
    orphan_clks = int((orphan["LogType"] == 2).sum())
    orphan_convs = int((orphan["LogType"] == 3).sum())
    imp_total = int((ev["LogType"] == 1).sum())
    if orphan_imps + orphan_clks + orphan_convs > 0:
        ratio = _pct(orphan_imps + orphan_clks, max(1, imp_total))
        sev = "critical" if ratio > float(qcfg.get("missing_parent_hard_ratio", 0.005)) else "warning"
        issues.append({
            "rule_id": "R1",
            "name": "主键完整性-孤儿事件",
            "severity": sev,
            "message": f"检测到 {orphan_imps} 条曝光 / {orphan_clks} 条点击 / "
                       f"{orphan_convs} 条转化没有对应的出价记录(BidID 在出价源中不存在)，"
                       f"疑似上游丢包或多源 join 失败。",
            "detail": {"orphan_imp": orphan_imps, "orphan_clk": orphan_clks,
                       "orphan_conv": orphan_convs, "imp_total": imp_total,
                       "ratio": round(ratio, 4)},
            "samples": _samples(ev, orphan["BidID"].head(3)),
        })

    # ---------------- R1b 出价后无曝光（bid->imp 比例突增） ----------------
    last48 = ev[ev["timestamp"] >= end_ts - pd.Timedelta(hours=48)]
    r_last24 = _bid_no_imp_ratio(last48, pd.Timedelta(hours=24), end_ts)
    r_prev24 = _bid_no_imp_ratio(last48, pd.Timedelta(hours=24), end_ts - pd.Timedelta(hours=24))
    delta = r_last24 - r_prev24
    if r_last24 > float(qcfg.get("orphan_bid_alert_ratio", 0.85)) or delta > 0.12:
        issues.append({
            "rule_id": "R1b",
            "name": "主键完整性-有出价无曝光",
            "severity": "warning",
            "message": f"最近24小时出价后无曝光占比 {r_last24:.1%}，较前一24h "
                       f"({r_prev24:.1%}) 变化 {delta:+.1%}，存在曝光上报/投放服务中断风险。",
            "detail": {"ratio_last24h": round(r_last24, 4),
                       "ratio_prev24h": round(r_prev24, 4),
                       "delta": round(delta, 4)},
            "samples": [],
        })

    # ---------------- R2 业务合理性：扣费 > 出价 ----------------
    imp_rows = ev[ev["LogType"] == 1]
    tol = int(qcfg.get("price_tolerance", 0))
    bad_price = imp_rows[(imp_rows["BiddingPrice"] > 0)
                         & (imp_rows["PayingPrice"] > imp_rows["BiddingPrice"] + tol)]
    if len(bad_price):
        issues.append({
            "rule_id": "R2",
            "name": "业务合理性-PayingPrice>BiddingPrice",
            "severity": "critical",
            "message": f"{len(bad_price)} 条曝光的实际扣费高于出价"
                       f"(PayingPrice>BiddingPrice)，可能存在计费异常或返点/加价规则 bug。",
            "detail": {"count": int(len(bad_price)),
                       "max_gap": int(bad_price["PayingPrice"].max() - bad_price["BiddingPrice"].max())},
            "samples": _samples(ev, bad_price["BidID"].head(3)),
        })

    # ---------------- R3 时间一致性：转化/点击早于曝光 ----------------
    t_issues = _time_reversal_issues(ev)
    for name, count, sids in t_issues:
        if count:
            issues.append({
                "rule_id": "R3",
                "name": name,
                "severity": "warning" if count < 50 else "critical",
                "message": f"{count} 条 {name} 时间早于其曝光时间，疑似客户端时钟/回传时间戳 bug。",
                "detail": {"count": count},
                "samples": sids[: int(qcfg.get("max_samples", 5))],
            })

    # ---------------- R4 统计异常：按广告单元聚合 CTR/CPC/消耗离群 ----------------
    outlier_units = _stat_outliers(ev, qcfg, ad_id)
    if outlier_units:
        issues.append({
            "rule_id": "R4",
            "name": "统计异常-广告单元指标离群",
            "severity": "warning",
            "message": f"{len(outlier_units)} 个广告单元日内聚合指标出现统计离群"
                       f"(robust-z>{qcfg.get('outlier_robust_z', 5)})，需关注 CTR 异常流量/CPC 成本失控。",
            "detail": {"units": outlier_units},
            "samples": [],
        })

    scope = {"ad_id": ad_id, "window_start": _fmt_dt(start_ts), "window_end": _fmt_dt(end_ts)}
    report = {
        "run_id": f"dq_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "scope": scope,
        "summary": _summarize(issues),
        "issues": issues,
        "totals": {
            "bids": int((ev["LogType"] == 0).sum()),
            "imps": imp_total,
            "clks": int((ev["LogType"] == 2).sum()),
            "convs": int((ev["LogType"] == 3).sum()),
        },
    }
    return report


def _empty_report(ad_id) -> dict:
    return {
        "run_id": "dq_empty", "scope": {"ad_id": ad_id, "note": "scope 内无数据"},
        "summary": "no data", "issues": [], "totals": {"bids": 0, "imps": 0, "clks": 0, "convs": 0},
    }


def _bid_no_imp_ratio(frame: pd.DataFrame, win: pd.Timedelta, end_ts: pd.Timestamp) -> float:
    w = frame[frame["timestamp"] >= end_ts - win]
    if w.empty:
        return 0.0
    bids = set(w.loc[w["LogType"] == 0, "BidID"])
    imps = set(w.loc[w["LogType"] == 1, "BidID"])
    if not bids:
        return 0.0
    return _pct(len(bids - imps), len(bids))


def _time_reversal_issues(ev: pd.DataFrame):
    imp_t = ev.loc[ev["LogType"] == 1, ["BidID", "timestamp"]].set_index("BidID")["timestamp"]
    out = []
    for lt, name in ((2, "点击"), (3, "转化")):
        sub = ev[ev["LogType"] == lt]
        if sub.empty:
            continue
        merged = sub.set_index("BidID")["timestamp"].to_frame("t_child").join(imp_t, how="inner")
        bad = merged[merged["t_child"] < merged["timestamp"]]
        out.append((f"时间一致性-{name}早于曝光", int(len(bad)),
                    [{"bid": k, "child_ts": _fmt_dt(v["t_child"]),
                      "imp_ts": _fmt_dt(v["timestamp"])} for k, v in bad.head(3).iterrows()]))
    return out


def _stat_outliers(ev: pd.DataFrame, qcfg: dict, ad_id: Optional[int]):
    """按广告单元(AdID)x 天 聚合，再在单元内部做时间维 robust-z 检测。"""
    imp = ev[ev["LogType"] == 1]
    if imp.empty:
        return []
    agg = imp.copy()
    agg["day"] = agg["timestamp"].dt.floor("D")
    clk = ev[ev["LogType"] == 2].groupby(["AdID"]).size()
    conv = ev[ev["LogType"] == 3].groupby(["AdID"]).size()
    spend = agg.groupby(["AdID", "day"])["PayingPrice"].sum()
    g = agg.groupby(["AdID", "day"]).agg(imps=("BidID", "count"))
    g["spend"] = spend
    g = g.reset_index()
    g["clks"] = g["AdID"].map(clk).fillna(0)
    g["convs"] = g["AdID"].map(conv).fillna(0)
    g["ctr"] = g["clks"] / g["imps"].replace(0, np.nan)
    g["cpc"] = g["spend"] / g["clks"].replace(0, np.nan)

    th = float(qcfg.get("outlier_robust_z", 5.0))
    res = []
    for metric in ("ctr", "cpc", "spend"):
        per_unit = []
        for u, sub in g.groupby("AdID"):
            if len(sub) < 3:
                continue
            z = _robust_z(sub[metric]).fillna(0)
            for _, row in sub[z.abs() > th].iterrows():
                per_unit.append({
                    "ad_id": int(u), "day": row["day"].strftime("%Y-%m-%d"),
                    "metric": metric, "value": round(float(row[metric]), 6),
                    "robust_z": round(float(z.loc[row.name]), 2),
                })
        res.extend(per_unit)
    # 过滤广告单元范围
    if ad_id is not None:
        res = [r for r in res if r["ad_id"] == ad_id]
    return res[:20]


def _samples(ev: pd.DataFrame, bid_ids) -> list:
    out = []
    sub = ev[ev["BidID"].isin(set(bid_ids))].head(5)
    for _, r in sub.iterrows():
        out.append({
            "bid": r["BidID"],
            "ts": _fmt_dt(r["timestamp"]),
            "ad_id": int(r["AdID"]),
            "logtype": int(r["LogType"]),
            "bid_price": int(r["BiddingPrice"]),
            "pay_price": int(r["PayingPrice"]),
        })
    return out


def _summarize(issues: list) -> str:
    if not issues:
        return "未发现异常：主键、价格、时间、统计指标均通过校验。"
    sev = sorted({x["severity"] for x in issues}, key=ISSUE_STATUS.index)
    names = "、".join(x["name"] for x in issues)
    return f"发现 {len(issues)} 类问题(最高级别:{sev[0]})：{names}。"


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def quality_to_markdown(report: dict) -> str:
    lines = [f"## 数据质量报告 {report['run_id']}",
             f"- 范围: {report['scope']}",
             f"- 结论: {report['summary']}",
             f"- 总量: {report['totals']}", ""]
    if not report["issues"]:
        lines.append("✅ 全部规则通过")
        return "\n".join(lines)
    lines.append("| 规则 | 级别 | 问题 |")
    lines.append("|---|---|---|")
    for it in report["issues"]:
        lines.append(f"| {it['rule_id']} {it['name']} | {it['severity']} | {it['message']} |")
    lines.append("")
    lines.append("### 样例")
    for it in report["issues"]:
        if it.get("samples"):
            lines.append(f"**{it['name']}** 样例: {json.dumps(it['samples'], ensure_ascii=False, default=str)}")
    return "\n".join(lines)


def snapshot_text(report: dict) -> str:
    """把质量报告转成可供知识库检索的文本片段。"""
    txt = (f"[数据质量报告] 范围={report['scope']} 结论={report['summary']} "
           f"总量={report['totals']} 问题数={len(report['issues'])}")
    for it in report["issues"]:
        txt += f" | {it['name']}({it['severity']}):{it['message']}"
    return txt
