"""阶段2：Agent 工具层（Mock）—— 模拟广告业务平台 API。

Agent 不能直接读原始大文件，必须通过这 4 个工具取数：
  1. get_campaign_events     取广告单元时间窗内的 event-level 原始事件
  2. get_campaign_metrics    小时/天粒度聚合指标（曝光/消耗/点击/转化/CTR/CPC）
  3. run_data_quality_check  调用阶段1质量校验，返回数据异常告警
  4. search_knowledge_base   检索故障案例 RAG 知识库

所有工具返回 JSON-serializable 的 dict / list。
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from . import warehouse as W
from . import quality as Q
from .knowledge_base import get_kb, ROOT_TAGS

_kb_cache = None


def _kb() -> "object":
    global _kb_cache
    if _kb_cache is None:
        _kb_cache = get_kb()
    return _kb_cache


# ===========================================================================
# 指标计算核心（工作流/评测/Agent 共用）
# ===========================================================================

def _pct_change(cur: float, prev: float) -> Optional[float]:
    if prev is None or prev == 0:
        return None if cur == 0 else (100.0 if cur > 0 else -100.0)
    return round((cur - prev) / abs(prev), 4)


def compute_metrics(ad_id: int, granularity: str = "hourly", window_hours: int = 48) -> dict:
    """某广告单元近 window_hours 的指标：当前24h vs 前一24h + 逐桶序列。"""
    ev = W.events_of_campaign(ad_id)
    if ev.empty:
        return {"ad_id": ad_id, "exists": False, "buckets": []}

    anchor = ev["timestamp"].max()
    ev = ev[ev["timestamp"] >= anchor - pd.Timedelta(hours=window_hours)].copy()

    m = pd.DataFrame({
        "flag_imp": (ev["LogType"] == 1).astype(int),
        "flag_clk": (ev["LogType"] == 2).astype(int),
        "flag_conv": (ev["LogType"] == 3).astype(int),
        "flag_bid": (ev["LogType"] == 0).astype(int),
        "spend_v": np.where(ev["LogType"] == 1, ev["PayingPrice"], 0),
        "bid_v": np.where(ev["LogType"] == 0, ev["BiddingPrice"], np.nan),
    }, index=ev.index)

    if granularity == "daily":
        bucket = ev["timestamp"].dt.floor("D")
        bucket_label = "day"
    else:
        bucket = ev["timestamp"].dt.floor("h")
        bucket_label = "hour"

    g = pd.DataFrame({
        "imps": m["flag_imp"], "clks": m["flag_clk"], "convs": m["flag_conv"],
        "bids": m["flag_bid"], "spend": m["spend_v"], "avg_bid_v": m["bid_v"],
        "bucket": bucket,
    }).groupby("bucket").agg(
        imps=("imps", "sum"), clks=("clks", "sum"), convs=("convs", "sum"),
        bids=("bids", "sum"), spend=("spend", "sum"), avg_bid=("avg_bid_v", "mean"),
    ).sort_index()

    buckets = []
    for ts, r in g.tail(24).iterrows():
        imp = int(r["imps"]); clk = int(r["clks"]); conv = int(r["convs"])
        buckets.append({
            "ts": ts.strftime("%m-%d %H:00" if bucket_label == "hour" else "%m-%d"),
            "imps": imp, "clks": clk, "convs": conv, "bids": int(r["bids"]),
            "spend_cents": int(r["spend"]),
            "avg_bid": round(float(r["avg_bid"]), 2) if pd.notna(r["avg_bid"]) else None,
            "ctr": round(clk / imp, 6) if imp else None,
            "cpc": round(float(r["spend"]) / clk, 2) if clk else None,
        })

    end24 = anchor - pd.Timedelta(hours=24)
    cur = ev[ev["timestamp"] >= end24]
    prev = ev[(ev["timestamp"] >= end24 - pd.Timedelta(hours=24)) & (ev["timestamp"] < end24)]

    def totals(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"imps": 0, "clks": 0, "convs": 0, "bids": 0, "spend_cents": 0,
                    "ctr": 0.0, "cpc": 0.0, "avg_bid": None}
        imps = int((df["LogType"] == 1).sum())
        clks = int((df["LogType"] == 2).sum())
        convs = int((df["LogType"] == 3).sum())
        bids = int((df["LogType"] == 0).sum())
        spend = int(df.loc[df["LogType"] == 1, "PayingPrice"].sum())
        bidv = df.loc[df["LogType"] == 0, "BiddingPrice"]
        return {
            "imps": imps, "clks": clks, "convs": convs, "bids": bids,
            "spend_cents": spend,
            "ctr": round(clks / imps, 6) if imps else 0.0,
            "cpc": round(spend / clks, 2) if clks else 0.0,
            "avg_bid": round(float(bidv.mean()), 2) if len(bidv) else None,
        }

    tc, tp = totals(cur), totals(prev)
    pct = {k: _pct_change(tc.get(k) or 0, tp.get(k) or 0) for k in
           ("imps", "clks", "convs", "bids", "spend_cents")}
    pct["ctr"] = _pct_change(tc["ctr"], tp["ctr"]) if tp["ctr"] else None
    pct["avg_bid"] = _pct_change(tc["avg_bid"] or 0, tp["avg_bid"] or 0) \
        if tp.get("avg_bid") else None

    return {
        "ad_id": ad_id, "exists": True, "granularity": granularity,
        "anchor_end": anchor.strftime("%Y-%m-%d %H:%M"),
        "current_24h": tc, "previous_24h": tp, "pct_change_24h": pct,
        "buckets": buckets,
        "summary": {
            "spend_drop": pct["spend_cents"] is not None and pct["spend_cents"] <= -0.4,
            "imp_drop": pct["imps"] is not None and pct["imps"] <= -0.4,
            "conv_drop": pct["convs"] is not None and pct["convs"] <= -0.4,
            "ctr_surge": (pct["ctr"] or 0) >= 1.5,
            "bid_drop": (pct["avg_bid"] or 0) <= -0.25,
        },
    }


# ===========================================================================
# 四个 Mock 工具
# ===========================================================================

def get_campaign_events(campaign_id: int, start_time: Optional[str] = None,
                        end_time: Optional[str] = None) -> dict:
    """返回广告单元时间范围内的 event-level 原始事件（抽样展示，不返回全量）。"""
    ev = W.events_of_campaign(campaign_id)
    if ev.empty:
        return {"campaign_id": campaign_id, "exists": False, "total_events": 0,
                "note": "campaign not found in the warehouse or no data in this window"}

    ts = pd.to_datetime(ev["timestamp"])
    if start_time:
        ev = ev[ts >= pd.to_datetime(start_time)]
    if end_time:
        ev = ev[ts <= pd.to_datetime(end_time)]
    if ev.empty:
        return {"campaign_id": campaign_id, "exists": True, "total_events": 0,
                "note": "no events in the given window"}

    counts = ev["LogType"].value_counts().sort_index().to_dict()
    bid_ids = set(ev.loc[ev["LogType"] == 0, "BidID"])
    orphan = int(ev.loc[ev["LogType"].isin([1, 2, 3]) & ~ev["BidID"].isin(bid_ids)].shape[0])

    samples = []
    for lt in (0, 1, 2, 3):
        sub = ev[ev["LogType"] == lt].head(3)
        for _, r in sub.iterrows():
            samples.append({
                "logtype": int(lt),
                "bid": r["BidID"],
                "ts": pd.Timestamp(r["timestamp"]).strftime("%m-%d %H:%M:%S"),
                "ad_id": int(r["AdID"]), "creative": int(r["CreativeID"]),
                "region": int(r["Region"]),
                "domain": str(r["Domain"])[:28],
                "bid_price_cents": int(r["BiddingPrice"]),
                "pay_price_cents": int(r["PayingPrice"]),
            })

    return {
        "campaign_id": campaign_id, "exists": True,
        "window": {"start": str(ev["timestamp"].min()), "end": str(ev["timestamp"].max())},
        "total_events": int(len(ev)),
        "counts": {f"logtype_{k}": int(v) for k, v in counts.items()},
        "events_without_parent_bid": orphan,
        "samples": samples,
        "note": "production data is huge; only a small sample is returned as evidence",
    }


def get_campaign_metrics(campaign_id: int, granularity: str = "hourly",
                         window_hours: int = 48) -> dict:
    """聚合指标：曝光/消耗/点击/转化/CTR/CPC，及同比上个窗口的变化率。"""
    if granularity not in ("hourly", "daily"):
        granularity = "hourly"
    return compute_metrics(int(campaign_id), granularity, int(window_hours))


def run_data_quality_check(campaign_id: int, window_hours: Optional[int] = None,
                           store_snapshot: bool = False) -> dict:
    """执行阶段1数据质量校验，返回该广告单元的告警。"""
    report = Q.run_quality_checks(ad_id=int(campaign_id), window_hours=window_hours)
    if store_snapshot and report["issues"]:
        try:
            _kb().add_doc({
                "doc_id": f"snap-{report['run_id']}",
                "title": f"Data-quality snapshot AdID={campaign_id} {report['run_id']}",
                "symptom": report["summary"],
                "root_cause_tag": "no_anomaly",
                "root_cause": "to be judged by the agent together with the metrics",
                "evidence": Q.snapshot_text(report),
                "actions": ["Combine with metric trends to judge whether it is a real business event",
                            "Raise a ticket to fix if it looks like a pipeline issue"],
                "tags": ["data quality", "snapshot"], "source": "quality-snapshot",
            })
        except Exception:
            pass
    return {
        "campaign_id": int(campaign_id),
        "summary": report["summary"],
        "totals": report["totals"],
        "issues": report["issues"],
        "scope": report["scope"],
        "run_id": report["run_id"],
    }


def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    """检索历史故障案例 / 行业排查经验知识库。"""
    hits = _kb().search(query, top_k=max(1, int(top_k)))
    return {"query": query, "hits": hits, "total_docs": _kb().count()}


# ===========================================================================
# 注册表：供 Planner/Executor/文档复用
# ===========================================================================

TOOL_SCHEMA = {
    "get_campaign_metrics": {
        "description": "Get aggregated metrics for a campaign (AdID): current 24h vs previous 24h totals, "
                       "pct_change_24h, and hourly buckets. Fields: imps, clks, convs, bids, spend_cents, "
                       "avg_bid, ctr, cpc. Call it first when spend/impressions change abruptly.",
        "params": {"campaign_id": "int", "granularity": "str(hourly|daily, default hourly)",
                   "window_hours": "int(default 48)"},
    },
    "get_campaign_events": {
        "description": "Get sampled event-level raw logs for a campaign in a time window: counts per LogType "
                       "(0 bid / 1 impression / 2 click / 3 conversion), orphan events without a parent bid, "
                       "and a few sample rows. Use it to verify whether the detail behind the metrics is credible.",
        "params": {"campaign_id": "int", "start_time": "str(optional)",
                   "end_time": "str(optional)"},
    },
    "run_data_quality_check": {
        "description": "Run the data-quality checks for a campaign (primary-key integrity / price sanity / "
                       "time consistency / statistical outliers) and return an issues array; rule_id values: "
                       "R1 orphan / R1b bid-without-impression / R2 overbilling / R3 time reversal / R4 outlier.",
        "params": {"campaign_id": "int", "window_hours": "int(optional)"},
    },
    "search_knowledge_base": {
        "description": "Search the troubleshooting knowledge base with a symptom description "
                       "(e.g. 'normal bidding but no impressions, spend collapse') and get the most relevant "
                       "historical root causes and action plans. Call it before finalizing root causes/advice.",
        "params": {"query": "str", "top_k": "int(default 3)"},
    },
}

INT_PARAMS = {"campaign_id", "window_hours", "top_k"}


def tool_names() -> list[str]:
    return list(TOOL_SCHEMA)


def call_tool(name: str, args: dict) -> dict:
    """执行工具；将字符串参数按 schema 做基础类型转换；错误包装成可读结果。"""
    fn = {"get_campaign_metrics": get_campaign_metrics,
          "get_campaign_events": get_campaign_events,
          "run_data_quality_check": run_data_quality_check,
          "search_knowledge_base": search_knowledge_base}.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    kwargs = dict(args or {})
    for k in list(kwargs):
        if k in INT_PARAMS:
            try:
                kwargs[k] = int(re.sub(r"\D", "", str(kwargs[k])) or 0)
            except (TypeError, ValueError):
                kwargs[k] = 0
    try:
        result = fn(**kwargs)
        return result if isinstance(result, dict) else {"result": result}
    except Exception as e:  # 工具异常不中断 Agent
        return {"error": f"{name} failed: {e!r}", "campaign_id": kwargs.get("campaign_id")}
