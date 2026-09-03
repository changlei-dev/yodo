"""阶段3：广告故障排查 Agent —— 工具调用 ReAct 主循环（LangGraph 编排）。

- LangGraph StateGraph：plan(思考) -> tool(行动/观察) 循环，直至 final(输出结构化报告)；
- 双 Planner：
    * MockPlanner  确定性规则(无 LLM 也可本地跑通，完全对齐 JD 示例剧情)
    * LLMPlanner   接 Qwen(OpenAI 兼容)，用 ReAct Prompt + 工具描述驱动；
- 默认 mode=auto：配置了 QWEN_API_KEY 用 LLM，否则自动 Mock。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TypedDict

import numpy as np
import pandas as pd

from . import config as C
from . import tools as T
from . import warehouse as W
from .knowledge_base import TAG_ZH, get_kb
from .llm_client import ChatLLM, extract_json_object
from .prompts import build_user_message

# ===========================================================================
# 基础工具：解析 campaign id / 摘要 / 画图
# ===========================================================================

_ID_EXPLICIT = [
    r"AdID[:：]?\s*(\d+)",
    r"广告单元\s*[:：#\s]*(\d+)",
    r"campaign[_\- ]?id[:：]?\s*(\d+)",
    r"广告组\s*[:：#\s]*(\d+)",
    r"单元\s*(\d{4,})",
]
KNOWN: list[int] = []


def _known_ids(cfg) -> list[int]:
    global KNOWN
    if not KNOWN:
        try:
            KNOWN = W.known_campaign_ids()
        except Exception:
            KNOWN = []
    return KNOWN


def extract_campaign_ids(text: str, cfg=None) -> list[int]:
    """从自然语言中提取广告单元ID。先显式格式，再回退到知识库内已知ID。"""
    cfg = cfg or C.load_config()
    ids: list[int] = []
    for pat in _ID_EXPLICIT:
        for m in re.finditer(pat, text, re.IGNORECASE):
            v = int(m.group(1))
            if v not in ids:
                ids.append(v)
    if ids:
        return ids
    known = _known_ids(cfg)
    for tok in re.findall(r"\d+", text):
        v = int(tok)
        if len(tok) >= 4 and v in known and v not in ids:
            ids.append(v)
    return ids


def _pct_str(v, digits=1) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:+.{digits}f}%"


def _summarize_metrics(args, r) -> str:
    if not r.get("exists"):
        return f"广告单元 {r.get('ad_id')} 不存在/无数据"
    p = r["pct_change_24h"]
    cur = r["current_24h"]
    bits = [
        f"AdID{r['ad_id']} 当前24h: imps={cur['imps']:,} spend_cents={cur['spend_cents']:,} "
        f"clks={cur['clks']} convs={cur['convs']} avg_bid={cur['avg_bid']} ctr={cur['ctr']}",
        f"同比前24h: imps={_pct_str(p.get('imps'))} spend={_pct_str(p.get('spend_cents'))} "
        f"clks={_pct_str(p.get('clks'))} convs={_pct_str(p.get('convs'))} "
        f"avg_bid={_pct_str(p.get('avg_bid'))} ctr={_pct_str(p.get('ctr'))}",
    ]
    tail = r["buckets"][-4:]
    if tail:
        last = tail[-1]
        bits.append(f"最近1小时: imps={last['imps']} spend_cents={last['spend_cents']} "
                    f"clks={last['clks']} convs={last['convs']}")
    return "；".join(bits)


def _summarize_dq(args, r) -> str:
    issues = r.get("issues") or []
    if not issues:
        return f"AdID{args.get('campaign_id')} 数据质量校验通过：主键/价格/时间/统计均无告警"
    parts = []
    for it in issues:
        parts.append(f"{it['rule_id']}({it['name']})[{it['severity']}] {it['message'][:80]}")
    return f"AdID{args.get('campaign_id')} 质量告警 {len(issues)} 条：" + " | ".join(parts)


def _summarize_events(args, r) -> str:
    if not r.get("exists"):
        return f"AdID{r.get('campaign_id')} 无原始事件"
    c = r.get("counts") or {}
    return (f"AdID{r['campaign_id']} 事件总量={r['total_events']}, 按类型 counts={c}, "
            f"无出价父记录={r['events_without_parent_bid']}; 已抽样展示")


def _summarize_kb(args, r) -> str:
    hits = r.get("hits") or []
    parts = [f"检索到 {len(hits)} 条案例"]
    for h in hits:
        parts.append(f"[{h['root_cause_tag']}] {h['title']} -> {h['actions'][0] if h.get('actions') else ''}")
    return "；".join(parts)


def summarize_result(name: str, args: dict, result: dict) -> str:
    if not isinstance(result, dict) or "error" in result:
        return f"工具错误: {result}"
    if name == "get_campaign_metrics":
        return _summarize_metrics(args, result)
    if name == "run_data_quality_check":
        return _summarize_dq(args, result)
    if name == "get_campaign_events":
        return _summarize_events(args, result)
    if name == "search_knowledge_base":
        return _summarize_kb(args, result)
    return json.dumps(result, ensure_ascii=False, default=str)[:400]


# ===========================================================================
# 决策（规则）：由指标 + 质量告警收敛到根因
# ===========================================================================

def _local_dip_info(buckets: list) -> tuple[bool, bool, float]:
    """(是否出现局部凹陷, 是否已恢复, 凹陷深度) — 用于识别随机丢包 vs 持续中断。"""
    bs = [b for b in buckets if (b.get("imps") or 0) > 0]
    if len(bs) < 8:
        return False, False, 1.0
    spend = np.array([b.get("spend_cents") or 0 for b in bs], dtype=float)
    med = float(np.median(spend))
    if med <= 0:
        return False, False, 1.0
    low = spend[spend < med * 0.72]
    dip = (float(np.min(spend)) / med) if len(spend) else 1.0
    tail_ok = bool(np.median(spend[-4:]) >= med * 0.8)
    return bool(len(low) >= 3), tail_ok, round(dip, 3)


def _decide_tag(metrics: dict, dq_report: dict, query: str) -> dict:
    """规则式根因判定。返回 dict(tag, reason, confidence)。"""
    if not metrics.get("exists"):
        return {"tag": "campaign_not_found", "reason": "该 AdID 仓库中不存在", "confidence": 0.99}

    issues = dq_report.get("issues") or []
    rule_ids = [it["rule_id"] for it in issues]
    pct = metrics.get("pct_change_24h") or {}
    cur = metrics.get("current_24h") or {}

    imp_p = pct.get("imps")
    spd_p = pct.get("spend_cents")
    clk_p = pct.get("clks")
    conv_p = pct.get("convs")
    bid_p = pct.get("avg_bid")
    ctr_p = pct.get("ctr")

    def neg(v, thr=-0.4):
        return v is not None and v <= thr

    def normalish(v):
        return v is None or abs(v) < 0.3

    # 1) 明确的字段级异常优先（与量级无关）
    if "R2" in rule_ids:
        return {"tag": "price_anomaly", "reason": "质量校验 R2：PayingPrice>BiddingPrice",
                "confidence": 0.97}
    if "R3" in rule_ids:
        return {"tag": "conv_clock_anomaly",
                "reason": "质量校验 R3：转化/点击时间早于曝光(时间倒挂)",
                "confidence": 0.95}

    dip, recovered, dip_depth = _local_dip_info(metrics.get("buckets") or [])
    severe = neg(imp_p, -0.45) and neg(spd_p, -0.45)

    # 2) 出价下调
    if bid_p is not None and bid_p <= -0.25 and (neg(imp_p, -0.2) or neg(spd_p, -0.2)):
        return {"tag": "bid_drop", "reason": f"均价下调 {_pct_str(bid_p)}，量级同步收缩",
                "confidence": 0.93}

    # 3) CTR 统计离群：CTR 大幅抬升且曝光未同步大跌(排除断量分母效应)
    if ctr_p is not None and ctr_p >= 0.5 and not severe and not neg(imp_p, -0.3):
        return {"tag": "ctr_stat_outlier",
                "reason": f"CTR 同比 {_pct_str(ctr_p)} 且曝光稳定，结合点击来源/质量 R4 判断是否存在异常流量",
                "confidence": 0.85}

    # 4) 持续中断：消耗/曝光双跌 + 出价正常
    if severe and normalish(bid_p):
        if "R1b" in rule_ids or "R1" in rule_ids:
            return {"tag": "delivery_outage",
                    "reason": f"消耗/曝光大幅下滑(imp {_pct_str(imp_p)}, spend {_pct_str(spd_p)})"
                              f"但出价正常，且质量校验提示有出价无曝光/孤儿事件，指向投放服务中断(素材/渠道/定向)",
                    "confidence": 0.9}
        return {"tag": "delivery_outage",
                "reason": f"消耗/曝光大幅下滑但出价正常，出价后无曝光占比上升，指向投放链路问题",
                "confidence": 0.85}

    # 5) 随机丢包：整体未暴跌但有局部凹陷且已恢复
    if dip and recovered and not severe and dip_depth <= 0.72:
        return {"tag": "imp_dataloss",
                "reason": f"整体量级未暴跌但小时序列出现凹陷(深度 {dip_depth})后已恢复，"
                          f"符合上报丢包特征(有出价无曝光、缺口随机、可自愈)",
                "confidence": 0.85}

    # 6) 转化下跌而点击正常
    if neg(conv_p, -0.4) and normalish(clk_p) and not neg(imp_p, -0.4):
        if "R3" in rule_ids:
            return {"tag": "conv_clock_anomaly", "reason": "转化下滑且质量校验命中时间倒挂", "confidence": 0.9}
        return {"tag": "attribution_loss",
                "reason": "点击/曝光正常但转化下滑，需排查归因回传与转化埋点", "confidence": 0.7}

    if neg(spd_p, -0.5) and normalish(bid_p) and not neg(imp_p, -0.3):
        return {"tag": "price_anomaly",
                "reason": "消耗口径异常(曝光正常但消耗大跌)，排查计费/成交价", "confidence": 0.7}

    # 7) 兜底
    if _looks_optimization(query):
        return {"tag": "optimization", "reason": "指标健康，属优化类咨询", "confidence": 0.9}
    return {"tag": "no_anomaly", "reason": "各维度变化在正常波动内，质量校验无告警",
            "confidence": 0.88}


def _looks_optimization(query: str) -> bool:
    kw = ["优化", "怎么提升", "如何提高", "如何提升", "提高ctr", "提升ctr", "建议怎么",
          "效果不好怎么办", "怎么优化", "提高转化", "提升转化"]
    return any(k in query.lower() for k in kw)


def _looks_health(query: str) -> bool:
    return any(k in query for k in ["健康", "是否正常", "检查", "有没有异常", "有没有问题", "体检"])


def _kb_query_for(tag: str, query: str, metrics: dict) -> str:
    cur = metrics.get("current_24h") or {}
    text = f"{TAG_ZH.get(tag, tag)} {query} imps={cur.get('imps')} spend={cur.get('spend_cents')}"
    return text[:220]


# ===========================================================================
# Planner
# ===========================================================================

@dataclass
class AgentState:
    query: str
    campaign_ids: list[int]
    transcript: list = field(default_factory=list)
    tool_calls: int = 0
    report: dict = field(default_factory=dict)
    mode: str = "mock"
    error: str = ""


class _GraphState(TypedDict, total=False):
    """LangGraph 共享状态 schema。"""
    query: str
    campaign_ids: list
    transcript: list
    tool_calls: int
    decision: dict | None
    report: dict


class Planner:
    mode = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.kb = get_kb(cfg)

    # ---------- 子类实现 ----------
    def plan(self, state: AgentState) -> dict:
        raise NotImplementedError


class MockPlanner(Planner):
    """确定性排障策略（无 LLM 兜底 / 离线演示）。"""
    mode = "mock"

    def plan(self, state: AgentState) -> dict:
        if not state.campaign_ids:
            return self._final(state, {"tag": "campaign_not_found",
                                       "reason": "未能从问题中解析出广告单元ID", "confidence": 0.8},
                               dq={}, metrics={})
        cid = state.campaign_ids[0]
        used = {s["tool"] for s in state.transcript}
        metrics = self._last(state, "get_campaign_metrics") or {}
        if not metrics:
            return {"action": "tool", "tool": "get_campaign_metrics",
                    "args": {"campaign_id": cid, "window_hours": 48}}

        # 目标单元不存在 -> 直接结论
        if not metrics.get("exists"):
            return self._final(state, self._decide(state), dq={}, metrics=metrics)

        opt_intent = _looks_optimization(state.query)
        if opt_intent and not self._last(state, "run_data_quality_check"):
            # 优化类问题：只需指标 + 知识库
            if "search_knowledge_base" not in used:
                return {"action": "tool", "tool": "search_knowledge_base",
                        "args": {"query": _kb_query_for("optimization", state.query, metrics), "top_k": 3}}
            return self._final(state, self._decide(state), dq={}, metrics=metrics)

        # 排障类：先质量校验
        if "run_data_quality_check" not in used:
            return {"action": "tool", "tool": "run_data_quality_check",
                    "args": {"campaign_id": cid}}

        decision = self._decide(state)
        tag = decision["tag"]
        # 核对原始事件：仅链路/丢包类根因且质量告警指向事件层时，或用户明确要求看明细
        rules = self._rule_ids(state)
        shape_tags = {"delivery_outage", "imp_dataloss"}
        need_events = ((tag in shape_tags and ("R1" in rules or "R1b" in rules))
                       or any(k in state.query for k in ["明细", "原始", "日志", "事件", "渠道", "核实"]))
        if need_events and "get_campaign_events" not in used:
            return {"action": "tool", "tool": "get_campaign_events",
                    "args": {"campaign_id": cid}}

        # 业务类根因在收尾前检索知识库
        if tag in ("delivery_outage", "imp_dataloss", "ctr_stat_outlier", "bid_drop",
                   "price_anomaly", "conv_clock_anomaly", "attribution_loss") \
                and "search_knowledge_base" not in used:
            return {"action": "tool", "tool": "search_knowledge_base",
                    "args": {"query": _kb_query_for(tag, state.query, metrics), "top_k": 3}}

        dq = self._last(state, "run_data_quality_check") or {}
        return self._final(state, decision, dq=dq, metrics=metrics)

    # ---------- 内部 ----------
    def _last(self, state: AgentState, tool: str) -> dict | None:
        for s in reversed(state.transcript):
            if s["tool"] == tool:
                return s.get("result")
        return None

    def _rule_ids(self, state: AgentState) -> list[str]:
        dq = self._last(state, "run_data_quality_check") or {}
        return [i.get("rule_id", "") for i in (dq.get("issues") or [])]

    def _decide(self, state: AgentState) -> dict:
        metrics = self._last(state, "get_campaign_metrics") or {}
        dq = self._last(state, "run_data_quality_check") or {}
        return _decide_tag(metrics, dq, state.query)

    def _final(self, state: AgentState, decision: dict, dq: dict, metrics: dict) -> dict:
        report = build_mock_report(
            state.query, state.campaign_ids, decision,
            metrics=metrics,
            dq=dq,
            events=self._last(state, "get_campaign_events"),
            kb=self._last(state, "search_knowledge_base"),
            kb_store=self.kb,
        )
        return {"action": "final", "report": report}


class LLMPlanner(Planner):
    """Qwen ReAct Planner。失败/不可用时回退 MockPlanner 保证闭环。"""
    mode = "llm"

    def __init__(self, cfg: dict, llm: ChatLLM | None = None):
        super().__init__(cfg)
        self.llm = llm or ChatLLM(cfg)
        self._fallback = MockPlanner(cfg)

    def plan(self, state: AgentState) -> dict:
        try:
            messages = build_user_message(state.query, state.campaign_ids,
                                          state.transcript, self.cfg["agent"]["max_steps"])
            obj = self.llm.chat_json(messages)
            if obj and obj.get("action") == "tool":
                tool = obj.get("tool")
                if tool in T.TOOL_SCHEMA:
                    args = obj.get("args") or {}
                    args.setdefault("campaign_id", state.campaign_ids[0] if state.campaign_ids else 0)
                    return {"action": "tool", "tool": tool, "args": args}
            if obj and obj.get("action") == "final" and obj.get("report"):
                report = normalize_report(obj["report"], state)
                return {"action": "final", "report": report}
            # 非法输出 -> 回退规则
        except Exception:
            pass
        fb = self._fallback.plan(state)
        return fb


# ===========================================================================
# 报告构建与归一化
# ===========================================================================

def _percent(x) -> str:
    return f"{x * 100:+.1f}%" if x is not None else "N/A"


def _pick_kb_docs(kb_store, tag: str, top: int = 2) -> list:
    exact = kb_store.search_tag(tag)
    if exact:
        return exact[:top]
    return kb_store.search(TAG_ZH.get(tag, tag), top_k=top)


def build_mock_report(query, campaign_ids, decision, metrics, dq, events, kb,
                      kb_store) -> dict:
    cid = campaign_ids[0] if campaign_ids else None
    tag = decision["tag"]
    cur = (metrics or {}).get("current_24h") or {}
    pct = (metrics or {}).get("pct_change_24h") or {}
    issues = (dq or {}).get("issues") or []

    def f(k):
        return _percent(pct.get(k))

    if tag == "campaign_not_found":
        phenomenon = [f"问题：{query}", f"目标广告单元 {cid} 在数据仓库中不存在或查询窗口无数据"]
        causes = [{"tag": tag, "desc": decision["reason"], "probability": "high"}]
        actions = ["核对广告单元ID是否正确(AdID 为整数)", "确认该单元是否在本账号/本数据源范围内",
                   "如确实存在，检查接入/同步是否延迟"]
        confirms = [f"人工在广告平台确认 AdID={cid} 是否有效"]
        conf = float(decision["confidence"])
    else:
        tag_zh = TAG_ZH.get(tag, tag)
        if tag == "no_anomaly":
            actions = ["无需处理，当前指标在正常波动范围内，保持观察",
                       "对比同主体下其他广告单元/大盘数据，判断是否为行业性波动",
                       "复核数据口径与质量报告，确认无告警后再下结论"]
        else:
            # 知识依据：检索命中文档 + 同标签规范文档（保证命中噪声时建议仍完整可执行）
            docs = (kb or {}).get("hits") or []
            if kb_store is not None:
                seen_ids = {d.get("doc_id") for d in docs if d.get("doc_id")}
                for d in kb_store.search_tag(tag, top_k=3):
                    if d.get("doc_id") not in seen_ids:
                        docs.append(d)
                        seen_ids.add(d.get("doc_id"))
            actions = []
            for d in docs:
                if d.get("root_cause_tag") == tag:
                    actions.extend(d.get("actions", []))
            if not actions:
                actions = ["结合指标与质量校验结果复核数据口径", "对比同主体下其他广告单元是否同步波动",
                           "必要时在广告平台侧进一步核对该单元配置与状态"]
        actions = _dedup(actions)[:5]

        phenomenon = [f"用户问题：{query}"]
        if metrics.get("exists"):
            phenomenon.append(
                f"指标：当前24h imps={cur.get('imps'):,} spend={cur.get('spend_cents'):,} "
                f"clks={cur.get('clks')} convs={cur.get('convs')} avg_bid={cur.get('avg_bid')}；"
                f"同比 imps {f('imps')} / spend {f('spend_cents')} / clks {f('clks')} / "
                f"convs {f('convs')} / avg_bid {f('avg_bid')}")
        if issues:
            phenomenon.append(f"数据质量：{dq.get('summary', '')}")
        phenomenon.append(f"判定：{decision['reason']}（标签 {tag_zh}）")

        causes = [{"tag": tag, "desc": decision["reason"], "probability": "high"}]
        confirms = ["需结合广告平台/媒体后台人工确认最终根因" if tag == "delivery_outage"
                    else "关注下个周期指标是否恢复以验证判断"]
        conf = float(decision["confidence"])

    return {
        "summary": f"广告单元 {cid}：{TAG_ZH.get(tag, tag)} —— {decision['reason'][:80]}",
        "query": query,
        "ad_id": cid,
        "phenomenon": phenomenon,
        "root_causes": causes,
        "root_cause_tags": [tag],
        "recommendations": actions,
        "needs_confirm": confirms,
        "confidence": conf,
        "evidence": {
            "metrics_current": cur,
            "metrics_pct": pct,
            "quality_summary": (dq or {}).get("summary", ""),
            "quality_rule_ids": [i.get("rule_id") for i in issues],
            "kb_used": bool(kb),
        },
    }


def normalize_report(raw: dict, state: AgentState) -> dict:
    """LLM 报告字段校验/补全，保证评估脚本字段健壮。"""
    from .knowledge_base import ROOT_TAGS
    tags = []
    for rc in raw.get("root_causes") or []:
        t = rc.get("tag") if isinstance(rc, dict) else None
        if t in ROOT_TAGS and t not in tags:
            tags.append(t)
    if not tags:
        tags = ["no_anomaly"]
    return {
        "summary": str(raw.get("summary") or ""),
        "query": state.query,
        "ad_id": state.campaign_ids[0] if state.campaign_ids else None,
        "phenomenon": [str(x) for x in (raw.get("phenomenon") or [])],
        "root_causes": raw.get("root_causes") or [],
        "root_cause_tags": tags,
        "recommendations": [str(x) for x in (raw.get("recommendations") or [])],
        "needs_confirm": [str(x) for x in (raw.get("needs_confirm") or [])],
        "confidence": float(raw.get("confidence") or 0.5),
        "evidence": {"llm_mode": True},
    }


def _dedup(seq: list) -> list:
    out = []
    for x in seq:
        if x and x not in out:
            out.append(x)
    return out


# ===========================================================================
# 编排入口：LangGraph StateGraph（langgraph 缺失时降级为普通循环）
# ===========================================================================

@dataclass
class DiagnosisResult:
    query: str
    campaign_ids: list[int]
    report: dict
    transcript: list
    mode: str
    n_tool_calls: int
    used_tools: list
    duration_sec: float
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query, "campaign_ids": self.campaign_ids,
            "report": self.report, "transcript": self.transcript,
            "mode": self.mode, "n_tool_calls": self.n_tool_calls,
            "used_tools": self.used_tools, "duration_sec": round(self.duration_sec, 3),
        }


def _make_planner(cfg: dict, mode: str) -> Planner:
    mode = (mode or cfg["agent"].get("mode", "auto")).lower()
    if mode == "llm":
        return LLMPlanner(cfg)
    if mode == "mock":
        return MockPlanner(cfg)
    # auto
    try:
        llm = ChatLLM(cfg)
        if llm.available:
            return LLMPlanner(cfg, llm)
    except Exception:
        pass
    return MockPlanner(cfg)


def _mock_final_from_transcript(query: str, cids: list[int], transcript: list,
                                kb_store=None) -> dict:
    """从 transcript 兜底生成 mock 报告（图异常/报告缺失时）。"""
    metrics = dq = events = kb = None
    for s in reversed(transcript):
        if s["tool"] == "get_campaign_metrics" and metrics is None:
            metrics = s["result"]
        elif s["tool"] == "run_data_quality_check" and dq is None:
            dq = s["result"]
        elif s["tool"] == "get_campaign_events" and events is None:
            events = s["result"]
        elif s["tool"] == "search_knowledge_base" and kb is None:
            kb = s["result"]
    metrics = metrics or {}
    dq = dq or {}
    if not metrics.get("exists"):
        decision = {"tag": "campaign_not_found", "reason": "广告单元在仓库中不存在", "confidence": 0.8}
    else:
        decision = _decide_tag(metrics, dq, query)
    return build_mock_report(query, cids, decision, metrics=metrics, dq=dq,
                             events=events, kb=kb, kb_store=kb_store or get_kb())


def run_diagnosis(query: str, cfg: dict | None = None, ad_ids: Optional[list[int]] = None,
                  mode: Optional[str] = None, max_steps: int | None = None,
                  return_only_report: bool = False) -> DiagnosisResult | dict:
    """Agent 主入口：输入自然语言问题 -> 结构化诊断报告。

    优先走 LangGraph StateGraph(plan<->tool 循环)；langgraph 缺失/异常时，
    降级为同一决策逻辑的普通循环，保证任何环境都能闭环。
    """
    cfg = cfg or C.load_config()
    t0 = time.time()
    if ad_ids is None:
        ad_ids = extract_campaign_ids(query, cfg)
    max_steps = max_steps or int(cfg["agent"].get("max_steps", 10))
    planner = _make_planner(cfg, mode)
    actual_mode = planner.mode if hasattr(planner, "mode") else mode
    cids = ad_ids[:1]

    out: dict | None = None
    fallback_reason = ""

    # ---------- LangGraph 路径 ----------
    try:
        from langgraph.graph import END, StateGraph

        def plan_node(s: dict) -> dict:
            decision = planner.plan(AgentState(
                query=query, campaign_ids=cids,
                transcript=s["transcript"], tool_calls=s["tool_calls"], mode=actual_mode))
            upd = {"decision": decision}
            if decision["action"] == "final":
                upd["report"] = decision.get("report") or {}
            return upd

        def tool_node(s: dict) -> dict:
            dec = s.get("decision") or {}
            result = T.call_tool(dec.get("tool"), dec.get("args") or {})
            step = {"tool": dec.get("tool"), "args": dec.get("args") or {},
                    "result": result,
                    "finding": summarize_result(dec.get("tool"), dec.get("args") or {}, result)}
            return {"transcript": s["transcript"] + [step],
                    "tool_calls": int(s["tool_calls"]) + 1, "decision": None}

        def route(s: dict) -> str:
            if int(s.get("tool_calls", 0)) >= max_steps:
                return "end"
            if (s.get("decision") or {}).get("action") == "final":
                return "end"
            return "tool"

        g = StateGraph(_GraphState)
        g.add_node("plan", plan_node)
        g.add_node("tool", tool_node)
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", route, {"tool": "tool", "end": "end"})
        g.add_edge("tool", "plan")
        g.add_edge("end", END)
        app = g.compile()
        out = app.invoke({"query": query, "campaign_ids": cids,
                          "transcript": [], "tool_calls": 0,
                          "decision": None, "report": {}})
    except Exception as e:  # noqa: BLE001
        fallback_reason = f"langgraph fallback: {e!r}"
        out = None

    # ---------- 降级路径 ----------
    if out is None:
        out = _run_loop(planner, AgentState(query=query, campaign_ids=cids, mode=actual_mode),
                        max_steps)

    transcript = [s for s in (out.get("transcript") or []) if s.get("tool")]
    final_report = out.get("report") or {}
    if not final_report:
        final_report = _mock_final_from_transcript(query, cids, transcript, kb_store=planner.kb)

    n_calls = int(out.get("tool_calls", len(transcript)))
    used = list(dict.fromkeys(s["tool"] for s in transcript))
    if return_only_report:
        return final_report
    return DiagnosisResult(
        query=query, campaign_ids=cids, report=final_report, transcript=transcript,
        mode=actual_mode, n_tool_calls=n_calls, used_tools=used,
        duration_sec=round(time.time() - t0, 3), error=fallback_reason)


def _run_loop(planner: Planner, state: AgentState, max_steps: int) -> dict:
    """等价于 LangGraph 编排的普通循环（降级用）。"""
    transcript: list = []
    tool_calls = 0
    report: dict = {}
    while tool_calls < max_steps:
        decision = planner.plan(AgentState(query=state.query, campaign_ids=state.campaign_ids,
                                           transcript=transcript, tool_calls=tool_calls,
                                           mode=state.mode))
        if decision["action"] == "final":
            report = decision.get("report") or {}
            break
        result = T.call_tool(decision["tool"], decision.get("args") or {})
        step = {"tool": decision["tool"], "args": decision.get("args") or {},
                "result": result, "finding": summarize_result(decision["tool"], decision.get("args") or {}, result)}
        transcript.append(step)
        tool_calls += 1
    if not report:
        decision = {"tag": "no_anomaly", "reason": "步数耗尽仍未收敛", "confidence": 0.4}
        m = {}
        for s_ in reversed(transcript):
            if s_["tool"] == "get_campaign_metrics":
                m = s_["result"]
        dq = {}
        for s_ in reversed(transcript):
            if s_["tool"] == "run_data_quality_check":
                dq = s_["result"]
        report = build_mock_report(state.query, state.campaign_ids, decision,
                                   metrics=m, dq=dq, events=None, kb=None,
                                   kb_store=planner.kb)
    return {"transcript": transcript, "tool_calls": tool_calls, "report": report}


def _wrap_result(query, state, out, planner, t0, error="") -> DiagnosisResult:
    transcript = out.get("transcript") or []
    n_calls = out.get("tool_calls", len([t for t in transcript if t.get("tool")]))
    used = [s.get("tool") for s in transcript if s.get("tool")]
    return DiagnosisResult(
        query=query, campaign_ids=state.campaign_ids, report=out.get("report") or {},
        transcript=transcript, mode=getattr(planner, "mode", "mock"),
        n_tool_calls=n_calls, used_tools=used,
        duration_sec=round(time.time() - t0, 3), error=error)


# ===========================================================================
# 报告 Markdown 渲染
# ===========================================================================

def report_to_markdown(result: DiagnosisResult) -> str:
    r = result.report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 广告故障排查报告",
        "",
        f"- 时间: {now}",
        f"- 查询: {r.get('query', result.query)}",
        f"- 广告单元: {r.get('ad_id', result.campaign_ids)}",
        f"- 排查模式: {result.mode}（工具调用 {result.n_tool_calls} 次）",
        f"- 置信度: {r.get('confidence')}",
        f"- 结论: {r.get('summary', '')}",
        "",
        "## 一、现象",
    ]
    for p in r.get("phenomenon") or []:
        lines.append(f"- {p}")

    lines += ["", "## 二、候选根因"]
    for rc in r.get("root_causes") or []:
        lines.append(f"- **{rc.get('tag')}**({rc.get('probability', '-')}): {rc.get('desc')}")

    lines += ["", "## 三、可执行建议"]
    for i, a in enumerate(r.get("recommendations") or [], 1):
        lines.append(f"{i}. {a}")

    lines += ["", "## 四、待确认事项"]
    for i, c in enumerate(r.get("needs_confirm") or [], 1):
        lines.append(f"{i}. {c}")

    if result.transcript:
        lines += ["", "## 五、排查过程（工具调用轨迹）"]
        for i, s in enumerate(result.transcript, 1):
            lines.append(f"{i}. 调用 **{s['tool']}** args={s.get('args')}")
            lines.append(f"   → {s.get('finding', '')}")
    return "\n".join(lines)
