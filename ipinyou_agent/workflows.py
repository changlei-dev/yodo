"""阶段4：业务工作流自动化。

1) 定时巡检工作流 inspect_all_campaigns：
   定时批量遍历全部广告单元 -> 检测消耗/CTR/数据质量突变 -> 自动生成 Markdown 巡检报告，
   标记风险广告单元（可 --watch 以守护进程方式周期运行）。

2) 自然语言分析工作流 nl_report：
   业务人员输入自然语言问题 -> Agent 自动完成数据查询/校验/根因分析 -> 输出业务报告，
   替代"人工写 SQL + 手工看报表"。
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd

from . import config as C
from . import quality as Q
from . import tools as T
from . import warehouse as W
from .agent_core import DiagnosisResult, _decide_tag, report_to_markdown, run_diagnosis

RISK_LEVELS = {"high": "高风险", "medium": "中风险", "low": "低风险"}


def risk_of(metrics: dict, dq: dict) -> tuple[str, str]:
    """把单个广告单元的指标+质量结果折叠成 (风险等级, 标签)。"""
    decision = _decide_tag(metrics, dq, "")
    tag = decision["tag"]
    issues = dq.get("issues") or []
    pct = metrics.get("pct_change_24h") or {}
    hard_dq = any(i.get("severity") in ("critical",) for i in issues)
    mild_dq = bool(issues)

    if tag in ("delivery_outage", "imp_dataloss", "price_anomaly", "conv_clock_anomaly"):
        return "high", tag
    if tag in ("ctr_stat_outlier", "bid_drop") or hard_dq:
        return "medium", tag
    if mild_dq and tag == "no_anomaly":
        return "low", "no_anomaly+minor_dq"
    if tag == "no_anomaly":
        return "low", "no_anomaly"
    return "low", tag


def scan_all(cfg: dict | None = None) -> list[dict]:
    """遍历全部广告单元，产出巡检行(仅只读计算)。"""
    cfg = cfg or C.load_config()
    ins = cfg["inspection"]
    W.load_events()  # 确保数据仓库就绪
    rows = []
    for ad_id in W.known_campaign_ids():
        metrics = T.compute_metrics(ad_id, granularity="hourly", window_hours=48)
        dq = Q.run_quality_checks(ad_id=ad_id)  # 不落快照，避免刷屏知识库
        level, tag = risk_of(metrics, dq)
        pct = metrics.get("pct_change_24h") or {}
        cur = metrics.get("current_24h") or {}
        issues = dq.get("issues") or []
        rows.append({
            "ad_id": ad_id,
            "risk": level,
            "risk_cn": RISK_LEVELS[level],
            "tag": tag,
            "cur_imps": cur.get("imps", 0), "cur_spend": cur.get("spend_cents", 0),
            "cur_clks": cur.get("clks", 0), "cur_convs": cur.get("convs", 0),
            "avg_bid": cur.get("avg_bid"),
            "d_imps": pct.get("imps"), "d_spend": pct.get("spend_cents"),
            "d_ctr": pct.get("ctr"), "d_conv": pct.get("convs"),
            "d_bid": pct.get("avg_bid"),
            "dq_count": len(issues),
            "dq_critical": sum(1 for i in issues if i["severity"] == "critical"),
            "dq_names": "、".join({i["name"] for i in issues}),
            "recommend": _auto_recommend(tag, issues),
        })
    rows.sort(key=lambda r: (r["risk"] != "high", r["risk"] != "medium", r["ad_id"]))
    return rows


def _auto_recommend(tag: str, issues: list) -> str:
    map_ = {
        "delivery_outage": "核对素材审核状态 / 渠道与定向配置，必要时联系平台侧",
        "imp_dataloss": "核查上报链路与补数，看缺口时段 access 日志",
        "price_anomaly": "对账排查扣费>出价记录，确认计费模式",
        "conv_clock_anomaly": "校准 SDK/服务端时间，重算归因窗口",
        "ctr_stat_outlier": "接反作弊过滤，核查点击来源聚集度",
        "bid_drop": "检查自动出价/oCPX 系数与目标出价",
        "no_anomaly": "保持观察，关注下个周期",
    }
    if tag == "no_anomaly" and issues:
        return "质量存在轻微告警，建议数据链路复核后再评估波动"
    return map_.get(tag, "结合指标复核口径")


def _fmt_pct(v) -> str:
    if v is None:
        return "-"
    return f"{v * 100:+.1f}%"


def inspection_report_md(rows: list[dict], cfg: dict | None = None) -> str:
    now = datetime.now()
    high = [r for r in rows if r["risk"] == "high"]
    med = [r for r in rows if r["risk"] == "medium"]
    lines = [
        f"# 广告单元定时巡检报告",
        "",
        f"- 巡检时间: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 覆盖广告单元: {len(rows)} 个",
        f"- 高风险: {len(high)} | 中风险: {len(med)} | 低风险: {len(rows) - len(high) - len(med)}",
        "",
        "## 一、风险概览",
        "",
        "| 风险 | AdID | 根因标签 | 当前24h imps/spend/clks/convs | "
        "Δimps | Δspend | Δctr | Δconv | Δ均价 | DQ告警 | 建议 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['risk_cn']} | {r['ad_id']} | {r['tag']} | "
            f"{r['cur_imps']:,}/{r['cur_spend']:,}/{r['cur_clks']}/{r['cur_convs']} | "
            f"{_fmt_pct(r['d_imps'])} | {_fmt_pct(r['d_spend'])} | {_fmt_pct(r['d_ctr'])} | "
            f"{_fmt_pct(r['d_conv'])} | {_fmt_pct(r['d_bid'])} | "
            f"{r['dq_count']}(危{r['dq_critical']}) | {r['recommend']} |")
    lines += ["", "## 二、重点关注（高/中风险明细）", ""]
    flagged = [r for r in rows if r["risk"] in ("high", "medium")]
    if not flagged:
        lines.append("> 本次未发现需要立即处理的广告单元。")
    for r in flagged:
        lines.append(f"### AdID {r['ad_id']}  [{r['risk_cn']}]")
        lines.append(f"- 根因标签: {r['tag']}")
        lines.append(f"- 质量告警: {r['dq_names'] or '无'}")
        lines.append(f"- 建议: {r['recommend']}")
        lines.append("")
    return "\n".join(lines)


def run_inspection(cfg: dict | None = None, watch: bool = False,
                   interval_sec: Optional[int] = None) -> str:
    """执行一次巡检，写 Markdown 报告；watch=True 时按周期循环（模拟定时任务）。"""
    cfg = cfg or C.load_config()
    interval_sec = interval_sec or int(cfg["inspection"].get("watch_interval_sec", 3600))
    last_path = ""
    while True:
        rows = scan_all(cfg)
        C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = C.REPORT_DIR / f"inspection_{stamp}.md"
        path.write_text(inspection_report_md(rows, cfg), encoding="utf-8")
        last_path = str(path)
        high = sum(1 for r in rows if r["risk"] == "high")
        print(f"[inspection] report written: {last_path} | units={len(rows)} high_risk={high}")
        if not watch:
            return last_path
        print(f"[inspection] next round in {interval_sec}s (Ctrl+C to stop)...")
        try:
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("[inspection] stopped")
            return last_path


# ===========================================================================
# 自然语言分析工作流
# ===========================================================================

def nl_report(query: str, cfg: dict | None = None, ad_ids: Optional[list[int]] = None,
              mode: Optional[str] = None, save: bool = True) -> DiagnosisResult:
    """业务自然语言 -> 结构化诊断 + Markdown 业务报告。"""
    cfg = cfg or C.load_config()
    result = run_diagnosis(query, cfg, ad_ids=ad_ids, mode=mode)
    if save:
        C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() else "_" for ch in query[:24])
        path = C.REPORT_DIR / f"nl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text(report_to_markdown(result), encoding="utf-8")
        result._saved_path = str(path)  # type: ignore[attr-defined]
    return result
