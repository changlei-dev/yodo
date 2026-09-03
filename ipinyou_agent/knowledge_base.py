"""阶段2/3：故障案例 RAG 知识库（FAISS 向量检索）。

- 内置行业排查经验种子文档（素材拒审/渠道故障/丢包/时钟倒挂/价格异常/CTR离群等）；
- 支持把每次数据质量报告快照 / 评测 bad-case 追加为可检索文档，形成迭代闭环；
- Embedding：默认本地字符 n-gram hashing 向量（无需 key、离线可用），
  配置 kb.embedding=qwen 且设置 QWEN_API_KEY 时调用 Qwen Embedding API。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import config as C

# ---------------------------------------------------------------- 标签词表
# 与评测 ground-truth 对齐的根因标签（英文小写）
ROOT_TAGS = [
    "delivery_outage",   # 投放中断：素材拒审/渠道故障/定向过窄/计划暂停
    "imp_dataloss",      # 曝光/计费日志上报丢包
    "conv_clock_anomaly",# 转化时间倒挂/时钟错乱
    "price_anomaly",     # 扣费>出价
    "ctr_stat_outlier",  # CTR 统计离群（疑似异常流量）
    "bid_drop",          # 出价/竞价力度下降
    "budget_cap",        # 预算耗尽/撞线
    "attribution_loss",  # 归因回传丢失
    "invalid_traffic",   # 无效流量/反作弊过滤
    "geo_performance",   # 定向地域/时段表现恶化
    "no_anomaly",        # 无异常/健康
    "optimization",      # 非故障，优化建议
    "campaign_not_found",# 广告单元ID不存在/查询条件异常
]
TAG_EN = {
    "delivery_outage": "delivery outage / creative rejection / channel failure",
    "imp_dataloss": "impression log loss",
    "conv_clock_anomaly": "conversion clock anomaly (timestamp reversal)",
    "price_anomaly": "billing above bid",
    "ctr_stat_outlier": "CTR statistical outlier / suspicious traffic",
    "bid_drop": "bid drop",
    "budget_cap": "budget cap reached",
    "attribution_loss": "attribution / callback loss",
    "invalid_traffic": "invalid traffic filtering",
    "geo_performance": "geo-targeting performance degradation",
    "no_anomaly": "no anomaly",
    "optimization": "optimization advice",
    "campaign_not_found": "campaign ID not found",
}

# ---------------------------------------------------------------- 种子案例
SEED_DOCS: list[dict] = [
    dict(doc_id="kb-delivery-01", title="Creative rejected by platform review causes spend cliff",
         root_cause_tag="delivery_outage",
         symptom="Bidding and competition remain normal with a stable average bid, but impressions/spend keep "
                 "dropping or hit zero after a certain timestamp",
         root_cause="The creative/copy was rejected or paused by the media/platform review due to compliance issues, "
                    "so traffic is no longer served",
         evidence="bid logs continue with normal prices; imp records break; PayingPrice billing disappears with them",
         actions=["Check creative review status and rejection reason",
                  "Replace or fix the creative and resubmit for review",
                  "Check whether the plan/account is capped or paused"],
         tags=["creative review", "spend drop", "zero impressions"], source="industry-case"),
    dict(doc_id="kb-delivery-02", title="Channel/media-side failure cuts off traffic supply",
         root_cause_tag="delivery_outage",
         symptom="Spend and impressions both drop 80%+, clicks/conversions hit zero, bid is not lowered",
         root_cause="Upstream traffic channel/ADX failure or supply cutoff; or the targeting audience package "
                    "abnormally narrowed",
         evidence="get_campaign_metrics shows imp/spend dropping sharply while avg_bid stays normal",
         actions=["Contact the channel/ADX to verify delivery status",
                  "Widen targeting/audience package and watch for recovery",
                  "Check whether geo-hour-device targeting conditions were changed by mistake"],
         tags=["channel failure", "traffic cutoff", "narrow targeting"], source="industry-case"),
    dict(doc_id="kb-delivery-03", title="Budget exhausted or frequency capping auto-stops delivery",
         root_cause_tag="budget_cap",
         symptom="Spend is truncated before the same daily time point, drops to zero late at night, "
                 "and recovers the next day",
         root_cause="Daily budget hit the cap, or frequency control / traffic shaping slowed or stopped delivery",
         evidence="Spend curve shows periodic truncation; bids stop being issued during the paused window",
         actions=["Review the budget consumption curve", "Adjust budget or daypart delivery plan",
                  "Check frequency cap / traffic-control thresholds"],
         tags=["budget", "frequency cap", "periodic zero"], source="industry-case"),
    dict(doc_id="kb-dataloss-01", title="Random impression log loss causes spend/impression gaps",
         root_cause_tag="imp_dataloss",
         symptom="Impressions/spend show irregular gaps in some hours then recover by themselves; bid records are "
                 "complete",
         root_cause="Loss in the reporting chain (log collection/cleaning/join), or inconsistency between server "
                    "and client accounting",
         evidence="Quality check hits R1: some BidIDs have bids without impressions, gaps are random and recover fast",
         actions=["Check reporting server logs and quota",
                  "Compare CDN/gateway access logs to backfill data",
                  "Recompute/re-report the affected hours",
                  "Fix the collection pipeline and add alerting"],
         tags=["log loss", "reporting gap", "spend gap"], source="industry-case"),
    dict(doc_id="kb-clock-01", title="Conversion earlier than impression (clock reversal)",
         root_cause_tag="conv_clock_anomaly",
         symptom="Clicks/conversions drop while impressions stay normal; some conversions are timestamped before "
                 "their impressions",
         root_cause="Client/SDK clock errors or a server-side callback timestamp bug push conversions into the wrong "
                    "time window",
         evidence="Quality check R3 hit: conv.timestamp < imp.timestamp",
         actions=["Fix SDK/server time calibration", "Clean reversed events and recompute attribution windows",
                  "Add timestamp sanity monitoring"],
         tags=["conversion", "time reversal", "attribution window"], source="industry-case"),
    dict(doc_id="kb-clock-02", title="Attribution window and click delay cause missed conversions",
         root_cause_tag="attribution_loss",
         symptom="Clicks are normal but conversions are lower than expected or volatile",
         root_cause="Attribution window too short / truncated across days, delayed conversions not credited to "
                    "the right day",
         evidence="Conversion delay distribution has a fat right tail; conversions drop sharply at window edges",
         actions=["Extend or switch to hourly attribution windows", "Review click-to-conversion delay distribution"],
         tags=["attribution", "missed conversions"], source="industry-case"),
    dict(doc_id="kb-price-01", title="Billing above bid (pricing anomaly)",
         root_cause_tag="price_anomaly",
         symptom="Reported spend mismatches theory (impressions x bid); some impressions bill above the bid price",
         root_cause="Billing bug: second-price calculation / rebate not applied / CPM vs CPC mode mixed up",
         evidence="Quality check R2 hit: PayingPrice > BiddingPrice",
         actions=["Verify bid and billing mode", "Jointly debug billing/reconciliation API",
                  "Compensate abnormal overcharges"],
         tags=["overbilling", "reconciliation", "R2"], source="industry-case"),
    dict(doc_id="kb-ctr-01", title="CTR spike (suspected invalid traffic / mis-trigger)",
         root_cause_tag="ctr_stat_outlier",
         symptom="CTR jumps several to ten times during low-impression hours, but clicks do not convert",
         root_cause="Suspected fraudulent traffic / bot clicks / creative-audience mismatch, or duplicated click "
                    "event reporting",
         evidence="Hourly CTR shows outlier spikes; impression volume drops correspondingly",
         actions=["Enable anti-fraud / invalid-traffic filtering",
                  "Check IP/UA/device fingerprint clustering of clicks",
                  "Verify whether click tracking is double-reported",
                  "Run A/B with a renewal-cycle creative"],
         tags=["CTR outlier", "fraud traffic", "stat anomaly"], source="industry-case"),
    dict(doc_id="kb-bid-01", title="Auto-lowered bids shrink delivery volume",
         root_cause_tag="bid_drop",
         symptom="Spend drops while avg_bid also drops; impressions fall as bid competitiveness weakens",
         root_cause="Auto-bid strategy / managed bid lowered the price, or oCPX bid factor behaves abnormally",
         evidence="get_campaign_metrics avg_bid moves in the same direction as spend",
         actions=["Check whether auto-bidding and target-bid config are enabled",
                  "Review cost-control coefficients",
                  "If scaling is needed, restore the bid manually and watch win rate"],
         tags=["bid drop", "cost control", "spend decline"], source="industry-case"),
    dict(doc_id="kb-geo-01", title="Geo/hour targeting changes degrade performance",
         root_cause_tag="geo_performance",
         symptom="CTR/CVR decline while volume stays stable; splitting by geo/hour reveals obvious differences",
         root_cause="Traffic mix changed: low-quality geo/hour segments take a larger share of delivery",
         evidence="CTR/CPC variance across regions widens",
         actions=["Slice reports to locate inefficient geo/hour segments",
                  "Add excluded regions or adjust daypart bids",
                  "Watch for creative lifecycle decay"],
         tags=["geo targeting", "daypart", "performance drift"], source="industry-case"),
    dict(doc_id="kb-normal-01", title="Normal fluctuation, no action needed",
         root_cause_tag="no_anomaly",
         symptom="Metrics stay within normal fluctuation and all data-quality checks pass",
         root_cause="Normal business fluctuation (traffic market, creative lifecycle, holiday effects)",
         evidence="All window deltas are below alert thresholds and the quality report has no warnings",
         actions=["Keep watching and set reasonable alert thresholds",
                  "Avoid over-interpreting small samples"],
         tags=["health check", "normal fluctuation"], source="industry-case"),
    dict(doc_id="kb-opt-01", title="General optimization path for low CTR",
         root_cause_tag="optimization",
         symptom="Business wants higher CTR/conversion performance (not an incident)",
         root_cause="Creative/targeting/bid strategy has room for optimization",
         evidence="No data-quality anomaly; this is a strategy question",
         actions=["Creative A/B testing and click-rate segment analysis",
                  "Remarketing and audience-package expansion",
                  "Fine-grained bidding by geo/daypart",
                  "Landing page and ad copy consistency optimization"],
         tags=["optimization", "CTR lift", "creative A/B"], source="industry-case"),
    dict(doc_id="kb-invalid-01", title="Invalid traffic filtered late, clicks without conversions",
         root_cause_tag="invalid_traffic",
         symptom="High clicks but very low conversions; clicks concentrate on low-value media",
         root_cause="Invalid/bot traffic is not filtered in time",
         evidence="Click IP/UA clustering; conversion-per-click far below normal",
         actions=["Turn on platform anti-fraud filtering",
                  "Block low-quality media/domains (domain-level check)",
                  "Add secondary validation to conversion callbacks"],
         tags=["invalid traffic", "click quality"], source="industry-case"),
    dict(doc_id="kb-ctr-02", title="Methodology for investigating CTR statistical outliers",
         root_cause_tag="ctr_stat_outlier",
         symptom="Daily/hourly CTR of a campaign shows a statistical outlier (robust-z > threshold)",
         root_cause="Must combine volume and neighboring campaigns to decide real lift vs anomaly",
         evidence="robust-z>5 with insufficient samples should be treated as suspected",
         actions=["Judge credibility against impression volume",
                  "Check whether click sources look abnormal before concluding",
                  "Run a strategy re-check for the campaign if needed"],
         tags=["outlier detection", "methodology"], source="industry-case"),
    dict(doc_id="kb-delivery-04", title="Mis-modified targeting conditions cause traffic cliff",
         root_cause_tag="delivery_outage",
         symptom="Traffic cliff without bid/creative changes",
         root_cause="Audience package/geo/device targeting conditions were modified by mistake, or daypart "
                    "settings are wrong",
         evidence="Dimension split reveals changes at the targeting layer",
         actions=["Compare targeting-setting change history", "Roll back to the historical effective targeting config"],
         tags=["targeting", "traffic cliff", "config change"], source="industry-case"),
    dict(doc_id="kb-general-01", title="Spend-drop troubleshooting SOP",
         root_cause_tag="delivery_outage",
         symptom="Generic flow for investigating a sudden spend drop",
         root_cause="Triage layer by layer: budget -> creative review -> channel/targeting -> data reporting -> "
                    "bidding strategy",
         evidence="Read metrics dimensions first, then run quality checks, and finally search the knowledge base",
         actions=["Step 1 check budget/frequency cap", "Step 2 check creative review status",
                  "Step 3 check channel and targeting", "Step 4 check data reporting quality",
                  "Step 5 check bid strategy changes"],
         tags=["SOP", "methodology"], source="industry-case"),
]


# ---------------------------------------------------------------- Embedding
_HASH_DIM = 1024


def _ngrams(text: str, n: int = 2):
    text = text.lower().strip()
    tokens = []
    for i in range(len(text) - n + 1):
        tokens.append(text[i:i + n])
    return tokens


def embed_text_offline(text: str, dim: int = _HASH_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for t in _ngrams(text, 1) + _ngrams(text, 2) + _ngrams(text, 3):
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _compose_doc_text(d: dict) -> str:
    parts = [d.get("title", ""), d.get("symptom", ""), d.get("root_cause", ""),
             d.get("evidence", ""), " ".join(d.get("actions", [])),
             " ".join(d.get("tags", []))]
    return " ".join(str(p) for p in parts if p)


# ---------------------------------------------------------------- 知识库
class KnowledgeBase:
    """FAISS(IndexFlatIP) 向量知识库，numpy/朴素检索做兼容降级。"""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or C.load_config()
        self.top_k = int(self.cfg["kb"].get("top_k", 3))
        self.docs: list[dict] = []
        self._extra_path = C.KB_DIR / "added_docs.json"
        self._vecs: np.ndarray | None = None
        self._index = None
        self._load()

    # ----- 增删 -----
    def _load(self):
        self.docs = [dict(d) for d in SEED_DOCS]
        if self._extra_path.exists():
            try:
                extra = json.loads(self._extra_path.read_text(encoding="utf-8"))
                self.docs.extend([dict(d) for d in extra])
            except Exception:
                pass

    def add_doc(self, doc: dict) -> str:
        """追加文档并持久化（bad-case / 质量快照落库）。"""
        doc_id = doc.get("doc_id") or f"add-{int(time.time() * 1000)}-{len(self.docs)}"
        doc = dict(doc)
        doc["doc_id"] = doc_id
        # 去重：同 source+title 覆盖
        self.docs = [d for d in self.docs if not (d.get("title") == doc.get("title")
                                                  and d.get("source") == doc.get("source"))]
        self.docs.append(doc)
        persist = [d for d in self.docs
                   if not (d["source"] == "industry-case" and d["doc_id"].startswith("kb-"))]
        C.KB_DIR.mkdir(parents=True, exist_ok=True)
        self._extra_path.write_text(
            json.dumps(persist, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc_id

    # ----- 向量索引（FAISS 优先，numpy 降级） -----
    def _vectors(self) -> np.ndarray:
        if self._vecs is None or len(self._vecs) != len(self.docs):
            self._vecs = np.vstack([embed_text_offline(_compose_doc_text(d))
                                    for d in self.docs]).astype("float32")
            self._index = None
        return self._vecs

    def _faiss_ok(self) -> bool:
        try:
            import faiss  # noqa: F401
            return True
        except Exception:
            return False

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = min(top_k or self.top_k, max(1, len(self.docs)))
        if not self.docs:
            return []
        qv = embed_text_offline(query).astype("float32")
        vecs = self._vectors()
        scores_flat: list[float]
        order: list[int]
        if self._faiss_ok():
            try:
                import faiss
                if self._index is None:
                    idx = faiss.IndexFlatIP(_HASH_DIM)
                    idx.add(vecs)
                    self._index = idx
                dist, idxs = self._index.search(np.ascontiguousarray(qv[None]), top_k)
                order = [int(i) for i in idxs[0]]
                scores_flat = [float(d) for d in dist[0]]
            except Exception:
                order, scores_flat = self._numpy_search(qv, vecs, top_k)
        else:
            order, scores_flat = self._numpy_search(qv, vecs, top_k)
        out = []
        for i, s in zip(order, scores_flat):
            d = dict(self.docs[i])
            d["score"] = round(s, 4)
            out.append(d)
        return out

    @staticmethod
    def _numpy_search(qv, vecs, top_k):
        sims = vecs @ qv
        order = np.argsort(sims)[::-1][:top_k]
        return [int(i) for i in order], [float(sims[i]) for i in order]

    def search_tag(self, tag: str, top_k: int = 1) -> list[dict]:
        hits = [d for d in self.docs if d.get("root_cause_tag") == tag]
        return hits[:top_k] if hits else []

    def count(self) -> int:
        return len(self.docs)


def get_kb(cfg: dict | None = None) -> KnowledgeBase:
    return KnowledgeBase(cfg)
