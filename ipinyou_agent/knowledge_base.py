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
TAG_ZH = {
    "delivery_outage": "投放中断/素材审核/渠道故障",
    "imp_dataloss": "曝光上报丢包",
    "conv_clock_anomaly": "转化时间倒挂",
    "price_anomaly": "扣费高于出价",
    "ctr_stat_outlier": "CTR统计离群/异常流量",
    "bid_drop": "出价下调",
    "budget_cap": "预算撞线",
    "attribution_loss": "归因/回传丢失",
    "invalid_traffic": "无效流量过滤",
    "geo_performance": "地域定向表现恶化",
    "no_anomaly": "无异常",
    "optimization": "优化建议",
    "campaign_not_found": "广告单元ID不存在",
}

# ---------------------------------------------------------------- 种子案例
SEED_DOCS: list[dict] = [
    dict(doc_id="kb-delivery-01", title="素材被平台拒审导致消耗骤降",
         root_cause_tag="delivery_outage",
         symptom="出价与竞价正常、出价均值稳定，但曝光/消耗在某一时刻后连续下跌或归零",
         root_cause="广告素材/文案因合规或审核问题被媒体或平台拒审，流量端不再放量",
         evidence="bid 日志继续产生且价格正常；imp 记录中断；PayingPrice 计费随之消失",
         actions=["核对素材审核状态与拒绝原因", "替换或修改素材后重新提交审核",
                  "检查广告计划/账户是否被限额或暂停"],
         tags=["素材审核", "消耗暴跌", "曝光归零"], source="industry-case"),
    dict(doc_id="kb-delivery-02", title="渠道/媒体侧故障导致流量断供",
         root_cause_tag="delivery_outage",
         symptom="消耗与曝光同跌 80%+，点击转化归零，出价未被调低",
         root_cause="上游流量渠道/ADX 侧故障或断供，竞价拿不到流量；或定向人群包异常收窄",
         evidence="get_campaign_metrics 显示 imp/spend 大幅下滑但 avg_bid 正常；",
         actions=["联系渠道/ADX 侧核对投放状态", "放宽定向/人群包并观察恢复",
                  "检查地域-时段-设备等定向条件是否被误改"],
         tags=["渠道故障", "流量断供", "定向过窄"], source="industry-case"),
    dict(doc_id="kb-delivery-03", title="预算耗尽或频控导致自动停投",
         root_cause_tag="budget_cap",
         symptom="消耗在每日固定时点前被截断，后半夜归零，第二天恢复",
         root_cause="日预算撞线或频控/流量控制自动降速停投",
         evidence="消耗曲线出现周期性截断，bid 在停投时段也不再发出",
         actions=["查看预算消耗曲线", "调整预算或分时投放计划", "检查频控/流量控制阈值"],
         tags=["预算", "频控", "周期性归零"], source="industry-case"),
    dict(doc_id="kb-dataloss-01", title="曝光日志随机丢包导致消耗与曝光缺口",
         root_cause_tag="imp_dataloss",
         symptom="某时段曝光/消耗出现无规律缺口后自行恢复，bid 记录完整",
         root_cause="上报链路(日志采集/清洗/拼接)丢包，或服务端与客户端口径不一致",
         evidence="质量校验命中 R1：部分 BidID 有出价无曝光且时间分布随机、不可恢复期短",
         actions=["核对上报服务端日志与配额", "比对 CDN/网关 access log 补数",
                  "对丢包时段做重算/重报", "修复采集通道并加告警"],
         tags=["日志丢包", "上报缺失", "消耗缺口"], source="industry-case"),
    dict(doc_id="kb-clock-01", title="转化时间早于曝光(时钟倒挂)",
         root_cause_tag="conv_clock_anomaly",
         symptom="点击/转化数量下滑但曝光正常；部分转化记录时间早于其曝光",
         root_cause="客户端/SDK 时钟错误或服务端回传时间戳 bug，导致转化落入错误时间窗",
         evidence="质量校验 R3 命中：conv.timestamp < imp.timestamp",
         actions=["修复 SDK/服务端时间校准", "清洗倒挂事件并做归因窗口重算",
                  "加时间合理性监控"],
         tags=["转化", "时间倒挂", "归因窗口"], source="industry-case"),
    dict(doc_id="kb-clock-02", title="归因窗口与点击延迟导致的转化漏计",
         root_cause_tag="attribution_loss",
         symptom="点击正常，转化量低于预期或波动大",
         root_cause="归因窗口过短/跨天截断，延迟转化未计入当日",
         evidence="转化延迟分布右尾明显，窗口边界转化骤降",
         actions=["延长/调整为按小时归因窗口", "核对点击-转化延迟分布"],
         tags=["归因", "转化漏计"], source="industry-case"),
    dict(doc_id="kb-price-01", title="扣费高于出价(计费异常)",
         root_cause_tag="price_anomaly",
         symptom="报表消耗与理论(曝光x出价)不符，单条曝光扣费>出价",
         root_cause="计费口径 bug：次高价计算/返点未生效/CPM 与 CPC 模式串价",
         evidence="质量校验 R2 命中 PayingPrice>BiddingPrice",
         actions=["核对出价与计费模式", "联调扣费/对账接口", "对异常扣费做补偿单"],
         tags=["扣费异常", "对账", "R2"], source="industry-case"),
    dict(doc_id="kb-ctr-01", title="CTR 异常冲高(疑似异常流量/误触发)",
         root_cause_tag="ctr_stat_outlier",
         symptom="低曝光时段 CTR 冲到数倍甚至十倍，点进来却不转化",
         root_cause="疑似作弊流量/机器点击/素材与受众匹配错位，或点击事件重复上报",
         evidence="按小时 CTR 出现离群尖峰；对应曝光量骤减",
         actions=["接入反作弊/无效流量过滤", "查看点击 IP/UA/设备指纹聚集度",
                  "核查点击埋点是否重复上报", "配合回归周期素材做 A/B"],
         tags=["CTR离群", "作弊流量", "统计异常"], source="industry-case"),
    dict(doc_id="kb-bid-01", title="出价被自动调低导致量级收缩",
         root_cause_tag="bid_drop",
         symptom="消耗下降同时 avg_bid 明显下降，曝光随出价竞争力下降而减少",
         root_cause="系统自动出价策略/托管调价下调，或 oCPX 出价因子异常",
         evidence="get_campaign_metrics avg_bid 变化方向与消耗一致",
         actions=["检查是否开启自动出价及目标出价配置", "核对成本控制系数",
                  "如需放量手动恢复出价并观察竞价成功率"],
         tags=["出价下调", "成本控制", "消耗下降"], source="industry-case"),
    dict(doc_id="kb-geo-01", title="地域/时段定向变化导致效果恶化",
         root_cause_tag="geo_performance",
         symptom="CTR/转化率下降但量级稳定，拆分地域/时段后差异明显",
         root_cause="流量结构变化：低质地域/时段放量占比升高",
         evidence="按 region 聚合 CTR/CPC 差异扩大",
         actions=["拆分报表定位低效地域/时段", "添加排除地域或调整分时出价",
                  "关注素材生命周期衰减"],
         tags=["地域定向", "时段", "效果波动"], source="industry-case"),
    dict(doc_id="kb-normal-01", title="正常波动无需处理",
         root_cause_tag="no_anomaly",
         symptom="指标在正常波动范围内，数据质量校验全部通过",
         root_cause="正常业务波动（流量大盘、素材生命周期、节假日效应）",
         evidence="各时间窗口 deltas 均低于告警阈值，质量报告无告警",
         actions=["保持观察，设置合理告警阈值", "避免对小样本做过度解读"],
         tags=["健康检查", "正常波动"], source="industry-case"),
    dict(doc_id="kb-opt-01", title="CTR 偏低的一般优化路径",
         root_cause_tag="optimization",
         symptom="业务方希望提升 CTR/转化效果（非故障）",
         root_cause="创意/定向/出价策略存在优化空间",
         evidence="无数据质量异常，属策略优化",
         actions=["素材 A/B 与点击率分层分析", "重定向(remarketing)与人群包扩展",
                  "按地域/时段精细化出价", "落地页与广告语一致性优化"],
         tags=["优化", "CTR提升", "素材A/B"], source="industry-case"),
    dict(doc_id="kb-invalid-01", title="无效流量过滤导致点击后无转化",
         root_cause_tag="invalid_traffic",
         symptom="点击量高但转化极低，点击集中在低价值媒体",
         root_cause="无效流量/机器人流量未被及时过滤",
         evidence="点击 IP/UA 集中，转化/点击比显著低于正常",
         actions=["开启平台反作弊过滤", "屏蔽低质媒体/域名(domain 维度核查)",
                  "加转化回传埋点二次校验"],
         tags=["无效流量", "点击质量"], source="industry-case"),
    dict(doc_id="kb-ctr-02", title="CTR 统计离群的排查方法论",
         root_cause_tag="ctr_stat_outlier",
         symptom="某广告单元按日/小时 CTR 出现统计离群(robust-z>阈值)",
         root_cause="需结合量级与相邻单元判断是真提升还是异常",
         evidence="robust-z>5 且样本量不足以支撑时按疑似处理",
         actions=["结合曝光量判断可信度", "先看点击来源是否异常再定性", "必要时对该单元做策略复检"],
         tags=["离群检测", "方法论"], source="industry-case"),
    dict(doc_id="kb-delivery-04", title="定向条件被误改导致流量断崖",
         root_cause_tag="delivery_outage",
         symptom="无出价/素材变动情况下量级断崖",
         root_cause="人群包/地域/设备定向条件被误修改或投放时段设置错误",
         evidence="dimension 拆分发现定向层条件变化",
         actions=["对比定向设置变更记录", "回滚到历史生效定向配置"],
         tags=["定向", "断崖", "配置变更"], source="industry-case"),
    dict(doc_id="kb-general-01", title="消耗暴跌排查 SOP",
         root_cause_tag="delivery_outage",
         symptom="消耗骤降排查通用流程",
         root_cause="按『预算→素材审核→渠道/定向→数据上报→竞价出价』逐层定位",
         evidence="先看 metrics 各指标拆解(dimension)，再跑质量校验，最后检索知识库",
         actions=["Step1 查预算/频控", "Step2 查素材审核状态", "Step3 查渠道与定向",
                  "Step4 查数据上报质量", "Step5 查出价策略变化"],
         tags=["SOP", "方法论"], source="industry-case"),
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
