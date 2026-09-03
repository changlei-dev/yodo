"""阶段1 前置：iPinYou RTB 事件日志生成器（确定性、可复现、按 fault_mode 注入教学故障）。

说明：
- 真实 iPinYou 数据集(GB级)需要单独下载；本项目默认按同一业务语义生成
  bid/imp/clk/conv 多源日志（tsv.gz），字段对齐 iPinYou：BidID / Timestamp /
  LogType / AdvertiserID / AdID / BiddingPrice / PayingPrice / Region / UA / Domain。
- 若你把真实 iPinYou 文件放入 data/raw（文件名 ipinyou_*_{source}.tsv.gz），
  可由 warehouse.build_from_raw 直接读取（loader 兼容列映射），无需生成器。
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import config as C
from .schema import RAW_SCHEMA

UA_POOL = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)",
    "Mozilla/5.0 (Linux; Android 11; SM-G991B)",
    "Mozilla/5.0 (Linux; Android 12; Pixel 6)",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (iPad; CPU OS 15_1 like Mac OS X)",
]
DOMAIN_POOL = [
    "sports.sina.com.cn", "news.sohu.com", "m.qq.com", "www.baidu.com",
    "bbs.tianya.cn", "game.163.com", "tech.ifeng.com", "youku.com",
    "jd.com", "taobao.com", "weibo.com", "douyin.com",
]
REGION_POOL = [11, 12, 13, 21, 22, 23, 31, 32, 33, 44, 51, 52, 61, 62]


@dataclass
class Fault:
    """故障模板。mult_windows: [(hour_from, hour_to, multiplier), ...] 相对现在(负数)。
    kind: channel_outage | imp_dataloss | conv_clock_reversal | price_anomaly |
          ctr_outlier | bid_drop | none
    """
    kind: str = "none"
    mult_windows: list = field(default_factory=list)   # 曝光/投放量乘数窗口
    params: dict = field(default_factory=dict)


@dataclass
class CampaignPlan:
    ad_id: int
    advertiser: int
    base_hourly_imps: int          # 目标每小时曝光基数
    win_rate: float                # 竞价成功率
    ctr: float                     # 基础点击率
    cvr: float                     # 点击->转化率
    fault: Fault = field(default_factory=Fault)
    name: str = ""

    @property
    def healthy(self) -> bool:
        return self.fault.kind in ("none",)


def campaign_plan(fault_mode: str = "full") -> list[CampaignPlan]:
    """内置 8 个广告单元：2 健康 + 6 类教学故障场景。"""
    plans = [
        CampaignPlan(ad_id=2345, advertiser=9, base_hourly_imps=850, win_rate=0.38,
                     ctr=0.0050, cvr=0.12, name="出海游戏-新马素材组",
                     fault=Fault(kind="channel_outage",
                                 mult_windows=[(-30, -19, 0.45), (-18, 0, 0.04)],
                                 params={"note": "出价正常，竞价仍参与，但曝光服务中断"})),
        CampaignPlan(ad_id=1001, advertiser=9, base_hourly_imps=260, win_rate=0.42,
                     ctr=0.0062, cvr=0.10, name="北美策略-对照组A"),
        CampaignPlan(ad_id=1002, advertiser=9, base_hourly_imps=300, win_rate=0.33,
                     ctr=0.0070, cvr=0.14, name="欧洲休闲-稳定跑量组",
                     fault=Fault(kind="conv_clock_reversal",
                                 params={"shift_hours": (22, 27), "sample_hours": 48})),
        CampaignPlan(ad_id=1003, advertiser=9, base_hourly_imps=340, win_rate=0.36,
                     ctr=0.0048, cvr=0.09, name="东南亚SLG-扩量测试",
                     fault=Fault(kind="price_anomaly",
                                 params={"ratio_range": (1.15, 2.0), "count": 500})),
        CampaignPlan(ad_id=1004, advertiser=10, base_hourly_imps=420, win_rate=0.45,
                     ctr=0.0055, cvr=0.11, name="日韩二次元-放量组",
                     fault=Fault(kind="imp_dataloss",
                                 mult_windows=[(-24, -8, 0.68)],   # 最近24h内随机丢30%曝光
                                 params={"note": "曝光/计费日志随机丢包，近8h已恢复"})),
        CampaignPlan(ad_id=1005, advertiser=10, base_hourly_imps=190, win_rate=0.30,
                     ctr=0.0042, cvr=0.13, name="拉美网赚-高转化素材",
                     fault=Fault(kind="ctr_outlier",
                                 mult_windows=[(-26, -24, 0.55), (-9, -7, 0.5)],
                                 params={"spike_ctr": 0.06, "day_mult": 3.0})),
        CampaignPlan(ad_id=1006, advertiser=10, base_hourly_imps=240, win_rate=0.35,
                     ctr=0.0060, cvr=0.08, name="中东工具-效果优先",
                     fault=Fault(kind="bid_drop",
                                 params={"factor": 0.45, "since_hours": 12})),
        CampaignPlan(ad_id=1007, advertiser=10, base_hourly_imps=210, win_rate=0.40,
                     ctr=0.0081, cvr=0.15, name="电商导流-优质流量池"),
    ]

    if fault_mode in ("full", "all"):
        return plans
    if fault_mode == "none":
        for p in plans:
            p.fault = Fault()
        return plans
    # 按 kind 白名单过滤故障（测试单类场景用）
    allowed = set(fault_mode.split(","))
    if "none" in allowed or "healthy" in allowed:
        allowed.add("none")
    return [CampaignPlan(ad_id=p.ad_id, advertiser=p.advertiser,
                         base_hourly_imps=p.base_hourly_imps, win_rate=p.win_rate,
                         ctr=p.ctr, cvr=p.cvr, name=p.name,
                         fault=p.fault if p.fault.kind in allowed else Fault())
            for p in plans]


# --------------------------------------------------------------------------
# 生成主逻辑
# --------------------------------------------------------------------------

def _random_uuid(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.array([uuid.uuid4().hex[:16] for _ in range(n)], dtype=object)


def _hourly_part_mult(rng: np.random.Generator, n_hours: int) -> np.ndarray:
    """昼夜流量波动 0.75~1.3。"""
    base = 1.0 + 0.25 * np.sin(np.linspace(0, 2 * np.pi * (n_hours / 24.0), n_hours) + 1.2)
    return base * (0.9 + 0.2 * rng.random(n_hours))


def _mult_for(ri: int, windows: list) -> float:
    m = 1.0
    for a, b, v in windows:
        if a <= ri <= b:
            m *= v
    return m


def generate_dataset(cfg: dict | None = None, force: bool = False) -> dict:
    """生成多源日志 tsv.gz 到 data/raw，返回统计摘要。

    - force=True 时无视已存在文件重新生成（注入故障，保证场景稳定）
    - fault_mode 来自 cfg['data']['fault_mode']
    """
    cfg = cfg or C.load_config()
    seed = int(cfg["data"]["seed"])
    days = int(cfg["data"]["days"])
    fault_mode = str(cfg["data"].get("fault_mode", "full"))

    raw_dir = C.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_files = {s: raw_dir / f"ipinyou_{s}.tsv.gz" for s in RAW_SCHEMA}
    if not force and all(f.exists() and f.stat().st_size > 100 for f in out_files.values()):
        return {"skipped": True, "files": {k: str(v) for k, v in out_files.items()}}

    rng = np.random.default_rng(seed)
    plans = campaign_plan(fault_mode)
    end = pd.Timestamp.now().floor("h")
    hours = int(days * 24)
    starts = [end - pd.Timedelta(hours=hours - i - 1) for i in range(hours)]  # 负ri对应最近
    rel = list(range(-hours + 1, 1))                                          # -167..0

    daypart = _hourly_part_mult(rng, hours)
    if len(daypart) != hours:
        daypart = np.ones(hours)

    # 每来源累计行
    frames: dict[str, list[pd.DataFrame]] = {s: [] for s in RAW_SCHEMA}
    stat = {"bid": 0, "imp": 0, "clk": 0, "conv": 0}

    for plan in plans:
        # 预生成每次 auction 需要的数据
        for hi, (ts, ri) in enumerate(zip(starts, rel)):
            target_imps = plan.base_hourly_imps * daypart[hi]
            attempts = int(max(8, round(target_imps / plan.win_rate)))

            # --- 出价记录：所有参与竞价(含赢/输/故障期) ---
            bid_ids = _random_uuid(rng, attempts)
            price_bid = rng.normal(45, 10, attempts)
            price_bid = np.clip(price_bid, 5, 120).astype(np.int64)  # 出价(分/CPM)
            if plan.fault.kind == "bid_drop":
                since = int(plan.fault.params.get("since_hours", 12))
                if ri >= -since:
                    price_bid = (price_bid * plan.fault.params.get("factor", 0.45)).astype(np.int64)
                    price_bid = np.maximum(price_bid, 1)

            bid_t = ts.value // 10 ** 6 + (rng.random(attempts) * 3599).astype(np.int64)

            # --- 竞价结果 ---
            wins = rng.random(attempts) < plan.win_rate
            mult = _mult_for(ri, plan.fault.mult_windows)
            served = wins & (rng.random(attempts) < min(1.0, mult))
            n_imp = int(served.sum())
            n_bid = attempts

            imp_idx = np.where(served)[0]
            bid_id_imp = bid_ids[imp_idx]
            price_bid_imp = price_bid[imp_idx]
            ts_imp_ms = bid_t[imp_idx] + (rng.random(n_imp) * 300).astype(np.int64) \
                if n_imp else bid_t[:0]

            # 实际成交价(<=出价为主)，少数次高成交
            pay = np.round(price_bid_imp * np.clip(
                rng.lognormal(mean=-0.55, sigma=0.5, size=n_imp), 0.1, 1.6)).astype(np.int64) \
                if n_imp else np.zeros(0, dtype=np.int64)
            pay = np.minimum(pay, price_bid_imp).astype(np.int64)

            # price_anomaly 故障稍后在整表上随机挑一批覆盖(避免小时粒度)
            # --- 字段填充 ---
            advertisers = np.full(attempts, plan.advertiser, dtype=np.int64)
            ad_ids = np.full(attempts, plan.ad_id, dtype=np.int64)
            creatives = rng.integers(plan.ad_id * 10, plan.ad_id * 10 + 9, attempts)
            regions = rng.choice(REGION_POOL, attempts)
            uas = rng.choice(UA_POOL, attempts)
            domains = rng.choice(DOMAIN_POOL, attempts)
            slots = rng.integers(1, 6, attempts)

            def _pick(a, mask):
                return a[mask]

            # bid 行(全部 attempts)
            frames["bid"].append(pd.DataFrame({
                "BidID": bid_ids, "TimestampMs": bid_t,
                "AdvertiserID": advertisers, "AdID": ad_ids,
                "CreativeID": creatives, "Region": regions,
                "UserAgent": uas, "Domain": domains, "SlotID": slots,
                "BiddingPrice": price_bid,
            }))
            stat["bid"] += n_bid

            if n_imp:
                imp_ad = _pick(ad_ids, served)
                imp_adv = _pick(advertisers, served)
                imp_cre = _pick(creatives, served)
                imp_reg = _pick(regions, served)
                imp_ua = _pick(uas, served)
                imp_dom = _pick(domains, served)
                imp_slot = _pick(slots, served)
                imp_frame = pd.DataFrame({
                    "BidID": bid_id_imp, "TimestampMs": ts_imp_ms,
                    "AdvertiserID": imp_adv, "AdID": imp_ad,
                    "CreativeID": imp_cre, "Region": imp_reg,
                    "UserAgent": imp_ua, "Domain": imp_dom, "SlotID": imp_slot,
                    "BiddingPrice": price_bid_imp, "PayingPrice": pay,
                })
                frames["imp"].append(imp_frame)
                stat["imp"] += n_imp

                # --- 点击(LogType2) ---
                ctr = plan.ctr
                # ctr_outlier：指定窗口 ctr 突刺 / 当天整体抬升
                if plan.fault.kind == "ctr_outlier":
                    for a, b, _m in plan.fault.mult_windows:
                        if a <= ri <= b:
                            ctr = plan.fault.params.get("spike_ctr", 0.05)
                    if ri >= -24 and plan.fault.params.get("day_mult"):
                        ctr = min(ctr * plan.fault.params.get("day_mult", 3.0), 0.03)
                click_flags = rng.random(n_imp) < ctr
                n_clk = int(click_flags.sum())
                if n_clk:
                    clk_sel = np.where(click_flags)[0]
                    frames["clk"].append(pd.DataFrame({
                        "BidID": bid_id_imp[clk_sel],
                        "TimestampMs": ts_imp_ms[clk_sel]
                        + (rng.random(n_clk) * 800).astype(np.int64) + 1,
                        "AdID": _pick(imp_ad, click_flags),
                        "CreativeID": _pick(imp_cre, click_flags),
                    }))
                    stat["clk"] += n_clk

                    # --- 转化(LogType3) ---
                    conv_flags = rng.random(n_clk) < plan.cvr
                    n_conv = int(conv_flags.sum())
                    if n_conv:
                        cv_sel = np.where(conv_flags)[0]
                        cv_bid = bid_id_imp[clk_sel][cv_sel]
                        cv_imp_ms = ts_imp_ms[clk_sel][cv_sel]
                        cv_ad = _pick(imp_ad, click_flags)[cv_sel]
                        cv_cre = _pick(imp_cre, click_flags)[cv_sel]
                        delay = (rng.integers(30, 900, n_conv)).astype(np.int64)
                        frames["conv"].append(pd.DataFrame({
                            "BidID": cv_bid,
                            "TimestampMs": cv_imp_ms + delay,
                            "AdID": cv_ad, "CreativeID": cv_cre,
                        }))
                        stat["conv"] += n_conv

    # 合并每来源
    raw_tables = {}
    for s in RAW_SCHEMA:
        if not frames[s]:
            raw_tables[s] = pd.DataFrame(columns=RAW_SCHEMA[s])
        else:
            raw_tables[s] = pd.concat(frames[s], ignore_index=True)
        # 保持列顺序
        raw_tables[s] = raw_tables[s].reindex(columns=RAW_SCHEMA[s])

    # --- 故障注入收尾（跨表级） ---
    _inject_orphans(raw_tables, rng)
    _inject_price_anomaly(raw_tables, plans, rng)
    _inject_clock_reversal(raw_tables, plans, rng, end, days)

    # 写盘
    for s in RAW_SCHEMA:
        raw_tables[s].to_csv(out_files[s], sep="\t", index=False, compression="gzip")
    return {"skipped": False, "stat": stat,
            "files": {k: str(v) for k, v in out_files.items()}}


def _inject_orphans(raw: dict[str, pd.DataFrame], rng: np.random.Generator) -> None:
    """制造「孤儿事件」：有曝光/点击/转化，但没有对应出价记录(模拟上游丢包/拼接失败)。"""
    imp = raw["imp"]
    if imp.empty:
        return
    # 仅从健康/正常单元抽取少量，避免污染主要故障单元口径
    pool = imp[imp["AdID"].isin([1001, 1007])]
    n = min(70, len(pool))
    if n == 0:
        return
    idx = rng.choice(pool.index, n, replace=False)
    orphan = pool.loc[idx].copy()
    orphan["BidID"] = [f"orphan{uuid.uuid4().hex[:12]}" for _ in range(n)]
    raw["imp"] = pd.concat([raw["imp"], orphan], ignore_index=True)

    clk = raw["clk"]
    if not clk.empty:
        cpool = clk[clk["AdID"].isin([1001, 1007])]
        if len(cpool):
            m = min(20, len(cpool))
            cidx = rng.choice(cpool.index, m, replace=False)
            corph = cpool.loc[cidx].copy()
            corph["BidID"] = [f"orphan{uuid.uuid4().hex[:12]}" for _ in range(m)]
            raw["clk"] = pd.concat([raw["clk"], corph], ignore_index=True)


def _inject_price_anomaly(raw: dict[str, pd.DataFrame], plans: list[CampaignPlan],
                          rng: np.random.Generator) -> None:
    """实际扣费(PayingPrice) > 出价(BiddingPrice)：业务逻辑异常。"""
    for p in plans:
        if p.fault.kind != "price_anomaly":
            continue
        imp = raw["imp"]
        sub = imp[imp["AdID"] == p.ad_id]
        lo, hi = p.fault.params.get("ratio_range", (1.15, 2.0))
        count = int(p.fault.params.get("count", 400))
        count = min(count, len(sub))
        if count == 0:
            continue
        idx = rng.choice(sub.index, count, replace=False)
        ratio = rng.uniform(lo, hi, count)
        raw["imp"].loc[idx, "PayingPrice"] = np.maximum(
            1, (raw["imp"].loc[idx, "BiddingPrice"] * ratio).astype(np.int64))


def _inject_clock_reversal(raw: dict[str, pd.DataFrame], plans: list[CampaignPlan],
                           rng: np.random.Generator, end: pd.Timestamp, days: int) -> None:
    """转化时间早于曝光时间：把最近样本小时内的转化整体向前拨 ~1 天。"""
    conv = raw["conv"]
    imp = raw["imp"]
    if conv.empty or imp.empty:
        return
    for p in plans:
        if p.fault.kind != "conv_clock_reversal":
            continue
        sub = conv[conv["AdID"] == p.ad_id].reset_index()  # 保留原 conv 行索引
        if sub.empty:
            continue
        imp_t = imp[imp["AdID"] == p.ad_id][["BidID", "TimestampMs"]].rename(
            columns={"TimestampMs": "ImpMs"})
        merged = sub.merge(imp_t, on="BidID", how="inner")
        if merged.empty:
            continue
        sample_hours = int(p.fault.params.get("sample_hours", 48))
        cutoff = (end - pd.Timedelta(hours=sample_hours)).value // 10 ** 6
        merged = merged[(merged["TimestampMs"] >= cutoff) & (merged["ImpMs"] >= cutoff)]
        if merged.empty:
            continue
        lo, hi = p.fault.params.get("shift_hours", (22, 27))
        shifts = rng.integers(lo * 3600 * 1000, hi * 3600 * 1000, len(merged))
        orig_idx = merged["index"].to_numpy()
        new_ts = (merged["ImpMs"].to_numpy() - shifts)
        raw["conv"].loc[orig_idx, "TimestampMs"] = new_ts
