"""阶段1：数据仓库 —— 多源日志(bid/imp/clk/conv)加载，按 BidID join 成完整事件宽表。

- 支持两种输入：a) 生成器产物 tsv.gz；b) 真实 iPinYou 文件(同名格式)放入 data/raw。
- 落盘 processed/events.parquet(长表) 与 processed/wide.parquet(宽表)，
  后续质量校验 / Agent 工具统一从本仓库取数（Agent 不直接读原始大文件）。
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from . import config as C
from .schema import EVENT_COLUMNS, LOGTYPE_CODE, RAW_SCHEMA, TS_MS
from .generator import generate_dataset

_EVENTS_DF: pd.DataFrame | None = None
_WIDE_DF: pd.DataFrame | None = None


def raw_files() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for s in RAW_SCHEMA:
        hits = sorted(glob.glob(str(C.RAW_DIR / f"ipinyou_{s}.tsv.gz")))
        if hits:
            out[s] = Path(hits[0])
    return out


def _read_source(path: Path, source: str) -> pd.DataFrame:
    cols = RAW_SCHEMA[source]
    df = pd.read_csv(path, sep="\t", dtype={TS_MS: "int64"}, compression="gzip")
    df = df.reindex(columns=cols)
    df["LogType"] = LOGTYPE_CODE[source]
    df["timestamp"] = pd.to_datetime(df[TS_MS], unit="ms")
    return df.drop(columns=[TS_MS])


def load_events(force_gen: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
    """加载/构建事件长表。返回规范 EVENT_COLUMNS 的 DataFrame。"""
    global _EVENTS_DF
    if _EVENTS_DF is not None and not force_rebuild:
        return _EVENTS_DF

    parquet = C.PROCESSED_DIR / "events.parquet"
    if parquet.exists() and not force_rebuild:
        _EVENTS_DF = pd.read_parquet(parquet)
        _EVENTS_DF["timestamp"] = pd.to_datetime(_EVENTS_DF["timestamp"])
        return _EVENTS_DF

    files = raw_files()
    if len(files) < 4 or force_gen:
        gen = generate_dataset(force=True)
        if gen.get("skipped"):
            raise RuntimeError("raw 日志不完整且无法生成")

    frames = []
    for s, p in raw_files().items():
        frames.append(_read_source(p, s))
    events = pd.concat(frames, ignore_index=True)

    # 规范类型
    for col in ("AdvertiserID", "AdID", "CreativeID", "LogType", "BiddingPrice", "PayingPrice", "SlotID"):
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce").fillna(0).astype(int)
    for col in ("Region",):
        events[col] = events[col].fillna(0).astype(int)
    events["UserAgent"] = events["UserAgent"].fillna("-")
    events["Domain"] = events["Domain"].fillna("-")
    # 补齐缺失列(如 clk/conv 无出价字段)
    events = events.reindex(columns=EVENT_COLUMNS)
    events["BiddingPrice"] = events["BiddingPrice"].fillna(0).astype(int)
    events["PayingPrice"] = events["PayingPrice"].fillna(0).astype(int)
    events = events.sort_values("timestamp").reset_index(drop=True)

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(parquet, index=False)
    _EVENTS_DF = events
    return events


def load_wide(force_rebuild: bool = False) -> pd.DataFrame:
    """按 BidID join 出价/曝光/点击/转化 -> 宽表：一行 = 一条广告请求全生命周期。"""
    global _WIDE_DF
    if _WIDE_DF is not None and not force_rebuild:
        return _WIDE_DF

    parquet = C.PROCESSED_DIR / "wide.parquet"
    if parquet.exists() and not force_rebuild:
        _WIDE_DF = pd.read_parquet(parquet)
        for c in ("t_bid", "t_imp", "t_clk", "t_conv"):
            _WIDE_DF[c] = pd.to_datetime(_WIDE_DF[c])
        return _WIDE_DF

    ev = load_events()
    pivot = ev.pivot_table(index="BidID", columns="LogType", values="timestamp",
                           aggfunc="min")
    pivot = pivot.rename(columns={0: "t_bid", 1: "t_imp", 2: "t_clk", 3: "t_conv"})

    # 上下文从 bid 记录取，无 bid(孤儿) 时回退 imp
    ctx = ev[ev["LogType"].isin([0, 1])].sort_values(["BidID", "LogType"])
    ctx = ctx.drop_duplicates("BidID", keep="first").set_index("BidID")
    ctx = ctx.drop(columns=["LogType", "timestamp"])

    wide = pivot.join(ctx, how="outer")
    wide = wide.reset_index().rename(columns={"index": "BidID"})
    for lt, tcol in ((0, "t_bid"), (1, "t_imp"), (2, "t_clk"), (3, "t_conv")):
        wide["has_" + tcol[2:]] = wide[tcol].notna()
    wide["AdID"] = wide["AdID"].fillna(0).astype(int)
    wide["AdvertiserID"] = wide["AdvertiserID"].fillna(0).astype(int)
    wide["BiddingPrice"] = wide["BiddingPrice"].fillna(0).astype(int)
    wide["PayingPrice"] = wide["PayingPrice"].fillna(0).astype(int)

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(parquet, index=False)
    _WIDE_DF = wide
    return wide


def known_campaign_ids() -> list[int]:
    """仓库中真实存在的广告单元。"""
    ev = load_events()
    return sorted(int(x) for x in ev["AdID"].unique() if x > 0)


def events_of_campaign(ad_id: int) -> pd.DataFrame:
    """某广告单元全部事件。Agent 工具通过它取数。"""
    ev = load_events()
    return ev[ev["AdID"] == ad_id].reset_index(drop=True)


def campaign_overview(ad_id: int) -> dict:
    ev = events_of_campaign(ad_id)
    if ev.empty:
        return {"ad_id": ad_id, "exists": False, "rows": 0}
    grp = ev.groupby("LogType")["BidID"].nunique().to_dict()
    return {
        "ad_id": ad_id,
        "exists": True,
        "rows": int(len(ev)),
        "bid_auctions": int(grp.get(0, 0)),
        "imps": int(grp.get(1, 0)),
        "clks": int(grp.get(2, 0)),
        "convs": int(grp.get(3, 0)),
        "first_ts": str(ev["timestamp"].min()),
        "last_ts": str(ev["timestamp"].max()),
    }
