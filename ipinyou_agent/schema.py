"""事件 Schema：对齐 iPinYou RTB 核心字段 + 简化 LogType 语义.

LogType: 1=曝光(imp) 2=点击(clk) 3=转化(conv)；0=出价(bid) 为扩展占位，
用于串联「出价->曝光->点击->转化」整条广告全生命周期事件（同一 BidID）。
"""
from __future__ import annotations

# 事件类型编码（用户定义的 1/2/3 保持不变，0 代表出价记录）
LOGTYPE_NAMES = {0: "bid", 1: "imp", 2: "clk", 3: "conv"}
LOGTYPE_CODE = {v: k for k, v in LOGTYPE_NAMES.items()}

RAW_SOURCES = ("bid", "imp", "clk", "conv")

# 各多源日志原始文件字段（模拟 iPinYou bid/imp/clk 独立文件）
RAW_SCHEMA = {
    "bid": ["BidID", "TimestampMs", "AdvertiserID", "AdID", "CreativeID",
            "Region", "UserAgent", "Domain", "SlotID", "BiddingPrice"],
    "imp": ["BidID", "TimestampMs", "AdvertiserID", "AdID", "CreativeID",
            "Region", "UserAgent", "Domain", "SlotID", "BiddingPrice", "PayingPrice"],
    "clk": ["BidID", "TimestampMs", "AdID", "CreativeID"],
    "conv": ["BidID", "TimestampMs", "AdID", "CreativeID"],
}

# 生成后的宽表/事件规范列（按 BidID join 后统一口径）
EVENT_COLUMNS = [
    "BidID", "LogType", "timestamp",
    "AdvertiserID", "AdID", "CreativeID",
    "Region", "UserAgent", "Domain", "SlotID",
    "BiddingPrice", "PayingPrice",
]

TS_MS = "TimestampMs"

def logtype_of(source: str) -> int:
    return LOGTYPE_CODE[source]

def source_of(logtype: int) -> str:
    return LOGTYPE_NAMES[logtype]
