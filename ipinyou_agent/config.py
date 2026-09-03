"""全局配置：加载 config.yaml + 环境变量覆盖，并统一工作区路径。"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

# 项目根目录（本文件位于 <root>/ipinyou_agent/config.py）
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KB_DIR = DATA_DIR / "kb"
EVAL_DIR = DATA_DIR / "eval"
REPORT_DIR = ROOT / "reports"

# 运行时目录清单
RUNTIME_DIRS = [DATA_DIR, RAW_DIR, PROCESSED_DIR, KB_DIR, EVAL_DIR, REPORT_DIR]

_DEFAULTS: dict = {
    "data": {"seed": 20260903, "days": 7, "fault_mode": "full"},
    "quality": {
        "price_tolerance": 0,
        "missing_parent_hard_ratio": 0.005,
        "orphan_bid_alert_ratio": 0.85,
        "outlier_robust_z": 5.0,
        "cross_unit_robust_z": 6.0,
        "max_samples": 5,
    },
    "kb": {"top_k": 3, "embedding": "offline"},
    "llm": {
        "model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
        "temperature": 0.1,
        "timeout": 90,
    },
    "agent": {"mode": "auto", "max_steps": 10},
    "inspection": {
        "hours_window": 24,
        "spend_drop_flag": -0.5,
        "imp_drop_flag": -0.5,
        "ctr_surge_flag": 2.0,
        "ctr_drop_flag": -0.5,
        "watch_interval_sec": 3600,
    },
    "eval": {"llm_judge": False},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


_cfg_cache: dict | None = None


def load_config(force: bool = False) -> dict:
    """加载配置（默认合并 config.yaml，环境变量 <ENV_> 或 <ENV_KEY> 覆盖）。"""
    global _cfg_cache
    if _cfg_cache is not None and not force:
        return _cfg_cache

    cfg = _deep_merge(_DEFAULTS, _load_yaml())

    # 环境变量覆盖：QWEN_API_KEY / IPINYOU_* 等
    key_env = cfg["llm"]["api_key_env"]
    if key_env and os.environ.get(key_env):
        cfg["llm"]["api_key"] = os.environ[key_env]
    for prefix in ("IPINYOU_",):
        for k, v in os.environ.items():
            if k.startswith(prefix):
                cfg.setdefault("env", {})[k[len(prefix):].lower()] = v

    _cfg_cache = cfg
    return cfg


def ensure_dirs() -> None:
    for d in RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def reset_config_cache() -> None:
    global _cfg_cache
    _cfg_cache = None


def cfg_get(cfg: dict, dotted: str, default=None):
    """按 a.b.c 取值。"""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
