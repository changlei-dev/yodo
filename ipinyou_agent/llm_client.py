"""Qwen(OpenAI 兼容/DashScope) 客户端封装。

- 未配置 QWEN_API_KEY / 网络不可用 / 依赖缺失时 available=False，
  Agent 自动降级到内置规则型 MockPlanner（保证本地无网可跑通全流程）。
"""
from __future__ import annotations

import json
import os
import time

from . import config as C


class ChatLLM:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or C.load_config()
        llm = self.cfg["llm"]
        self.model = llm.get("model", "qwen-plus")
        self.base_url = llm.get("base_url")
        self.api_key = llm.get("api_key") or os.environ.get(llm.get("api_key_env", "QWEN_API_KEY"))
        self.temperature = llm.get("temperature", 0.1)
        self.timeout = llm.get("timeout", 90)
        self.available = bool(self.api_key and self.base_url)

    def chat_text(self, messages: list[dict], temperature: float | None = None) -> str:
        if not self.available:
            raise RuntimeError("LLM 未配置")
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature if temperature is None else temperature,
            timeout=self.timeout,
            max_retries=1,
        )
        t0 = time.time()
        resp = client.invoke(messages)
        meta = getattr(resp, "usage_metadata", None) or {}
        print(f"[llm] model={self.model} in_tokens={meta.get('input_tokens')} "
              f"out_tokens={meta.get('output_tokens')} total={meta.get('total_tokens')} "
              f"sec={time.time() - t0:.1f}")
        return (resp.content or "").strip()

    def chat_json(self, messages: list[dict]) -> dict | None:
        """要求模型只输出 JSON；自动抽取首个 JSON 块。"""
        text = self.chat_text(messages, temperature=0.0)
        return extract_json_object(text)


def extract_json_object(text: str) -> dict | None:
    """从模型输出中稳妥地提取 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 从文本中抓取第一个 { ... } 并逐层剥括号
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None
