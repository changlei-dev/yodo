"""Agent 提示词：故障排查 ReAct + 结构化报告约束（LLM 模式使用）。"""
from __future__ import annotations

from . import config as C
from .knowledge_base import TAG_ZH, ROOT_TAGS
from .tools import TOOL_SCHEMA

REPORT_SCHEMA_HINT = """final 报告 JSON 字段(必须完整)：
{
  "summary": "一句话结论(中文)",
  "phenomenon": ["现象描述(中文)..."],
  "root_causes": [{"tag": "候选根因标签", "desc": "中文描述", "probability": "high|medium|low"}],
  "recommendations": ["可执行的业务建议(中文)..."],
  "needs_confirm": ["需要线下确认/进一步验证的事项(中文)..."],
  "confidence": 0.0~1.0
}"""

SYSTEM_PROMPT = """你是某广告投放平台的资深广告运维/数据工程师，负责基于事件级日志排查广告单元(AdID)的
指标异常(消耗/曝光/点击/转化/CTR/CPC 突变)，并给出可执行建议。

【工作方式】像真实生产环境一样，你【不能】直接读原始日志/数据文件，必须调用下面 4 个工具拿数据。
请遵循 ReAct：思考(Think) -> 工具调用(Action) -> 观察(Observation) 的闭环，逐步收敛。

可用工具:
{tool_docs}

【根因标签词表】(final 的 root_causes[].tag 只能从这里选，可多个)
{tags}

【排障路径建议】
1. 先 get_campaign_metrics 拿当前24h vs 前24h 变化率(pct_change_24h)与逐小时 buckets，
   判断是消耗/曝光/点击/转化/CTR/出价哪个维度的突变；
2. 变化明显时 run_data_quality_check 排除数据质量问题(R1孤儿事件/R1b出价无曝光/R2扣费>出价/R3时间倒挂/R4统计离群)；
3. 需要核对明细时 get_campaign_events(注意：仅抽样样例，数量大不代表全量)；
4. 最终定性前 search_knowledge_base 检索行业/历史案例，给出有依据的根因与建议；
5. 不要重复调用同一工具获取同样的信息，避免冗余调用。

【输出格式】每轮你只输出一个 JSON 对象，二选一：
- 调用工具: {{"action":"tool", "tool":"get_campaign_metrics", "args":{{"campaign_id":2345}}}}
- 完成分析: {{"action":"final", "report": {REPORT_SCHEMA}}}
禁止输出 JSON 以外的任何内容(不要 markdown 围栏)。
"""


def build_tool_docs() -> str:
    lines = []
    for name, meta in TOOL_SCHEMA.items():
        params = ", ".join(f"{k}:{v}" for k, v in meta["params"].items())
        lines.append(f"- {name}({params})\n  {meta['description']}")
    return "\n".join(lines)


def build_tags_hint() -> str:
    return "、".join(f"{t}({TAG_ZH.get(t, '')})" for t in ROOT_TAGS)


def build_system(cfg: dict | None = None) -> str:
    return SYSTEM_PROMPT.format(tool_docs=build_tool_docs(), tags=build_tags_hint())


def history_lines(transcript: list[dict]) -> str:
    """把已发生的工具调用与结果压成紧凑历史。"""
    lines = []
    for i, step in enumerate(transcript, 1):
        t = step.get("tool")
        lines.append(f"[{i}] 调用了 {t}({step.get('args')})")
        finding = step.get("finding")
        lines.append(f"    观察: {finding if finding else step.get('result')}")
    return "\n".join(lines)


def build_user_message(query: str, campaign_ids: list[int], transcript: list[dict],
                       max_steps: int) -> list[dict]:
    history = history_lines(transcript)
    if history:
        history = "已获取的信息：\n" + history + "\n\n"
    else:
        history = ""
    content = (
        f"【待排查问题】\n{query}\n\n"
        f"【目标广告单元】{campaign_ids}\n\n"
        f"{history}"
        f"剩余可用轮次不超过 {max_steps} 步。请输出下一轮 JSON(要么 tool 要么 final)。"
    )
    return [{"role": "system", "content": build_system()},
            {"role": "user", "content": content}]
