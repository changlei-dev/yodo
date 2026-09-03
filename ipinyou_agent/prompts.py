"""Agent prompts: ReAct troubleshooting + structured-report constraints (used by LLM mode)."""
from __future__ import annotations

from . import config as C
from .knowledge_base import TAG_EN, ROOT_TAGS
from .tools import TOOL_SCHEMA

REPORT_SCHEMA_HINT = """Final report JSON fields (must be complete and written in ENGLISH):
{
  "summary": "one-sentence conclusion",
  "phenomenon": ["observed phenomena..."],
  "root_causes": [{"tag": "root-cause tag", "desc": "description", "probability": "high|medium|low"}],
  "recommendations": ["actionable business recommendations..."],
  "needs_confirm": ["items that need offline confirmation / further verification..."],
  "confidence": 0.0~1.0
}"""

SYSTEM_PROMPT = """You are a senior ad-ops / data engineer at an ad delivery platform. Your job is to
troubleshoot metric anomalies (spend/impressions/clicks/conversions/CTR/CPC changes) for an ad unit
(AdID) using event-level logs, and to give actionable recommendations.

[Working mode] As in a real production environment, you CANNOT read raw logs/data files directly;
you MUST use the 4 tools below to get data. Follow ReAct: Think -> Action -> Observation, converging
step by step.

Available tools:
{tool_docs}

[Root-cause tag vocabulary] (root_causes[].tag in final must only use these, multiple allowed)
{tags}

[Suggested troubleshooting path]
1. First call get_campaign_metrics to get current-24h vs previous-24h changes (pct_change_24h) and
   hourly buckets, and decide which dimension (spend/impressions/clicks/conversions/CTR/bid) changed;
2. When a change is significant, run run_data_quality_check to rule out data-quality issues
   (R1 orphan events / R1b bid-without-impression / R2 overbilling / R3 time reversal / R4 outlier);
3. When event detail is needed, call get_campaign_events (note: only a sample is returned,
   not the full volume);
4. Before finalizing, search_knowledge_base for industry/historical cases to give evidence-backed
   root causes and recommendations;
5. Do not call the same tool twice for the same information; avoid redundant calls.

[Output format] Each round output exactly ONE JSON object, one of two kinds:
- To call a tool: {{"action":"tool", "tool":"get_campaign_metrics", "args":{{"campaign_id":2345}}}}
- To finish: {{"action":"final", "report": {schema}}}
No content other than JSON (no markdown fences). The report text MUST be written in English,
regardless of the language of the question.
"""


def build_tool_docs() -> str:
    lines = []
    for name, meta in TOOL_SCHEMA.items():
        params = ", ".join(f"{k}:{v}" for k, v in meta["params"].items())
        lines.append(f"- {name}({params})\n  {meta['description']}")
    return "\n".join(lines)


def build_tags_hint() -> str:
    return ", ".join(f"{t}({TAG_EN.get(t, '')})" for t in ROOT_TAGS)


def build_system(cfg: dict | None = None) -> str:
    return SYSTEM_PROMPT.format(tool_docs=build_tool_docs(), tags=build_tags_hint(),
                                schema=REPORT_SCHEMA_HINT)


def history_lines(transcript: list[dict]) -> str:
    """把已发生的工具调用与结果压成紧凑历史。"""
    lines = []
    for i, step in enumerate(transcript, 1):
        t = step.get("tool")
        lines.append(f"[{i}] called {t}({step.get('args')})")
        finding = step.get("finding")
        lines.append(f"    observation: {finding if finding else step.get('result')}")
    return "\n".join(lines)


def build_user_message(query: str, campaign_ids: list[int], transcript: list[dict],
                       max_steps: int) -> list[dict]:
    history = history_lines(transcript)
    if history:
        history = "Information already gathered:\n" + history + "\n\n"
    else:
        history = ""
    content = (
        f"[Question to troubleshoot]\n{query}\n\n"
        f"[Target campaign (AdID)]{campaign_ids}\n\n"
        f"{history}"
        f"Remaining turns available: no more than {max_steps}. Output the next-round JSON "
        f"(either a tool call or final)."
    )
    return [{"role": "system", "content": build_system()},
            {"role": "user", "content": content}]
