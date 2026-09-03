# iPinYou RTB Ad Delivery Troubleshooting Agent (full mock pipeline)

> **English** | [中文文档](./README.zh-CN.md)

Locally simulates an ad-DSP data full pipeline: **data generation → data-quality checks →
fault-case knowledge base → ReAct troubleshooting Agent → business report → evaluation loop**.
No real DSP API is called; the `agent` layer decides, per configuration, between an
**LLM (Qwen, OpenAI-compatible)** and the **built-in rule-based Planner (Mock)**.

---

## 1. End-to-End Call Flow

```
                        ┌────────────────────────────────────────────┐
                        │ main.py <command> (single CLI entry)        │
                        └───────────────┬────────────────────────────┘
                                        │ dispatch by subcommand
      ┌────────────────┬───────────────┼─────────────────┬──────────────────┐
      ▼                ▼               ▼                 ▼                  ▼
  pipeline / gen   quality / tools  diagnose / report  inspect        eval / absorb
  (phase1 data)     (phase1 checks)  (phase3/4 agent)  (phase4 watch) (phase5 loop)
      │                │               │                 │                  │
      ▼                ▼               ▼                 ▼                  ▼
 warehouse  ───────►  quality      agent_core        workflows          evaluation
 events wide table    R1~R4 alerts  run_diagnosis     scan_all           build cases→run
      │                │               │                 │              →score→bad cases
      │                │               ▼                 │                  │
      │                │       LangGraph StateGraph      │                  ▼
      │                │       plan ⇄ tool loop          │              knowledge_base
      │                │               │                 │            (bad-case absorb)
      │                │               ▼                 ▼
      │                ▼          Planner × 2        knowledge_base
      │         knowledge_base   (LLM or Mock)  ←—— vector search / canonical docs
      │         (quality snapshots)      │
      ▼                                ▼
 report ──────────────────────────► reports/*.md (structured business reports)
```

### Phase → Command map

| Phase | Command | What it does | Artifacts |
|---|---|---|---|
| 1 data | `pipeline` / `gen --force` | build event wide table + quality baseline | `data/processed/*.parquet`, `reports/quality_baseline.md` |
| 1 quality | `quality --ad-id N` | per-campaign R1~R4 checks | `reports/dq_adN.md` |
| 2 tools | `tools-demo` | demo the 4 Mock tools | terminal output |
| 3 agent | `diagnose "<question>" [--mode] [--save]` | one-shot troubleshooting | terminal / `reports/diagnosis_report.md` |
| 4 inspect | `inspect [--watch]` | batch-scan all campaigns into a risk report | `reports/inspection_*.md` |
| 4 report | `report "<question>" [--mode]` | natural language → business report | `reports/nl_report_*.md` |
| 5 eval | `eval [--mode] [--judge] [--absorb]` | quantify root-cause/advice/tool-path | `data/eval/eval_*.json`, `reports/eval_report_*.md` |
| 5 loop | `absorb` | sink failed cases into KB | `data/kb/added_docs.json` |

---

## 2. Agent Call Chain (core)

`diagnose` / `report` / `eval` all converge into `agent_core.run_diagnosis()`:

```
natural-language question query
   │
   ▼
extract_campaign_ids()        parse AdID from text (explicit formats → KB-known-ID fallback)
   │
   ▼
_make_planner(mode)           pick Planner by auto/llm/mock
   │
   ▼
LangGraph StateGraph          (langgraph missing/error → same-logic plain loop _run_loop)
   plan ⇄ tool loop, bounded by agent.max_steps=10
   │
   ├── plan_node: Planner.plan() emits {action: tool|final}
   ├── tool_node: tools.call_tool() executes and appends transcript
   └── final     : normalized report → DiagnosisResult
   │
   ▼
report_to_markdown()          reports/*.md
```

### Two Planners

| Planner | mode | Behavior |
|---|---|---|
| `LLMPlanner` | `llm` | each plan sends system prompt + tool docs + question + history to Qwen, asks for one JSON only: `{"action":"tool",...}` or `{"action":"final","report":{...}}`; **invalid output / exceptions fall back to MockPlanner** |
| `MockPlanner` | `mock` | deterministic rule path: `get_campaign_metrics` → `run_data_quality_check` → rule-based root cause → `get_campaign_events` when needed → `search_knowledge_base` → assemble report |

The four Agent tools (`ipinyou_agent/tools.py`; the Agent cannot read the raw big files directly):

1. `get_campaign_metrics` — current-24h vs previous-24h metrics + hourly buckets
2. `get_campaign_events` — sampled raw events (LogType 0 bid / 1 imp / 2 click / 3 conv)
3. `run_data_quality_check` — R1 orphan / R1b bid-without-impression / R2 overbilling / R3 time reversal / R4 outlier
4. `search_knowledge_base` — fault-case RAG (FAISS vectors)

---

## 3. LLM API: entry points, switching and fallback

### 3.1 Where the LLM is actually called

| Call site | Trigger |
|---|---|
| `LLMPlanner.plan()` (agent_core) | `diagnose` / `report` / `eval` with mode resolving to `llm` |
| `_llm_judge()` (evaluation) | `eval --judge` or `eval.llm_judge: true`; LLM reviews recommendation reasonableness |
| KB embedding | only when `kb.embedding: qwen` (default `offline` local n-gram, no API) |

### 3.2 `config.yaml`

```yaml
llm:                      # phase 3: Qwen inference (OpenAI-compatible / DashScope)
  model: qwen-plus
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1   # switch vendors here
  api_key_env: QWEN_API_KEY                                     # env var holding the key
  temperature: 0.1
  timeout: 90

agent:
  mode: auto              # auto=use LLM when key exists else Mock | llm force | mock force
```

### 3.3 Switching steps

1. **Switch vendor/model**: change `model` + `base_url` in `config.yaml` (OpenAI-compatible; DeepSeek /
   OpenAI / local vLLM all work).
2. **Provide the key** (either):
   - env var: `$env:QWEN_API_KEY='sk-xxx'` (persist with `setx QWEN_API_KEY "sk-xxx"`)
   - plaintext: add `api_key: sk-xxx` under `llm` in `config.yaml` (takes precedence over env)
3. **Force which planner**: `agent.mode` or `--mode llm|mock|auto` on the CLI.

### 3.4 Precedence and fallback (`config.py` / `llm_client.py` / `agent_core.py`)

```
QWEN_API_KEY env ──overrides──► config.llm.api_key
ChatLLM.available = bool(api_key and base_url)     # no key → available=False
_make_planner():
  mode=llm  → LLMPlanner (its internals still fall back to Mock when no key)
  mode=mock → MockPlanner (no network/key needed)
  mode=auto → ChatLLM.available ? LLMPlanner : MockPlanner
LLMPlanner.plan() parse failure / LLM error / timeout → auto-fallback to MockPlanner.plan()
```

> Note: `config.load_config()` caches; start a new process after changing the key.

---

## 4. Data / Knowledge Flow per Phase

1. **Generate** `generator`: injects fault plots per `data.fault_mode`
   (creative rejection / log loss / clock reversal / overbilling / CTR outlier / bid drop / healthy control).
2. **Warehouse** `warehouse`: joins the 4 raw log types (bid/imp/clk/conv tsv.gz) by BidID into
   `data/processed/events.parquet`; the Agent only reads data through `tools`.
3. **Quality** `quality`: R1~R4 rules emit structured issue arrays;
   `run_data_quality_check(store_snapshot=True)` may write snapshots into the KB.
4. **Knowledge base** `knowledge_base`: built-in industry seed docs + `added_docs.json` incremental
   docs (quality snapshots / eval bad-cases). Retrieval defaults to offline n-gram vectors;
   FAISS failure auto-falls-back to numpy inner products.
5. **Evaluation** `evaluation`: `CASE_LIB` builds natural-language cases by fault distribution →
   runs the Agent → computes root-cause accuracy / action-keyword recall / tool redundancy &
   coverage → failed cases can be `absorb`ed back into the KB for iteration.

---

## 5. Quick Start

```powershell
# Dependencies live in the project .venv. No activation needed; call the venv python by full path:
# (if .venv is activated / execution policy allows, you may shorten to: python main.py ...)
# Install first or after adding deps (may be skipped if already installed):
#   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
# (CN mirrors, e.g. https://pypi.tuna.tsinghua.edu.cn/simple, can speed this up)

.\.venv\Scripts\python.exe main.py pipeline                    # 1. build data + quality baseline
.\.venv\Scripts\python.exe main.py tools-demo                  # 2. demo the 4 Mock tools
.\.venv\Scripts\python.exe main.py diagnose "Campaign AdID:2345 spend collapsed in the last 24h, investigate" --save
.\.venv\Scripts\python.exe main.py inspect                     # 4. full-campaign inspection
.\.venv\Scripts\python.exe main.py eval --mode mock --absorb   # 5. evaluate and absorb bad-cases

# To use a real LLM:
$env:QWEN_API_KEY='sk-xxx'
.\.venv\Scripts\python.exe main.py report "AdID 2345 spend dropped 85% with no bid change; verify" --mode llm
# The terminal prints mode=llm when the real API is used; falls back to mode=mock on missing key / network errors
```

---

## 6. Key File Index

| File | Responsibility |
|---|---|
| `main.py` | CLI entry, subcommand dispatch (single entry) |
| `config.yaml` / `ipinyou_agent/config.py` | global config + env overrides |
| `ipinyou_agent/generator.py` | simulated log generation / fault injection |
| `ipinyou_agent/warehouse.py` | event warehouse / wide table |
| `ipinyou_agent/quality.py` | R1~R4 quality checks |
| `ipinyou_agent/tools.py` | the 4 Agent tools + schema registry |
| `ipinyou_agent/knowledge_base.py` | FAISS RAG knowledge base + root-cause tag table |
| `ipinyou_agent/prompts.py` | LLM ReAct system prompt / report JSON constraints |
| `ipinyou_agent/llm_client.py` | ChatOpenAI wrapper (Qwen & other OpenAI-compatible endpoints) |
| `ipinyou_agent/agent_core.py` | LangGraph ReAct orchestration / dual Planners / root-cause rules |
| `ipinyou_agent/workflows.py` | inspection `inspect` + natural-language report `nl_report` |
| `ipinyou_agent/evaluation.py` | evaluation loop + bad-case absorption |
