# Execution Process & Measured Results

An end-to-end, self-improving **ad-delivery troubleshooting agent** built from scratch
(mock/local RTB pipeline — no real DSP dependency), plus the measured results of every stage.
Repository: <https://github.com/changlei-dev/yodo>

---

## 1. What was built (execution process)

### Stage 0 — Data foundation
- Modelled a real DSP event schema (based on the **iPinYou RTB** benchmark): 4 raw log types
  (bid / impression / click / conversion), **1.26M bids** generated per run across 7 simulated days.
- `warehouse` reconciles all 4 types into a single event-level wide table keyed by **BidID**,
  so every downstream metric and diagnosis traces back to one source of truth.

### Stage 1 — Data quality & baseline
- Implemented **R1–R4 rule checks**: orphan events, bid-without-impression fill-rate gaps (R1b),
  over-billing above the winning price (PayingPrice > BiddingPrice), time reversals, and
  cross-campaign statistical outliers (robust-z) — each issue returns actionable, sample-backed rows.
- A **quality baseline** is computed for the whole corpus and refreshed like a nightly job
  (see `reports/quality_baseline.md`): 5 issue classes found, incl. a critical billing-rule violation.

### Stage 2 — Fault injection & knowledge base
- `generator` injects **7 teaching fault modes** (creative rejection, impression log loss, clock
  reversal, over-billing, CTR outlier, bid drop, optimization opportunity, plus healthy control).
- `knowledge_base` = canonical industry seed docs + incremental bad-case absorption; retrieval via
  offline n-gram vectors (FAISS with numpy fallback) — zero external API dependency.

### Stage 3 — Agent loop
- A **LangGraph StateGraph ReAct loop** with two interchangeable planners:
  - `MockPlanner` — deterministic rule path (metrics → DQ → events → KB → report), used for eval/CI;
  - `LLMPlanner` — real LLM (OpenAI-compatible: Qwen / GLM) that plans tool calls per step.
- Agent tools are the **only** way to touch data: metrics, raw-event sampling, DQ checks, KB search.
- A natural-language question ("spend collapsed in the last 24h, bid looks normal — find the root
  cause") becomes a structured business report with root cause + evidence + recommendations.

### Stage 4 — Eval & self-improvement loop
- 26 scripted cases (9 root-cause classes × difficulty) → score root-cause accuracy,
  recommendation recall, and tool-path economy → **failed cases are absorbed back into the KB**
  so the next iteration starts smarter.

### Stage 5 — Real-LLM hardening (bugs found & fixed while shipping)
Running the *real* LLM in `--mode llm` exposed a genuine production-style bug:
- **Bug**: the model repeatedly called `search_knowledge_base` with an unsupported `campaign_id`
  kwarg → every call errored → it burned its entire 10-step budget → the system silently fell back
  to a **`no_anomaly` verdict on a real delivery-outage case** (worst possible failure: mis-report).
- **Fix (2 prongs, commit `f938efd`)**:
  1. tolerant tool-argument handling (unknown params stripped instead of hard error);
  2. **rule-based fallback verdict on step-budget exhaustion** — the system now degrades to an
     evidence-grounded guess instead of a wrong "all-clear".
- After the fix the same query converges in **4 tool calls** with correct tags
  (`imp_dataloss` + `delivery_outage`, confidence 0.88).

### Stage 6 — Application polish
- English-first output layer (reports/prompts/KB/tools/eval) + bilingual README (`1e083aa`).
- `config.yaml` comments translated to English (`e14a717`), repo pushed to GitHub.

---

## 2. Measured results

### Offline evaluation (26 cases, mock planner, commit `f938efd`+)
| Metric | Value |
|---|---|
| Root-cause accuracy | **96.2%** (macro precision 0.96 / recall 0.96) |
| Actionable-recommendation coverage (kw-recall ≥ 50%) | **96.2%** (macro action recall 0.83) |
| Tool calls per case (avg) | **2.77** (target: minimal tool set) |
| Redundant tool calls (avg) | **0.077** |
| Minimal-tool coverage | 0.95 |
| Broken cases | 1 / 26 (`optimization` mis-judged as `no_anomaly`) |

Accuracy by group:
- by difficulty: easy 92.9% (13/14) · medium 100% · hard 100%
- by root cause: 8/9 classes at 100%; `optimization` at 66.7% — the remaining known gap.

### Data-quality baseline (whole corpus)
- Volume: **1,259,743 bids / 448,694 impressions / 2,666 clicks / 342 conversions**
- 5 issue classes detected (1 critical): orphan events, 73.3% bid-without-impression in last 24h,
  500 over-billing rows, 17 conversion-before-impression, 11 outlier campaigns.
  → evidence the checks catch the exact failure modes injected.

### Real-LLM end-to-end run (after fix)
- Command: `main.py report "Campaign AdID:2345 spend collapsed in the last 24h while the bid is normal; find the root cause" --mode llm`
- Result: **4 tool calls**, tags `imp_dataloss` + `delivery_outage`, confidence 0.88, report written
  to `reports/nl_report_20260904_004606.md`.
- Evidence quality (from the report): R1b 94.6% bid-without-impression (+12.9% vs prior window),
  32.4% fill rate from raw events, **0 orphan events** (rules out attribution/callback loss),
  avg_bid stable (44.44¢, -0.2%) → narrows to logging/delivery loss, not bidding.

### Work-product / evidence index
| File | What it shows |
|---|---|
| `reports/eval_report_20260904_001404.md` | 26-case eval scores (96.2% root-cause) |
| `reports/quality_baseline.md` | corpus-wide DQ baseline (5 issue classes) |
| `reports/nl_report_20260904_004606.md` | real-LLM structured business report |
| `docs/cover_letter.md` | role-specific application letter |

---

## 3. Known gaps & next steps (honest list)
1. `optimization` cases are occasionally mis-read as `no_anomaly` — needs a KB doc + a weaker
   "opportunity" signal so absence of fault ≠ absence of action.
2. Current pipeline is **batch/offline**; no streaming, no real DSP feed. Natural follow-up:
   plug a real event stream and turn the agent loop into a service with alerting.
3. RAG retrieval is n-gram based; swap to an embedding model + keep the FAISS path hot.
4. Data volume (~1M bids) is demo-scale; profiling on 100M+ rows would validate the warehouse/DQ joins.
