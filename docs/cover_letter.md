# Cover Letter — Applied AI Engineer (AdTech) at Yodo1

> **Subject:** Application — Applied AI Engineer (AdTech) · Lei Chang — LLM agents for production incident investigation

Dear Yodo1 Hiring Team,

I'm Lei Chang, an applied-AI engineer with **7+ years shipping NLP/LLM systems end-to-end** — currently building LLM-powered cockpit voice assistants at Bosch, after leading LLM development and vertical applications at Ultrapower Software (Beijing). Your Applied AI Engineer – AdTech role caught my attention because it asks for exactly the muscle I've been training: **turning messy event-level data into trusted metrics, catching data regressions before they cost revenue, and giving LLM agents the tools and feedback loops to investigate incidents autonomously.**

Three things from your job description match my experience directly:

**1. Trusted data foundation & quality checks (JD 1, 2, 7).** Across my projects I've owned the whole "data → model → evaluation" loop: building standardized evaluation pipelines (data collection → filtering → labeling → model iteration → assessment) for LLM alignment at Voyah, cleaning/labeling/augmenting large raw corpora for SFT at Ultrapower, and processing ~4M alarm events/day for telecom knowledge-graph mining. To prove the same discipline in your domain, I built an open-source demo mirroring an ad-DSP pipeline from raw logs up: it reconciles four RTB log types (bid/impression/click/conversion) into one event-level wide table by BidID, then runs the data-quality checks a campaign owner actually needs — orphan events, **bid-without-impression fill-rate gaps**, over-billing above the winning price, clock reversals, and cross-campaign statistical outliers — with a quality baseline and scheduled full-campaign inspection.

**2. Agents with context, tools, and feedback loops (JD 5; nice-to-have: agent frameworks).** In the cockpit assistant project I helped build the **Planner module** that decomposes compound voice commands and routes them to domains, and combined **RAG + few-shot** generation to push new-scenario handling up 40%. My demo takes that further into autonomous investigation: a LangGraph ReAct planner drives four domain tools (metrics, raw-event sampling, data-quality checks, a vector-retrieval fault knowledge base) to go from a question like *"spend collapsed while bid looks normal"* to a diagnosed root cause and a written business report — and **failed evaluation cases are absorbed back into the knowledge base**, so the agent measurably improves across iterations.

**3. Investigating failures and defining "good" (JD 4, 7).** I'm metric-driven by habit (intent accuracy 98%+ on the cockpit system; SQL-generation accuracy 99.98% on a 26-scenario NL2SQL assistant), and the demo scores every diagnosis on root-cause accuracy, recommendation coverage, and tool-path redundancy. It even surfaced a real robustness bug I had to fix — an LLM repeatedly calling a tool with an unsupported parameter and exhausting its step budget — which I resolved with tolerant tool handling **plus a rule-based fallback verdict**, so the system degrades gracefully instead of mis-reporting. That's the reliability mindset I'd bring to your revenue pipelines.

AdTech is a new domain for me, but the fundamentals — event-level data, reconciliation, latency, and now LLM agents on top — are the same problems I've solved in automotive and enterprise NLP, and my demo already uses the real iPinYou RTB schema (bidding vs. paying price, fill rates) so I can speak your operators' language from day one. I'm fully remote-ready, comfortable with async work, and impact-driven rather than title-driven — values I see mirrored in how Yodo1 describes itself.

I'd love to walk your team through the system, including a **live run where the agent investigates a deliberately injected outage**. Project and README: **https://github.com/changlei-dev/yodo**

Best regards,
**Lei Chang**
changlei0406@163.com · (+86) 178-0029-0863
