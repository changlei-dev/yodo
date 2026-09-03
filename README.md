# iPinYou RTB 广告故障排查 Agent（Mock 全流程）

本地模拟广告 DSP 数据全链路：**数据生成 → 质量校验 → 故障案例知识库 → ReAct 排查 Agent → 业务报告 → 评测闭环**。
全程不调用真实 DSP API；`agent` 层按配置决定走 **LLM（Qwen，OpenAI 兼容）** 还是 **内置规则 Planner（Mock）**。

---

## 1. 端到端调用流程

```
                        ┌────────────────────────────────────────────┐
                        │ main.py <command>（唯一命令行入口）          │
                        └───────────────┬────────────────────────────┘
                                        │ 按子命令分发
      ┌────────────────┬───────────────┼─────────────────┬──────────────────┐
      ▼                ▼               ▼                 ▼                  ▼
  pipeline / gen   quality / tools  diagnose / report  inspect        eval / absorb
  (阶段1 建数)      (阶段1 校验)     (阶段3/4 Agent)    (阶段4 巡检)    (阶段5 评测闭环)
      │                │               │                 │                  │
      ▼                ▼               ▼                 ▼                  ▼
 warehouse  ───────►  quality      agent_core        workflows          evaluation
 events宽表/parquet    R1~R4 告警    run_diagnosis     scan_all           构造case→跑Agent
      │                │               │                 │               →评分→bad-case
      │                │               ▼                 │                  │
      │                │       LangGraph StateGraph      │                  ▼
      │                │       plan ⇄ tool 循环          │              knowledge_base
      │                │               │                 │            (bad-case 沉淀)
      │                │               ▼                 ▼
      │                ▼          Planner 二选一      knowledge_base
      │         knowledge_base   (LLM 或 Mock)   ←—— 向量检索 / 规范文档
      │         (质量快照可写入)         │
      ▼                                ▼
 report ──────────────────────────► reports/*.md（结构化业务报告）
```

### 五阶段命令对应

| 阶段 | 命令 | 干什么 | 产物 |
|---|---|---|---|
| 1 数据 | `pipeline` / `gen --force` | 建事件宽表 + 质量基线 | `data/processed/*.parquet`、`reports/quality_baseline.md` |
| 1 质量 | `quality --ad-id N` | 单单元 R1~R4 校验 | `reports/dq_adN.md` |
| 2 工具 | `tools-demo` | 演示 4 个 Mock 工具 | 终端输出 |
| 3 Agent | `diagnose "问题" [--mode] [--save]` | 单次故障排查 | 终端 / `reports/diagnosis_report.md` |
| 4 巡检 | `inspect [--watch]` | 批量扫描全部单元产出风险报告 | `reports/inspection_*.md` |
| 4 报告 | `report "问题" [--mode]` | 自然语言 → 业务报告 | `reports/nl_report_*.md` |
| 5 评测 | `eval [--mode] [--judge] [--absorb]` | 量化根因/建议/工具路径 | `data/eval/eval_*.json`、`reports/eval_report_*.md` |
| 5 闭环 | `absorb` | 失败 case 沉淀进知识库 | `data/kb/added_docs.json` |

---

## 2. Agent 排查调用链（核心）

无论 `diagnose` / `report` / `eval`，最终都汇聚到 `agent_core.run_diagnosis()`：

```
自然语言问题 query
   │
   ▼
extract_campaign_ids()        从文本解析 AdID（显式格式 → 知识库已知 ID 回退）
   │
   ▼
_make_planner(mode)           按 auto/llm/mock 决定 Planner
   │
   ▼
LangGraph StateGraph          （langgraph 缺失/异常 → 降级同逻辑普通循环 _run_loop）
   plan ⇄ tool 循环，上限 agent.max_steps=10
   │
   ├── plan_node: Planner.plan() 产出 {action: tool|final}
   ├── tool_node: tools.call_tool() 执行并追加 transcript
   └── final     : 归一化 report → DiagnosisResult
   │
   ▼
report_to_markdown()          reports/*.md
```

### Planner 二选一

| Planner | mode | 行为 |
|---|---|---|
| `LLMPlanner` | `llm` | 每次 plan 把 系统提示词 + 工具描述 + 问题 + 历史轨迹 发给 Qwen，要求只输出一个 JSON：`{"action":"tool",...}` 或 `{"action":"final","report":{...}}`；**非法输出/异常自动回退 MockPlanner** |
| `MockPlanner` | `mock` | 确定性规则路径：先 `get_campaign_metrics` → `run_data_quality_check` → 按规则判定根因 → 必要时 `get_campaign_events` 核对 → `search_knowledge_base` 检索案例 → 组装报告 |

四个 Agent 工具（`ipinyou_agent/tools.py`，Agent 不能直接读原始大文件）：

1. `get_campaign_metrics` — 当前24h vs 前24h 指标与逐小时序列
2. `get_campaign_events` — 抽样原始事件（LogType 0出价/1曝光/2点击/3转化）
3. `run_data_quality_check` — R1孤儿 / R1b有出价无曝光 / R2扣费>出价 / R3时间倒挂 / R4统计离群
4. `search_knowledge_base` — 检索故障案例 RAG（FAISS 向量）

---

## 3. LLM API：接入点、切换与降级

### 3.1 在哪里真正调用 LLM

| 调用点 | 触发条件 |
|---|---|
| `LLMPlanner.plan()`（agent_core） | `diagnose` / `report` / `eval` 且 mode 落到 `llm` |
| `_llm_judge()`（evaluation） | `eval --judge` 或 `eval.llm_judge: true`，用 LLM 复核建议合理性 |
| 知识库 Embedding | `kb.embedding: qwen` 时（默认 `offline` 本地 n-gram，不调 API） |

### 3.2 配置文件 `config.yaml`

```yaml
llm:                      # 阶段3：Qwen 推理（OpenAI 兼容 / DashScope）
  model: qwen-plus
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1   # 换厂商改这里
  api_key_env: QWEN_API_KEY                                     # key 从哪个环境变量读
  temperature: 0.1
  timeout: 90

agent:
  mode: auto              # auto=有key用LLM否则Mock | llm 强制 | mock 强制
```

### 3.3 切换步骤

1. **换服务商/模型**：改 `config.yaml` 的 `model` + `base_url`（协议为 OpenAI 兼容，DeepSeek / OpenAI / 本地 vLLM 均可）。
2. **配 Key**（二选一）：
   - 环境变量：`$env:QWEN_API_KEY='sk-xxx'`（永久用 `setx QWEN_API_KEY "sk-xxx"`）
   - 明文：在 `config.yaml` 的 `llm` 段直接加 `api_key: sk-xxx`（优先级高于环境变量）
3. **决定是否真走 LLM**：`agent.mode` 或命令行 `--mode llm|mock|auto`。

### 3.4 读取优先级与降级（`config.py` / `llm_client.py` / `agent_core.py`）

```
QWEN_API_KEY 环境变量 ──覆盖──► config.llm.api_key
ChatLLM.available = bool(api_key and base_url)     # 无 key → available=False
_make_planner():
  mode=llm  → LLMPlanner（无 key 时其内部仍会回退 Mock）
  mode=mock → MockPlanner（完全不需要网络/Key）
  mode=auto → ChatLLM.available ? LLMPlanner : MockPlanner
LLMPlanner.plan() 解析失败 / LLM 报错 / 超时 → 自动调用 MockPlanner.plan() 兜底闭环
```

> 提示：改完 key 后 `config.load_config()` 有缓存，新起一个进程即可生效。

---

## 4. 各阶段数据/知识流动

1. **生成** `generator`：按 `data.fault_mode` 注入故障剧情（素材拒审/丢包/时钟倒挂/扣费超价/CTR离群/出价下调/健康对照）。
2. **仓库** `warehouse`：4 类原始日志（bid/imp/clk/conv tsv.gz）按 BidID join → `data/processed/events.parquet`；Agent 全部经 `tools` 取数。
3. **质量** `quality`：R1~R4 规则产出结构化 issue 数组；`run_data_quality_check(store_snapshot=True)` 可将快照写入知识库。
4. **知识库** `knowledge_base`：内置行业排查种子文档 + `added_docs.json` 增量文档（质量快照 / 评测 bad-case）。检索默认离线 n-gram 向量，FAISS 命中失败自动降级 numpy 内积。
5. **评测** `evaluation`：`CASE_LIB` 按 fault 分布构造自然语言用例 → 跑 Agent → 计算根因准确率 / 建议关键词命中 / 工具冗余与覆盖率 → 失败 case 可 `absorb` 回知识库迭代。

---

## 5. 快速开始

```powershell
# 依赖装在项目 .venv 内。无需激活，直接用 venv 的 python 全路径调用：
# （若已激活 .venv 或执行策略允许，可简写为 python main.py ...）
# 首次或新增依赖时才需要执行（已装可跳过），可用国内源加速：
#   .\.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

.\.venv\Scripts\python.exe main.py pipeline                    # 1. 建数 + 质量基线
.\.venv\Scripts\python.exe main.py tools-demo                  # 2. 4 个 Mock 工具示例
.\.venv\Scripts\python.exe main.py diagnose "广告单元 AdID:2345 最近24小时消耗暴跌，请排查" --save
.\.venv\Scripts\python.exe main.py inspect                     # 4. 全单元巡检
.\.venv\Scripts\python.exe main.py eval --mode mock --absorb   # 5. 评测并沉淀 bad-case

# 想走真实 LLM：
$env:QWEN_API_KEY='sk-xxx'
.\.venv\Scripts\python.exe main.py report "广告单元 AdID:2345 消耗骤降 85%，出价没动，请核实" --mode llm
# 终端会打印 mode=llm 说明已走真实 API；无 key / 网络异常时自动落回 mode=mock
```

---

## 6. 关键文件索引

| 文件 | 职责 |
|---|---|
| `main.py` | CLI 入口，子命令分发（唯一入口） |
| `config.yaml` / `ipinyou_agent/config.py` | 全局配置 + env 覆盖 |
| `ipinyou_agent/generator.py` | 模拟日志生成 / 故障注入 |
| `ipinyou_agent/warehouse.py` | 事件仓库 / 宽表 |
| `ipinyou_agent/quality.py` | R1~R4 质量校验 |
| `ipinyou_agent/tools.py` | 4 个 Agent 工具 + schema 注册表 |
| `ipinyou_agent/knowledge_base.py` | FAISS RAG 知识库 + 根因标签词表 |
| `ipinyou_agent/prompts.py` | LLM ReAct 系统提示词 / 报告 JSON 约束 |
| `ipinyou_agent/llm_client.py` | ChatOpenAI 封装（Qwen 等 OpenAI 兼容端点） |
| `ipinyou_agent/agent_core.py` | LangGraph ReAct 编排 / 双 Planner / 根因判定 |
| `ipinyou_agent/workflows.py` | 巡检 `inspect` + 自然语言报告 `nl_report` |
| `ipinyou_agent/evaluation.py` | 评测闭环 + bad-case 沉淀 |
