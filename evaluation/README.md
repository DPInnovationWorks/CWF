# PSCB 评测模块（Personalized Science Communication Benchmark）

CWF 的评测模块，实现对个性化科普写作的 **LLM-as-a-Judge** 评估，包含两条并行流水线：

- **三指标 Batch 打分**：基于 `CL / PA / RA` 三指标（各含细粒度维度）做 0-5 分评分；
- **Pairwise Baseline Ranking**：在相同 `domain + audience + index` 下，直接比较两个 baseline 孰优。

实现遵循两阶段协议：

- Stage 1：元评估器根据 `topic + persona` 冻结每个指标的维度权重（动态维度权重）；
- Stage 2：打分时复用冻结权重，模型每次只输出一个分数。

## 指标

| 指标 | 含义 | 维度 |
| --- | --- | --- |
| `CL` | Cognitive Load（认知负荷） | INTR / EXTR / GERM |
| `PA` | Personalization Alignment（个性化对齐） | CONT / KNOW / STYLE / CONTX |
| `RA` | Reader Attitude（读者态度） | ENG / TRU / CONTI |

维度权重按 `metric + domain + audience + topic` 冻结；同一主题与人群的权重跨 baseline 复用。LLM 分数可通过 MAE / MARD 与人工问卷（540 份）对齐。

## 环境配置

```bash
pip install openai tqdm
cp .env.example .env    # 填入 DASHSCOPE_API_KEY / OPENAI_API_KEY
```

支持模型（见 `scripts/_shared.py` 中 `MODEL_ENDPOINT_CONFIGS`）：

| 模型 | 路径 | 说明 |
| --- | --- | --- |
| `qwen-plus` / `qwen3.6-plus` / `deepseek-v3.2` | DashScope Batch / 直接 | Batch 自动注入 `enable_thinking=false` |
| `deepseek-v4-pro` / `MiniMax-M2.5` | DashScope 直接 | MiniMax 为仅思考模式，需 `--allow-thinking-models` |
| `gpt-4o-mini` / `gpt-5.5` | OpenAI 直接 | 标准 Chat Completions |

## 评测数据格式（需自行准备）

> 评测数据（PSCB 数据集）按 License 另行发布，此处仅说明约定格式。

- 目录形态：`data/<domain>/<baseline>_<audience>_*.json`
- `<domain>` ∈ `ai | biology | medicine`；`<audience>` ∈ `child | teen | adult`
- `<baseline>` ∈ `base | lora | moe | qwen2.5-14B | domain | revised`（文件名第一个 `_` 前的内容）
- 每条样本字段：
  - `index`（int）、`status`（str）
  - `original_input`：非 `domain`/`revised` baseline 的选题文本
  - `input`：`domain` / `revised` baseline 的选题文本
  - `output`：生成的科普文章
  - `revised_ouput` / `revised_output`：`revised` baseline 事实核查修订后的文章
  - `age_group`（可选，与文件名人群不一致会报错）
  - `final_instruction`（可选）

`data/generation/` 等旁路目录会被自动忽略。

## 三指标 Batch 打分全流程

```bash
# 1. 冻结动态维度权重（首次；--allow-extend 可补齐新 topic）
python scripts/freeze_metric_weights.py

# 1.1 （可选）生成完全平均权重版本，做动态 vs 平均对照
python scripts/build_equal_metric_weights.py

# 2. 构建 Batch JSONL（每条样本拆成 CL/PA/RA 三个请求）
python scripts/build_metric_batch_jsonl.py

# 3. 创建 Batch 任务（默认 24h completion window）
python scripts/create_batch_job.py

# 4. 轮询并下载结果
python scripts/poll_batch_job.py

# 5. 解析结果 → metric_scores / sample_level_summary
python scripts/parse_batch_results.py

# 6. 汇总分析
python scripts/analyze_results.py
```

指定模型与路径的示例：

```bash
python scripts/build_metric_batch_jsonl.py --model deepseek-v3.2 \
  --weights-file artifacts/weights/metric_weights.v1.json \
  --output-file batch_deepseek.jsonl \
  --manifest-file artifacts/manifests/batch_manifest_deepseek.jsonl

python scripts/run_metric_direct_scoring.py --model gpt-4o-mini \
  --manifest-file artifacts/manifests/batch_manifest.jsonl \
  --output-file artifacts/direct_outputs/gpt-4o-mini.output.jsonl
```

## Pairwise Ranking 全流程

```bash
python scripts/build_pairwise_comparisons.py   # 构建两两比较
python scripts/run_pairwise_judging.py         # 多线程调用 qwen-plus 判定（只允许输出 A/B）
python scripts/parse_pairwise_results.py       # 解析（容错 A./Answer: B 等格式）
python scripts/analyze_pairwise_rankings.py    # Copeland + 胜率排名，输出四层排行
```

常用参数：`run_pairwise_judging.py --max-workers 6 --max-retries 3 --request-timeout 120`

## 分数输出约束

打分 prompt 强制模型只输出 `0.0~5.0` 的一位小数；非法输出（如 `score=4.6`、附带解释）记入 `invalid_score`。

## 引用

论文：*CWF: A Collaborative Writing Framework for Personalized and Reliable Popular Science Writing*（PSCB 定义与问卷详见论文「PSCB Benchmark」一节及附录）。