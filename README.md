# CWF: Collaborative Writing Framework for Personalized and Reliable Popular Science Writing

本仓库是论文 **"CWF: A Collaborative Writing Framework for Personalized and Reliable Popular Science Writing"** 的开源代码，覆盖 CWF 的三个核心模块：

| 模块           | 目录                               | 说明                                                                                                      |
| -------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 训练（DA-MoE） | [`training/`](training/)           | 基于 MoE-PEFT 的领域知识 / 读者适配专家训练与 AlphaNet 融合门控训练                                       |
| 事实核查       | [`fact_checking/`](fact_checking/) | 多智能体辩论（Leader / Trust / Skeptic）+ 图上置信度传播 + 写作修订                                       |
| 评测（PSCB）   | [`evaluation/`](evaluation/)       | LLM-as-a-Judge 个性化评测（Cognitive Load / Personalization Alignment / Reader Attitude）与 Pairwise 排行 |

> 论文中的「协作写作（Collaborative Writing）」部分（起草与扩写）见 `fact_checking/forum/base_forum.py` 的用户-专家多轮论坛示例；评测数据（PSCB）与训练数据按 License 另行发布，请按下文中约定的数据格式自行准备。

---

## 整体 Pipeline

```text
                        ┌───────────────────────────┐
                        │   训练阶段 training/        │
                        │  1) 专家适配器训练           │
                        │     - 科学知识专家           │
                        │     - 读者风格专家(儿童/少年/成人) │
                        │  2) AlphaNet 融合门控训练    │
                        └────────────┬──────────────┘
                                     │ 导出 MixLoRA 适配器
                                     ▼
  科普文章草稿 ──► 事实核查阶段 fact_checking/
                       │
                       │  1) ExtractClaimAgent：原子声明分解（带权重）
                       │  2) MarkovForum：多智能体辩论（3 轮）：
                       │       Leader(判定) / Trust(补证) / Skeptic(质疑)
                       │       └ 联网搜索 / 知识库检索补充证据
                       │  3) forum2graph：辩论结果 → 推理图
                       │  4) CommGraph：置信度自底向上传播，输出 [0,5] 置信度
                       │  5) RewriteAgent：对低置信度声明进行修订
                       ▼
  经修订的科普文章 ──► 评测阶段 evaluation/
                       │
                       │  1) freeze_metric_weights：动态维度权重冻结
                       │  2) build_metric_batch_jsonl：CL/PA/RA 批量打分
                       │  3) create/poll Batch 任务、解析结果
                       │  4) analyze：一致性、排行等分析
                       ▼
               PSCB 分数（个性化 + 事实准确性）
```

---

## 目录结构

```text
.
├── fact_checking/          # 多智能体事实核查（辩论 + 图 + 修订）
│   ├── agents/
│   │   ├── claim/            # ExtractClaimAgent 声明分解
│   │   ├── check/            # RewriteAgent 写作修订
│   │   ├── forum/            # UserAgent / ExpertAgent 协作写作示例
│   │   └── markov/           # Leader / Trust / Skeptic 辩论智能体
│   ├── forum/                # MarkovForum 辩论编排（Markov 状态转移）
│   ├── graph/                # CommGraph：Claim-Comment-Evidence 推理图
│   ├── tools/                # 联网搜索（ddgo / Tavily）、知识库检索
│   ├── utils/                # forum2graph、证据处理、读者画像等
│   ├── tests/                # 可运行示例与离线测试
│   ├── config/               # 环境变量与 LLM 工厂
│   ├── .env.example
│   └── requirements.txt
├── training/                # DA-MoE 训练（MoE-PEFT 定制版）
│   ├── DA_MoE_training/      # 适配器训练 + AlphaNet 训练脚本
│   └── moe_peft/             # MoE-PEFT 框架（MixLoRA 等）
├── evaluation/              # PSCB 评测
│   ├── scripts/              # 权重冻结 / Batch 打分 / 解析 / 分析
│   ├── .env.example
│   └── README.md
├── .gitignore
└── README.md
```

---

## 快速开始

### 1. 环境配置

```bash
cd fact_checking
pip install -r requirements.txt
cp .env.example .env.local   # 填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY 等
```

```bash
cd evaluation
pip install openai tqdm
cp .env.example .env         # 填入 DASHSCOPE_API_KEY / OPENAI_API_KEY
```

```bash
cd training
pip install -r requirements.txt   # 训练环境（torch / transformers / mixlora 等）
```

### 2. 训练 DA-MoE（可选）

```bash
cd training

# 阶段一：训练科学知识适配器与读者风格适配器
python DA_MoE_training/train_science_age_adapters.py --prepare_only
python DA_MoE_training/train_science_age_adapters.py --base_model /path/to/base/model

# 阶段二：训练 AlphaNet 融合门控
python DA_MoE_training/prepare_alphanet_training_data.py
python DA_MoE_training/train_alphanet.py \
  --base_model_path /path/to/base/model \
  --adapter_a_path /path/to/science/adapter \
  --adapter_b_path /path/to/age/adapter \
  --train_json /path/to/train.json \
  --val_json /path/to/validation.json
```

详见 [`training/README.md`](training/README.md) 与 [`training/DA_MoE_training/README.md`](training/DA_MoE_training/README.md)。

### 3. 事实核查

```bash
cd fact_checking

# 端到端示例：声明分解 → 多智能体辩论 → 结果写出
python -B tests/forum/markov_forum.test.py

# 离线图测试：辩论结果 → 推理图 → 置信度 → 可视化
python -B tests/forum/forum2graph.test.py

# 批量运行：对 51 条声明并行辩论并构图
python -B tests/forum/claims_51_forum2graph_batch.test.py --limit 10 --max-workers 4
```

### 4. PSCB 评测

```bash
cd evaluation

# 批量打分（CL / PA / RA）
python scripts/freeze_metric_weights.py
python scripts/build_metric_batch_jsonl.py
python scripts/create_batch_job.py
python scripts/poll_batch_job.py
python scripts/parse_batch_results.py

# Pairwise 排行
python scripts/build_pairwise_comparisons.py
python scripts/run_pairwise_judging.py
python scripts/parse_pairwise_results.py
python scripts/analyze_pairwise_rankings.py
```

详见 [`evaluation/README.md`](evaluation/README.md)。

---

## 各模块说明

### DA-MoE 训练（`training/`）

论文提出的 DA-MoE 将**读者适配**与**领域知识**解耦为两组可插拔专家：

- **知识专家**：在特定科学领域数据（生物 / 医学 / AI 等）上微调，负责领域知识。
- **风格专家**：在特定读者人群数据（儿童 / 少年 / 成人）上微调，负责表达适配。
- **AlphaNet**：轻量 MLP 门控，按输入动态融合两组专家的隐状态：`h(x) = α·h^s(x) + (1−α)·h^k(x)`。

训练基于 [MoE-PEFT](https://github.com/TUDB-Labs/MoE-PEFT)（MixLoRA 路由），适配器可即插即用，扩到新领域只需轻量训练门控。

### 多智能体事实核查（`fact_checking/`）

- **声明分解**：`ExtractClaimAgent` 将段落拆为可独立核查的原子声明，并输出归一化权重（和为 1）。
- **Markov 辩论**：`MarkovForum` 依据 Leader 的事实性信号做状态转移（支持→Skeptic 先行；不确定/反对→Trust 先行），最多 3 轮，全部智能体立场一致则提前终止。
- **图上推理**：`CommGraph`（claim–comment–evidence 三层图）自底向上传播置信度，证据节点固定为 5，其余从 2.5 初值开始，最终输出 [0,5] 的可量化置信度与可追溯推理路径。
- **修订**：`RewriteAgent` 依辩论结论修订低置信度或自相矛盾的声明。

每一轮对话中的智能体均可按需调用联网搜索（默认 ddgo，可切换 Tavily）或本地知识库检索来补充证据。

### PSCB 评测（`evaluation/`）

PSCB 从两个分支联合评估：

1. **个性化适应**：CL（认知负荷）/ PA（个性化对齐）/ RA（读者态度）三个指标，采用「动态维度权重冻结 + 维度级打分」两阶段协议，LLM-as-a-judge 分数通过 MAE / MARD 与人工问卷对齐。
2. **事实准确性**：基于推理图的声明验证，惩罚无证据或错误声明。

评测脚本支持 DashScope Batch（qwen-plus / deepseek-v3.2 等）与直接调用（gpt-4o-mini / MiniMax-M2.5 等）两条路径。

---

## 数据格式约定

### 评测数据（`evaluation/data/`，需自行准备）

- 目录：`evaluation/data/<domain>/<baseline>_<audience>_*.json`
- `<domain>` ∈ `ai | biology | medicine`
- `<baseline>` ∈ `base | lora | moe | qwen2.5-14B | domain | revised`（文件名第一个 `_` 前的内容即 baseline）
- `<audience>` ∈ `child | teen | adult`
- 每条样本字段：
  - `index`（int）、`status`（str）
  - `original_input`：非 domain/revised baseline 的选题文本
  - `input`：domain / revised baseline 的选题文本
  - `output`：生成的科普文章
  - `revised_ouput`（或 `revised_output`）：revised baseline 经事实核查修订后的文章
  - `age_group`：与文件名人群一致（可选，存在则校验）
  - `final_instruction`（可选）

### 训练数据（`training/train_data/`，需自行准备）

`train_science_age_adapters.py` 默认读取 `training/train_data/` 下的科学语料与 `training/train_data/processed/{child,teens,adult}.json` 读者改写语料，格式为 `instruction/output` 对。

---

## 引用

```bibtex
@article{fu2026cwf,
  title     = {CWF: A Collaborative Writing Framework for Personalized and Reliable Popular Science Writing},
  author    = {Fu, Ruibiao and Tang, Di and Yang, Yunlong and Wang, Ran and Lu, Sicheng and Wu, Peixuan and Fan, Xiaoyu and Ma, Jiacheng and Luo, Haozhe and Xiao, Yang},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
}
```

## License

代码以 Apache 2.0 协议开源（`training/` 部分沿用 MoE-PEFT 的 Apache 2.0）。评测数据与训练数据另行发布。
