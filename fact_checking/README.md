# CWF 事实核查模块（Multi-Agent Fact-Checking）

CWF 的多智能体事实核查与修订模块：将科普文章草稿分解为原子声明，通过 **Leader / Trust / Skeptic** 三角色马尔可夫辩论 + 检索工具补证，把辩论结果构造成 **Claim-Comment-Evidence** 推理图并自底向上传播置信度，最后对低置信度声明进行写作修订。

## 模块结构

```text
agents/
├── claim/extract_claim_agent.py    # 声明分解（原子化 + 归一化权重）
├── check/rewrite_agent.py          # 低置信度声明修订
├── forum/                          # 用户-专家多轮论坛（协作写作示例）
│   ├── user_agent.py
│   └── expert_agent.py
└── markov/                         # 辩论三智能体
    ├── leader_agent.py             # Leader：专家判定（总结论）
    ├── trust_agent.py              # Trust：读者视角补证 / 补充解释
    └── skeptic_agent.py            # Skeptic：质疑无依据声明，纠正弱证据

forum/
├── base_forum.py                   # 用户-专家论坛编排
└── markov_forum.py                 # Markov 辩论编排（状态转移 + 提前终止）

graph/comm_graph.py                 # Claim-Comment-Evidence 推理图
                                    # - evidence 置信度固定 5；claim/comment 初值 2.5
                                    # - 自底向上传播，输出刻度 [0, 5]
tools/                              # 联网搜索（ddgo / Tavily）+ 知识库检索
utils/                              # forum2graph、证据解析、读者画像、搜索限额兜底
config/                             # .env.local 加载 + LangChain 模型工厂
tests/                              # 可运行示例与离线测试
```

## 辩论流程

1. **声明分解**（`ExtractClaimAgent`）：将复杂语句拆为 1~3 条可核查声明，权重和为 1。
2. **Markov 辩论**（`MarkovForum`）：按 Leader 最新事实信号做状态转移：
   - Leader 支持（`stance ≥ 0.2`）→ 先由 **Skeptic** 质疑，避免过早接受；
   - Leader 中性/反对 → 先由 **Trust** 补证，探索缺失证据或替代解释；
   - 每轮结束后 Leader 综合双方论点更新判定，默认最多 3 轮（`R_max=3`）。
3. **构图与置信度**（`forum2graph` + `CommGraph`）：辩论轮次中的意见作为 comment 节点，检索证据作为 evidence 节点；置信度自底向上传播，输出可量化的 claim 置信度（0~5）与可追溯推理链。
4. **修订**（`RewriteAgent`）：对低置信度或冲突声明按编辑建议改写，保留原文风格与读者适配。

## 环境配置

```bash
pip install -r requirements.txt
cp .env.example .env.local    # 填入 DeepSeek / DashScope / OpenAI / Tavily 密钥
```

- 辩论默认使用 `deepseek-chat`（在测试中可通过 `model` 参数切换，支持 `qwen-plus`、`gpt-4o-mini` 等）。
- 联网搜索默认 ddgo（免费），切换 Tavily 见 `tools/web_search.py`；知识库检索默认读取 `data/knowledge` 与 `data/knowledge_index`（按需构建）。

## 快速开始

```bash
# 端到端示例：声明分解 → 多智能体辩论（无检索，可离线跑通 API 流程）
python -B tests/forum/markov_forum.test.py

# 离线图测试：辩论结果 → 推理图 → 置信度 → 可视化 PNG
python -B tests/forum/forum2graph.test.py

# 批量：对 51 条声明并行辩论 + 构图（需要 claims_51.json 数据集）
python -B tests/forum/claims_51_forum2graph_batch.test.py --limit 10 --max-workers 4

# 协作写作示例：用户与专家多轮论坛（base_forum.test.py 或直接调用 run_user_expert_forum）
```

## 测试

离线测试（不调用 LLM，仅依赖 fixture）：

```bash
python -B tests/forum/forum2graph.test.py
```

需要 API Key 的测试（示例/端到端）：

```bash
python -B tests/agents/agent.test.py
python -B tests/agents/extract_claim_agent.test.py
python -B tests/agents/rewrite_agent.test.py
python -B tests/forum/markov_forum.test.py
```

## 引用

论文：*CWF: A Collaborative Writing Framework for Personalized and Reliable Popular Science Writing*（多智能体事实核查机制详见论文「Fact-checking and Revision」一节与附录 Prompt 模板）。