"""
声明处理 Agent：默认保留单条声明；仅在确有多个独立可判命题或前提/结论需分开审查时拆成少量子声明，并为每条分配权重（和为 1）。
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import load_env, create_langchain_agent
from config.llm_factory import Provider
from utils.label_process import extract_all_label


def _coerce_ai_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(x) if not isinstance(x, dict) else str(x.get("text", x))
            for x in content
        )
    return str(content)


def _parse_weight_strings(weight_strs: List[str], n: int) -> List[float]:
    """将模型输出的 weight 文本解析为 n 个非负数，再归一化使和为 1。"""
    vals: List[float] = []
    for s in weight_strs[:n]:
        try:
            v = float(str(s).strip().replace(",", "."))
            vals.append(max(0.0, min(1.0, v)))
        except (TypeError, ValueError):
            vals.append(0.0)
    while len(vals) < n:
        vals.append(0.0)
    vals = vals[:n]
    total = sum(vals)
    if total <= 0:
        return [1.0 / n] * n if n else []
    return [v / total for v in vals]


class ExtractClaimAgent:
    """在确有必要时把复合声明拆成少量子声明；多数情况下保留为一条，并为每条分配权重（和为 1）。"""

    SYSTEM_PROMPT = """
        你是「声明处理」助手。用户会给出一段**声明**（可能是单句，也可能是多句、含因果或并列）。

        **默认原则：能不分就不分**
        - **优先只输出 1 条** <claim>：把整段话整理成一条书面、简洁、可单独判断真假的命题即可；不要为了「显得专业」而硬拆。
        - 仅在同时满足以下条件时才考虑拆成多条：
          - 原文里存在**彼此独立**、**真值可能不一致**的多个判断（例如并列的 A、B、C 各自可单独核实）；或
          - 不拆开会导致**无法分别核实**（例如明显的前提 + 「因此」推出的结论，且审查时需要分开取证）。
        - **禁止**把同一句子的同义改写、修饰语剥离、常识性铺垫拆成多条；**禁止**为凑条数而细分。
        - 确需拆分时：不要输出 4 条及以上。

        其它要求：
        - 保留原意，不添加原文没有的信息；不凭空补充数据或来源。
        - 每条 <claim> 用语书面、简洁。

        **权重（与每条 <claim> 一一对应）**：
        - 每一条 <claim> 后紧跟一个 <weight>，值为 **0～1 之间的两位小数**。
        - 含义：该条对**原始声明整体语义**的覆盖贡献；只有 1 条时该 <weight> 应为 **1.00**。
        - **所有 <weight> 之和必须等于 1**（两位小数，最后一项可用来凑齐）。

        输出格式（严格交替，不要其它说明文字）：
        <claim>…</claim>
        <weight>…</weight>
        （仅 1 对时也要成对输出）
        - <claim> 与 <weight> 条数必须相同，至少 1 对。
    """

    def __init__(
        self,
        provider: Provider = "deepseek",
        model: str = "deepseek-chat",
        max_tokens: int = 1200,
    ):
        load_env()
        self._agent = create_langchain_agent(
            provider,
            tools=[],
            system_prompt=self.SYSTEM_PROMPT,
            model=model,
            max_tokens=max_tokens,
        )

    def extract(self, complex_claim: str) -> Dict[str, Any]:
        """
        处理声明：优先单条输出；必要时拆成少量子声明。为每条分配权重（归一化后和为 1）。
        """
        text = (complex_claim or "").strip()
        if not text:
            return {"claims": [], "weights": [], "items": [], "raw": ""}

        user_msg = (
            "请按系统要求处理以下声明（默认一条即可，勿过度拆分），输出 <claim> 与配对的 <weight>：\n\n"
            f"{text}"
        )
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]}
        )
        messages = result.get("messages", [])
        raw = _coerce_ai_content(
            next(
                (
                    getattr(m, "content", "")
                    for m in reversed(messages)
                    if getattr(m, "type", "") == "ai"
                ),
                "",
            )
        )
        claims = extract_all_label("claim", raw, default=[])
        claims = [c.strip() for c in claims if c and str(c).strip()]
        weight_strs = extract_all_label("weight", raw, default=[])

        if not claims and raw.strip():
            claims = [raw.strip()]
            weight_strs = []

        n = len(claims)
        weights = _parse_weight_strings(weight_strs, n)
        items = [{"claim": c, "weight": w} for c, w in zip(claims, weights)]

        return {
            "claims": claims,
            "weights": weights,
            "items": items,
            "raw": raw,
        }
