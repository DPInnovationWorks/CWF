"""
联网检索次数触顶时：仅根据本轮已成功返回的检索结果补全标签格式输出。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_chat_model
from config.llm_factory import Provider
from utils.get_evidence import extract_retrieved_evidences

_LIMIT_MARKERS = (
    "Tool call limit reached",
    "run limit exceeded",
)


def coerce_ai_content(content: Any) -> str:
    """将 Agent 最后一条 AI 的 content 规范为 str。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for x in content:
            if isinstance(x, dict):
                parts.append(str(x.get("text", x)))
            else:
                parts.append(str(x))
        return "".join(parts)
    return str(content)


def is_search_limit_error(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(m.lower() in t for m in _LIMIT_MARKERS)


def _format_retrieved_blocks(retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return (
            "（本轮尚未有任何成功的检索条目返回。请结合上文 user 消息中的 claim 与观点，"
            "在证据不足的前提下仍按系统要求的标签格式输出，<cited> 填 无。）"
        )
    lines: List[str] = []
    for block in retrieved:
        b = block.get("batch", "?")
        q = block.get("query", "")
        lines.append(f"--- 第 {b} 次检索 | 查询: {q} ---")
        for r in block.get("results") or []:
            if not isinstance(r, dict):
                continue
            lines.append(
                f"  [{r.get('rank', '')}] {r.get('title', '')}\n"
                f"  {r.get('snippet', '')}\n"
                f"  来源: {r.get('url', '')}"
            )
    return "\n".join(lines)


def _history_dicts_to_lc(history: List[Dict[str, Any]]) -> List[Any]:
    out: List[Any] = []
    for m in history:
        role = m.get("role", "")
        c = str(m.get("content", ""))
        if role == "user":
            out.append(HumanMessage(content=c))
        elif role == "assistant":
            out.append(AIMessage(content=c))
    return out


def fallback_reply_after_search_limit(
    agent_raw: Any,
    invoke_messages: Sequence[Any],
    *,
    system_prompt: str,
    history_dicts: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    provider: Provider = "deepseek",
) -> Tuple[str, bool]:
    """
    若 agent_raw 为检索次数触顶类错误文案，则用无工具 LLM 根据已检索摘要补答。

    :return: (最终用于展示的文本, 是否走了兜底补答且得到了非空新文本)
    """
    text = coerce_ai_content(agent_raw)
    if not is_search_limit_error(text):
        return text, False

    retrieved = extract_retrieved_evidences(invoke_messages)
    evidence = _format_retrieved_blocks(retrieved)

    recovery = (
        "[系统] 联网检索调用次数已达上限，**禁止**再使用任何检索或联网工具。\n"
        "请仅根据下方「本轮已成功返回的检索摘要」完成你的角色任务。\n"
        "输出必须**只**包含系统提示中要求的标签（<opinion> <stance> <severity> <cited>），"
        "不要其它说明文字。\n"
        "<cited> 中的批次-条号须与摘要中「第 n 次检索」及 [rank] 一致；无可用条目则填 无。\n\n"
        f"{evidence}"
    )

    lc: List[Any] = [SystemMessage(content=system_prompt.strip())]
    lc.extend(_history_dicts_to_lc(history_dicts))
    lc.append(HumanMessage(content=recovery))

    llm = get_chat_model(provider, model=model, max_tokens=max_tokens)
    resp = llm.invoke(lc)
    out = getattr(resp, "content", None)
    if isinstance(out, list):
        out = "".join(
            str(x) if not isinstance(x, dict) else str(x.get("text", x)) for x in out
        )
    recovered = (out or "").strip() if out is not None else ""
    if recovered:
        return recovered, True
    return text, False
