"""
DeepSeek 等接口在输入/上下文触发内容安全策略时可能返回 400：Content Exists Risk。
此处检测该类错误并生成符合标签格式的占位输出，使辩论流程可继续。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

from openai import BadRequestError

from config.llm_factory import Provider
from utils.search_limit import fallback_reply_after_search_limit

Role = Literal["trust", "skeptic", "leader"]


def is_content_risk_error(exc: BaseException) -> bool:
    """判断是否为「内容存在风险」类 API 拒绝（不依赖具体 SDK 版本的字段结构）。"""
    s = str(exc).lower()
    if "content exists risk" in s:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            m = str(err.get("message", "")).lower()
            if "content exists risk" in m:
                return True
    return False


def content_risk_fallback_raw(role: Role) -> str:
    """与各角色提示一致的标签占位，立场中立以便流程继续。"""
    opinions = {
        "trust": (
            "（接口内容安全策略拦截，本轮无法生成模型输出；为保持辩论流程继续，"
            "暂以中立立场占位；未进行检索。）"
        ),
        "skeptic": (
            "（接口内容安全策略拦截，本轮无法生成模型输出；为保持辩论流程继续，"
            "暂以中立立场占位；未进行检索。）"
        ),
        "leader": (
            "（接口内容安全策略拦截，本轮无法生成模型输出；为保持辩论流程继续，"
            "暂以中立立场占位；未进行检索。）"
        ),
    }
    op = opinions.get(role, opinions["leader"])
    return (
        f"<opinion>{op}</opinion>\n"
        "<stance>0</stance>\n"
        "<severity>3</severity>\n"
        "<cited>无</cited>"
    )


def invoke_markov_agent_with_fallbacks(
    agent: Any,
    messages_after_user_appended: List[Dict[str, Any]],
    *,
    role: Role,
    system_prompt: str,
    provider: Provider = "deepseek",
    model: str,
    max_tokens: int,
) -> Tuple[List[Any], str, bool, bool]:
    """
    调用 Agent；若遇内容风险 400 则返回占位标签文本。

    :return: (invoke 返回的 messages 列表, 最终 raw 文本,
             search_limit_fallback_used, content_risk_fallback_used)
    """
    try:
        result = agent.invoke({"messages": messages_after_user_appended})
    except BadRequestError as e:
        if not is_content_risk_error(e):
            raise
        return [], content_risk_fallback_raw(role), False, True

    messages = result.get("messages", [])
    raw_agent = next(
        (
            getattr(m, "content", "")
            for m in reversed(messages)
            if getattr(m, "type", "") == "ai"
        ),
        "",
    )
    raw, fallback_used = fallback_reply_after_search_limit(
        raw_agent,
        messages,
        system_prompt=system_prompt,
        history_dicts=list(messages_after_user_appended),
        provider=provider,
        model=model,
        max_tokens=max_tokens,
    )
    return messages, raw, fallback_used, False
