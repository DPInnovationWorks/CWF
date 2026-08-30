"""
统一创建 LangChain 聊天模型 / Agent 的工厂。
"""

from typing import Literal, Optional
from langchain.agents import create_agent
from config.env import ensure_loaded

Provider = Literal["qwen", "deepseek", "gpt"]


def get_chat_model(
    provider: Provider,
    *,
    model: str | None = None,
    max_tokens: int = 3000,
    temperature: float = 0,
    **kwargs,
):
    """
    根据 provider 创建 LangChain ChatModel 实例。
    """
    ensure_loaded()

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model or "deepseek-chat",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "qwen":
        from langchain_qwq import ChatQwen

        return ChatQwen(
            model=model or "qwen-plus",
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "gpt":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    raise ValueError(f"Unknown provider: {provider}")


def create_langchain_agent(
    provider: Provider,
    tools: list,
    system_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 3000,
    temperature: float = 0,
    enable_web_search: bool = False,
    enable_knowledge_base: bool = False,
    max_search_counts: Optional[int] = 2,
    **model_kwargs,
):
    """
    使用统一配置创建 LangChain Agent（create_agent）。
    """
    all_tools = list(tools)
    if enable_web_search or enable_knowledge_base:
        from tools import get_agent_extra_tools

        extra = get_agent_extra_tools(
            web_search=enable_web_search,
            knowledge_base=enable_knowledge_base,
        )
        all_tools = all_tools + extra

    chat_model = get_chat_model(
        provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **model_kwargs,
    )

    middleware = []
    if max_search_counts is not None and max_search_counts > 0:
        from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware

        middleware.append(
            ToolCallLimitMiddleware(run_limit=max_search_counts, exit_behavior="end")
        )

    return create_agent(
        model=chat_model,
        tools=all_tools,
        system_prompt=system_prompt,
        middleware=middleware if middleware else (),
    )
