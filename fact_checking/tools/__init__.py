"""
Agent 通用工具：联网搜索、知识库检索等。
"""

from pathlib import Path
from typing import List, Optional

from tools.web_search import build_web_search_tools
from tools.knowledge_retrieval import (
    build_knowledge_retriever_tool,
    get_default_knowledge_tools,
)


def get_agent_extra_tools(
    *,
    web_search: bool = True,
    knowledge_base: bool = False,
    knowledge_base_path: Optional[str | Path] = None,
    knowledge_persist_dir: Optional[str | Path] = None,
    web_search_max_results: int = 5,
) -> List:
    """
    返回「联网搜索 + 知识库检索」工具列表，供 create_langchain_agent 的 tools 参数拼接使用。

    :param web_search: 是否启用联网搜索
    :param knowledge_base: 是否启用知识库检索
    :param knowledge_base_path: 知识库文档目录（默认 data/knowledge）
    :param knowledge_persist_dir: 向量索引持久化目录（默认 data/knowledge_index）
    :param web_search_max_results: 联网搜索返回条数
    """
    extra: List = []
    if web_search:
        extra.extend(build_web_search_tools(max_results=web_search_max_results))
    if knowledge_base:
        if knowledge_base_path is not None or knowledge_persist_dir is not None:
            from pathlib import Path

            root = Path(__file__).resolve().parent.parent
            base = (
                Path(knowledge_base_path)
                if knowledge_base_path
                else root / "data" / "knowledge"
            )
            persist = (
                Path(knowledge_persist_dir)
                if knowledge_persist_dir
                else root / "data" / "knowledge_index"
            )
            extra.extend(build_knowledge_retriever_tool(base, persist))
        else:
            extra.extend(get_default_knowledge_tools())
    return extra


__all__ = [
    "build_web_search_tools",
    "build_knowledge_retriever_tool",
    "get_default_knowledge_tools",
    "get_agent_extra_tools",
]
