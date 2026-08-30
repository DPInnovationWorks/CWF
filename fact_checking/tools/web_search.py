"""
联网搜索工具：DuckDuckGo / Tavily
"""

from typing import List

from config.env import get
from ddgs import DDGS
from langchain_core.tools import tool
from tavily import TavilyClient


# DuckDuckGo
def _duckduckgo_search(query: str, max_results: int = 5, verbose: bool = True) -> str:
    """
    网络/限流等导致搜索失败时返回说明文本，**不抛异常**，以便 Agent 继续对话。
    """
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        if verbose:
            print(f"[DuckDuckGo 搜索失败] 依据当前对话中已有信息与常识继续完成回答 {type(e).__name__}: {e}")
        return (
            "[联网检索失败]请不要再调用搜索工具；请仅依据当前对话中已有信息与常识继续完成回答。"
        )
    if not results:
        print("[DuckDuckGo 搜索]：未找到相关结果") if verbose else None
        return "未找到相关结果。"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"[{i}] {title}\n{body}\n来源: {href}")
    out = "\n\n".join(lines)
    # if verbose:
        # print("[DuckDuckGo 搜索完成]")
        # print("\n[联网搜索结果]\n查询:", query, "\n", out, "\n")
    return out


def _tavily_search(
    query: str,
    max_results: int = 4,
    verbose: bool = True,
    search_depth: str = "basic",
) -> str:
    """使用 Tavily API 联网搜索"""
    api_key = get("TAVILY_KEY")
    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )
    except Exception as e:
        if verbose:
            print(f"[Tavily 搜索失败] {type(e).__name__}: {e}")
        return (
            "【联网检索暂不可用】搜索请求失败。请不要再调用搜索工具，"
            "仅依据已有对话与常识继续完成回答。"
        )
    results = (
        response.get("results")
        if isinstance(response, dict)
        else getattr(response, "results", []) or []
    )
    if not results:
        if verbose:
            print("[Tavily 搜索完成]：未找到相关结果")
        return "未找到相关结果。"

    lines = []
    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            title = r.get("title", "")
            content = r.get("content", "") or r.get("body", "")
            url = r.get("url", "") or r.get("href", "")
        else:
            title = getattr(r, "title", "")
            content = getattr(r, "content", "") or getattr(r, "body", "")
            url = getattr(r, "url", "") or getattr(r, "href", "")
        lines.append(f"[{i}] {title}\n{content}\n来源: {url}")
    out = "\n\n".join(lines)
    #if verbose:
        # print("[Tavily 联网搜索完成]")
        # print("\n[联网搜索结果]\n查询:", query, "\n", out, "\n")
    return out


def build_web_search_tools(max_results: int = 5) -> List:
    """
    返回联网搜索工具列表
    """

    @tool
    def web_search(query: str) -> str:
        """在互联网上搜索最新信息。查询实时新闻、事实、当前事件。"""
        return _duckduckgo_search(query, max_results=max_results)

    return [web_search]
