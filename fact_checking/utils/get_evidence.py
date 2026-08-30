"""
从 LangChain Agent invoke 返回的 messages 中提取联网检索轨迹；
applied_evidences 由模型在 <cited> 中给出「批次-条号」，在此解析映射。
每条证据的 cite id 格式：r{debate_round}-{agent_slug}-{batch}-{rank}，避免跨轮次/角色冲突。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.label_process import extract_label


def _msg_type(msg: Any) -> str:
    t = getattr(msg, "type", None)
    if t:
        return str(t).lower()
    name = type(msg).__name__.lower()
    if "human" in name:
        return "human"
    if "ai" in name and "tool" not in name:
        return "ai"
    if "tool" in name:
        return "tool"
    return name


def _tool_call_entries(msg: Any) -> List[Dict[str, Any]]:
    raw = getattr(msg, "tool_calls", None) or []
    out: List[Dict[str, Any]] = []
    for call in raw:
        if isinstance(call, dict):
            name = call.get("name", "") or ""
            args = call.get("args", {}) or {}
            cid = call.get("id", "")
        else:
            name = getattr(call, "name", "") or ""
            args = getattr(call, "args", {}) or {}
            cid = getattr(call, "id", "") or ""
        if not isinstance(args, dict):
            args = {}
        out.append({"name": name, "args": args, "id": cid})
    return out


def _parse_ddg_style_results(content: str) -> List[Dict[str, Any]]:
    """解析 web_search 工具返回格式。"""
    items: List[Dict[str, Any]] = []
    if not content or not str(content).strip():
        return items
    text = str(content).strip()
    if "未找到相关结果" in text:
        return items
    parts = re.split(r"\n\n+", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(
            r"\[(\d+)\]\s*(.+?)\n(.+?)\n来源:\s*(\S+)",
            part,
            re.S,
        )
        if m:
            items.append(
                {
                    "rank": int(m.group(1)),
                    "title": m.group(2).strip(),
                    "snippet": m.group(3).strip(),
                    "url": m.group(4).strip(),
                }
            )
        else:
            items.append(
                {
                    "rank": len(items) + 1,
                    "title": "",
                    "snippet": part,
                    "url": "",
                }
            )
    return items


def extract_retrieved_evidences(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """
    从一次 invoke 的 messages 中提取各轮工具返回的检索结果。
    每项: { "tool", "query", "results": [ {rank, title, snippet, url}, ... ] }
    """
    pending: List[Dict[str, Any]] = []
    retrieved: List[Dict[str, Any]] = []

    for msg in messages:
        mt = _msg_type(msg)
        if mt == "ai":
            for tc in _tool_call_entries(msg):
                q = ""
                args = tc.get("args") or {}
                if isinstance(args, dict):
                    q = str(args.get("query", "") or "")
                pending.append({"name": tc.get("name", ""), "query": q})
        elif mt == "tool":
            name = getattr(msg, "name", None) or ""
            if pending:
                meta = pending.pop(0)
                q = meta.get("query", "")
                if not name:
                    name = str(meta.get("name", "") or "unknown")
            else:
                q = ""
            content = getattr(msg, "content", "") or ""
            content = content if isinstance(content, str) else str(content)
            results = _parse_ddg_style_results(content)
            batch_no = len(retrieved) + 1
            retrieved.append(
                {
                    "batch": batch_no,
                    "tool": name or "web_search",
                    "query": q,
                    "results": results,
                }
            )

    return retrieved


def parse_cited_specs(cited_text: str) -> List[Tuple[int, int]]:
    """
    解析模型输出的引用，格式：批次-条号，对应本次发言中第几次检索工具返回里的 [n]。
    支持：1-2,2-1 / 1-2 2-1 / 中文逗号分隔。未检索或未引用时可写 无 / none / 空。
    """
    s = (cited_text or "").strip()
    if not s:
        return []
    lower = s.lower()
    if lower in ("无", "没有", "none", "n/a", "na", "-", "无引用", "未引用"):
        return []
    # 统一分隔符为逗号
    normalized = re.sub(r"[,，;；\s]+", ",", s)
    specs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*[-_.]\s*(\d+)$", part)
        if not m:
            continue
        b, r = int(m.group(1)), int(m.group(2))
        if b < 1 or r < 1:
            continue
        key = (b, r)
        if key not in seen:
            seen.add(key)
            specs.append(key)
    return specs


def format_evidence_cite_id(
    debate_round: int, agent_slug: str, batch_no: int, rank: int
) -> str:
    """
    全局唯一引用 id：轮次 + 发言角色 + 本次 invoke 内批次 + 条号。
    例：r1-leader-1-1 表示第 1 轮辩论中 leader 第 1 次检索的第 1 条。
    初判（S0）使用 debate_round=0。
    """
    slug = (agent_slug or "unknown").strip().lower().replace(" ", "-")
    return f"r{int(debate_round)}-{slug}-{batch_no}-{rank}"


def applied_evidences_from_cited(
    retrieved_evidences: List[Dict[str, Any]],
    cited_text: str,
    *,
    debate_round: int = 0,
    agent_slug: str = "leader",
) -> List[Dict[str, Any]]:
    """根据 <cited> 解析结果，从 retrieved_evidences 中取出对应条目；cite 为带轮次/角色的全局 id。"""
    applied: List[Dict[str, Any]] = []
    by_batch: Dict[int, Dict[str, Any]] = {}
    for block in retrieved_evidences:
        if not isinstance(block, dict):
            continue
        b = int(block.get("batch", 0) or 0)
        if b > 0:
            by_batch[b] = block

    for batch_no, rank in parse_cited_specs(cited_text):
        cite_id = format_evidence_cite_id(debate_round, agent_slug, batch_no, rank)
        block = by_batch.get(batch_no)
        if not block:
            applied.append(
                {
                    "batch": batch_no,
                    "rank": rank,
                    "debate_round": debate_round,
                    "agent": agent_slug,
                    "cite": cite_id,
                    "error": "batch_not_found",
                }
            )
            continue
        results = block.get("results") or []
        item = next(
            (x for x in results if isinstance(x, dict) and int(x.get("rank", -1)) == rank),
            None,
        )
        if not item:
            applied.append(
                {
                    "batch": batch_no,
                    "rank": rank,
                    "debate_round": debate_round,
                    "agent": agent_slug,
                    "cite": cite_id,
                    "error": "rank_not_found",
                }
            )
            continue
        applied.append(
            {
                "batch": batch_no,
                "rank": rank,
                "debate_round": debate_round,
                "agent": agent_slug,
                "cite": cite_id,
                "tool": block.get("tool", ""),
                "query": block.get("query", ""),
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("url", ""),
            }
        )
    return applied


def attach_evidence_fields(
    parsed: Dict[str, Any],
    invoke_messages: Sequence[Any],
    *,
    debate_round: int = 0,
    agent_slug: str = "leader",
) -> None:
    """
    就地写入 retrieved_evidences、applied_evidences、cited。
    cited 对外为完整引用 id 列表（与 applied_evidences[].cite 一致）；模型原文仍在 parsed['raw'] 的 <cited> 中。
    """
    retrieved = extract_retrieved_evidences(invoke_messages)
    parsed["retrieved_evidences"] = retrieved
    raw = str(parsed.get("raw", "") or "")
    cited_from_model = extract_label("cited", raw, default="").strip()
    applied = applied_evidences_from_cited(
        retrieved,
        cited_from_model,
        debate_round=debate_round,
        agent_slug=agent_slug,
    )
    parsed["applied_evidences"] = applied
    parsed["cited_specs_raw"] = cited_from_model
    if applied:
        parts = [a["cite"] for a in applied if a.get("cite")]
        parsed["cited"] = ",".join(parts) if parts else cited_from_model
    else:
        parsed["cited"] = cited_from_model


def slim_opinion_for_prompt(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """供下一轮 user 消息使用，避免把检索明细重复注入上下文。"""
    if not data:
        return {}
    keys = ("opinion", "stance", "factuality", "Error severity")
    return {k: data[k] for k in keys if k in data}
