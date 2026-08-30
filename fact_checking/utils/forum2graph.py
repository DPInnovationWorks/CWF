"""
Markov 论坛结果 JSON → ``CommGraph.from_dict`` 可用的图字典。

仅导出::

    graphs = markov_forum_result_to_comm_graph_dicts(path_or_dict)
    for cg in graphs:
        g = CommGraph.from_dict(cg)
        g.infer_claim_confidence()
        g.draw(layout=\"hierarchy\", title=\"...\")  # 或 fig.savefig(...)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

JsonSource = Union[Dict[str, Any], str, Path]


def markov_forum_result_to_comm_graph_dicts(data: JsonSource) -> List[Dict[str, Any]]:
    """
    将 ``markov_forum_result.json`` 整份对象（或文件路径）转为
    **可直接** ``CommGraph.from_dict(cg)`` 的字典列表。

    返回 ``{\"claim\": {...}}`` 列表：
    - 第 1 条为原始声明聚合图（``claim_original``），其顶层 comments 对应各子声明，并按抽取权重设置 influence。
    - 后续条目与 ``sub_claim_debates`` 顺序一一对应。
    """
    obj = _load_json(data)
    sub_graphs: List[Dict[str, Any]] = []
    for sub in obj.get("sub_claim_debates") or []:
        if isinstance(sub, dict):
            sub_graphs.append(_sub_claim_debate_to_comm_graph_dict(sub))
    root_graph = _build_original_claim_graph(obj, sub_graphs)
    if root_graph is None:
        return sub_graphs
    return [root_graph, *sub_graphs]


def _load_json(data: JsonSource) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    p = Path(data)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _parse_stance(raw: Any, *, default_weak: float = 0.2) -> float:
    """
    统一 stance 到 {-1, -0.2, 0, 0.2, 1}。
    非法值回退为 default_weak（默认 0.2）。
    """
    if raw is None:
        return default_weak
    s = str(raw).strip().lower()
    if s in ("1", "1.0", "支持", "support", "true"):
        return 1.0
    if s in ("0.2", "+0.2", "弱支持", "weak_support"):
        return 0.2
    if s in ("-0.2", "弱反对", "weak_oppose"):
        return -0.2
    if s in ("-1", "-1.0", "反对", "oppose", "false"):
        return -1.0
    if s in ("0", "0.0", "中立", "neutral"):
        return 0.0
    return default_weak


def _evidence_text_from_applied(ae: Dict[str, Any]) -> str:
    title = (ae.get("title") or "").strip()
    snippet = (ae.get("snippet") or "").strip()
    url = (ae.get("url") or "").strip()
    parts = [p for p in (title, snippet) if p]
    body = " ".join(parts)[:2000]
    if url:
        return f"{body}\n{url}" if body else url
    return body or ""


def _evidence_dicts_from_applied_list(
    applied: List[Dict[str, Any]],
    id_prefix: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ae in enumerate(applied or []):
        cite = ae.get("cite")
        if cite is None or str(cite).strip() == "":
            cite = f"{id_prefix}_ae_{i}"
        eid = str(cite).strip()
        content = _evidence_text_from_applied(ae)
        if not content:
            content = eid
        out.append({"id": eid, "content": content, "stance": 1})
    return out


def _iter_debate_opinion_steps(
    debate: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    init = debate.get("initial")
    if isinstance(init, dict):
        out.append(("initial", init))
    for r in debate.get("rounds") or []:
        if not isinstance(r, dict):
            continue
        ri = r.get("round", 0)
        for step in r.get("sequence") or []:
            if not isinstance(step, dict):
                continue
            agent = step.get("agent") or "unknown"
            op = step.get("output")
            if not isinstance(op, dict):
                continue
            out.append((f"r{ri}_{agent}", op))
    return out


def _comment_node_from_op(
    label_key: str, op: Dict[str, Any], prefix: str
) -> Dict[str, Any]:
    opinion_text = (op.get("opinion") or "").strip()
    evs = _evidence_dicts_from_applied_list(
        op.get("applied_evidences") or [], f"{prefix}_{label_key}"
    )
    weak_default = -0.2 if "skeptic" in label_key else 0.2
    return {
        "id": f"{prefix}_{label_key}",
        "content": opinion_text,
        "stance": _parse_stance(op.get("stance"), default_weak=weak_default),
        "influence": 1.0,
        "evidence": evs,
        "comments": [],
    }


def _build_markov_comment_tree(debate: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    top_level: List[Dict[str, Any]] = []

    init = debate.get("initial")
    if isinstance(init, dict):
        initial_node = _comment_node_from_op("initial", init, prefix)
        initial_node["comments"] = []
        nodes_by_id[initial_node["id"]] = initial_node
        top_level.append(initial_node)

    for round_block in debate.get("rounds") or []:
        if not isinstance(round_block, dict):
            continue
        r = int(round_block.get("round", 0) or 0)
        if r < 1:
            continue
        seq = round_block.get("sequence") or []
        if len(seq) < 3:
            continue
        state = str(round_block.get("state") or "S1").strip().upper()

        anchor_id = f"{prefix}_initial" if r == 1 else f"{prefix}_r{r - 1}_leader"
        anchor = nodes_by_id.get(anchor_id)
        if anchor is None:
            continue

        if state == "S1":
            t_op = (seq[0].get("output") or {}) if isinstance(seq[0], dict) else {}
            s_op = (seq[1].get("output") or {}) if isinstance(seq[1], dict) else {}
            l_op = (seq[2].get("output") or {}) if isinstance(seq[2], dict) else {}
            t_node = _comment_node_from_op(f"r{r}_trust", t_op, prefix)
            s_node = _comment_node_from_op(f"r{r}_skeptic", s_op, prefix)
            t_node["comments"] = [s_node]
            nodes_by_id[t_node["id"]] = t_node
            nodes_by_id[s_node["id"]] = s_node
            anchor.setdefault("comments", []).append(t_node)
        elif state == "S2":
            sk_op = (seq[0].get("output") or {}) if isinstance(seq[0], dict) else {}
            tr_op = (seq[1].get("output") or {}) if isinstance(seq[1], dict) else {}
            l_op = (seq[2].get("output") or {}) if isinstance(seq[2], dict) else {}
            sk_node = _comment_node_from_op(f"r{r}_skeptic", sk_op, prefix)
            tr_node = _comment_node_from_op(f"r{r}_trust", tr_op, prefix)
            sk_node["comments"] = [tr_node]
            nodes_by_id[sk_node["id"]] = sk_node
            nodes_by_id[tr_node["id"]] = tr_node
            anchor.setdefault("comments", []).append(sk_node)
        else:
            continue

        l_node = _comment_node_from_op(f"r{r}_leader", l_op, prefix)
        l_node["comments"] = []
        nodes_by_id[l_node["id"]] = l_node
        top_level.append(l_node)

    return top_level


def _flat_comments_from_steps(debate: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    comment_dicts: List[Dict[str, Any]] = []
    for label, op in _iter_debate_opinion_steps(debate):
        opinion_text = (op.get("opinion") or "").strip()
        if not opinion_text:
            continue
        evs = _evidence_dicts_from_applied_list(
            op.get("applied_evidences") or [], f"{prefix}_{label}"
        )
        comment_dicts.append(
            {
                "id": f"{prefix}_{label}",
                "content": opinion_text,
                "stance": _parse_stance(
                    op.get("stance"), default_weak=(-0.2 if "skeptic" in label else 0.2)
                ),
                "influence": 1.0,
                "evidence": evs,
            }
        )
    return comment_dicts


def _final_opinion_block(debate: Dict[str, Any]) -> Dict[str, Any]:
    fin = debate.get("final")
    if isinstance(fin, dict) and fin:
        return fin
    steps = _iter_debate_opinion_steps(debate)
    for label, op in reversed(steps):
        if label.endswith("_leader"):
            return op
    init = debate.get("initial")
    if isinstance(init, dict):
        return init
    return {}


def _sub_claim_debate_to_comm_graph_dict(sub_entry: Dict[str, Any]) -> Dict[str, Any]:
    idx = int(sub_entry.get("index", 0))
    claim_text = (sub_entry.get("claim") or "").strip()
    debate = sub_entry.get("debate") or {}
    if not isinstance(debate, dict):
        debate = {}

    prefix = f"claim_{idx}"
    final_op = _final_opinion_block(debate)

    comment_tree = _build_markov_comment_tree(debate, prefix)
    if not comment_tree:
        comment_tree = _flat_comments_from_steps(debate, prefix)

    return {
        "claim": {
            "id": f"{prefix}",
            "content": claim_text,
            "stance": _parse_stance(final_op.get("stance"), default_weak=0.2),
            "evidence": [],
            "comments": comment_tree,
        },
    }


def _normalize_weights(raw_weights: List[float]) -> List[float]:
    cleaned = [max(0.0, float(w)) for w in raw_weights]
    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned)] * len(cleaned) if cleaned else []
    return [w / total for w in cleaned]


def _build_original_claim_graph(
    obj: Dict[str, Any], sub_graphs: List[Dict[str, Any]]
) -> Dict[str, Any] | None:
    if not sub_graphs:
        return None
    subs = [s for s in (obj.get("sub_claim_debates") or []) if isinstance(s, dict)]
    if not subs:
        return None
    count = min(len(subs), len(sub_graphs))
    if count <= 0:
        return None

    raw_weights = [float((subs[i].get("weight", 1.0) or 1.0)) for i in range(count)]
    norm_weights = _normalize_weights(raw_weights)
    root_comments: List[Dict[str, Any]] = []
    for i in range(count):
        sub = subs[i]
        sub_graph = sub_graphs[i]
        sub_claim = sub_graph.get("claim") or {}
        root_comments.append(
            {
                "id": f"sub_{sub_claim.get('id') or f'claim_{i}'}",
                "content": (sub.get("claim") or sub_claim.get("content") or "").strip(),
                # 子声明是对原始声明的语义分解，故其到 original 的关系固定为支持。
                "stance": 1,
                "influence": norm_weights[i],
                "evidence": [],
                "comments": copy.deepcopy(sub_claim.get("comments") or []),
            }
        )

    root_text = (obj.get("original_complex_claim") or "").strip()
    if not root_text:
        root_text = (obj.get("claim") or "").strip()
    return {
        "claim": {
            "id": "claim_original",
            "content": root_text,
            "stance": 1,
            "evidence": [],
            "comments": root_comments,
            "conflict_penalty_lambda": 0.15,
        }
    }


__all__ = ["markov_forum_result_to_comm_graph_dicts"]
