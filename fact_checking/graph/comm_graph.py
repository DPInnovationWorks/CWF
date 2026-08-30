"""
Claim–Comment–Evidence
节点类型：claim、comment、evidence
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import networkx as nx
from typing import Any


class NodeType(str, Enum):
    CLAIM = "claim"
    COMMENT = "comment"
    EVIDENCE = "evidence"


# evidence 固定置信度
EVIDENCE_CONFIDENCE = 5
# claim / comment 初始置信度
INITIAL_CLAIM_COMMENT_CONFIDENCE = 2.5
# 仅用于「无子评论时」证据分与初值 2.5 混合：w 为证据分一侧占比；(1-w) 为初值 2.5。
INITIAL_BLEND_WEIGHT = 0.15
# 叶子 comment（无子评论）在「证据 vs 初值」中的动态证据权重范围与增长速度。
LEAF_EVIDENCE_W_MIN = 0.25
LEAF_EVIDENCE_W_MAX = 0.5
LEAF_EVIDENCE_N_SCALE = 3.0
# 「本地证据 vs 子评论 / claim 直连证据 vs 顶层评论」混合：证据条数 n 越多，证据侧占比越高（更信自己）。
# 鲁棒默认：提高低样本保守性，并限制证据侧上限，避免单边信息过快压制评论侧。
EVIDENCE_VS_PEER_W_MIN = 0.15
EVIDENCE_VS_PEER_W_MAX = 0.75
EVIDENCE_VS_PEER_N_SCALE = 4.0  # 越大则随 n 增长越慢、越晚贴近 W_MAX
# 证据条数 n 对聚合强度的放大：在「带符号均值」上乘以 (1 + k*log1p(n))，n 越大证据侧信号越强（再 clamp）。
EVIDENCE_COUNT_STRENGTH_COEF = 0.2
# 弱立场值：不允许完全中立（0），统一使用弱支持/弱质疑。
WEAK_SUPPORT_STANCE = 0.2
WEAK_OPPOSE_STANCE = -0.2
EDGE_NEUTRAL_EPS = 0.1


# stance：允许 {-1, -0.2, 0, 0.2, 1}；非法值映射为默认弱支持 0.2。
def _parse_stance(v: Any, *, default_weak: float = WEAK_SUPPORT_STANCE) -> float:
    if v is None:
        return default_weak
    s = str(v).strip().lower()
    if s in ("1", "1.0", "支持", "support", "true"):
        return 1.0
    if s in ("0.2", "+0.2", "弱支持", "weak_support"):
        return WEAK_SUPPORT_STANCE
    if s in ("-0.2", "弱反对", "weak_oppose"):
        return WEAK_OPPOSE_STANCE
    if s in ("-1", "-1.0", "反对", "oppose", "false"):
        return -1.0
    if s in ("0", "0.0", "中立", "neutral"):
        return 0.0
    return default_weak


@dataclass
class Evidence:
    """证据节点：置信度恒为 EVIDENCE_CONFIDENCE（默认 5）"""

    id: str
    content: str = ""
    confidence: float = EVIDENCE_CONFIDENCE
    stance: float = WEAK_SUPPORT_STANCE  # 1=强支持, 0.2=弱支持, -0.2=弱反对, -1=强反对

    def node_id(self) -> str:
        return f"evidence:{self.id}"


@dataclass
class Comment:
    """评论节点：初始置信度为 INITIAL_CLAIM_COMMENT_CONFIDENCE；推理后 confidence 更新，initial_confidence 不变。"""

    id: str
    content: str = ""
    confidence: float = INITIAL_CLAIM_COMMENT_CONFIDENCE
    initial_confidence: float = INITIAL_CLAIM_COMMENT_CONFIDENCE
    influence: float = 1.0  # 对父节点（claim 或父 comment）的影响力权重
    stance: float = WEAK_SUPPORT_STANCE  # 1=强支持, 0.2=弱支持, -0.2=弱反对, -1=强反对
    children: list["Comment"] = field(default_factory=list)  # 子评论

    def node_id(self) -> str:
        return f"comment:{self.id}"

    def all_comments(self) -> list["Comment"]:
        """扁平化返回自身及所有后代 comment"""
        out: list[Comment] = [self]
        for ch in self.children:
            out.extend(ch.all_comments())
        return out


@dataclass
class Claim:
    """主张节点：初始置信度为 INITIAL_CLAIM_COMMENT_CONFIDENCE；推理后 confidence 更新，initial_confidence 不变。"""

    id: str
    content: str = ""
    confidence: float = INITIAL_CLAIM_COMMENT_CONFIDENCE
    initial_confidence: float = INITIAL_CLAIM_COMMENT_CONFIDENCE
    stance: float = WEAK_SUPPORT_STANCE  # 1=强支持, 0.2=弱支持, -0.2=弱反对, -1=强反对
    conflict_penalty_lambda: float = 0.0

    def node_id(self) -> str:
        return f"claim:{self.id}"


def _parse_evidence_list(items: list[Any]) -> list[Evidence]:
    """从 [{"id":"e1","content":"...","stance":1}, ...] 解析出 Evidence 列表。"""
    out: list[Evidence] = []
    for i, e in enumerate(items or []):
        if not isinstance(e, dict):
            continue
        eid = e.get("id") or f"evidence_{i}"
        out.append(
            Evidence(
                id=eid,
                content=e.get("content", ""),
                stance=_parse_stance(e.get("stance"), default_weak=WEAK_SUPPORT_STANCE),
            )
        )
    return out


def _evidence_ids_from_list(items: list[Any]) -> list[str]:
    """从 evidence 列表提取 id 列表。"""
    ids: list[str] = []
    for i, e in enumerate(items or []):
        if isinstance(e, dict):
            eid = e.get("id") or f"evidence_{i}"
        else:
            eid = str(e)
        ids.append(eid)
    return ids


def _parse_comment_tree(
    comment_dicts: list[Any],
    comment_evidence_ids: dict[str, list[str]],
    evidence_by_id: dict[str, Evidence],
) -> list[Comment]:
    """递归解析 comment 树"""
    result: list[Comment] = []
    for i, c in enumerate(comment_dicts or []):
        if not isinstance(c, dict):
            continue
        cid = c.get("id") or f"comment_{i}"
        evidence_raw = c.get("evidence") or c.get("evidence_ids") or []
        ev_list = evidence_raw if isinstance(evidence_raw, list) else []
        eids = _evidence_ids_from_list(ev_list)
        comment_evidence_ids[cid] = eids
        for e in _parse_evidence_list(ev_list):
            if e.id not in evidence_by_id:
                evidence_by_id[e.id] = e
        children = _parse_comment_tree(
            c.get("comments") or [], comment_evidence_ids, evidence_by_id
        )
        result.append(
            Comment(
                id=cid,
                content=c.get("content", ""),
                confidence=INITIAL_CLAIM_COMMENT_CONFIDENCE,
                initial_confidence=INITIAL_CLAIM_COMMENT_CONFIDENCE,
                influence=float(c.get("influence", 1.0)),
                stance=_parse_stance(c.get("stance"), default_weak=WEAK_SUPPORT_STANCE),
                children=children,
            )
        )
    return result


@dataclass
class CommGraph:
    """
    一个 claim、树形 comments、若干 evidence 的图。
    - claim 下有 evidence 与 comments；comments 可无限嵌套（comments 内嵌 comments）。
    - 边：claim/comment -> evidence；comment -> 父节点（claim 或父 comment），带 influence。
    - 刻度：evidence 置信度恒为 EVIDENCE_CONFIDENCE（5）；claim/comment 初始为 INITIAL（2.5）；
      图推后 claim/comment 的 confidence 落在 [0, 5]。
    """

    claim: Claim
    comments: list[Comment] = field(
        default_factory=list
    )  # 仅顶层；嵌套在 comment.children 中
    evidence: list[Evidence] = field(default_factory=list)
    claim_evidence_ids: list[str] = field(default_factory=list)
    comment_evidence_ids: dict[str, list[str]] = field(default_factory=dict)
    _inferred_claim_confidence: float | None = None

    def _evidence_by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.evidence}

    def _all_comments_flat(self) -> list[Comment]:
        """扁平化所有 comment（含嵌套）。"""
        out: list[Comment] = []
        for c in self.comments:
            out.extend(c.all_comments())
        return out

    @staticmethod
    def _evidence_stance_weight(stance: float) -> float:
        """证据 stance 直接作为权重参与传播（允许 ±0.2 弱立场）。"""
        s = float(stance)
        return max(-1.0, min(1.0, s))

    @staticmethod
    def _signed_aggregate_to_scale05(signed: float) -> float:
        """
        将「带符号」的聚合（约在 [-EVIDENCE_CONFIDENCE, EVIDENCE_CONFIDENCE]）线性映射到 [0, 5]。
        """
        s = max(
            -EVIDENCE_CONFIDENCE,
            min(EVIDENCE_CONFIDENCE, float(signed)),
        )
        return (s + EVIDENCE_CONFIDENCE) / (2.0 * EVIDENCE_CONFIDENCE) * EVIDENCE_CONFIDENCE

    @staticmethod
    def _blend_with_initial(
        evidence_score_05: float, weight: float = INITIAL_BLEND_WEIGHT
    ) -> float:
        """与初始 2.5 混合，得到 [0,5] 内结果。weight 为证据分一侧占比。"""
        w = max(0.0, min(1.0, weight))
        out = (1.0 - w) * INITIAL_CLAIM_COMMENT_CONFIDENCE + w * evidence_score_05
        return max(0.0, min(EVIDENCE_CONFIDENCE, out))

    @staticmethod
    def _count_effective_evidence(ev: dict[str, Evidence], eids: list[str]) -> int:
        return sum(1 for eid in eids if eid in ev)

    @staticmethod
    def _evidence_vs_peer_blend_weight(n: int) -> float:
        """
        证据条数 n 越多，越提高「证据侧」在「证据 vs 子评论/顶层评论」中的占比。
        n=1 时为 EVIDENCE_VS_PEER_W_MIN；n 增大时渐近 EVIDENCE_VS_PEER_W_MAX。
        """
        w_min = EVIDENCE_VS_PEER_W_MIN
        w_max = EVIDENCE_VS_PEER_W_MAX
        if n <= 0:
            return w_min
        t = 1.0 - math.exp(-(n - 1) / max(EVIDENCE_VS_PEER_N_SCALE, 1e-9))
        t = max(0.0, min(1.0, t))
        return w_min + (w_max - w_min) * t

    @staticmethod
    def _leaf_evidence_blend_weight(n: int) -> float:
        """
        叶子 comment（无子评论）在「证据 vs 初值」中的证据侧权重。
        n 越大证据权重越高，范围 [LEAF_EVIDENCE_W_MIN, LEAF_EVIDENCE_W_MAX]。
        """
        w_min = LEAF_EVIDENCE_W_MIN
        w_max = LEAF_EVIDENCE_W_MAX
        if n <= 0:
            return w_min
        t = 1.0 - math.exp(-n / max(LEAF_EVIDENCE_N_SCALE, 1e-9))
        t = max(0.0, min(1.0, t))
        return w_min + (w_max - w_min) * t

    def _aggregate_evidence_signed(
        self, ev: dict[str, Evidence], eids: list[str]
    ) -> float | None:
        """对证据做带符号聚合：先均值，再按条数 n 放大（纯均值时 n 不改变数值）。"""
        if not eids:
            return None
        vals: list[float] = []
        for eid in eids:
            if eid not in ev:
                continue
            e = ev[eid]
            vals.append(
                float(e.confidence) * self._evidence_stance_weight(e.stance)
            )
        if not vals:
            return None
        n = len(vals)
        mean_signed = sum(vals) / n
        strength = 1.0 + EVIDENCE_COUNT_STRENGTH_COEF * math.log1p(n)
        signed = mean_signed * strength
        return max(-EVIDENCE_CONFIDENCE, min(EVIDENCE_CONFIDENCE, signed))

    def _update_comment_confidence_recursive(
        self, c: Comment, ev: dict[str, Evidence]
    ) -> None:
        """自底向上更新 comment 置信度，输出刻度 [0, EVIDENCE_CONFIDENCE]。"""
        for ch in c.children:
            self._update_comment_confidence_recursive(ch, ev)
        eids = self.comment_evidence_ids.get(c.id) or []

        ev_signed = self._aggregate_evidence_signed(ev, eids)
        ev_05 = (
            self._signed_aggregate_to_scale05(ev_signed)
            if ev_signed is not None
            else None
        )

        if c.children:
            inf_sum = sum(ch.influence for ch in c.children)
            child_signed = (
                sum(
                    ch.confidence * ch.influence * ch.stance
                    for ch in c.children
                )
                / max(inf_sum, 1e-9)
            )
            child_05 = self._signed_aggregate_to_scale05(child_signed)

            if ev_05 is not None:
                n_ev = self._count_effective_evidence(ev, eids)
                w = self._evidence_vs_peer_blend_weight(n_ev)
                c.confidence = w * ev_05 + (1.0 - w) * child_05
            else:
                # 无本地证据时，与根 claim 一致：直接采用子评论树聚合，避免再与初值 2.5 混合导致“父低于子”失真
                c.confidence = child_05
        else:
            if ev_05 is not None:
                n_ev = self._count_effective_evidence(ev, eids)
                w_leaf = self._leaf_evidence_blend_weight(n_ev)
                c.confidence = self._blend_with_initial(ev_05, weight=w_leaf)
            else:
                c.confidence = INITIAL_CLAIM_COMMENT_CONFIDENCE

        c.confidence = max(0.0, min(EVIDENCE_CONFIDENCE, c.confidence))

    def infer_claim_confidence(self) -> float:
        """
        自底向上更新 comment，再算 claim。claim/comment 的推理置信度均在 [0, 5]，
        与 evidence=5、初始 2.5 同一刻度。
        证据聚合在「带符号均值」上按条数放大（见 _aggregate_evidence_signed）。
        「直连证据 vs 顶层评论」混合权重随直连证据条数增多而提高证据侧（见 _evidence_vs_peer_blend_weight）。
        若无直连证据仅有顶层评论，根置信度取 comment_05（主要反映辩论结果），不与初值 2.5 做 _blend_with_initial。
        """
        ev = self._evidence_by_id()
        for c in self.comments:
            self._update_comment_confidence_recursive(c, ev)

        direct_signed = self._aggregate_evidence_signed(ev, self.claim_evidence_ids)
        direct_05 = (
            self._signed_aggregate_to_scale05(direct_signed)
            if direct_signed is not None
            else None
        )

        total_influence = sum(c.influence for c in self.comments)
        comment_05: float | None = None
        if total_influence > 0:
            # 顶层评论已在 [0,5] 刻度，直接做（带 stance）加权均值，不再做二次 [-5,5]→[0,5] 映射。
            comment_05 = (
                sum(c.confidence * c.influence * c.stance for c in self.comments)
                / total_influence
            )
            comment_05 = max(0.0, min(EVIDENCE_CONFIDENCE, comment_05))

        if direct_05 is not None and comment_05 is not None:
            n_direct = self._count_effective_evidence(ev, self.claim_evidence_ids)
            w = self._evidence_vs_peer_blend_weight(n_direct)
            claim_conf = w * direct_05 + (1.0 - w) * comment_05
        elif direct_05 is not None:
            claim_conf = self._blend_with_initial(direct_05)
        elif comment_05 is not None:
            claim_conf = comment_05
        else:
            claim_conf = INITIAL_CLAIM_COMMENT_CONFIDENCE

        # 根节点冲突惩罚：顶层观点分歧越大，整体置信度越保守。
        if self.claim.conflict_penalty_lambda > 0 and len(self.comments) >= 2:
            weights = [max(float(c.influence), 0.0) for c in self.comments]
            w_sum = sum(weights)
            if w_sum > 0:
                values = [float(c.confidence) for c in self.comments]
                mean_val = sum(v * w for v, w in zip(values, weights)) / w_sum
                var_val = (
                    sum(w * ((v - mean_val) ** 2) for v, w in zip(values, weights))
                    / w_sum
                )
                claim_conf -= self.claim.conflict_penalty_lambda * var_val

        claim_conf = max(0.0, min(EVIDENCE_CONFIDENCE, claim_conf))
        self.claim.confidence = claim_conf
        self._inferred_claim_confidence = claim_conf
        return claim_conf

    def get_claim_confidence(self) -> float:
        """返回当前 claim 置信度；若尚未推理则先执行一次推理。"""
        if self._inferred_claim_confidence is None:
            self.infer_claim_confidence()
        return self.claim.confidence

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommGraph:
        """从字典构建图"""
        claim_data = data.get("claim") or {}
        claim = Claim(
            id=claim_data.get("id", "claim_0"),
            content=claim_data.get("content", ""),
            confidence=INITIAL_CLAIM_COMMENT_CONFIDENCE,
            initial_confidence=INITIAL_CLAIM_COMMENT_CONFIDENCE,
            stance=_parse_stance(claim_data.get("stance"), default_weak=WEAK_SUPPORT_STANCE),
            conflict_penalty_lambda=float(
                claim_data.get("conflict_penalty_lambda", 0.0) or 0.0
            ),
        )
        claim_evidence_raw = (
            claim_data.get("evidence") or claim_data.get("evidence_ids") or []
        )
        if isinstance(claim_evidence_raw, list):
            claim_evidence_ids = _evidence_ids_from_list(claim_evidence_raw)
        else:
            claim_evidence_ids = []

        evidence_by_id: dict[str, Evidence] = {}
        for e in _parse_evidence_list(
            claim_evidence_raw if isinstance(claim_evidence_raw, list) else []
        ):
            evidence_by_id[e.id] = e
        comment_evidence_ids = {}
        comments = _parse_comment_tree(
            claim_data.get("comments") or [], comment_evidence_ids, evidence_by_id
        )
        evidence = list(evidence_by_id.values())

        return cls(
            claim=claim,
            comments=comments,
            evidence=evidence,
            claim_evidence_ids=claim_evidence_ids,
            comment_evidence_ids=comment_evidence_ids,
        )

    @classmethod
    def from_json(cls, json_str: str) -> CommGraph:
        """从 JSON 字符串构建图"""
        return cls.from_dict(json.loads(json_str))

    def _comment_parent_pairs(self) -> list[tuple[Comment, str]]:
        """(comment, parent_node_id) 列表，parent 为 claim 或父 comment 的 node_id。"""
        out: list[tuple[Comment, str]] = []
        claim_nid = self.claim.node_id()

        def walk(comments: list[Comment], parent_nid: str) -> None:
            for c in comments:
                out.append((c, parent_nid))
                walk(c.children, c.node_id())

        walk(self.comments, claim_nid)
        return out

    def to_networkx(self) -> "nx.DiGraph":
        """构建 NetworkX 有向图；comment 连向父节点（claim 或父 comment）及自身 evidence。"""
        try:
            import networkx as nx
        except ImportError:
            raise RuntimeError("需要安装 networkx：pip install networkx")
        G = nx.DiGraph()
        ev = self._evidence_by_id()

        G.add_node(
            self.claim.node_id(),
            node_type=NodeType.CLAIM,
            confidence=self.claim.confidence,
            initial_confidence=self.claim.initial_confidence,
            stance=self.claim.stance,
            conflict_penalty_lambda=self.claim.conflict_penalty_lambda,
        )
        for c in self._all_comments_flat():
            G.add_node(
                c.node_id(),
                node_type=NodeType.COMMENT,
                confidence=c.confidence,
                initial_confidence=c.initial_confidence,
                influence=c.influence,
                stance=c.stance,
            )
        for e in self.evidence:
            G.add_node(
                e.node_id(),
                node_type=NodeType.EVIDENCE,
                confidence=e.confidence,
                stance=e.stance,
            )

        for eid in self.claim_evidence_ids:
            if eid in ev:
                G.add_edge(self.claim.node_id(), ev[eid].node_id(), relation="supports")
        for c, parent_nid in self._comment_parent_pairs():
            G.add_edge(
                c.node_id(), parent_nid, weight=c.influence, relation="influences"
            )
            for eid in self.comment_evidence_ids.get(c.id) or []:
                if eid in ev:
                    G.add_edge(c.node_id(), ev[eid].node_id(), relation="supports")
        return G

    def _hierarchy_layout(self, G: "nx.DiGraph") -> dict:
        """分层布局：claim 在上，comments 按深度分层，evidence 在最下，减少边交叉。"""
        claim_nid = self.claim.node_id()
        ev_nids = {e.node_id() for e in self.evidence}
        layers: list[list[str]] = [[claim_nid]]

        def assign_layer(comments: list[Comment], layer: int) -> None:
            for c in comments:
                nid = c.node_id()
                while len(layers) <= layer:
                    layers.append([])
                layers[layer].append(nid)
                assign_layer(c.children, layer + 1)

        assign_layer(self.comments, 1)
        layers.append([n for n in G if n in ev_nids])
        layers = [L for L in layers if L]

        pos = {}
        n_layers = len(layers)
        for li, layer_nodes in enumerate(layers):
            y = 1.0 - (li + 0.5) / n_layers
            n_per_layer = len(layer_nodes)
            for xi, n in enumerate(layer_nodes):
                x = (xi + 1) / (n_per_layer + 1)
                pos[n] = (x, y)
        return pos

    def draw(
        self,
        ax=None,
        node_size: int = 1200,
        font_size: int = 8,
        layout: str = "spring",
        title: str | None = "Claim-Comment-Evidence Graph",
    ) -> Any:
        """
        layout: 'hierarchy' | 'spring' | 'shell' | 'kamada_kawai'
        边按 stance 着色：绿=支持，红=反对，灰=中立。
        claim/comment→evidence 的 supports 边按「支持」着色（绿），与证据节点 stance 无关。
        """
        # 先推理一次，使图中显示的置信度为推理结果
        self.infer_claim_confidence()
        G = self.to_networkx()

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 9))
        else:
            fig = ax.figure

        if layout == "hierarchy":
            pos = self._hierarchy_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G, seed=42, k=2.0)
        elif layout == "shell":
            shells = [[self.claim.node_id()]]
            shells.append(
                [n for n in G if G.nodes[n].get("node_type") == NodeType.COMMENT]
            )
            shells.append(
                [n for n in G if G.nodes[n].get("node_type") == NodeType.EVIDENCE]
            )
            pos = nx.shell_layout(G, nlist=shells)
        else:
            pos = nx.kamada_kawai_layout(G)

        # 节点颜色：claim 红，comment 蓝，evidence 琥珀色（与边色区分）
        color_map = []
        for n in G:
            t = G.nodes[n].get("node_type")
            if t == NodeType.CLAIM:
                color_map.append("#e74c3c")
            elif t == NodeType.COMMENT:
                color_map.append("#3498db")
            else:
                color_map.append("#f39c12")  # evidence 琥珀色

        conf_map = {n: G.nodes[n].get("confidence", 0) for n in G}
        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=node_size, ax=ax)

        # 边按 stance 分色：强支持(深绿)、弱支持(浅绿)、中性(灰)、弱质疑(浅红)、强质疑(深红)
        stance_edge_color = {
            2: "#27ae60",   # strong support
            1: "#7DCEA0",   # weak support
            0: "#95a5a6",   # near neutral
            -1: "#F1948A",  # weak oppose
            -2: "#c0392b",  # strong oppose
        }
        edges_by_stance: dict[int, list[tuple[str, str]]] = {
            2: [],
            1: [],
            0: [],
            -1: [],
            -2: [],
        }
        for u, v in G.edges():
            rel = G.edges[u, v].get("relation", "")
            if rel == "influences":
                stance = G.nodes[u].get("stance", 0)
            elif rel == "supports":
                # 声明/评论 -> evidence：边颜色跟随 evidence.stance（绿=支持，灰=中立，红=反对）
                stance = G.nodes[v].get("stance", 0)
            else:
                stance = G.nodes[v].get("stance", 0)
            s = float(stance)
            if s >= 0.6:
                stance_key = 2
            elif s > EDGE_NEUTRAL_EPS:
                stance_key = 1
            elif s <= -0.6:
                stance_key = -2
            elif s < -EDGE_NEUTRAL_EPS:
                stance_key = -1
            else:
                stance_key = 0
            edges_by_stance[stance_key].append((u, v))
        for stance, edge_list in edges_by_stance.items():
            if edge_list:
                color = stance_edge_color.get(stance, "#95a5a6")
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=edge_list,
                    edge_color=color,
                    arrows=True,
                    arrowsize=12,
                    ax=ax,
                    width=1.5,
                )

        labels: dict[str, str] = {}
        for n in G:
            short = n.split(":", 1)[-1]
            t = G.nodes[n].get("node_type")
            if t == NodeType.CLAIM or t == NodeType.COMMENT:
                inf_c = float(G.nodes[n].get("confidence", 0) or 0)
                labels[n] = f"{short}\ninf={inf_c:.2f}"
            else:
                labels[n] = f"{short}\nconf={conf_map[n]:.2f}"
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)
        if title:
            ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        return fig
