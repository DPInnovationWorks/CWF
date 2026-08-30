"""
markov_forum_result.json → CommGraph：仅用 ``markov_forum_result_to_comm_graph_dicts`` + ``CommGraph``。

python -B tests/forum/forum2graph.test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLE_JSON = ROOT / "tests" / "forum" / "example" / "markov_forum_result.json"
GRAPH_OUT_DIR = ROOT / "tests" / "forum" / "example" / "forum_graphs"

import matplotlib.pyplot as plt

from graph.comm_graph import CommGraph
from utils.forum2graph import markov_forum_result_to_comm_graph_dicts


def test_one_entrypoint_converts_and_matches_sub_claims_count():
    graphs = markov_forum_result_to_comm_graph_dicts(EXAMPLE_JSON)
    raw = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))
    assert len(graphs) == len(raw.get("sub_claim_debates") or []) + 1


def test_root_claim_graph_exists_and_uses_sub_claim_weights():
    graphs = markov_forum_result_to_comm_graph_dicts(EXAMPLE_JSON)
    root = graphs[0]["claim"]
    raw = json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))
    subs = raw.get("sub_claim_debates") or []
    assert root["id"] == "claim_original"
    assert len(root.get("comments") or []) == len(subs)
    if subs:
        total = sum(float(c.get("influence", 0.0) or 0.0) for c in root["comments"])
        assert abs(total - 1.0) < 1e-9


def test_comm_graph_from_dict_and_infer():
    graphs = markov_forum_result_to_comm_graph_dicts(EXAMPLE_JSON)
    for cg in graphs:
        claim = cg["claim"]
        assert claim.get("id")
        assert claim.get("content")
        assert claim.get("evidence") == []
        assert isinstance(claim.get("comments"), list)

        g = CommGraph.from_dict(cg)
        conf = g.infer_claim_confidence()
        assert 0.0 <= conf <= 5.0
        G = g.to_networkx()
        assert G.number_of_nodes() >= 1
        assert G.number_of_edges() >= 1


def test_trust_skeptic_nested_under_initial():
    graphs = markov_forum_result_to_comm_graph_dicts(EXAMPLE_JSON)
    cg = graphs[1]
    debate = (json.loads(EXAMPLE_JSON.read_text(encoding="utf-8"))["sub_claim_debates"][0])[
        "debate"
    ]
    initial = cg["claim"]["comments"][0]
    assert initial["id"].endswith("_initial")
    assert initial.get("comments"), "Trust/Skeptic 应嵌在 initial 下"
    chain0 = initial["comments"][0]
    state = (debate.get("rounds") or [{}])[0].get("state")
    if state == "S1":
        assert "r1_trust" in chain0["id"]
        assert chain0.get("comments") and "skeptic" in chain0["comments"][0]["id"]
    elif state == "S2":
        assert "r1_skeptic" in chain0["id"]
        assert chain0.get("comments") and "trust" in chain0["comments"][0]["id"]
    assert "r1_leader" in cg["claim"]["comments"][1]["id"]


def test_draw_with_comm_graph():
    """绘制原始声明聚合图并保存为 ``claim_original.png``。"""
    graphs = markov_forum_result_to_comm_graph_dicts(EXAMPLE_JSON)
    GRAPH_OUT_DIR.mkdir(parents=True, exist_ok=True)

    cg = graphs[0]
    assert (cg.get("claim") or {}).get("id") == "claim_original"

    g = CommGraph.from_dict(cg)
    g.infer_claim_confidence()
    conf = g.claim.confidence
    fig = g.draw(
        layout="hierarchy",
        title=f"claim_original  claim conf={conf:.3f}",
    )
    out = GRAPH_OUT_DIR / "claim_original.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out.is_file()
    assert out.stat().st_size > 0


if __name__ == "__main__":
    test_one_entrypoint_converts_and_matches_sub_claims_count()
    test_root_claim_graph_exists_and_uses_sub_claim_weights()
    test_comm_graph_from_dict_and_infer()
    test_trust_skeptic_nested_under_initial()
    test_draw_with_comm_graph()
    print("forum2graph tests OK")
    print(f"  PNG: {GRAPH_OUT_DIR / 'claim_original.png'}")
