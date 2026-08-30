"""
ExtractClaimAgent 测试：声明 → 一条或少量子声明 + 权重（和为 1）。
python -B tests/agents/extract_claim_agent.test.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_env
from agents.claim.extract_claim_agent import ExtractClaimAgent


def test_extract_claim_agent():
    load_env()

    agent = ExtractClaimAgent(max_tokens=800)

    complex_claim = (
        "长期熬夜会扰乱昼夜节律，进而影响代谢与免疫，因此可能增加肥胖与心血管疾病风险，"
        "而规律运动与充足睡眠有助于降低这些风险。"
    )

    out = agent.extract(complex_claim)

    print("initial claim:")
    print(complex_claim)
    print()
    print("extracted claims + weights:")
    for i, it in enumerate(out.get("items") or [], 1):
        print(f"  {i}. [{it.get('weight', 0):.2f}] {it.get('claim', '')}")

    claims = out.get("claims") or []
    weights = out.get("weights") or []
    assert isinstance(claims, list)
    assert len(claims) >= 1, "应至少有一条简单声明"
    assert len(weights) == len(claims)
    assert abs(sum(weights) - 1.0) < 1e-5, "权重和应为 1"


if __name__ == "__main__":
    test_extract_claim_agent()
