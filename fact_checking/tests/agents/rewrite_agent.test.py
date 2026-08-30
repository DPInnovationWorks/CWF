"""
RewriteAgent 测试。
python -B tests/agents/rewrite_agent.test.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_env
from agents.check.rewrite_agent import RewriteAgent


def test_rewrite_agent_rewrite():
    """测试：根据修改建议改写原文，返回非空且为字符串。"""
    load_env()

    agent = RewriteAgent()
    paragraph = "人工智能会改变我们的工作方式。很多人担心失业。"
    suggestion = "将第二句改得更中性，并补充一句积极前景。"

    result = agent.rewrite(paragraph, suggestion)

    print("【改写测试】")
    print("原文:", paragraph)
    print("建议:", suggestion)
    print("结果:", result[:200] + "..." if len(result) > 200 else result)


if __name__ == "__main__":
    test_rewrite_agent_rewrite()
