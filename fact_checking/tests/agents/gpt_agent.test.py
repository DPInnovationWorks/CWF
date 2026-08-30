"""
最小 GPT Agent 测试。

运行方式：
python -B tests/agents/gpt_agent.test.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_env, create_langchain_agent, get


if __name__ == "__main__":
    load_env()

    model = get("OPENAI_MODEL", "gpt-4o-mini")
    agent = create_langchain_agent(
        "gpt",
        tools=[],
        system_prompt="你是一个简洁的中文助手。请直接回答，不要使用工具。",
        model=model,
        max_tokens=300,
    )

    print(f"当前模型: {model}")
    print("提问: 请用一句中文介绍你自己。")
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "请用一句中文介绍你自己。"}
            ]
        }
    )

    for msg in reversed(result.get("messages", [])):
        if getattr(msg, "type", "") == "ai":
            print("助手:", getattr(msg, "content", ""))
            break
