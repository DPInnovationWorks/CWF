"""
测试挂载了「联网搜索」与「知识库」的 DeepSeek Agent。
python -B tests/agents/agent.test.py
"""

import sys
from pathlib import Path

# 将项目根加入 path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_env, create_langchain_agent

KNOWLEDGE_PATH = ROOT / "data" / "knowledge"
PERSIST_DIR = ROOT / "data" / "knowledge_index"

if __name__ == "__main__":
    load_env()
    ag = create_langchain_agent(
        "deepseek",
        tools=[],
        system_prompt=(
            "你是助手。可联网搜索、从知识库检索。用中文简短回答。"
            "当用户问今天日期、当前时间、实时信息或你不确定的内容时，必须先调用联网搜索获取后再回答。"
        ),
        model="deepseek-chat",
        max_tokens=500,
        enable_web_search=True,
        enable_knowledge_base=True,
    )
    print("【联网搜索测试】提问: 今天的日期是多少？")
    r = ag.invoke({"messages": [{"role": "user", "content": "今天的日期是多少？"}]})
    for msg in reversed(r.get("messages", [])):
        if getattr(msg, "type", "") == "ai":
            print("助手:", getattr(msg, "content", ""))
            break
    print("\n【知识库测试】提问: 知识库里有什么说明？请简短说一句。")
    r2 = ag.invoke(
        {
            "messages": [
                {"role": "user", "content": "知识库里有什么说明？请简短说一句。"}
            ]
        }
    )
    for msg in reversed(r2.get("messages", [])):
        if getattr(msg, "type", "") == "ai":
            print("助手:", getattr(msg, "content", ""))
            break
