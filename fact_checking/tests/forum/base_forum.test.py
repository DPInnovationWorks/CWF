"""
用户-专家多轮辩论测试
python -B tests/forum/base_forum.test.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_env
from forum.base_forum import DEFAULT_TOPIC, run_user_expert_forum


def run_base_forum_test():
    load_env()

    num_rounds = 3
    topic = DEFAULT_TOPIC
    profiles_name = "teenager_profile.json"

    run_user_expert_forum(
        topic=topic,
        profiles_name=profiles_name,
        num_rounds=num_rounds,
    )


if __name__ == "__main__":
    run_base_forum_test()
