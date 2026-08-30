"""
用户 Agent 与专家 Agent 的多轮论坛对话编排。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import load_env
from utils.profile_select import select_profile

DEFAULT_TOPIC = (
    "越来越多研究提示，长期的低度慢性炎症可能与多种代谢性疾病有关，因此通过调节肠道微生态、"
    "优化线粒体功能和降低系统性炎症水平，可能有助于从根本上改善人体的代谢健康状态。"
)


def run_user_expert_forum(
    topic: str = DEFAULT_TOPIC,
    profile: Optional[Dict[str, Any]] = None,
    profiles_name: str = "teenager_profile.json",
    num_rounds: int = 3,
    *,
    user_max_tokens: int = 200,
    expert_max_tokens: int = 500,
) -> None:
    """
    用户与专家进行多轮交互，并打印每轮双方回复。

    :param topic: 讨论话题（首轮用户开场会围绕该话题）。
    :param profile: 读者画像；为 None 时从 profiles_name 随机抽一条。
    :param profiles_name: 画像 JSON 文件名（位于 utils/config/）。
    :param num_rounds: 对话轮数（每轮：用户一条 + 专家一条）。
    :param user_max_tokens: UserAgent max_tokens。
    :param expert_max_tokens: ExpertAgent max_tokens。
    """
    from agents.forum.expert_agent import ExpertAgent
    from agents.forum.user_agent import UserAgent

    load_env()
    if profile is None:
        selected = select_profile(profiles_name, num_profiles=1)
        profile = {**selected[0], "topic": topic} if selected else {"topic": topic}
    else:
        profile = {**profile, "topic": profile.get("topic", topic)}

    user_agent = UserAgent(profile=profile, max_tokens=user_max_tokens)
    expert_agent = ExpertAgent(max_tokens=expert_max_tokens)

    print("话题:", topic)
    print("读者画像:", profile)
    print("轮数:", num_rounds)
    print("-" * 60)

    expert_reply_text = ""

    for round_no in range(1, num_rounds + 1):
        if round_no == 1:
            user_out = user_agent.respond(topic=topic)
        else:
            user_out = user_agent.respond(expert_message=expert_reply_text)

        user_reply = user_out.get("reply", "") if isinstance(user_out, dict) else str(user_out)
        user_stance = user_out.get("stance", "") if isinstance(user_out, dict) else ""
        print(f"[第{round_no}轮] 用户 [{user_stance}]: {user_reply}")

        expert_out = expert_agent.respond(user_reply, profile=profile)
        expert_reply_text = (
            expert_out.get("reply", "") if isinstance(expert_out, dict) else str(expert_out)
        )
        expert_stance = expert_out.get("stance", "") if isinstance(expert_out, dict) else ""
        print(f"[第{round_no}轮] 专家 [{expert_stance}]: {expert_reply_text}")
        print("-" * 60)

    print(f"{num_rounds} 轮对话结束。")
