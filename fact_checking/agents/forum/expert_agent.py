"""
专家 Agent：基于 DeepSeek，支持与用户多轮对话，带对话历史。
支持每次回复时传入不同的用户画像（读者画像）。
"""

from typing import Optional, List, Dict, Any

from config import load_env, create_langchain_agent
from config.llm_factory import Provider
from utils.label_process import extract_label


class ExpertAgent:
    """使用 DeepSeek 的专家 Agent，支持多轮对话；每次回复可指定本轮的读者画像。"""

    DEBATE_PROMPT = """
        你是一名该话题领域的「科普专家」，需要对用户的发言进行专业回应。
        【任务目标】
        - 对用户观点进行专业讨论
        - 重点在于：澄清事实、指出可能误解、提示潜在风险
        - 保持科普沟通的理性与友好
        【读者适配】
        用户可能提供【本轮读者画像】，请根据读者背景调整：
        - 语气
        - 解释深度
        - 术语使用程度
        【回复原则】
        1. 只基于常识性科学知识或广泛共识进行回应
        2. 不编造具体论文、数据或机构结论
        3. 避免情绪化、攻击性表达
        4. 保持回应简洁清晰
        【输出格式】
        <stance>支持|中立|质疑</stance>
        <reply>专家回应内容（不超过160字）</reply>
        <prior>0~1之间的小数，表示你对本次回应正确性的主观确信度：0.9+ = 科学共识 / 0.7~0.9 = 较可靠常识 / 0.5~0.7 = 可能存在争议</prior>
        【重要规则】
        - stance 表示你对「用户论点」的立场，而不是对文章主题的态度
        - 仅输出上述XML结构，不要添加任何解释或额外文本
    """

    def __init__(
        self,
        provider: Provider = "deepseek",
        model: str = "deepseek-chat",
        max_tokens: int = 3000,
        max_search_counts: Optional[int] = 2,
    ):
        load_env()
        self._agent = create_langchain_agent(
            provider,
            tools=[],
            system_prompt=self.DEBATE_PROMPT,
            model=model,
            max_tokens=max_tokens,
            enable_web_search=True,
            enable_knowledge_base=False,
            max_search_counts=max_search_counts,
        )
        self._messages: List[Dict[str, Any]] = []

    def respond(
        self,
        user_input: str,
        profile: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        多轮对话：传入用户输入，返回专家回复内容。
        profile：本轮的读者/用户画像。传则在本轮用户消息中带上画像，便于模型调整语气与深度。
        """

        def _format_profile(profile: Dict[str, str]) -> str:
            """将用户画像格式化为一段说明文字。"""
            return f"""
                读者画像：年龄 {profile.get('age', 'Unknown')}，教育程度 {profile.get('education', 'Unknown')}，专长 {profile.get('expertise', 'Unknown')}，
                阅读理解水平 {profile.get('reading_level', 'Unknown')}，人格特质 {profile.get('personality', 'Unknown')}，职业 {profile.get('career', 'Unknown')}
            """

        if profile:
            header = (
                "【本轮读者画像】"
                + _format_profile(profile)
                + "\n\n【需要回应的评论】\n"
            )
            content = header + user_input
        else:
            content = user_input
        self._messages.append({"role": "user", "content": content})
        r = self._agent.invoke({"messages": self._messages})
        messages = r.get("messages", [])
        raw = next(
            (
                getattr(m, "content", "")
                for m in reversed(messages)
                if getattr(m, "type", "") == "ai"
            ),
            "",
        )
        self._messages.append({"role": "assistant", "content": raw})
        return {
            "stance": extract_label("stance", raw),
            "reply": extract_label("reply", raw),
            "prior": extract_label("prior", raw),
        }
