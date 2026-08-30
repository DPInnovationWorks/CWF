"""
改写 Agent：基于 DeepSeek，根据原段落与修改建议进行科普向改写/润色。
"""

from config import load_env, create_langchain_agent
from config.llm_factory import Provider
from utils.label_process import extract_label


class RewriteAgent:
    """使用 DeepSeek 的改写 Agent，按修改建议输出修改后的完整段落。"""

    REWRITE_PROMPT = """
        你是一名专业的科普编辑，擅长在保持原文风格的前提下进行精准改写。

        任务：阅读用户提供的【原段落】和【修改建议】，对原段落进行合理修改。

        改写要求：
        1. 保留原文中事实正确、表达清晰的内容。
        2. 根据【修改建议】修正存在的问题（如：表达不清、逻辑问题、错误信息等）。
        3. 保持原有的写作风格、语气和面向的读者群体，不要改变整体文风。
        4. 保证段落逻辑通顺、语言自然流畅。
        5. 尽量控制字数，与原段落长度接近（一般不超过原文的 1.2 倍）。
        6. 不要添加原文和建议中没有的新事实或新观点。
        7. 只输出修改后的完整段落，不要输出解释或其他内容。

        输出格式：
        <text>修改后的完整段落</text>
    """

    def __init__(
        self,
        provider: Provider = "deepseek",
        model: str = "deepseek-chat",
    ):
        load_env()
        self._agent = create_langchain_agent(
            provider,
            tools=[],
            system_prompt=self.REWRITE_PROMPT,
            model=model,
        )

    def rewrite(self, text: str, suggestion: str) -> str:
        """
        根据修改建议改写原文，返回修改后的完整段落。
        """
        user_input = f"""【原段落】{text}\n\n【修改建议】\n{suggestion}"""
        r = self._agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        messages = r.get("messages", [])
        content = next(
            (
                getattr(m, "content", "")
                for m in reversed(messages)
                if getattr(m, "type", "") == "ai"
            ),
            "",
        )
        text = extract_label("text", content)
        return text
