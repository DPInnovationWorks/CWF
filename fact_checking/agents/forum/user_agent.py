"""
用户 Agent：模拟与科普专家对话的读者，支持多轮；用于测试或仿真对话。
"""

from typing import Optional, List, Dict, Any

from config import load_env, create_langchain_agent
from config.llm_factory import Provider
from utils.label_process import extract_label


class UserAgent:
    """模拟读者/用户，与专家多轮对话时生成用户侧的发言。"""

    USER_PROMPT = """
        作为社交媒体用户，你正在与一名专家讨论，针对一个段落中的论点发表看法。
        【用户画像】
        你是一名 {age} 岁、受教育程度为 {education}、主要专长是 {expertise}、阅读理解水平 {reading_level}、人格特质 {personality} 的 {career}。
        【写作语气】
        - 口吻自然、接地气，符合你的画像；允许使用少量口语化表达。
        - 观点明确，可以是赞同、质疑、补充经历或提问等。
        - 如果 topic 中存在潜在风险或争议点，可以适当表达担忧或提出反面观点。
        【内容要求】
        - 输出内容一定要紧扣主题，不要偏离主题内容。
        - 若你引用了数据/事实，请保持克制、避免虚构具体来源；可以用“有研究指出/业内常见说法”等中性表述。
        - 给出你对该内容的“立场标签”：支持/中立/质疑（只选其一）
        【输出格式】
        <stance>支持/中立/质疑</stance>
        <reply>你的单条评论，150字以内，不要出现标签外内容</reply>
        <prior>0到1之间的数字，表示你对评论内容的确信度</prior>
    """

    def __init__(
        self,
        profile: Optional[Dict[str, str]] = None,
        provider: Provider = "deepseek",
        model: str = "deepseek-chat",
        max_tokens: int = 300,
    ):
        load_env()
        user_prompt = self.USER_PROMPT.format(
            age=profile.get("age", "Unknown") if profile else "Unknown",
            education=profile.get("education", "Unknown") if profile else "Unknown",
            expertise=profile.get("expertise", "Unknown") if profile else "Unknown",
            reading_level=(
                profile.get("reading_level", "Unknown") if profile else "Unknown"
            ),
            personality=profile.get("personality", "Unknown") if profile else "Unknown",
            career=profile.get("career", "Unknown") if profile else "Unknown",
        )
        self._agent = create_langchain_agent(
            provider,
            tools=[],
            system_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
        )
        self._messages: List[Dict[str, Any]] = []

    def respond(self, topic: str = "", expert_message: Optional[str] = None) -> dict:
        """
        生成用户发言。expert_message 为空或 None 时生成开场；否则根据专家回复生成下一条。
        传入topic默认为 user 的开场白；传入 expert_message 则在 user 消息中带上专家回复，生成针对专家回复的用户评论。
        """
        if topic:
            content = f"请就以下话题以读者身份说一句开场（问题或看法）：{topic}"
            self._messages = [{"role": "user", "content": content}]
        else:
            content = (
                f"专家说：\n{expert_message}\n\n请以读者身份简短回复或追问（1～2 句）。"
            )
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
