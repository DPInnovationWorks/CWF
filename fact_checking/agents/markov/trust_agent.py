"""
Trust Agent：在 Markov 辩论中尽量信任前序观点并扩展论证。
"""

import json
import re
from typing import Any, Dict, List

from config import load_env, create_langchain_agent
from config.llm_factory import Provider
from utils.content_risk import invoke_markov_agent_with_fallbacks
from utils.get_evidence import attach_evidence_fields, slim_opinion_for_prompt
from utils.label_process import extract_label


class TrustAgent:
    """信任者 Agent：倾向接受前序观点，并在证据基础上补充与强化。"""

    TRUST_PROMPT = """
        你是三方辩论中的 *Trust（信任者）*。

        你会收到前序智能体的观点。请结合 claim[text]，必要时自行调用检索工具获取证据，
        在尽量信任前序观点的前提下，指出其合理部分并继续补充分析，形成你的判断。

        要求：
        1. 不要复述前序观点，要在其基础上推进论证。
        2. 结论必须由证据支撑。
        3. 输出简洁、明确。

        引用证据（必须）：
        - 若本轮你调用了检索工具，按调用先后给每次检索编号：第1次为批次1，第2次为批次2，以此类推。
        - 条号对应该次工具返回正文里的 [n] 编号。
        - 在输出末尾增加一行：<cited>批次-条号</cited>，多条用英文逗号分隔，例如 <cited>1-1,1-3,2-2</cited>。
        - 系统会将每条引用映射为全局唯一 id（含辩论轮次与当前角色，如 r1-trust-1-1）；你只需按上规则写批次-条号。
        - 若本轮完全未调用检索，输出 <cited>无</cited>。

        stance 取值限制：{1：强支持, 0.2：中立（弱支持）, 0：中立, -0.2：中立（弱反对）, -1：强反对}

        你必须仅按以下标签格式输出，不要输出标签以外的任何内容：
        <opinion>你的信任扩展意见</opinion>
        <stance>你对前序观点的立场：仅可为 1 / 0.2 / 0 / -0.2 / -1</stance>
        <cited>如 1-1,2-2 或 无</cited>
    """

    def __init__(
        self,
        provider: Provider = "deepseek",
        model: str = "deepseek-chat",
        max_tokens: int = 1200,
        max_search_counts: int | None = 2,
    ):
        load_env()
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._agent = create_langchain_agent(
            provider,
            tools=[],
            system_prompt=self.TRUST_PROMPT,
            model=model,
            max_tokens=max_tokens,
            enable_web_search=True,
            enable_knowledge_base=False,
            max_search_counts=max_search_counts,
        )
        self._messages: List[Dict[str, Any]] = []

    @staticmethod
    def _extract_output(text: str) -> Dict[str, Any]:
        if not text:
            return {
                "opinion": "",
                "stance": "0.2",
                "factuality": False,
            }

        opinion = extract_label("opinion", text, default="").strip()
        stance_raw = extract_label("stance", text, default="0.2").strip()

        # 回退兼容：若模型输出 JSON，则解析兜底
        if not opinion:
            content = text.strip()
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S | re.I)
            if fenced:
                content = fenced.group(1).strip()
            if not content.startswith("{"):
                obj = re.search(r"\{.*\}", content, re.S)
                if obj:
                    content = obj.group(0).strip()
            try:
                parsed = json.loads(content)
                opinion = str(parsed.get("opinion", "")).strip()
                stance_raw = str(parsed.get("stance", "0.2")).strip()
            except Exception:
                opinion = text.strip()

        s = str(stance_raw).strip().lower()
        if s in ("1", "1.0", "支持", "strong_support", "support", "true"):
            stance = "1"
            factuality = True
        elif s in ("0.2", "+0.2", "弱支持", "weak_support"):
            stance = "0.2"
            factuality = True
        elif s in ("-0.2", "弱反对", "weak_oppose"):
            stance = "-0.2"
            factuality = False
        elif s in ("-1", "-1.0", "反对", "strong_oppose", "oppose", "false"):
            stance = "-1"
            factuality = False
        elif s in ("0", "0.0", "中立", "neutral"):
            stance = "0"
            factuality = False
        else:
            # 非法值回退为弱支持
            stance = "0.2"
            factuality = True
        return {
            "opinion": opinion,
            "stance": stance,
            "factuality": factuality,
        }

    def respond(
        self,
        claim: str,
        previous_opinion: Dict[str, Any],
        *,
        debate_round: int = 1,
    ) -> Dict[str, Any]:
        """
        输入 claim/前序观点，输出 Trust 的结构化判断。
        """
        previous_text = json.dumps(
            slim_opinion_for_prompt(previous_opinion), ensure_ascii=False, indent=2
        )

        content = (
            f"[text]\n{claim}\n\n"
            f"[previous opinion]\n{previous_text}\n\n"
            "你可以自行检索相关信息。请按约定的标签格式返回你的信任扩展判断。"
        )

        self._messages.append({"role": "user", "content": content})
        messages, raw, fallback_used, content_risk_used = (
            invoke_markov_agent_with_fallbacks(
                self._agent,
                self._messages,
                role="trust",
                system_prompt=self.TRUST_PROMPT,
                provider=self._provider,
                model=self._model,
                max_tokens=self._max_tokens,
            )
        )
        self._messages.append({"role": "assistant", "content": raw})

        parsed = self._extract_output(raw)
        parsed["raw"] = raw
        parsed["search_limit_fallback_used"] = fallback_used
        parsed["content_risk_fallback_used"] = content_risk_used
        attach_evidence_fields(
            parsed, messages, debate_round=debate_round, agent_slug="trust"
        )
        return parsed
