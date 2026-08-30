"""统一配置：从 .env.local 加载环境变量，并提供 Agent/ChatModel 工厂。"""

from config.env import load_env, ensure_loaded, get
from config.llm_factory import get_chat_model, create_langchain_agent

__all__ = [
    "load_env",
    "ensure_loaded",
    "get",
    "get_chat_model",
    "create_langchain_agent",
]
