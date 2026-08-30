"""
统一环境配置：从 .env.local 加载环境变量。
"""

import os
from pathlib import Path

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_LOCAL = _PROJECT_ROOT / ".env.local"
_loaded = False


def load_env(env_path: str | Path | None = None) -> bool:
    """加载 .env.local"""
    global _loaded
    path = Path(env_path) if env_path else _ENV_LOCAL
    if not path.exists():
        _loaded = True
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    except ImportError:
        pass

    if os.getenv("DS_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = os.environ["DS_KEY"]
    if os.getenv("QWEN_KEY") and not os.getenv("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = os.environ["QWEN_KEY"]
    if os.getenv("GPT_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["GPT_KEY"]
    
    _loaded = True
    return True


def ensure_loaded() -> None:
    if not _loaded:
        load_env()


def get(key: str, default: str = "") -> str:
    """获取环境变量（确保已加载 .env.local）"""
    ensure_loaded()
    return os.getenv(key, default)
