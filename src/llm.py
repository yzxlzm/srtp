from typing import Optional

from langchain_openai import ChatOpenAI

from app_config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)


# 可扩展的模型注册表：前端传入 key，后端按 key 选不同大模型
MODEL_REGISTRY = {
    "deepseek-chat": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "model": "deepseek-chat",
    },
    "deepseek-reasoner": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "model": "deepseek-reasoner",
    },
    "openai-gpt-4o": {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "model": "gpt-4o",
    },
    "openai-gpt-4o-mini": {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "model": "gpt-4o-mini",
    },
}


def get_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    构造 LLM。

    Args:
        model: 模型 key，可选值见 MODEL_REGISTRY；为空则使用默认 DEEPSEEK_MODEL。
        api_key: 可选，覆盖注册表里的 api_key。
        base_url: 可选，覆盖注册表里的 base_url。
    """
    # 如果传了注册表 key，则优先按 key 取配置
    if model and model in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[model]
        api_key = api_key or cfg.get("api_key")
        base_url = base_url or cfg.get("base_url")
        model_name = cfg.get("model")
    else:
        # 回落到默认 DeepSeek 配置（保持旧逻辑兼容）
        model_name = model or DEEPSEEK_MODEL
        api_key = api_key or DEEPSEEK_API_KEY
        base_url = base_url or DEEPSEEK_BASE_URL

    if not api_key:
        raise ValueError("LLM API Key 未设置，请在 .env 中配置对应的密钥")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.2,
    )

