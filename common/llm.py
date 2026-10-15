"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI


from dotenv import load_dotenv


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    load_dotenv(override=True)
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("API key is not set — please set DEEPSEEK_API_KEY or OPENROUTER_API_KEY in .env")

    model = os.environ.get("LLM_MODEL")
    base_url = os.environ.get("LLM_BASE_URL")

    if os.environ.get("DEEPSEEK_API_KEY") and not base_url:
        base_url = "https://api.deepseek.com"
        if not model:
            model = "deepseek-chat"

    return ChatOpenAI(
        model=model or "openai/gpt-4o-mini",
        base_url=base_url or "https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=temperature,
    )
