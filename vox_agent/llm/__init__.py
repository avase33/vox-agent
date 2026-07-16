"""LLM provider registry / factory."""

from __future__ import annotations

from ..config import Settings
from .base import BaseLLM, LLMEvent
from .mock import MockLLM


def build_llm(settings: Settings) -> BaseLLM:
    provider = settings.llm_provider.lower()
    if provider in ("mock", "offline", ""):
        return MockLLM()
    if provider == "groq":
        from groq import AsyncGroq  # type: ignore

        from .openai_compat import OpenAICompatLLM

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY required for the Groq LLM adapter")
        return OpenAICompatLLM(AsyncGroq(api_key=settings.groq_api_key), settings.llm_model, "groq")
    if provider == "openai":
        from openai import AsyncOpenAI  # type: ignore

        from .openai_compat import OpenAICompatLLM

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY required for the OpenAI LLM adapter")
        model = settings.llm_model if settings.llm_model.startswith("gpt") else "gpt-4o-mini"
        return OpenAICompatLLM(AsyncOpenAI(api_key=settings.openai_api_key), model, "openai")
    raise ValueError(f"unknown LLM provider: {settings.llm_provider!r}")


__all__ = ["BaseLLM", "LLMEvent", "MockLLM", "build_llm"]
