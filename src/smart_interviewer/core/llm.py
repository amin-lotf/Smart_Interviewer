from __future__ import annotations

from langchain_openai import ChatOpenAI
from smart_interviewer.settings import settings

LLM = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    api_key=settings.OPENAI_API_KEY,
    seed=settings.RANDOM_SEED
)
