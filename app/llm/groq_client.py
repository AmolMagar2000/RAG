"""Groq (gpt-oss-120b) fallback. Reasoning stays server-side; only message.content is used."""
import asyncio
from typing import Tuple

from groq import AsyncGroq

from app.config import get_settings

settings = get_settings()
_client = None


async def generate_json(system: str, prompt: str, temperature: float) -> Tuple[str, int, int, str]:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    resp = await asyncio.wait_for(
        _client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
            reasoning_effort=settings.GROQ_REASONING_EFFORT,
        ),
        timeout=settings.LLM_TIMEOUT_S,
    )
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
        settings.GROQ_MODEL,
    )
