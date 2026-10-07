"""Gemini client: JSON generation + shared client for embeddings."""
import asyncio
from typing import Tuple

from google import genai
from google.genai import types

from app.config import get_settings

settings = get_settings()
_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


async def generate_json(system: str, prompt: str, temperature: float) -> Tuple[str, int, int, str]:
    resp = await asyncio.wait_for(
        get_client().aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=temperature, response_mime_type="application/json",
            ),
        ),
        timeout=settings.LLM_TIMEOUT_S,
    )
    usage = resp.usage_metadata
    return (
        resp.text or "",
        getattr(usage, "prompt_token_count", 0) or 0,
        getattr(usage, "candidates_token_count", 0) or 0,
        settings.GEMINI_MODEL,
    )
