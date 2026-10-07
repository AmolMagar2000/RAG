"""Provider routing: Gemini first, Groq on any failure (429/503/timeout/invalid JSON). Every call is metered."""
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class LLMMeta:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    fallback_reason: Optional[str] = None


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def _cost(provider: str, tin: int, tout: int) -> float:
    pin, pout = (
        (settings.GEMINI_PRICE_IN, settings.GEMINI_PRICE_OUT) if provider == "gemini"
        else (settings.GROQ_PRICE_IN, settings.GROQ_PRICE_OUT)
    )
    return (tin * pin + tout * pout) / 1_000_000


async def complete_structured(
    schema: Type[T], system: str, prompt: str, *, temperature: float = 0.0
) -> Tuple[T, LLMMeta]:
    from app.llm import gemini, groq_client

    providers = []
    if settings.GEMINI_API_KEY:
        providers.append(("gemini", gemini.generate_json))
    if settings.GROQ_API_KEY:
        providers.append(("groq", groq_client.generate_json))
    if not providers:
        raise LLMUnavailable("No LLM API key configured (set GEMINI_API_KEY and/or GROQ_API_KEY).")

    full_prompt = (
        f"{prompt}\n\nReturn ONLY one JSON object (no markdown, no commentary) that validates against "
        f"this JSON Schema:\n{json.dumps(schema.model_json_schema())}"
    )
    errors = []
    for name, fn in providers:
        t0 = time.perf_counter()
        try:
            text, tin, tout, model = await fn(system, full_prompt, temperature)
            data = schema.model_validate(_parse_json(text))
            meta = LLMMeta(name, model, tin, tout, (time.perf_counter() - t0) * 1000,
                           _cost(name, tin, tout), "; ".join(errors) or None)
            return data, meta
        except Exception as exc:  # noqa: BLE001 - any failure -> next provider
            errors.append(f"{name}:{type(exc).__name__}")
            logger.warning("LLM provider %s failed: %s", name, exc)
    raise LLMUnavailable("All LLM providers failed: " + "; ".join(errors))
