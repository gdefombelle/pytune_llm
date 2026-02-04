# moonshot_backend.py
import base64
import httpx
import time
import asyncio
from typing import Optional, List, Dict, Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from simple_logger import get_logger, SimpleLogger
from pytune_configuration.sync_config_singleton import config, SimpleConfig


# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────

config = config or SimpleConfig()
logger: SimpleLogger = get_logger()

MOONSHOT_API_KEY = getattr(config, "MOONSHOT_API_KEY", None)
if not MOONSHOT_API_KEY:
    raise RuntimeError("MOONSHOT_API_KEY is not configured")

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MOONSHOT_MODEL = "kimi-k2.5"

client = OpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url=MOONSHOT_BASE_URL,
)

VALID_ROLES = {"system", "user", "assistant"}


# ────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────

async def _fetch_image_as_base64(url: str) -> str:
    """Moonshot requires base64 data URLs."""
    async with httpx.AsyncClient(timeout=30) as http:
        resp = await http.get(url)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "image/jpeg")
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return f"data:{mime};base64,{b64}"


async def _image_blocks_from_context(context: dict) -> List[Dict[str, Any]]:
    """
    Convert images to Moonshot-compatible base64 blocks.
    Images are fetched in parallel for performance.
    """
    images = context.get("images", [])
    blocks: List[Dict[str, Any]] = []

    async def _one(img: dict):
        if "bytes" in img:
            b64 = base64.b64encode(img["bytes"]).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime']};base64,{b64}"}
            }

        if "url" in img:
            data_url = await _fetch_image_as_base64(img["url"])
            return {
                "type": "image_url",
                "image_url": {"url": data_url}
            }

        return None

    tasks = [_one(img) for img in images]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, dict):
            blocks.append(r)

    return blocks


def sanitize_messages(messages: list[dict]) -> list[dict]:
    cleaned = []
    for m in messages:
        role = str(m.get("role", "user")).lower().strip()
        if role not in VALID_ROLES:
            role = "user"
        cleaned.append({
            "role": role,
            "content": m.get("content", "")
        })
    return cleaned


# ────────────────────────────────────────────────────────────────
# Main entry point (HOMOTHÉTIQUE OpenAI)
# ────────────────────────────────────────────────────────────────

async def call_moonshot_llm(
    prompt: str | None = None,
    context: dict | None = None,
    messages: list[dict] | None = None,
    model: str | None = None,
    vision: bool = False,
    reporter=None,  # ⚠️ volontairement ignoré
) -> str:
    """
    Moonshot / Kimi 2.5 backend
    Signature compatible avec call_openai_llm
    Logging via SimpleLogger uniquement
    """

    t0 = time.perf_counter()
    context = context or {}
    model = model or DEFAULT_MOONSHOT_MODEL

    logger.info(f"[Moonshot] start model={model} vision={vision}")

    try:
        if not prompt and not messages:
            raise ValueError("You must provide either prompt or messages")

        system_prompt = context.get(
            "system_prompt",
            "You are an expert assistant for PyTune professional users."
        )

        if not messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = sanitize_messages(messages)

        final_messages: List[Dict[str, Any]] = []

        for m in messages:
            if m["role"] == "user" and vision:
                t_img = time.perf_counter()
                image_blocks = await _image_blocks_from_context(context)
                logger.info(
                    f"[Moonshot] images processed in {time.perf_counter() - t_img:.2f}s "
                    f"({len(image_blocks)} images)"
                )

                final_messages.append({
                    "role": "user",
                    "content": [
                        *image_blocks,
                        {"type": "text", "text": m["content"]}
                    ]
                })
            else:
                final_messages.append(m)

        t_llm = time.perf_counter()

        completion = client.chat.completions.create(
            model=model,
            messages=cast(List[ChatCompletionMessageParam], final_messages),
            temperature=1,   # ⚠️ Kimi constraint
            timeout=100,
        )

        logger.info(
            f"[Moonshot] LLM responded in {time.perf_counter() - t_llm:.2f}s"
        )

        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Moonshot returned empty response")

        logger.info(
            f"[Moonshot] total time {time.perf_counter() - t0:.2f}s"
        )

        return content.strip()

    except Exception as e:
        logger.error(
            f"[Moonshot] FAILED after {time.perf_counter() - t0:.2f}s → {e}"
        )
        raise