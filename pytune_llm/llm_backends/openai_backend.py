# openai_backend.py
import json
from typing import Optional
import httpx

from pytune_llm.llm_utils import estimate_tokens
from pytune_llm.settings import get_openai_key
from pytune_llm.task_reporting.reporter import TaskReporter
from pytune_configuration.sync_config_singleton import config, SimpleConfig

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────

config = config or SimpleConfig()

LLM_DEFAULT_MODEL = getattr(config, "LLM_DEFAULT_MODEL", "gpt-5-mini")

LLM_MAX_OUTPUT_TOKENS_TEXT = int(
    getattr(config, "LLM_MAX_OUTPUT_TOKENS_TEXT", "1024")
)
LLM_MAX_OUTPUT_TOKENS_VISION = int(
    getattr(config, "LLM_MAX_OUTPUT_TOKENS_VISION", "2048")
)
LLM_MAX_OUTPUT_TOKENS_REASONING = int(
    getattr(config, "LLM_MAX_OUTPUT_TOKENS_REASONING", "3072")
)

VALID_ROLES = {"system", "user", "assistant"}

# ────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────

def sanitize_messages(messages: list[dict]) -> list[dict]:
    cleaned = []
    for m in messages:
        role = str(m.get("role", "user")).lower().strip()
        content = m.get("content", "")

        if role not in VALID_ROLES:
            role = "user"

        if isinstance(content, (str, list)):
            cleaned.append({"role": role, "content": content})
        else:
            raise ValueError(f"Invalid message content type: {type(content)}")

    return cleaned


def to_responses_input(messages: list[dict]) -> list[dict]:
    """
    Convert chat-like messages to Responses API input format
    """
    out = []
    for m in messages:
        content = m.get("content")

        if isinstance(content, str):
            out.append({
                "role": m.get("role", "user"),
                "content": [
                    {
                        "type": "input_text",   # ✅ FIX ICI
                        "text": content
                    }
                ]
            })
        else:
            # multimodal already structured (images, etc.)
            out.append(m)

    return out


def extract_text_from_response(json_data: dict) -> str:
    """
    Robust extraction of assistant text from Responses API
    """
    for item in json_data.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text", "")
                    if text:
                        return text.strip()

    # Fallback: reasoning-only responses
    if json_data.get("status") == "incomplete":
        raise RuntimeError(
            f"Model stopped early (reason: {json_data.get('incomplete_details')})"
        )

    raise ValueError(f"No assistant text found in response: {json_data}")


# ────────────────────────────────────────────────────────────────
# Main call
# ────────────────────────────────────────────────────────────────

async def call_openai_llm(
    prompt: str | None = None,
    context: dict | None = None,
    messages: list[dict] | None = None,
    model: str = LLM_DEFAULT_MODEL,
    vision: bool = False,
    reporter: Optional[TaskReporter] = None,
) -> str:

    api_key = get_openai_key()
    reasoning_enabled = model.startswith("gpt-5")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if not messages:
        if not prompt:
            raise ValueError("You must provide either 'messages' or 'prompt'")
        messages = [
            {"role": "system", "content": "You are a helpful assistant for PyTune users."},
            {"role": "user", "content": prompt},
        ]

    messages = sanitize_messages(messages)

    # Token budget
    if vision:
        max_output_tokens = LLM_MAX_OUTPUT_TOKENS_VISION
    elif reasoning_enabled:
        max_output_tokens = LLM_MAX_OUTPUT_TOKENS_REASONING
    else:
        max_output_tokens = LLM_MAX_OUTPUT_TOKENS_TEXT

    reporter and await reporter.step(f"📡 Calling OpenAI Responses API ({model})") # type: ignore

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json={
                "model": model,
                "input": to_responses_input(messages),
                "max_output_tokens": max_output_tokens,
                "reasoning": {"effort": "low"} if reasoning_enabled else None,
                "text": {"verbosity": "medium"},
            },
        )

        if response.status_code >= 400:
            print("❌ STATUS:", response.status_code)
            print("❌ BODY:", response.text)

        response.raise_for_status()
        json_data = response.json()

    reporter and await reporter.step("✅ Received OpenAI response") # type: ignore

    return extract_text_from_response(json_data)