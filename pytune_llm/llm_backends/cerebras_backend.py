# cerebras_backend.py

from typing import Optional, List, Dict, Any, cast
import os
import asyncio

from cerebras.cloud.sdk import Cerebras

from pytune_llm.task_reporting.reporter import TaskReporter
from pytune_configuration.sync_config_singleton import config, SimpleConfig


# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────

config = config or SimpleConfig()

CEREBRAS_DEFAULT_MODEL = getattr(
    config,
    "CEREBRAS_DEFAULT_MODEL",
    "gpt-oss-120b"
)

CEREBRAS_MAX_TOKENS = int(
    getattr(config, "CEREBRAS_MAX_TOKENS", "1024")
)

CEREBRAS_TEMPERATURE = float(
    getattr(config,
    "CEREBRAS_TEMPERATURE",
    "0.0")
)


# ────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────

def format_messages(messages: List[Dict]) -> List[Dict]:
    """
    Ensure Cerebras-compatible chat messages.
    Cerebras follows OpenAI-style chat format.
    """
    formatted: List[Dict[str, str]] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if content:
            formatted.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return formatted


# ────────────────────────────────────────────────────────────────
# Main call
# ────────────────────────────────────────────────────────────────

async def call_cerebras_llm(
    prompt: str | None = None,
    context: dict | None = None,  # reserved for future use
    messages: list[dict] | None = None,
    model: str | None = None, 
    json_mode: bool = False,  # kept for API symmetry (not used)
    reporter: Optional[TaskReporter] = None,
) -> str:
    """
    Cerebras connector optimized for:
    - very low latency
    - structured factual tasks (model resolver)
    - deterministic output
    """
    if model is None:
        model = CEREBRAS_DEFAULT_MODEL

    api_key = os.environ.get("CEREBRAS_API_KEY") or getattr(
        config, "CEREBRAS_API_KEY", None
    )

    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not configured")

    if not messages:
        if not prompt:
            raise ValueError("You must provide either 'messages' or 'prompt'")
        messages = [{"role": "user", "content": prompt}]

    messages = format_messages(messages)

    if reporter is not None:
        await reporter.step(f"⚡ Calling Cerebras API ({model})")

    # Cerebras SDK is synchronous → run in thread for async safety
    def _run():
        client = Cerebras(api_key=api_key)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=CEREBRAS_MAX_TOKENS,
            temperature=CEREBRAS_TEMPERATURE,
        )

    try:
        completion = await asyncio.to_thread(_run)
    except Exception as e:
        if reporter is not None:
            await reporter.step("⚠️ Cerebras call failed")
        raise e

    if reporter is not None:
        await reporter.step("✅ Received Cerebras response")

    # Defensive extraction (SDK is weakly typed)
    try:
        data = cast(Any, completion)

        content = data.choices[0].message.content

        if not isinstance(content, str):
            raise ValueError("Cerebras response content is not a string")

        return content.strip()

    except Exception as e:
        raise ValueError(
            f"Invalid Cerebras response format: {completion}"
        ) from e