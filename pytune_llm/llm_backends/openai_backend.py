# openai_backend.py
import json
from typing import Optional
import httpx
from pytune_llm.llm_utils import estimate_tokens
from pytune_llm.settings import get_openai_key
from pytune_llm.task_reporting.reporter import TaskReporter

VALID_ROLES = {"system", "user", "assistant"}
def sanitize_messages(messages: list[dict]) -> list[dict]:
    result = []
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = m.get("content", "")

        # fallback si role non conforme
        if role not in VALID_ROLES:
            role = "user"

        # ✅ NE PAS transformer les listes multimodales en string
        if isinstance(content, str) or isinstance(content, list):
            result.append({"role": role, "content": content})
        else:
            raise ValueError(f"Invalid message content type: {type(content)}")
    return result

async def call_openai_llm(
    prompt: str = None,
    context: dict = None,
    messages: list[dict] = None,
    model: str = "gpt-3.5-turbo",
    vision: bool = False,
    reporter: Optional[TaskReporter]= None
) -> dict | str:
    api_key = get_openai_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    if not messages:
        if not prompt:
            raise ValueError("You must provide either 'messages' or 'prompt'")
        messages = [
            {"role": "system", "content": "You are a helpful assistant for PyTune users."},
            {"role": "user", "content": prompt}
        ]

    # Clean message roles
    messages = sanitize_messages(messages)
    # ✅ Token-aware adjustment
    if vision:
        max_tokens = 4096 * 2  # GPT-4o accepte jusqu’à ~128k selon version
    else:
        total_prompt_tokens = estimate_tokens(messages, model)
        max_tokens = max(256, 4096 - total_prompt_tokens)
        print(f"📏 Prompt tokens: {total_prompt_tokens} | max_tokens set to: {max_tokens}")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    

    reporter and await reporter.step(f"📡 Calling OpenAI API ({model})")
    async with httpx.AsyncClient(timeout=20) as client:
        print(json.dumps(payload, indent=2))

        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        json_data = response.json()
        reporter and await reporter.step("✅ Received OpenAI response")
        return json_data if vision else json_data["choices"][0]["message"]["content"].strip()
