# openai_backend.py
import json
import httpx
from pytune_llm.settings import get_openai_key

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
    vision: bool = False
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

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }

    async with httpx.AsyncClient(timeout=20) as client:
        print(json.dumps(payload, indent=2))

        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        json_data = response.json()

        return json_data if vision else json_data["choices"][0]["message"]["content"].strip()
