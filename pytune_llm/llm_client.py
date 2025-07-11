# pytune_llm/llm_client.py

import openai
from string import Template
from typing import Optional

from pytune_llm.llm_backends.openai_backend import call_openai_llm
from pytune_llm.settings import get_llm_backend, get_supported_llm_models
from pytune_configuration.sync_config_singleton import config, SimpleConfig

config = config or SimpleConfig()
openai.api_key = config.OPEN_AI_PYTUNE_API_KEY


async def ask_llm(prompt_template: str, user_input: str, context: dict) -> str:
    """
    Envoie une requête au LLM avec un prompt enrichi du contexte utilisateur.
    Le prompt_template provient de la policy YAML (ex: policy.prompt_template).
    """
    full_prompt = Template(prompt_template).safe_substitute(
        user_input=user_input,
        **context
    )

    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are PyTune AI assistant."},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content.strip()


async def call_llm_vision(prompt: str, image_urls: list[str], metadata: Optional[dict] = None) -> dict:
    metadata = metadata or {}

    backend = metadata.get("llm_backend") or get_llm_backend()
    model = metadata.get("llm_model") or "gpt-4o"

    if backend != "openai":
        raise NotImplementedError("Only OpenAI backend supports vision for now.")

    supported_models = get_supported_llm_models().get("openai", set())
    if model not in supported_models:
        raise ValueError(f"Model {model} not supported for OpenAI backend.")

    messages = [
        {
            "role": "system",
            "content": "You are an expert in identifying pianos from images. Be precise and confident."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": url}  # ✅ CORRECTION ICI
                    }
                    for url in image_urls
                ]
            ]
        }
    ]

    return await call_openai_llm(messages=messages, model=model, vision=True)
