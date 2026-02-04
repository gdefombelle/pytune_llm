# pytune_llm/llm_client.py

from typing import Optional, List, Dict, Any
from string import Template
import os

from openai import AsyncOpenAI, OpenAI

from pytune_llm.settings import get_llm_backend
from pytune_llm.task_reporting.reporter import TaskReporter
from pytune_configuration.sync_config_singleton import config, SimpleConfig

from pytune_llm.llm_backends.moonshot_backend import call_moonshot_llm
from pytune_llm.llm_backends.gemini_backend import call_gemini_llm

# ---------------------------------------------------------------------
# Config & Clients
# ---------------------------------------------------------------------

config = config or SimpleConfig()

# OpenAI (Responses API)
openai_client = AsyncOpenAI(
    api_key=config.OPEN_AI_PYTUNE_API_KEY
)

# Moonshot (Chat Completions compatible OpenAI SDK)
moonshot_client = OpenAI(
    api_key=getattr(config, "MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
)

LLM_TEXT_MODEL = getattr(config, "LLM_TEXT_MODEL", "gpt-4.1-mini")
LLM_VISION_MODEL = getattr(config, "LLM_VISION_MODEL", "gpt-4o")
MOONSHOT_VISION_MODEL = "kimi-k2.5"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_text(response) -> str:
    """
    Robust text extraction for OpenAI Responses API.
    """
    if getattr(response, "output_text", None):
        return response.output_text.strip()

    texts: List[str] = []
    for item in response.output or []:
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))

    return "\n".join(t for t in texts if t).strip()


# ---------------------------------------------------------------------
# TEXT LLM (unchanged behaviour)
# ---------------------------------------------------------------------

async def ask_llm(
    prompt_template: str,
    user_input: str,
    context: dict,
    reporter: Optional[TaskReporter] = None,
    *,
    model: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:

    backend = backend or get_llm_backend()
    model = model or LLM_TEXT_MODEL

    if backend != "openai":
        raise NotImplementedError(f"LLM backend '{backend}' not supported for TEXT")

    full_prompt = Template(prompt_template).safe_substitute(
        user_input=user_input,
        **context
    )

    reporter and await reporter.step(f"🤖 Asking LLM ({model})")  # type: ignore

    response = await openai_client.responses.create(
        model=model, # type: ignore
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": "You are PyTune AI assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": full_prompt}]
            }
        ]
    )

    text = extract_text(response)
    if not text:
        raise RuntimeError("LLM returned empty response")

    return text


# ---------------------------------------------------------------------
# VISION ROUTER
# ---------------------------------------------------------------------

async def call_llm_vision(
    prompt: str,
    image_urls: List[str],
    voice_url: str | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    reporter: Optional[TaskReporter] = None
) -> Dict[str, Any]:
    """
    Unified Vision LLM entry point.
    ⚠️ Signature MUST remain stable.
    """

    metadata = metadata or {}
    backend = metadata.get("llm_backend") or get_llm_backend()

    if backend == "openai":
        return await _call_openai_vision(
            prompt=prompt,
            image_urls=image_urls,
            metadata=metadata,
            reporter=reporter,
        )

    if backend == "moonshot":
        # 🔥 DELEGATION PROPRE AU BACKEND OFFICIEL
        raw_text = await call_moonshot_llm(
            prompt=prompt,
            context={
                "images": metadata.get("images", []),
                "system_prompt": (
                    "You are an expert piano identifier. "
                    "Be precise, factual, and conservative. "
                    "Do not guess missing information."
                ),
            },
            model=metadata.get("llm_model"),
            vision=True,
            reporter=reporter,
        )

        return {
            "raw_text": raw_text,
            "model": metadata.get("llm_model") or "kimi-k2.5",
            "backend": "moonshot",
        }
    
    if backend == "gemini":
        # 1. Construction dynamique de la liste de contenu (Texte + Images)
        content_parts = [{"type": "text", "text": prompt}]
        
        for url in image_urls:
            content_parts.append({"type": "image_url", "url": url})

        # 2. C. Le bloc Audio (NOUVEAU)
        if voice_url:
            if reporter: await reporter.step("🎤 Analyzing voice note context")
            content_parts.append({
                "type": "audio_url",
                "url": voice_url
            })

        # 3. Appel au backend avec le contenu multimodal
        raw_text = await call_gemini_llm(
            messages=[
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            model=metadata.get("llm_model") or "gemini-2.0-flash",
            json_mode=True,
            use_tools=True, # ACTIVÉ pour l'Agent de Datation (le code Python qui cherche l'année)
            reporter=reporter,
        )

        return {
            "raw_text": raw_text,
            "model": metadata.get("llm_model") or "gemini-2.0-flash",
            "backend": "gemini",
        }

    raise NotImplementedError(f"Vision backend '{backend}' not supported")


# ---------------------------------------------------------------------
# OPENAI VISION (Responses API)
# ---------------------------------------------------------------------

async def _call_openai_vision(
    prompt: str,
    image_urls: List[str],
    metadata: Dict[str, Any],
    reporter: Optional[TaskReporter],
) -> Dict[str, Any]:

    model = metadata.get("llm_model") or LLM_VISION_MODEL

    reporter and await reporter.step("👁️ OpenAI Vision analysis")  # type: ignore

    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        *[
            {"type": "input_image", "image_url": url}
            for url in image_urls
        ]
    ]

    response = await openai_client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "You are an expert piano identifier. "
                        "Be precise, factual, and conservative. "
                        "Do not guess missing information."
                    )
                }]
            },
            {
                "role": "user",
                "content": user_content
            } # type: ignore
        ]
    )

    output_text = extract_text(response)
    if not output_text:
        raise RuntimeError("OpenAI Vision returned empty response")

    return {
        "raw_text": output_text,
        "model": model,
        "backend": "openai",
    }