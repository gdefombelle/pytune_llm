# gemini_backend.py
import json
import base64
import asyncio
import random
import httpx
import requests
import mimetypes 
from typing import Optional, Tuple, Union
from httpx import HTTPStatusError

from pytune_llm.task_reporting.reporter import TaskReporter
from pytune_configuration.sync_config_singleton import config, SimpleConfig

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────

config = config or SimpleConfig()

# ON UTILISE TON NOUVEAU MODÈLE "RICHE" (Rapide & Multimodal)
# Si tu as accès à la 2.5 via ta liste : "gemini-2.5-flash"
# Sinon la valeur sûre actuelle : "gemini-2.0-flash"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash-001" 

GEMINI_MAX_OUTPUT_TOKENS = int(
    getattr(config, "GEMINI_MAX_OUTPUT_TOKENS", "1024")
)

# ────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────

def encode_image(image_path_or_bytes: Union[str, bytes]) -> str:
    """Encode une image OU un fichier audio en Base64."""
    if isinstance(image_path_or_bytes, str):
        if image_path_or_bytes.startswith("http"):
            resp = requests.get(image_path_or_bytes, timeout=10)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("utf-8")
        else:
            with open(image_path_or_bytes, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    return base64.b64encode(image_path_or_bytes).decode("utf-8")

def format_gemini_messages(messages: list[dict]) -> Tuple[list[dict], dict | None]:
    """Convertit les messages (Texte, Image, Audio) pour l'API Gemini."""
    contents = []
    system_instruction = None

    for m in messages:
        role = m.get("role", "user").lower()
        content = m.get("content", "")

        if role == "system":
            system_instruction = {"parts": [{"text": content}]}
            continue 

        gemini_role = "model" if role == "assistant" else "user"
        parts = []

        if isinstance(content, str):
            parts.append({"text": content})
        
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict):
                    c_type = item.get("type")

                    if c_type == "text":
                        parts.append({"text": item["text"]})
                    
                    # GESTION IMAGE
                    elif c_type == "image_url":
                        source = item.get("url") or item.get("source")
                        if source:
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg", 
                                    "data": encode_image(source)
                                }
                            })

                    # GESTION AUDIO (Ta nouveauté)
                    elif c_type == "audio_url":
                        source = item.get("url") or item.get("source")
                        if source:
                            mime_type, _ = mimetypes.guess_type(source)
                            if not mime_type or not mime_type.startswith("audio"):
                                mime_type = "audio/mp3" # Fallback
                            
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encode_image(source)
                                }
                            })
        
        contents.append({
            "role": gemini_role,
            "parts": parts
        })

    return contents, system_instruction


def extract_text_from_gemini_response(json_data: dict) -> str:
    if json_data.get("promptFeedback", {}).get("blockReason"):
        raise ValueError(f"Blocked by safety filters: {json_data['promptFeedback']}")

    candidates = json_data.get("candidates", [])
    if not candidates:
        error_msg = json.dumps(json_data, indent=2)
        print("DEBUG - Gemini Error Response:", error_msg)
        raise ValueError(f"No candidates returned. Check logs.")

    candidate = candidates[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [])

    if parts and "text" in parts[0]:
        return parts[0]["text"].strip()
    
    return ""
# ────────────────────────────────────────────────────────────────
# Main call (VERSION FINALE & FLEXIBLE)
# ────────────────────────────────────────────────────────────────
async def call_gemini_llm(
    prompt: str | None = None,
    context: dict | None = None,
    messages: list[dict] | None = None,
    model: str | None = None, 
    max_tokens: int | None = None, # <--- 1. NOUVEL ARGUMENT ICI
    json_mode: bool = False,
    use_tools: bool = False,
    reporter: Optional[TaskReporter] = None,
) -> str:
    if model is None:
        model = GEMINI_DEFAULT_MODEL
        
    api_key = getattr(config, "GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    if not messages:
        if not prompt:
            raise ValueError("You must provide either 'messages' or 'prompt'")
        messages = [{"role": "user", "content": prompt}]

    # Préparation du payload
    contents, system_instruction = format_gemini_messages(messages)
    
    # 2. LOGIQUE INTELLIGENTE POUR LA LIMITE DE TOKENS
    # Si on te donne une valeur spécifique à l'appel, on l'utilise.
    # Sinon, on prend la config globale.
    # Sinon, on met 8192 (standard confortable pour Gemini 2.0).
    final_max_tokens = max_tokens or int(getattr(config, "GEMINI_MAX_OUTPUT_TOKENS", "8192"))

    generation_config = {
        "temperature": 0.0,
        "maxOutputTokens": final_max_tokens, # <--- ON UTILISE LA VARIABLE CALCULÉE
    }

    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    if system_instruction:
        payload["systemInstruction"] = system_instruction

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    reporter and await reporter.step(f"⚡ Calling Gemini API ({model})")  # type: ignore

    max_retries = 3

    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(max_retries + 1):
            try:
                response = await client.post(
                    url,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                if response.status_code == 429:
                     raise HTTPStatusError("Rate Limit (HTTP)", request=response.request, response=response)
                
                if response.status_code == 404:
                    print(f"❌ Erreur 404: Modèle '{model}' introuvable.")
                    response.raise_for_status()

                if response.status_code >= 500:
                    response.raise_for_status()

                try:
                    json_data = response.json()
                except json.JSONDecodeError:
                    raise HTTPStatusError("Invalid JSON", request=response.request, response=response)

                if "error" in json_data:
                    err = json_data["error"]
                    code = err.get("code")
                    status = err.get("status")
                    
                    if code == 429 or status == "RESOURCE_EXHAUSTED":
                        raise HTTPStatusError(f"Rate Limit JSON: {err.get('message')}", request=response.request, response=response)
                    
                    raise ValueError(f"Gemini API Error: {err.get('message')}")

                reporter and await reporter.step("✅ Received Gemini response") # type: ignore
                return extract_text_from_gemini_response(json_data)

            except (HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                is_rate_limit = False
                if isinstance(e, HTTPStatusError):
                    if e.response and e.response.status_code == 429:
                        is_rate_limit = True
                    if "Rate Limit" in str(e):
                        is_rate_limit = True

                if attempt >= max_retries:
                    print(f"❌ Gemini Failure: {e}")
                    return json.dumps({"error": "Service unavailable"}) if json_mode else "Service unavailable."

                delay = (5.0 if is_rate_limit else 2.0) * (2 ** attempt) + random.uniform(0, 1)
                print(f"⚠️ Gemini Retry ({attempt + 1}/{max_retries}) dans {delay:.1f}s...")
                await asyncio.sleep(delay)
    
    return "Error: Unreachable code"