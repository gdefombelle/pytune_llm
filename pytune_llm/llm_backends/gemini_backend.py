# gemini_backend.py
import json
from typing import Optional, Tuple
import httpx

# On suppose que vous créerez cette fonction dans vos settings
# from pytune_llm.settings import get_gemini_key 
from pytune_llm.task_reporting.reporter import TaskReporter
from pytune_configuration.sync_config_singleton import config, SimpleConfig

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────

config = config or SimpleConfig()

# Le modèle Flash est le plus adapté (vitesse/coût) pour votre usage
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash" 

# Gemini a une fenêtre de contexte énorme, mais on garde des limites de sécurité
GEMINI_MAX_OUTPUT_TOKENS = int(
    getattr(config, "GEMINI_MAX_OUTPUT_TOKENS", "1024")
)

# ────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────

def format_gemini_messages(messages: list[dict]) -> Tuple[list[dict], dict | None]:
    """
    Convertit le format standard Chat (role/content) vers le format Gemini REST API.
    Extrait également le System Prompt car il est traité séparément dans Gemini.
    """
    contents = []
    system_instruction = None

    for m in messages:
        role = m.get("role", "user").lower()
        content = m.get("content", "")

        # Gestion du System Prompt (Extraction)
        if role == "system":
            system_instruction = {
                "parts": [{"text": content}]
            }
            continue # On ne l'ajoute pas aux 'contents'

        # Mapping des rôles (OpenAI -> Gemini)
        # user -> user
        # assistant -> model
        gemini_role = "model" if role == "assistant" else "user"

        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            # Gestion basique du multi-modal si votre input est déjà structuré
            # Note: Pour les images, Gemini attend "inline_data" ou "file_data"
            parts = content 
        
        contents.append({
            "role": gemini_role,
            "parts": parts
        })

    return contents, system_instruction


def extract_text_from_gemini_response(json_data: dict) -> str:
    """
    Extraction robuste de la réponse texte Gemini.
    """
    # Vérification des filtres de sécurité
    if json_data.get("promptFeedback", {}).get("blockReason"):
        raise ValueError(f"Blocked by safety filters: {json_data['promptFeedback']}")

    candidates = json_data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates returned. Response: {json_data}")

    candidate = candidates[0]
    
    # Vérification si le modèle a fini pour une raison étrange
    if candidate.get("finishReason") not in ["STOP", "MAX_TOKENS", None]:
         # Warning silencieux ou log ici si nécessaire
         pass

    content = candidate.get("content", {})
    parts = content.get("parts", [])

    if parts and "text" in parts[0]:
        return parts[0]["text"].strip()
    
    return ""

# ────────────────────────────────────────────────────────────────
# Main call
# ────────────────────────────────────────────────────────────────

async def call_gemini_llm(
    prompt: str | None = None,
    context: dict | None = None, # Réservé pour usage futur
    messages: list[dict] | None = None,
    model: str = GEMINI_DEFAULT_MODEL,
    json_mode: bool = False, # Ajout utile pour l'extraction de données
    reporter: Optional[TaskReporter] = None,
) -> str:
    
    # Récupération de la clé (à adapter selon votre gestion de config)
    # api_key = get_gemini_key() 
    # Pour l'exemple, on suppose qu'elle est dans l'environnement ou config
    api_key = getattr(config, "GEMINI_API_KEY", "") 

    if not messages:
        if not prompt:
            raise ValueError("You must provide either 'messages' or 'prompt'")
        messages = [
            {"role": "user", "content": prompt},
        ]

    # Préparation du payload
    contents, system_instruction = format_gemini_messages(messages)
    
    generation_config = {
        "temperature": 0.0,
        "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
    }

    # Activation du mode JSON strict si demandé (très puissant sur Gemini Flash)
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": contents,
        "generationConfig": generation_config
    }

    if system_instruction:
        payload["systemInstruction"] = system_instruction

    # URL de l'API REST Google (v1beta est souvent requise pour les derniers modèles)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    reporter and await reporter.step(f"⚡ Calling Gemini API ({model})") # type: ignore

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json"
            },
            json=payload,
        )

        if response.status_code >= 400:
            print("❌ GEMINI STATUS:", response.status_code)
            print("❌ GEMINI BODY:", response.text)
            response.raise_for_status()

        json_data = response.json()

    reporter and await reporter.step("✅ Received Gemini response") # type: ignore

    return extract_text_from_gemini_response(json_data)