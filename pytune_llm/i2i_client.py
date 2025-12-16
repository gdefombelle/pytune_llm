# pytune_llm/i2i_client.py

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os
import base64
import random
import time
import mimetypes
import httpx

from pytune_configuration.sync_config_singleton import config, SimpleConfig

config = config or SimpleConfig()

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

I2I_MODEL    = getattr(config, "I2I_MODEL", "gpt-image-1")
I2I_TIMEOUT  = float(getattr(config, "I2I_TIMEOUT", 240))
I2I_RETRIES  = int(getattr(config, "I2I_RETRIES", 2))

SIZE_PRESETS: Dict[str, str] = {
    "landscape_hd": "1536x1024",
    "square_hd":    "1024x1024",
    "portrait_hd":  "1024x1536",
}

ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}


# ─────────────────────────────────────────────────────────────
# ERRORS
# ─────────────────────────────────────────────────────────────

class I2IError(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _resolve_size(size: str) -> str:
    resolved = SIZE_PRESETS.get(size, size)
    return resolved if resolved in ALLOWED_SIZES else "auto"


def _is_heic_like(mime: str | None, filename: str) -> bool:
    name = filename.lower()
    return (
        (mime or "").lower() in {"image/heic", "image/heif", "image/heic-sequence"}
        or name.endswith(".heic")
        or name.endswith(".heif")
    )


def _try_convert_heic_to_jpeg(img_bytes: bytes) -> Optional[bytes]:
    """
    Convertit HEIC/HEIF → JPEG si pillow-heif + Pillow sont installés.
    """
    try:
        import pillow_heif  # type: ignore
        from PIL import Image  # type: ignore
        import io

        heif = pillow_heif.read_heif(img_bytes)
        img = Image.frombytes(heif.mode, heif.size, heif.data)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return None


def _coerce_for_openai(
    img_bytes: bytes,
    filename: str = "image.jpg",
    mime: str | None = None,
) -> Optional[Tuple[str, bytes, str]]:
    """
    S'assure que l'image est acceptable par OpenAI Images API.
    Retourne (filename, bytes, mime) ou None.
    """
    mime = (mime or mimetypes.guess_type(filename)[0] or "").lower()

    # HEIC → JPEG
    if _is_heic_like(mime, filename):
        converted = _try_convert_heic_to_jpeg(img_bytes)
        if not converted:
            return None
        return (os.path.splitext(filename)[0] + ".jpg", converted, "image/jpeg")

    # Normalisation mime
    if mime not in ALLOWED_MIMES:
        if mime.startswith("image/"):
            mime = "image/jpeg"
        else:
            return None

    # Normalisation filename
    if not os.path.splitext(filename)[1]:
        filename += ".jpg"
        mime = "image/jpeg"

    return (filename, img_bytes, mime)


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

async def call_image_to_image(
    *,
    prompt: str,
    images: List[bytes],
    size: str = "landscape_hd",
    extra: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Image-to-Image OpenAI (Images / edits)

    - AUCUN téléchargement réseau
    - images = LISTE DE BYTES
    - retourne TOUJOURS des BYTES (PNG/JPEG)
    """

    if not images:
        raise I2IError("images must not be empty")

    api_key = (
        getattr(config, "OPEN_AI_PYTUNE_API_KEY", None)
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise I2IError("OPENAI_API_KEY missing")

    resolved_size = _resolve_size(size)

    data_fields: Dict[str, Any] = {
        "model": I2I_MODEL,
        "prompt": prompt,
        "size": resolved_size,
    }

    files: List[Tuple[str, Tuple[str, bytes, str]]] = []

    for idx, img_bytes in enumerate(images):
        coerced = _coerce_for_openai(
            img_bytes,
            filename=f"base_{idx}.jpg",
            mime="image/jpeg",
        )

        if not coerced:
            if idx == 0:
                raise I2IError("Base image unsupported or conversion failed")
            continue

        fname, bts, mime = coerced
        files.append(("image[]", (fname, bts, mime)))

    if not files:
        raise I2IError("No usable images after coercion")

    attempt = 0
    last_err: Exception | None = None

    while attempt <= I2I_RETRIES:
        attempt += 1
        try:
            async with httpx.AsyncClient(timeout=I2I_TIMEOUT) as client:
                r = await client.post(
                    "https://api.openai.com/v1/images/edits",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=data_fields,
                    files=files,
                )

            if r.status_code >= 400:
                raise I2IError(
                    f"OpenAI I2I HTTP {r.status_code}: {r.text[:800]}"
                )

            payload = r.json()
            data = payload.get("data") or []
            b64 = (data[0] or {}).get("b64_json") if data else None

            if not b64:
                raise I2IError("OpenAI I2I response missing b64_json")

            return base64.b64decode(b64)

        except Exception as e:
            last_err = e
            if attempt > I2I_RETRIES:
                break
            time.sleep(0.6 * attempt + random.uniform(0, 0.4))

    raise I2IError(f"I2I failed after {I2I_RETRIES + 1} attempts: {last_err}")