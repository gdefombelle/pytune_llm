# pytune_llm/i2i_client.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
import os, base64, random, time, mimetypes
import httpx

from pytune_configuration.sync_config_singleton import config, SimpleConfig
config = config or SimpleConfig()

I2I_PROVIDER = (getattr(config, "I2I_PROVIDER", None) or "openai").lower()
I2I_MODEL    = getattr(config, "I2I_MODEL", "gpt-image-1")
I2I_TIMEOUT  = float(getattr(config, "I2I_TIMEOUT", 240))
I2I_RETRIES  = int(getattr(config, "I2I_RETRIES", 2))

# OpenAI Images sizes autorisées
SIZE_PRESETS: Dict[str, str] = {
    "landscape_hd": "1536x1024",
    "square_hd":    "1024x1024",
    "portrait_hd":  "1024x1536",
}
ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}

ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}

class I2IError(RuntimeError): ...
class I2IUnsupported(RuntimeError): ...

def _resolve_size(size: str) -> str:
    wanted = SIZE_PRESETS.get(size, size)
    return wanted if wanted in ALLOWED_SIZES else "auto"

async def _download(url: str, timeout: float) -> Tuple[bytes, str, str]:
    """
    Télécharge une URL -> (bytes, guessed_mime, filename)
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        content = r.content
    mime, _ = mimetypes.guess_type(url)
    filename = os.path.basename(url) or "image"
    return content, (mime or "application/octet-stream"), filename

def _is_heic_like(mime: str, filename: str) -> bool:
    name = filename.lower()
    return (
        (mime or "").lower() in {"image/heic", "image/heif", "image/heic-sequence"}
        or name.endswith(".heic")
        or name.endswith(".heif")
    )

def _try_convert_heic_to_jpeg(img_bytes: bytes) -> Optional[bytes]:
    """
    Convertit HEIC -> JPEG si pillow-heif + Pillow sont dispos.
    Retourne bytes JPEG ou None si conversion impossible.
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

def _coerce_file_for_openai(
    img_bytes: bytes,
    mime: str,
    filename: str,
) -> Optional[Tuple[str, bytes, str]]:
    """
    S'assure que le fichier est dans un mimetype accepté par OpenAI.
    Convertit HEIC -> JPEG si besoin. Renvoie (filename, bytes, mime) ou None si inutilisable.
    """
    # HEIC → JPEG
    if _is_heic_like(mime, filename):
        converted = _try_convert_heic_to_jpeg(img_bytes)
        if not converted:
            return None
        return (os.path.splitext(filename)[0] + ".jpg", converted, "image/jpeg")

    # Si mime pas clair, essaie de deviner par extension
    ext_mime = (mimetypes.guess_type(filename)[0] or "").lower()
    final_mime = (mime or ext_mime or "").lower()

    # Si toujours pas OK, force en JPEG (meilleur pari)
    if final_mime not in ALLOWED_MIMES:
        # on ne sait pas convertir arbitrairement; tente de passer quand même en JPEG si c'est déjà JPEG
        if final_mime.startswith("image/"):
            # laisser passer; beaucoup d'images "unknown" marcheront si réellement jpeg
            final_mime = "image/jpeg"
        else:
            # trop risqué -> refuser
            return None

    # normalise filename
    if not os.path.splitext(filename)[1]:
        filename += ".jpg"
        final_mime = "image/jpeg"

    return (filename, img_bytes, final_mime)

async def call_image_to_image(
    prompt: str,
    image_urls: List[str],
    size: str = "landscape_hd",
    extra: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Appel pur Images/Edits (pas de chat). Retourne TOUJOURS des bytes (PNG/JPEG).
    - supporte plusieurs URLs
    - auto-convertit HEIC -> JPEG si possible
    """
    if not image_urls:
        raise I2IError("image_urls must not be empty")

    if I2I_PROVIDER != "openai":
        raise I2IUnsupported(f"Unsupported provider: {I2I_PROVIDER}")

    api_key = getattr(config, "OPEN_AI_PYTUNE_API_KEY", None) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise I2IError("OPENAI_API_KEY missing")

    resolved_size = _resolve_size(size)

    # Prépare multipart
    data_fields: Dict[str, Any] = {
        "model": I2I_MODEL,
        "prompt": prompt,
        "size": resolved_size,
    }

    files: List[Tuple[str, Tuple[str, bytes, str]]] = []
    for idx, url in enumerate(image_urls):
        try:
            raw, mime, name = await _download(url, I2I_TIMEOUT)
        except Exception as e:
            if idx == 0:
                raise I2IError(f"Failed to download base image: {e}") from e
            continue

        coerced = _coerce_file_for_openai(raw, mime, name)
        if not coerced:
            # impossible à convertir (ex: HEIC sans libs) -> skip sauf si c'est la première
            if idx == 0:
                raise I2IError(
                    f"Base image unsupported or conversion failed (mime={mime}, name={name}). "
                    "Install pillow-heif + Pillow to convert HEIC/HEIF."
                )
            continue

        fname, bts, final_mime = coerced
        # Pour plusieurs images, OpenAI accepte `image[]`. La première suffirait
        # mais on envoie toutes (les non-HEIC) pour maximiser la guidance.
        files.append(("image[]", (fname, bts, final_mime)))

    if not files:
        raise I2IError("No usable images to send (all unsupported/failed to download).")

    # Retry + backoff
    attempt, last_err = 0, None
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
                # on inclut un extrait du body pour debug
                raise I2IError(f"OpenAI I2I HTTP {r.status_code}: {r.text[:600]}")

            payload = r.json()
            data = payload.get("data") or []
            b64 = (data[0] or {}).get("b64_json") if data else None
            if not b64:
                raise I2IError("OpenAI I2I: missing data[0].b64_json")

            return base64.b64decode(b64)

        except Exception as e:
            print("call_image_to_image exception: ", e)
            last_err = e
            if attempt > I2I_RETRIES:
                break
            time.sleep(0.6 * attempt + random.uniform(0, 0.4))

    raise I2IError(f"I2I failed after {I2I_RETRIES+1} attempts: {last_err}")
