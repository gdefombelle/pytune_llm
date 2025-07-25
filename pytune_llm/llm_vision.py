import asyncio
import json
import re
from typing import List
from pytune_llm.llm_client import call_llm_vision


async def label_image_from_url(image_url: str, filename: str = None) -> dict:
    prompt = """
You are a visual piano expert. Your job is to label photos of acoustic pianos with precise visual descriptors.

Return a strict JSON object with the following fields:
- angle: angle of view (front, side, top, angled, unknown)
- view_type: type of shot (keyboard, full, logo, internal, other)
- lighting: lighting quality (well-lit, dark, blurry, partial)
- content: list of visible piano-related objects (e.g. logo, bench, pedals, keys, cover, music stand)
- notes: freeform notes on clarity or issues (e.g. “logo obscured”, “too close to frame”, etc.)
- If the image appears to show a serial number (e.g., a sequence of digits engraved or printed inside the piano), mention this in the notes field, e.g. "possible serial number visible".

Respond ONLY with JSON. Do not explain or apologize.
""".strip()

    try:
        response = await call_llm_vision(
            prompt=prompt,
            image_urls=[image_url],
            metadata={"llm_model": "gpt-4o"}
        )

        content = response["choices"][0]["message"]["content"]
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)

        if not match:
            raise ValueError("No JSON block found in response")

        json_str = match.group(1)
        parsed = json.loads(json_str)

        # Ajoute has_serial_number = True si les notes mentionnent un serial
        notes = parsed.get("notes", "").lower()
        if "serial number" in notes or "numéro de série" in notes:
            parsed["has_serial_number"] = True
        else:
            parsed["has_serial_number"] = False

        return parsed

    except Exception as e:
        return {
            "error": str(e),
            "filename": filename or image_url,
            "image_url": image_url
        }


async def label_images_from_urls(image_data: List[dict]) -> List[dict]:
    """
    image_data = [{"url": ..., "filename": ...}, ...]
    """
    tasks = [
        label_image_from_url(img["url"], img.get("filename", f"photo_{i+1}.jpg"))
        for i, img in enumerate(image_data)
    ]
    return await asyncio.gather(*tasks)
