import asyncio
import json
from typing import List
from pytune_llm.llm_client import call_llm_vision

async def label_image_from_url(image_url: str) -> dict:
    prompt = """
You are a visual piano expert. Your job is to label photos of acoustic pianos with precise visual descriptors.

Return a strict JSON object with the following fields:
- angle: angle of view (front, side, top, angled, unknown)
- view_type: type of shot (keyboard, full, logo, internal, other)
- lighting: lighting quality (well-lit, dark, blurry, partial)
- content: list of visible piano-related objects (e.g. logo, bench, pedals, keys, cover, music stand)
- notes: freeform notes on clarity or issues (e.g. “logo obscured”, “too close to frame”, etc.)

Respond ONLY with JSON. Do not explain or apologize.
"""

    try:
        response_text = await call_llm_vision(prompt=prompt, image_urls=[image_url])
        return json.loads(response_text)
    except Exception as e:
        return {"error": str(e), "source_image": image_url}

async def label_images_from_urls(image_urls: List[str]) -> List[dict]:
    tasks = [label_image_from_url(url) for url in image_urls]
    return await asyncio.gather(*tasks)