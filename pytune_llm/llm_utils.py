# llm_utils.py
import json, zlib, hashlib
from typing import List, Dict
from pydantic import BaseModel
from tiktoken import encoding_for_model

def compress_json(data: dict | str) -> bytes:
    if isinstance(data, dict):
        data = json.dumps(data)
    return zlib.compress(data.encode())

def decompress_json(blob: bytes) -> str:
    return zlib.decompress(blob).decode()

def serialize_messages(messages: List[BaseModel]) -> List[Dict[str, str]]:
    return [
        {"role": m.role.strip().lower(), "content": m.content.strip()} for m in messages
    ]

def make_cache_key(messages: List[Dict], model: str) -> str:
    raw = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return "llmcache:chat:" + hashlib.sha256(raw.encode()).hexdigest()


def estimate_tokens(messages: list[dict], model_name: str) -> int:
    try:
        encoding = encoding_for_model(model_name)
    except KeyError:
        encoding = encoding_for_model("gpt-3.5-turbo")  # fallback
    total = 0
    for message in messages:
        total += 4  # per-message overhead
        for key, value in message.items():
            total += len(encoding.encode(str(value)))
    total += 2  # priming
    return total

