# llm_utils.py
import json, zlib, hashlib
from typing import List, Dict
from pydantic import BaseModel

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
