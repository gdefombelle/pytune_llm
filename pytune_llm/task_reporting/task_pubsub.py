import asyncio
import json
from typing import Any

# Une file par agent
_task_queues: dict[str, asyncio.Queue] = {}

def get_queue(agent: str) -> asyncio.Queue:
    if agent not in _task_queues:
        _task_queues[agent] = asyncio.Queue()
    return _task_queues[agent]

async def send_task(
    agent: str,
    message: str,
    *,
    progress: float | None = None,
    extra: dict[str, Any] = {},
    visible: bool = False,   # 👈 nouveau flag
):
    """
    Envoie un message structuré vers le flux SSE de l'agent donné.
    """
    payload = {
        "message": message,
        "progress": progress,
        "visible": visible,   # 👈 ajouté
        **extra
    }
    await get_queue(agent).put(json.dumps(payload))