from typing import Callable, Awaitable, Sequence
import asyncio
from pytune_llm.task_reporting.task_pubsub import send_task

StepFunc = Callable[[], Awaitable[None]]

async def wrap_backend_task(
    agent: str,
    steps: Sequence[tuple[str, Callable[[], Awaitable[None]]]],
    delay_after_step: float = 0.0,  # ⏳ délai entre les étapes
):
    """
    Exécute des étapes asynchrones avec messages SSE de progression.
    Chaque étape est (label, async_fn)
    """
    total = len(steps)
    for i, (label, func) in enumerate(steps):
        progress = round(i / total, 3)
        await send_task(agent, f"{label}...", progress=progress)
        await func()
        if delay_after_step > 0:
            await asyncio.sleep(delay_after_step)

    await send_task(agent, "✅ Terminé", progress=1.0)

class TaskReporter:
    def __init__(
        self,
        agent: str,
        total_steps: int = 1,
        delay_after_step: float = 0.05,
        auto_progress: bool = False
    ):
        self.agent = agent
        self.total_steps = total_steps
        self.current_step = 0
        self.base_progress = 0.0  # Reserved for future offset logic
        self.delay_after_step = delay_after_step
        self.auto_progress = auto_progress

    def set_total(self, total: int):
        self.total_steps = total

    async def send(self, message: str, *, step: int | None = None, progress: float | None = None):
        if progress is None:
            if step is not None:
                progress = step / self.total_steps
            elif self.auto_progress and self.total_steps > 0:
                progress = self.current_step / self.total_steps
        await send_task(self.agent, message, progress=progress)

    async def step(self, label: str, *, progress: float | None = None):
        self.current_step += 1
        await self.send(f"{label}...", step=self.current_step, progress=progress)
        if self.delay_after_step > 0:
            await asyncio.sleep(self.delay_after_step)

    async def done(self, message: str = "✅ Completed", delay_after: float = 0.0):
        await self.send(message, progress=1.0)
        if delay_after > 0:
            await asyncio.sleep(delay_after)

    def decorator(self, label: str):
        """Decorateur pour envelopper une fonction asynchrone avec un step automatique."""
        def wrapper(func: Callable[..., Awaitable[None]]):
            async def inner(*args, **kwargs):
                await self.step(label)
                return await func(*args, **kwargs)
            return inner
        return wrapper
