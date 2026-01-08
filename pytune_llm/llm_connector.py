# llm_connector.py
from typing import Optional, Dict

from pytune_llm.llm_backends.cerebras_backend import call_cerebras_llm
from pytune_llm.llm_backends.gemini_backend import call_gemini_llm
from pytune_llm.llm_backends.openai_backend import call_openai_llm
from pytune_llm.llm_backends.ollama_backend import call_ollama_llm

from pytune_llm.settings import get_llm_backend
from pytune_llm.task_reporting.reporter import TaskReporter


async def call_llm(
    prompt: str,
    context: Dict,
    model: Optional[str] = None,
    metadata: Optional[Dict] = None,
    reporter: Optional[TaskReporter] = None,
) -> str:
    """
    Unified LLM entry point.
    - Resolves backend and model
    - Delegates default model choice to each backend
    """

    if metadata is None:
        metadata = {}

    # 1. Resolve backend
    backend: str = metadata.get("llm_backend") or get_llm_backend()

    # 2. Resolve model (explicit > metadata > context > None)
    resolved_model: Optional[str] = (
        model
        or metadata.get("llm_model")
        or context.get("llm_model")
    )

    # 3. Inject resolved info into context (traceability)
    context["llm_backend"] = backend
    if resolved_model is not None:
        context["llm_model"] = resolved_model

    # 4. Route to backend
    if backend == "cerebras":
        return await call_cerebras_llm(
            prompt=prompt,
            context=context,
            model=resolved_model,
            reporter=reporter,
        )

    if backend == "openai":
        return await call_openai_llm(
            prompt=prompt,
            context=context,
            model=resolved_model,
            reporter=reporter,
        )

    if backend == "ollama":
        return await call_ollama_llm(
            prompt=prompt,
            context=context,
            model=resolved_model,
            reporter=reporter,
        )

    if backend == "gemini":
        return await call_gemini_llm(
            prompt=prompt,
            context=context,
            model=resolved_model,
            reporter=reporter,
        )

    raise ValueError(f"❌ Unsupported LLM backend: {backend}")