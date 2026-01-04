# llm_connector.py
from typing import Optional
from pytune_llm.llm_backends.gemini_backend import call_gemini_llm
from pytune_llm.settings import get_llm_backend, get_supported_llm_models
from pytune_llm.llm_backends.openai_backend import call_openai_llm
from pytune_llm.llm_backends.ollama_backend import call_ollama_llm
from pytune_llm.task_reporting.reporter import TaskReporter

async def call_llm(
        prompt: str, 
        context: dict, 
        metadata: dict = None,
        reporter: Optional[TaskReporter] = None) -> str:

    metadata = metadata or {}

    # 1. Récupérer backend et modèle
    backend = metadata.get("llm_backend") or get_llm_backend()
    llm_model = metadata.get("llm_model") or context.get("llm_model") or "gpt-5-mini"

    # 2. Valider contre les modèles supportés
    # supported_models = get_supported_llm_models()
    # supported = supported_models.get(backend, set())

    # if llm_model not in supported:
    #     raise ValueError(f"❌ LLM model '{llm_model}' is not supported for backend '{backend}'")

    # 3. Injecter dans le contexte (optionnel mais utile)
    context["llm_model"] = llm_model
    context["llm_backend"] = backend

    # 4. Appel du backend correspondant
    if backend == "openai":
        return await call_openai_llm(prompt, context, reporter=reporter)

    elif backend == "ollama":
        return await call_ollama_llm(prompt, context, reporter=reporter)
    
    elif backend == 'gemini':
        return await call_gemini_llm(prompt, context, reporter=reporter)

    else:
        raise ValueError(f"❌ Unsupported LLM backend: {backend}")
