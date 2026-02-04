# pytune_llm/prompting.py

from typing import Dict, Any
from jinja2 import Environment, StrictUndefined


_jinja_env = Environment(
    autoescape=False,
    undefined=StrictUndefined,  # 🔥 très important pour debug
    trim_blocks=True,
    lstrip_blocks=True,
)

def render_prompt(
    template_source: str,
    context: Dict[str, Any],
) -> str:
    """
    Render a Jinja prompt from a raw template string and a context dict.
    No filesystem access. No business logic.
    """
    try:
        template = _jinja_env.from_string(template_source)
        return template.render(context)
    except Exception as e:
        print("⚠️ Prompt rendering error")
        print("🧩 Context keys:", list(context.keys()))
        raise