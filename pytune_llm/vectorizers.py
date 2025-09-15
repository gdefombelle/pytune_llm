import openai
from pytune_configuration.sync_config_singleton import SimpleConfig, config

config = config or SimpleConfig()

openai.api_key = config.OPEN_AI_PYTUNE_API_KEY

# 🔁 Modèles disponibles pour les embeddings OpenAI :
# - "text-embedding-3-small" : léger, rapide, 1536 dimensions (recommandé pour la plupart des cas)
# - "text-embedding-3-large" : plus précis, mais plus lent et plus cher (3072 dimensions)

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Retourne l'embedding OpenAI pour un texte donné."""
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    return response["data"][0]["embedding"]

def get_embedding_batch(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Retourne les embeddings OpenAI pour une liste de textes."""
    response = openai.Embedding.create(
        input=texts,
        model=model
    )
    return [d["embedding"] for d in response["data"]]

# ✅ Optionnel : fallback local avec sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    _local_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    _local_model = None

def get_embedding_local(text: str) -> list[float]:
    """Retourne un embedding local si sentence-transformers est installé."""
    if _local_model is None:
        raise ImportError("sentence-transformers n'est pas installé")
    return _local_model.encode(text).tolist()
