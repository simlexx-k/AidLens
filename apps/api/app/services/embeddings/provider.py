from functools import lru_cache

from app.services.embeddings.sentence_transformer import SentenceTransformerEncoder


@lru_cache(maxsize=4)
def get_embedding_encoder(model_name: str) -> SentenceTransformerEncoder:
    """Return the process-local semantic encoder singleton for one model."""

    return SentenceTransformerEncoder(model_name)
