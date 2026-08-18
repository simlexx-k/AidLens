from typing import Any


class SentenceTransformerEncoder:
    """Lazy SentenceTransformers wrapper so the core API can run without ML extras."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic search requires the 'ml' optional dependencies. "
                "Rebuild the API with AIDLENS_API_EXTRAS=ml."
            ) from exc
        self.model_name = model_name
        self.model: Any = SentenceTransformer(model_name)

    def encode_query(self, query: str) -> list[float]:
        text = query
        if "bge-" in self.model_name.lower():
            text = f"Represent this sentence for searching relevant passages: {query}"
        vector = self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return vector.tolist()

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(
            documents,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
