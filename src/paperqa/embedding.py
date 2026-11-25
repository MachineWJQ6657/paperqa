"""Thin wrapper around sentence-transformers so ingest and retrieval embed the same way."""

from __future__ import annotations

import numpy as np

from .config import EMBED_MODEL, QUERY_PREFIXES


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # slow import, keep it lazy

        self.model_name = model_name
        self.query_prefix = QUERY_PREFIXES.get(model_name, "")
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True)[0]
