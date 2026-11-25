"""Retrieval. For now only dense (vector) search; BM25 and hybrid come later."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from .config import CHROMA_DIR, collection_name
from .embedding import Embedder


@dataclass
class Hit:
    chunk_id: str
    paper_id: str
    kind: str
    section_name: str
    document: str  # "[Section] paragraph text", as shown to the LLM
    score: float  # cosine similarity, higher is better


class DenseRetriever:
    name = "dense"

    def __init__(self, split: str = "dev", embedder: Embedder | None = None):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        name = collection_name(split)
        try:
            self.collection = client.get_collection(name, embedding_function=None)
        except Exception as e:  # chromadb raises different types across versions
            raise RuntimeError(
                f"Chroma collection '{name}' not found in {CHROMA_DIR}. "
                f"Build it first: uv run python -m paperqa.ingest --split {split}"
            ) from e
        index_model = self.collection.metadata["embed_model"]
        if embedder is not None and embedder.model_name != index_model:
            raise ValueError(f"index was built with {index_model}, but embedder is {embedder.model_name}")
        self.embedder = embedder or Embedder(index_model)

    def search(self, query: str, paper_id: str | None = None, k: int = 5) -> list[Hit]:
        """Top-k chunks for the query. QASPER questions are about one paper, so filter by paper_id."""
        vec = self.embedder.embed_query(query)
        res = self.collection.query(
            query_embeddings=[vec.tolist()],
            n_results=k,
            where={"paper_id": paper_id} if paper_id else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(
                chunk_id=cid,
                paper_id=meta["paper_id"],
                kind=meta["kind"],
                section_name=meta["section_name"],
                document=doc,
                score=1.0 - dist,
            )
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
