"""Retrieval: dense (vectors in Chroma), BM25 (keywords), and hybrid (RRF fusion of both).

All retrievers share one interface: `search(query, paper_id=None, k=5) -> list[Hit]`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace

import chromadb
import numpy as np

from .config import CHROMA_DIR, collection_name
from .embedding import Embedder


@dataclass
class Hit:
    chunk_id: str
    paper_id: str
    kind: str
    section_name: str
    document: str  # "[Section] paragraph text", as shown to the LLM
    score: float  # higher is better; the scale depends on the retriever


def open_collection(split: str = "dev") -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name(split)
    try:
        return client.get_collection(name, embedding_function=None)
    except Exception as e:  # chromadb raises different types across versions
        raise RuntimeError(
            f"Chroma collection '{name}' not found in {CHROMA_DIR}. "
            f"Build it first: uv run python -m paperqa.ingest --split {split}"
        ) from e


# --------------------------------------------------------------------------- dense


class DenseRetriever:
    name = "dense"

    def __init__(self, split: str = "dev", embedder: Embedder | None = None, collection=None):
        self.collection = collection if collection is not None else open_collection(split)
        index_model = self.collection.metadata["embed_model"]
        if embedder is not None and embedder.model_name != index_model:
            raise ValueError(f"index was built with {index_model}, but embedder is {embedder.model_name}")
        self.embedder = embedder or Embedder(index_model)

    def describe(self) -> dict:
        return {"retriever": self.name, "embed_model": self.embedder.model_name}

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
                score=1.0 - dist,  # cosine similarity
            )
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]


# --------------------------------------------------------------------------- BM25

STOPWORDS = frozenset(
    "a an and are as at be been by can did do does for from has have how in is it its of on or "
    "that the their them these they this those to was were what when where which who whom why "
    "will with would".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over a small list of tokenized documents.

    Uses the Lucene-style IDF, ln(1 + (N - df + 0.5) / (df + 0.5)), which is never
    negative. That matters here: one paper is only ~55 chunks, so many terms appear
    in more than half of them, and the classic formula would give those a negative weight.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n and self.doc_len.sum() else 1.0
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                postings.setdefault(term, []).append((i, tf))
        # term -> (doc indices, term frequencies, idf)
        self.postings = {
            term: (
                np.array([i for i, _ in plist]),
                np.array([tf for _, tf in plist], dtype=float),
                math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5)),
            )
            for term, plist in postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n)
        for term in set(query_tokens):
            if term not in self.postings:
                continue
            idx, tf, idf = self.postings[term]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
            scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


class BM25Retriever:
    name = "bm25"

    def __init__(self, split: str = "dev", collection=None):
        self.collection = collection if collection is not None else open_collection(split)
        self._cache: dict[str | None, tuple[BM25Index, list[Hit]]] = {}

    def describe(self) -> dict:
        return {"retriever": self.name, "embed_model": None}

    def _index(self, paper_id: str | None) -> tuple[BM25Index, list[Hit]]:
        """BM25 index over one paper's chunks (or the whole collection), built on first use."""
        if paper_id not in self._cache:
            got = self.collection.get(
                where={"paper_id": paper_id} if paper_id else None, include=["documents", "metadatas"]
            )
            rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]))  # stable order for ties
            chunks = [Hit(cid, m["paper_id"], m["kind"], m["section_name"], doc, 0.0) for cid, doc, m in rows]
            self._cache[paper_id] = (BM25Index([tokenize(c.document) for c in chunks]), chunks)
        return self._cache[paper_id]

    def search(self, query: str, paper_id: str | None = None, k: int = 5) -> list[Hit]:
        index, chunks = self._index(paper_id)
        scores = index.scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:k]
        # A chunk that shares no term with the query was not retrieved; don't return it.
        return [replace(chunks[i], score=float(scores[i])) for i in order if scores[i] > 0]


# --------------------------------------------------------------------------- hybrid


def rrf_fuse(rankings: list[list[Hit]], rrf_k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(chunk) = sum over rankings of 1 / (rrf_k + rank)."""
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [replace(hits[cid], score=scores[cid]) for cid in ordered]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, split: str = "dev", n_candidates: int = 20, rrf_k: int = 60):
        collection = open_collection(split)
        self.dense = DenseRetriever(split, collection=collection)
        self.bm25 = BM25Retriever(split, collection=collection)
        self.n_candidates, self.rrf_k = n_candidates, rrf_k

    def describe(self) -> dict:
        return self.dense.describe() | {
            "retriever": self.name,
            "n_candidates": self.n_candidates,
            "rrf_k": self.rrf_k,
        }

    def search(self, query: str, paper_id: str | None = None, k: int = 5) -> list[Hit]:
        n = max(self.n_candidates, k)
        fused = rrf_fuse([self.dense.search(query, paper_id, n), self.bm25.search(query, paper_id, n)], self.rrf_k)
        return fused[:k]


RETRIEVERS = {"dense": DenseRetriever, "bm25": BM25Retriever, "hybrid": HybridRetriever}


def make_retriever(name: str, split: str = "dev"):
    return RETRIEVERS[name](split)
