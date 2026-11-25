"""Project-wide defaults. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("PAPERQA_DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = Path(os.getenv("PAPERQA_CHROMA_DIR", DATA_DIR / "chroma"))
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# Embedding model (sentence-transformers). bge-small is ~130MB and runs fine on CPU.
EMBED_MODEL = os.getenv("PAPERQA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Some embedding models expect an instruction in front of the query (not the documents).
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

# How LLM calls are made: "claude-code" runs `claude -p` and counts against the Claude
# subscription; "api" uses an Anthropic API key and is billed to that key.
LLM_BACKEND = os.getenv("PAPERQA_LLM_BACKEND", "claude-code")

# LLM used by the Writer, and by the LLM-as-judge during evaluation.
LLM_MODEL = os.getenv("PAPERQA_LLM_MODEL", "claude-sonnet-5-5")
JUDGE_MODEL = os.getenv("PAPERQA_JUDGE_MODEL", LLM_MODEL)

# Bump this when the chunking logic changes, so stale indexes are rebuilt.
CHUNKING_VERSION = "v1-paragraph+caption+abstract"


def collection_name(split: str) -> str:
    return f"qasper_{split}"
