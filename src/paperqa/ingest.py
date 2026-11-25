"""Load QASPER, chunk every paper, embed the chunks and store them in Chroma.

    uv run python -m paperqa.ingest                # dev split (default)
    uv run python -m paperqa.ingest --rebuild      # force a rebuild
"""

from __future__ import annotations

import argparse
import time

import chromadb

from .config import CHROMA_DIR, CHUNKING_VERSION, EMBED_MODEL, collection_name
from .embedding import Embedder
from .qasper import Chunk, load_split, paper_chunks


def build_index(
    split: str = "dev",
    embed_model: str = EMBED_MODEL,
    rebuild: bool = False,
    limit_papers: int | None = None,
    batch_size: int = 64,
) -> chromadb.Collection:
    papers = load_split(split)
    if limit_papers:
        papers = dict(list(papers.items())[:limit_papers])
    chunks: list[Chunk] = [c for pid, p in papers.items() for c in paper_chunks(pid, p)]

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name(split)
    wanted_meta = {"embed_model": embed_model, "chunking": CHUNKING_VERSION, "split": split}

    existing = {c.name for c in client.list_collections()}
    if name in existing:
        col = client.get_collection(name, embedding_function=None)
        same_config = all(col.metadata.get(k) == v for k, v in wanted_meta.items())
        if not rebuild and same_config and col.count() == len(chunks):
            print(f"Collection '{name}' is up to date ({col.count()} chunks). Use --rebuild to force.")
            return col
        client.delete_collection(name)

    col = client.create_collection(
        name,
        embedding_function=None,  # we always pass our own embeddings
        configuration={"hnsw": {"space": "cosine"}},
        metadata=wanted_meta,
    )

    print(f"Embedding {len(chunks)} chunks from {len(papers)} papers with {embed_model} ...")
    t0 = time.perf_counter()
    embedder = Embedder(embed_model)
    vectors = embedder.embed_documents([c.document for c in chunks], batch_size=batch_size, show_progress=True)

    step = 2000  # stay below Chroma's max batch size
    for i in range(0, len(chunks), step):
        part = chunks[i : i + step]
        col.add(
            ids=[c.chunk_id for c in part],
            embeddings=vectors[i : i + step].tolist(),
            documents=[c.document for c in part],
            metadatas=[c.metadata() for c in part],
        )
    print(f"Indexed {col.count()} chunks into '{name}' at {CHROMA_DIR} in {time.perf_counter() - t0:.0f}s")
    return col


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit-papers", type=int, default=None, help="index only the first N papers (smoke test)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    build_index(args.split, args.embed_model, args.rebuild, args.limit_papers, args.batch_size)


if __name__ == "__main__":
    main()
