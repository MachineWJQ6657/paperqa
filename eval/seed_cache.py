"""Fill the LLM reply cache from saved result files, so runs made before the cache existed
(or on another machine) are not paid for again.

    uv run python eval/seed_cache.py eval/results/baseline_dense_cc_sonnet.json [more.json ...]

For every answered question it rebuilds the exact Writer and judge requests and stores the
saved reply under them. If two files contain the same request, the first file wins.
Grader calls are not seeded. Needs the Chroma index (`uv run python -m paperqa.ingest`).
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from paperqa.agents import Writer
from paperqa.config import LLM_CACHE_PATH
from paperqa.llm import LLMCache, LLMResult
from paperqa.metrics import LLMJudge
from paperqa.retrieve import open_collection


class Recorder:
    """Stands in for an LLM and remembers the request, so the prompts come from the real agents."""

    def complete(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        self.request = (system, user, schema)
        return LLMResult("", None, "", None, 0, 0, 0.0)


def seed(result: dict, cache: LLMCache) -> tuple[int, int]:
    c = result["config"]
    if c["retrieval_only"]:
        return 0, 0
    backend = c.get("backend") or "api"
    collection = open_collection(c["split"])
    recorder = Recorder()
    before = len(cache.entries)
    seen = 0
    for rec in result["per_question"]:
        if rec["error"]:
            continue
        ids = rec.get("context") or [r["chunk_id"] for r in rec["retrieved"][: c["k"]]]
        got = collection.get(ids=ids, include=["documents"])
        text = dict(zip(got["ids"], got["documents"]))
        hits = [SimpleNamespace(chunk_id=i, document=text[i]) for i in ids]

        Writer(recorder).answer(rec["question"], hits)
        data = {"answer": rec["answer"], "citations": [ids.index(i) + 1 for i in rec["cited"]]}
        reply = LLMResult(
            text=json.dumps(data, ensure_ascii=False),
            data=data,
            model=rec.get("writer_model") or c["writer_model"],
            stop_reason=None,
            input_tokens=rec["tokens"]["writer_in"],
            output_tokens=rec["tokens"]["writer_out"],
            latency_s=rec["latency_s"]["writer"],
        )
        cache.put(cache.key(backend, c["writer_model"], c["writer_effort"], *recorder.request), reply)
        seen += 1

        if rec.get("judge_correct") is None:
            continue
        references = [SimpleNamespace(answer=r["answer"]) for r in rec["references"]]
        LLMJudge(recorder).judge(rec["question"], references, rec["answer"])
        data = {"correct": rec["judge_correct"], "reason": rec["judge_reason"]}
        reply = LLMResult(
            text=json.dumps(data, ensure_ascii=False),
            data=data,
            model=c["judge_model"],
            stop_reason=None,
            input_tokens=rec["tokens"].get("judge_in", 0),
            output_tokens=rec["tokens"].get("judge_out", 0),
            latency_s=0.0,  # judge latency was never recorded
        )
        cache.put(cache.key(backend, c["judge_model"], c["judge_effort"], *recorder.request), reply)
        seen += 1
    return seen, len(cache.entries) - before


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cache = LLMCache(LLM_CACHE_PATH)
    for path in sys.argv[1:]:
        seen, added = seed(json.load(open(path)), cache)
        print(f"{path}: {seen} replies, {added} new in cache")
    print(f"{LLM_CACHE_PATH}: {len(cache.entries)} entries")


if __name__ == "__main__":
    main()
