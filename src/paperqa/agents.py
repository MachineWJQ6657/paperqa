"""Agents: the Writer, and the Grader with its re-query loop. A Router will be added later."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .llm import LLM, LLMResult
from .retrieve import Hit

WRITER_SYSTEM = """\
You answer questions about a research paper. You are given numbered passages retrieved \
from that paper; use only these passages.

- Answer as briefly as possible: a short phrase, a number, a comma-separated list, or \
"Yes"/"No" for a yes/no question. Write a sentence only when the question asks for an \
explanation.
- If the passages do not contain the answer, set the answer to "Unanswerable" and leave \
the citations empty.
- In "citations", list the numbers of the passages that support your answer."""

WRITER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


@dataclass
class WriterOutput:
    answer: str
    cited_chunk_ids: list[str]
    llm: LLMResult
    error: str | None = None


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.document}" for i, h in enumerate(hits, start=1))


class Writer:
    def __init__(self, llm: LLM):
        self.llm = llm

    def answer(self, question: str, hits: list[Hit]) -> WriterOutput:
        user = f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"
        res = self.llm.complete(WRITER_SYSTEM, user, schema=WRITER_SCHEMA)
        if res.data is None:
            return WriterOutput("", [], res, error=f"no valid JSON (stop_reason={res.stop_reason})")
        # Citation numbers are 1-based; ignore any that point outside the passage list.
        cited = []
        for n in res.data["citations"]:
            if 1 <= n <= len(hits) and hits[n - 1].chunk_id not in cited:
                cited.append(hits[n - 1].chunk_id)
        return WriterOutput(res.data["answer"].strip(), cited, res)


# --------------------------------------------------------------------------- Grader

GRADER_SYSTEM = """\
You check whether passages retrieved from a research paper are enough to answer a \
question about that paper. You do not answer the question.

- Set "sufficient" to true only if the passages contain the specific information the \
answer needs. Passages that are on the same topic but do not contain the answer are \
not sufficient.
- If the passages are not sufficient, write in "query" one search query for the missing \
information. Phrase it the way the paper itself would state the answer: use likely \
section names, method or dataset names, and key terms, rather than repeating the question.
- If the passages are sufficient, set "query" to an empty string."""

GRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "query": {"type": "string"},
    },
    "required": ["sufficient", "query"],
    "additionalProperties": False,
}


@dataclass
class Grade:
    sufficient: bool
    query: str  # new search query when not sufficient, else ""
    llm: LLMResult
    error: str | None = None


class Grader:
    def __init__(self, llm: LLM):
        self.llm = llm

    def grade(self, question: str, hits: list[Hit]) -> Grade:
        user = f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"
        res = self.llm.complete(GRADER_SYSTEM, user, schema=GRADER_SCHEMA)
        if res.data is None:
            # No verdict: keep the passages we have instead of guessing a new query.
            return Grade(True, "", res, error=f"no valid JSON (stop_reason={res.stop_reason})")
        return Grade(bool(res.data["sufficient"]), res.data["query"].strip(), res)


@dataclass
class Gathered:
    hits: list[Hit]  # passages for the Writer: the first retrieval, then anything added
    grades: list[Grade] = field(default_factory=list)
    added: list[list[str]] = field(default_factory=list)  # chunk ids added by each re-query
    search_latency_s: float = 0.0  # time spent in re-query searches


def gather_passages(
    question: str,
    hits: list[Hit],
    search: Callable[[str, int], list[Hit]],
    grader: Grader,
    k: int,
    max_requeries: int = 1,
) -> Gathered:
    """Grade the passages; while they are not sufficient, search again with the Grader's query.

    Each re-query adds up to `k` passages that are not already in the list, so the Writer
    gets at most k * (1 + max_requeries) passages. `search(query, n)` returns the top n hits.
    """
    out = Gathered(hits=list(hits))
    for _ in range(max_requeries):
        grade = grader.grade(question, out.hits)
        out.grades.append(grade)
        if grade.sufficient or not grade.query:
            break
        seen = {h.chunk_id for h in out.hits}
        t0 = time.perf_counter()
        found = search(grade.query, k + len(seen))
        out.search_latency_s += time.perf_counter() - t0
        new = [h for h in found if h.chunk_id not in seen][:k]
        out.added.append([h.chunk_id for h in new])
        if not new:
            break
        out.hits.extend(new)
    return out
