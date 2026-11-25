"""Agents. Baseline has only the Writer; Router and Grader will be added later."""

from __future__ import annotations

from dataclasses import dataclass

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
