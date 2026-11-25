"""Evaluation metrics.

`normalize_answer`, `token_f1` and `paragraph_f1` follow the official QASPER evaluator:
for every question take the max score over all annotators' references.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from .llm import LLM, LLMResult
from .qasper import Reference

# --------------------------------------------------------------------------- official QASPER


def normalize_answer(s: str) -> str:
    """Lowercase, drop punctuation and articles (a/an/the), collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    num_same = sum((Counter(pred) & Counter(gold)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def paragraph_f1(prediction: list[str], ground_truth: list[str]) -> float:
    if not ground_truth and not prediction:
        return 1.0  # unanswerable and nothing cited
    num_same = len(set(ground_truth) & set(prediction))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def max_answer_f1(prediction: str, references: list[Reference]) -> tuple[float, str]:
    """Best token F1 over references, and the answer type of the reference that achieved it."""
    scored = [(token_f1(prediction, r.answer), r.type) for r in references]
    return max(scored, key=lambda x: x[0])


def max_evidence_f1(predicted_evidence: list[str], references: list[Reference]) -> float:
    return max(paragraph_f1(predicted_evidence, r.evidence) for r in references)


# --------------------------------------------------------------------------- retrieval


def evidence_recall(retrieved: list[str], gold_sets: list[set[str]]) -> float | None:
    """Fraction of one annotator's evidence found in `retrieved`, max over annotators.

    `retrieved` and `gold_sets` hold normalized chunk texts. Returns None when the
    question has no gold evidence that maps to an indexed chunk.
    """
    if not gold_sets:
        return None
    got = set(retrieved)
    return max(len(g & got) / len(g) for g in gold_sets)


def evidence_hit(retrieved: list[str], gold_sets: list[set[str]]) -> float | None:
    """1.0 if at least one gold evidence chunk (from any annotator) was retrieved."""
    if not gold_sets:
        return None
    got = set(retrieved)
    return float(any(g & got for g in gold_sets))


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------- LLM-as-judge

JUDGE_SYSTEM = """\
You grade answers to questions about research papers. Compare the predicted answer with \
the reference answers written by human annotators.

The prediction is correct if it gives the same information as at least one reference \
answer. Wording may differ, and extra detail is fine as long as it does not contradict \
that reference. A prediction that is only partly right, or that misses the main point, \
is incorrect. If a reference is "Unanswerable" and the prediction also says the question \
cannot be answered from the paper, that counts as a match."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "correct": {"type": "boolean"},
    },
    "required": ["reason", "correct"],
    "additionalProperties": False,
}


@dataclass
class Judgement:
    correct: bool | None  # None if the judge failed to return a verdict
    reason: str
    llm: LLMResult


class LLMJudge:
    def __init__(self, llm: LLM):
        self.llm = llm

    def judge(self, question: str, references: list[Reference], prediction: str) -> Judgement:
        refs = "\n".join(f"- {r.answer}" for r in references)
        user = f"Question: {question}\n\nReference answers:\n{refs}\n\nPredicted answer: {prediction or '(empty)'}"
        res = self.llm.complete(JUDGE_SYSTEM, user, schema=JUDGE_SCHEMA)
        if res.data is None:
            return Judgement(None, f"no verdict (stop_reason={res.stop_reason})", res)
        return Judgement(bool(res.data["correct"]), res.data["reason"], res)
