import pytest

from paperqa.metrics import (
    evidence_hit,
    evidence_recall,
    max_answer_f1,
    max_evidence_f1,
    normalize_answer,
    paragraph_f1,
    token_f1,
)
from paperqa.qasper import Reference


def test_normalize_answer_drops_case_punctuation_articles():
    assert normalize_answer("The  BERT-base, model!") == "bertbase model"
    assert normalize_answer("an apple a day") == "apple day"


def test_token_f1():
    assert token_f1("BERT", "BERT") == 1.0
    assert token_f1("the BERT model", "BERT") == pytest.approx(2 * 0.5 * 1 / 1.5)  # P=1/2, R=1
    assert token_f1("GPT", "BERT") == 0.0
    assert token_f1("", "BERT") == 0.0
    assert token_f1("Yes", "yes.") == 1.0


def test_paragraph_f1():
    assert paragraph_f1([], []) == 1.0
    assert paragraph_f1(["a"], []) == 0.0
    assert paragraph_f1([], ["a"]) == 0.0
    assert paragraph_f1(["a", "b"], ["a"]) == pytest.approx(2 * 0.5 * 1 / 1.5)


def test_max_over_references():
    refs = [
        Reference("Unanswerable", "none", []),
        Reference("BERT and ELMo", "extractive", ["p1", "p2"]),
    ]
    f1, kind = max_answer_f1("BERT and ELMo", refs)
    assert (f1, kind) == (1.0, "extractive")
    f1, kind = max_answer_f1("Unanswerable", refs)
    assert (f1, kind) == (1.0, "none")
    assert max_evidence_f1(["p1", "p2"], refs) == 1.0
    assert max_evidence_f1([], refs) == 1.0  # matches the unanswerable reference


def test_evidence_recall_takes_best_annotator():
    gold = [{"a", "b"}, {"c"}]
    assert evidence_recall(["a", "x", "y"], gold) == 0.5
    assert evidence_recall(["c"], gold) == 1.0
    assert evidence_recall(["x"], gold) == 0.0
    assert evidence_recall(["a"], []) is None


def test_evidence_hit():
    gold = [{"a", "b"}, {"c"}]
    assert evidence_hit(["x", "b"], gold) == 1.0
    assert evidence_hit(["x"], gold) == 0.0
    assert evidence_hit(["x"], []) is None
