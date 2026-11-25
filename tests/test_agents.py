"""Writer/LLM plumbing, tested with a fake client so no API key or network is needed."""

import json
from types import SimpleNamespace

from paperqa.agents import Writer
from paperqa.llm import LLM
from paperqa.retrieve import Hit


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=self.text)],
            model=kwargs["model"],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=20, output_tokens=30
            ),
        )


def fake_llm(text, stop_reason="end_turn"):
    messages = FakeMessages(text, stop_reason)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return LLM("test-model", effort="low", client=client), messages


HITS = [
    Hit("p#s0p0", "p", "paragraph", "Intro", "[Intro] We use BERT.", 0.9),
    Hit("p#s1p0", "p", "paragraph", "Data", "[Data] We use SQuAD.", 0.8),
]


def test_writer_parses_answer_and_maps_citations():
    llm, messages = fake_llm(json.dumps({"answer": " BERT ", "citations": [1, 1, 7]}))
    out = Writer(llm).answer("Which model?", HITS)
    assert out.answer == "BERT"
    assert out.cited_chunk_ids == ["p#s0p0"]  # duplicate and out-of-range citations dropped
    assert out.error is None
    assert out.llm.input_tokens == 120 and out.llm.output_tokens == 30

    call = messages.calls[0]
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "[1] [Intro] We use BERT." in call["messages"][0]["content"]
    assert "Question: Which model?" in call["messages"][0]["content"]


def test_writer_handles_truncated_output():
    llm, _ = fake_llm('{"answer": "BE', stop_reason="max_tokens")
    out = Writer(llm).answer("Which model?", HITS)
    assert out.answer == "" and out.cited_chunk_ids == []
    assert "max_tokens" in out.error


# --------------------------------------------------------------------------- Grader


def hit(i: int) -> Hit:
    return Hit(f"p#{i}", "p", "paragraph", "S", f"[S] passage {i}", 1.0 - i / 100)


class FakeGrader:
    """Returns the given (sufficient, query) verdicts in order and records what it was shown."""

    def __init__(self, verdicts):
        self.verdicts, self.seen = list(verdicts), []

    def grade(self, question, hits):
        from paperqa.agents import Grade

        self.seen.append([h.chunk_id for h in hits])
        sufficient, query = self.verdicts.pop(0)
        return Grade(sufficient, query, llm=None)


def fake_search(results):
    calls = []

    def search(query, n):
        calls.append((query, n))
        return results[:n]

    return search, calls


def test_grader_parses_verdict():
    from paperqa.agents import Grader

    llm, messages = fake_llm(json.dumps({"sufficient": False, "query": " BLEU scores table "}))
    grade = Grader(llm).grade("What is the BLEU score?", HITS)
    assert (grade.sufficient, grade.query, grade.error) == (False, "BLEU scores table", None)
    assert "[1] [Intro] We use BERT." in messages.calls[0]["messages"][0]["content"]


def test_grader_without_verdict_keeps_passages():
    from paperqa.agents import Grader

    llm, _ = fake_llm("not json")
    grade = Grader(llm).grade("q", HITS)
    assert grade.sufficient and grade.query == "" and grade.error


def test_gather_sufficient_does_not_search():
    from paperqa.agents import gather_passages

    search, calls = fake_search([hit(9)])
    out = gather_passages("q", [hit(0), hit(1)], search, FakeGrader([(True, "")]), k=2)
    assert [h.chunk_id for h in out.hits] == ["p#0", "p#1"]
    assert calls == [] and out.added == []


def test_gather_requery_adds_only_new_passages():
    from paperqa.agents import gather_passages

    # The re-query returns one passage we already have, then three new ones; only k=2 are added.
    search, calls = fake_search([hit(1), hit(5), hit(6), hit(7)])
    grader = FakeGrader([(False, "new query")])
    out = gather_passages("q", [hit(0), hit(1)], search, grader, k=2, max_requeries=1)
    assert calls == [("new query", 4)]  # k + passages already held, so k new ones can survive the filter
    assert [h.chunk_id for h in out.hits] == ["p#0", "p#1", "p#5", "p#6"]
    assert out.added == [["p#5", "p#6"]]
    assert len(grader.seen) == 1  # one re-query allowed, so no second grading


def test_gather_second_round_grades_all_passages_so_far():
    from paperqa.agents import gather_passages

    search, calls = fake_search([hit(5), hit(6)])
    grader = FakeGrader([(False, "first"), (True, "")])
    out = gather_passages("q", [hit(0)], search, grader, k=1, max_requeries=2)
    assert grader.seen == [["p#0"], ["p#0", "p#5"]]
    assert [c[0] for c in calls] == ["first"] and len(out.hits) == 2


def test_gather_stops_when_requery_finds_nothing_new():
    from paperqa.agents import gather_passages

    search, calls = fake_search([hit(0)])
    grader = FakeGrader([(False, "again"), (False, "and again")])
    out = gather_passages("q", [hit(0)], search, grader, k=1, max_requeries=3)
    assert len(calls) == 1 and out.added == [[]] and [h.chunk_id for h in out.hits] == ["p#0"]
