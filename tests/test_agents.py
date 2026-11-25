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
