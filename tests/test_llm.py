"""ClaudeCodeLLM plumbing, tested with a fake `subprocess.run` so the CLI is never started."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from paperqa import llm as llm_module
from paperqa.llm import ClaudeCodeError, ClaudeCodeLLM

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def cli_output(**overrides) -> str:
    out = {
        "is_error": False,
        "subtype": "success",
        "result": '{"answer":"BERT"}',
        "structured_output": {"answer": "BERT"},
        "stop_reason": "tool_use",
        "duration_api_ms": 1500,
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 2000,
            "cache_read_input_tokens": 100,
            "output_tokens": 40,
        },
        "modelUsage": {"served-model": {"outputTokens": 40}},
    }
    return json.dumps(out | overrides)


def fake_run(monkeypatch, outputs):
    """Each call pops the next (stdout, returncode) or raises the next exception."""
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        nxt = outputs.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        stdout, returncode = nxt
        return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

    monkeypatch.setattr(llm_module.subprocess, "run", run)
    return calls


def test_command_and_parsed_result(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-passed")
    calls = fake_run(monkeypatch, [(cli_output(), 0)])
    res = ClaudeCodeLLM("test-model", effort="low").complete("SYS", "USER", schema=SCHEMA)

    cmd, kwargs = calls[0]
    assert cmd[:2] == ["claude", "-p"]
    for flag, value in [("--model", "test-model"), ("--system-prompt", "SYS"), ("--tools", ""), ("--effort", "low")]:
        assert cmd[cmd.index(flag) + 1] == value
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert kwargs["input"] == "USER"
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]  # otherwise the CLI would bill the API key

    assert res.data == {"answer": "BERT"}
    assert res.model == "served-model"
    assert (res.input_tokens, res.output_tokens) == (2102, 40)
    assert res.latency_s == 1.5


def test_no_schema_no_effort(monkeypatch):
    calls = fake_run(monkeypatch, [(cli_output(result="OK", structured_output=None), 0)])
    res = ClaudeCodeLLM("test-model", effort=None).complete("SYS", "USER")
    assert "--json-schema" not in calls[0][0] and "--effort" not in calls[0][0]
    assert (res.text, res.data) == ("OK", None)


def test_missing_structured_output_gives_no_data(monkeypatch):
    fake_run(monkeypatch, [(cli_output(structured_output=None), 0)])
    assert ClaudeCodeLLM("m").complete("SYS", "USER", schema=SCHEMA).data is None


def test_retries_then_succeeds(monkeypatch):
    outputs = [
        (cli_output(is_error=True, result="overloaded"), 1),
        subprocess.TimeoutExpired("claude", 1),
        (cli_output(), 0),
    ]
    calls = fake_run(monkeypatch, outputs)
    res = ClaudeCodeLLM("m", retries=2, backoff_s=0).complete("SYS", "USER", schema=SCHEMA)
    assert len(calls) == 3 and res.data == {"answer": "BERT"}


def test_raises_after_retries(monkeypatch):
    calls = fake_run(monkeypatch, [("not json", 1), (cli_output(is_error=True, result="Not logged in"), 1)])
    with pytest.raises(ClaudeCodeError, match="Not logged in"):
        ClaudeCodeLLM("m", retries=1, backoff_s=0).complete("SYS", "USER")
    assert len(calls) == 2


def test_missing_cli_is_not_retried(monkeypatch):
    calls = fake_run(monkeypatch, [FileNotFoundError("claude")])
    with pytest.raises(ClaudeCodeError, match="not found"):
        ClaudeCodeLLM("m", retries=3, backoff_s=0).complete("SYS", "USER")
    assert len(calls) == 1
