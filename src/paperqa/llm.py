"""Single place where the project calls an LLM. Records tokens and latency for every call.

Two backends with the same `complete()` signature:

- LLM: the Anthropic API. Credentials: PAPERQA_ANTHROPIC_API_KEY if set, otherwise the
  SDK defaults (ANTHROPIC_API_KEY, or an `ant auth login` profile). Billed to the API key.
- ClaudeCodeLLM: Claude Code in headless mode (`claude -p`). Uses the Claude Code login,
  so calls count against the Claude subscription instead of API credits.

To try another provider later, add a class with the same `complete()` signature here.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic

from .config import LLM_MODEL

# Server-side fallback: if the model declines for policy reasons, the API retries on a
# substitute model inside the same call. "default" lets the server pick the fallback.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class LLMResult:
    text: str
    data: dict | None  # parsed JSON when a schema was given and the output is complete
    model: str  # model that actually served the request (may differ after a fallback)
    stop_reason: str | None
    input_tokens: int  # includes cached input tokens
    output_tokens: int  # includes thinking tokens
    latency_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class LLM:
    backend = "api"

    def __init__(
        self,
        model: str = LLM_MODEL,
        effort: str | None = "medium",
        max_tokens: int = 16000,
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.effort = effort  # low | medium | high | xhigh | max; None = model default
        self.max_tokens = max_tokens
        # A project-specific variable keeps this key separate from other tools that
        # read ANTHROPIC_API_KEY (for example Claude Code itself in a cloud session).
        self.client = client or anthropic.Anthropic(api_key=os.getenv("PAPERQA_ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        """One user turn. With `schema`, the reply is constrained to JSON matching it."""
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        extra = {"output_config": output_config} if output_config else {}

        t0 = time.perf_counter()
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            betas=[FALLBACK_BETA],
            fallbacks="default",
            **extra,
        )
        latency = time.perf_counter() - t0

        text = "".join(b.text for b in resp.content if b.type == "text")
        data = None
        # On "refusal" or "max_tokens" the JSON may be missing or cut off.
        if schema and resp.stop_reason not in ("refusal", "max_tokens"):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None

        u = resp.usage
        input_tokens = u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0)
        return LLMResult(
            text=text,
            data=data,
            model=resp.model,
            stop_reason=resp.stop_reason,
            input_tokens=input_tokens,
            output_tokens=u.output_tokens,
            latency_s=latency,
        )


class ClaudeCodeError(RuntimeError):
    """A `claude -p` call failed (after retries)."""


class ClaudeCodeLLM:
    """Same interface as LLM, but every call runs `claude -p` (Claude Code headless mode).

    Differences from the API backend that show up in the results:
    - Claude Code adds its own context to every request (about 1.5-2k input tokens), and a
      call with a schema takes two model turns. Token counts and latency are therefore
      higher than with LLM and must not be compared across backends.
    - No server-side fallback: a refusal comes back as an error or as an answer without JSON.
    """

    backend = "claude-code"

    def __init__(
        self,
        model: str = LLM_MODEL,
        effort: str | None = "medium",
        timeout_s: float = 300,
        retries: int = 2,
        backoff_s: float = 2.0,
        executable: str = "claude",
    ):
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.executable = executable
        # Run outside the repo so no project CLAUDE.md or settings reach the model.
        self.cwd = tempfile.mkdtemp(prefix="paperqa-claude-code-")
        # With ANTHROPIC_API_KEY set, Claude Code would bill that key instead of the login.
        self.env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    def command(self, system: str, schema: dict | None) -> list[str]:
        cmd = [
            self.executable,
            "-p",
            "--model", self.model,
            "--system-prompt", system,
            "--tools", "",  # no built-in tools: this is a plain completion
            "--strict-mcp-config",  # no MCP servers
            "--setting-sources", "",  # ignore user/project settings (hooks, permissions)
            "--no-session-persistence",
            "--output-format", "json",
        ]  # fmt: skip
        if self.effort:
            cmd += ["--effort", self.effort]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        return cmd

    def run_once(self, cmd: list[str], user: str) -> tuple[dict, float]:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, input=user, capture_output=True, text=True, timeout=self.timeout_s, cwd=self.cwd, env=self.env
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCodeError(f"timed out after {self.timeout_s:.0f}s") from e
        wall = time.perf_counter() - t0
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            detail = (proc.stderr or proc.stdout).strip()[:300]
            raise ClaudeCodeError(f"exit code {proc.returncode}, no JSON output: {detail}") from e
        if proc.returncode != 0 or out.get("is_error"):
            raise ClaudeCodeError(f"{out.get('subtype')}: {str(out.get('result'))[:300]}")
        return out, wall

    def complete(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        """One user turn. With `schema`, the reply is constrained to JSON matching it."""
        cmd = self.command(system, schema)
        for attempt in range(self.retries + 1):
            try:
                out, wall = self.run_once(cmd, user)
                break
            except FileNotFoundError as e:
                raise ClaudeCodeError(f"'{self.executable}' not found; install Claude Code and log in") from e
            except ClaudeCodeError:
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff_s * 2**attempt)

        data = out.get("structured_output") if schema else None
        u = out.get("usage") or {}
        input_tokens = sum(
            u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        models = out.get("modelUsage") or {}
        model = max(models, key=lambda m: models[m].get("outputTokens", 0)) if models else self.model
        api_ms = out.get("duration_api_ms")
        return LLMResult(
            text=out.get("result") or "",
            data=data if isinstance(data, dict) else None,
            model=model,
            stop_reason=out.get("stop_reason"),
            input_tokens=input_tokens,
            output_tokens=u.get("output_tokens") or 0,
            # Time spent in model requests; excludes starting the CLI process.
            latency_s=api_ms / 1000 if api_ms is not None else wall,
        )


class LLMCache:
    """Replies keyed by the full request, kept in an append-only JSONL file.

    One instance is shared by every CachedLLM of a run. Safe to use from several threads.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.entries: dict[str, dict] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry["result"]

    @staticmethod
    def key(backend: str, model: str, effort: str | None, system: str, user: str, schema: dict | None) -> str:
        request = [backend, model, effort, system, user, schema]
        return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, key: str) -> LLMResult | None:
        with self.lock:
            result = self.entries.get(key)
            if result is None:
                self.misses += 1
                return None
            self.hits += 1
            return LLMResult(**result)

    def put(self, key: str, result: LLMResult) -> None:
        with self.lock:
            if key in self.entries:
                return
            self.entries[key] = result.to_dict()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "result": self.entries[key]}, ensure_ascii=False) + "\n")


class CachedLLM:
    """Wraps a backend: a request seen before is answered from the cache instead of the model.

    A cached reply keeps the tokens and latency of the original call, so efficiency numbers
    stay those of a real request. Replies without usable output are not stored.
    """

    def __init__(self, inner: LLM | ClaudeCodeLLM, cache: LLMCache):
        self.inner = inner
        self.cache = cache

    def complete(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        key = self.cache.key(self.inner.backend, self.inner.model, self.inner.effort, system, user, schema)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.inner.complete(system, user, schema)
        if schema is None or result.data is not None:
            self.cache.put(key, result)
        return result


BACKENDS = {"api": LLM, "claude-code": ClaudeCodeLLM}


def make_llm(
    backend: str, model: str, effort: str | None, cache: LLMCache | None = None
) -> LLM | ClaudeCodeLLM | CachedLLM:
    llm = BACKENDS[backend](model, effort=effort)
    return CachedLLM(llm, cache) if cache is not None else llm
