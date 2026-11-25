"""Single place where the project calls an LLM. Records tokens and latency for every call.

Credentials come from the environment (ANTHROPIC_API_KEY, or an `ant auth login` profile).
To try another provider later, add a class with the same `complete()` signature here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

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
        self.client = client or anthropic.Anthropic()

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
