"""Provider-agnostic chat client (any OpenAI-compatible API, e.g. Groq) with the bookkeeping this study needs.

- Latency: wall-clock time of the successful HTTP call, measured on the client (D-021). Retries and
  rate-limit waits are excluded and logged separately, so the latency reflects the provider, not our pacing.
- Cost: billed tokens x the configured price. Reasoning tokens are billed as output (D-022).
- Cache: every response is appended to a JSONL file keyed by a hash of (model, messages, generation
  params). Re-running an evaluation is free and reproducible; cache hits keep their original latency.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import openai


@dataclass
class LLMResponse:
    key: str
    model: str
    content: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    latency_ms: float
    attempts: int
    cost_usd: float
    from_cache: bool = field(default=False)


def request_key(model: str, messages: list[dict], params: dict) -> str:
    payload = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cost_usd(prompt_tokens: int, completion_tokens: int, cached_tokens: int, prices: dict) -> float:
    cached_price = prices.get("cached_input_per_million")
    if cached_price is None:
        cached_price = prices["input_per_million"]
    uncached = prompt_tokens - cached_tokens
    return (uncached * prices["input_per_million"] + cached_tokens * cached_price
            + completion_tokens * prices["output_per_million"]) / 1e6


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self._store[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self._store.get(key)

    def put(self, rec: dict) -> None:
        self._store[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)


class LLMClient:
    def __init__(self, cfg: dict, api_key: str | None, cache: ResponseCache | None = None, sdk_client=None):
        self.model = cfg["model"]
        self.prices = cfg["prices"]
        gen = cfg["generation"]
        self.params = {k: v for k, v in {
            "temperature": gen.get("temperature"),
            "max_completion_tokens": gen.get("max_completion_tokens"),
            "reasoning_effort": gen.get("reasoning_effort"),
        }.items() if v is not None}
        self.extra_body = gen.get("extra_body") or {}
        self.min_interval_s = 60.0 / cfg["client"]["requests_per_minute"]
        self.tokens_per_minute = cfg["client"].get("tokens_per_minute")  # None = no token pacing
        self._recent: deque[tuple[float, int]] = deque()  # (time, tokens) of calls in the last 60 s
        self.max_retries = cfg["client"]["max_retries"]
        # network outages last minutes, not seconds: connection errors get a longer retry budget
        self.max_connection_retries = cfg["client"].get("max_connection_retries", self.max_retries)
        self.cache = cache
        # max_retries=0: we retry ourselves so every attempt and wait is visible in the logs
        self.sdk = sdk_client or openai.OpenAI(api_key=api_key, base_url=cfg["base_url"],
                                               timeout=cfg["client"]["timeout_s"], max_retries=0)
        self._last_call = 0.0

    def _pace(self) -> None:
        wait = self._last_call + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        if self.tokens_per_minute and self._recent:
            # assume the next call costs about as much as the last one; wait until it fits in the window
            expected = self._recent[-1][1]
            while self._recent and sum(t for _, t in self._recent) + expected > self.tokens_per_minute:
                time.sleep(max(0.0, self._recent[0][0] + 60 - time.monotonic()) + 0.05)
                self._drop_old()
        self._last_call = time.monotonic()

    def _drop_old(self) -> None:
        while self._recent and self._recent[0][0] < time.monotonic() - 60:
            self._recent.popleft()

    def complete(self, messages: list[dict]) -> LLMResponse:
        key = request_key(self.model, messages, {**self.params, **self.extra_body})
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return LLMResponse(**{**hit, "from_cache": True})

        for attempt in range(1, max(self.max_retries, self.max_connection_retries) + 2):
            self._pace()
            start = time.perf_counter()
            try:
                resp = self.sdk.chat.completions.create(model=self.model, messages=messages,
                                                        extra_body=self.extra_body or None, **self.params)
            except RETRYABLE as exc:
                is_network = isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))
                if attempt > (self.max_connection_retries if is_network else self.max_retries):
                    raise
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None and retry_after > 600:
                    raise  # a daily quota, not a per-minute one: stop; the cache lets the run resume later
                backoff = retry_after if retry_after is not None else min(60.0, 2 ** attempt) + random.random()
                print(f"  {type(exc).__name__} (attempt {attempt}); retrying in {backoff:.1f}s")
                time.sleep(backoff)
                continue
            latency_ms = (time.perf_counter() - start) * 1000
            break

        usage = resp.usage
        self._recent.append((time.monotonic(), usage.total_tokens or usage.prompt_tokens + usage.completion_tokens))
        self._drop_old()
        details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        reasoning = (getattr(details, "reasoning_tokens", None) or 0) if details else 0
        cached = (getattr(prompt_details, "cached_tokens", None) or 0) if prompt_details else 0
        choice = resp.choices[0]
        out = LLMResponse(
            key=key, model=resp.model or self.model, content=choice.message.content,
            finish_reason=choice.finish_reason, prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens, reasoning_tokens=reasoning, cached_tokens=cached,
            latency_ms=latency_ms, attempts=attempt,
            cost_usd=cost_usd(usage.prompt_tokens, usage.completion_tokens, cached, self.prices),
        )
        if self.cache is not None:
            rec = asdict(out)
            rec.pop("from_cache")
            self.cache.put(rec)
        return out


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) + 0.5 if value is not None else None
    except ValueError:
        return None
