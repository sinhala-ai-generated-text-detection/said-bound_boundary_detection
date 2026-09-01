"""OpenRouter chat-completions client.

Request format follows the live API reference (fetched 2026-09-01):
  POST https://openrouter.ai/api/v1/chat/completions
  Authorization: Bearer <key>
  body: {model, messages, temperature, top_p, max_tokens, seed, reasoning{...}}
  response: choices[0].message.content, choices[0].message.reasoning,
            usage{prompt_tokens, completion_tokens, cost, ...}

Reasoning safety: for reasoning-capable models we send
`reasoning: {effort: ..., exclude: true}` so traces are never returned, AND we
defensively read only `message.content`, never `message.reasoning`. A trace can
therefore not leak into stored dataset text even if the provider ignores the
flag.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from utils import REPO_ROOT, append_jsonl


class OpenRouterError(RuntimeError):
    pass


class MissingAPIKey(OpenRouterError):
    pass


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader. Real environment always wins."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0


@dataclass
class Completion:
    text: str
    usage: Usage
    model: str
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


class RateLimiter:
    """Spaces request starts by at least `min_interval` seconds."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last = now


class OpenRouterClient:
    """Thread-safe client with retries, backoff, and cost logging."""

    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}

    def __init__(self, cfg: dict, dry_run: bool = False) -> None:
        self.cfg = cfg
        api = cfg["api"]
        self.api = api
        self.dry_run = dry_run
        self.base_url = api["base_url"].rstrip("/")
        self.endpoint = api["endpoint"]
        self.timeout = api["timeout_seconds"]
        self.max_retries = api["max_retries"]
        self.backoff_base = api["backoff_base_seconds"]
        self.backoff_max = api["backoff_max_seconds"]
        self.limiter = RateLimiter(api["min_request_interval"])
        self.cost_log = Path(cfg["paths"]["logs_dir"]) / "cost.jsonl"

        self._lock = threading.Lock()
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0
        self.n_retries = 0

        if dry_run:
            self.api_key = "DRY-RUN-NO-KEY"
            self._client = None
            return

        load_dotenv()
        key = os.environ.get(api["api_key_env"], "").strip()
        if not key:
            raise MissingAPIKey(
                f"Environment variable {api['api_key_env']} is not set.\n"
                f"Set it in your shell or place it in a .env file at the repo "
                f"root as:\n  {api['api_key_env']}=sk-or-v1-...\n"
                f"Refusing to continue: no API key, no generation."
            )
        self.api_key = key
        self._client = httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # Optional attribution headers documented by OpenRouter.
                "HTTP-Referer": api.get("referer", ""),
                "X-Title": api.get("title", ""),
            },
        )

    # ------------------------------------------------------------ helpers --
    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def verify_models(self, verbose: bool = True) -> dict[str, dict]:
        """Check configured slugs still exist and warn on price drift.

        Prices drift; the spec asks us to verify at runtime rather than trust
        the numbers pinned in config.yaml.
        """
        if self.dry_run:
            return {}
        url = self.base_url + self.api.get("models_endpoint", "/models")
        r = httpx.get(url, timeout=self.timeout)
        r.raise_for_status()
        live = {m["id"]: m for m in r.json()["data"]}

        out: dict[str, dict] = {}
        for name, g in self.cfg["generators"].items():
            slug = g["slug"]
            m = live.get(slug)
            if m is None:
                raise OpenRouterError(
                    f"Configured model '{slug}' ({name}) is not available on "
                    f"OpenRouter. Update config.yaml generators.{name}.slug."
                )
            p_in = float(m["pricing"]["prompt"])
            p_out = float(m["pricing"]["completion"])
            drift_in = abs(p_in - g["price_prompt"]) > 1e-12
            drift_out = abs(p_out - g["price_completion"]) > 1e-12
            if verbose:
                flag = "  <-- PRICE DRIFT" if (drift_in or drift_out) else ""
                print(f"  {name:<16s} {slug:<28s} "
                      f"in=${p_in*1e6:.4f}/M out=${p_out*1e6:.4f}/M{flag}")
                if drift_in or drift_out:
                    print(f"      config has in=${g['price_prompt']*1e6:.4f}/M "
                          f"out=${g['price_completion']*1e6:.4f}/M")
            out[name] = {"slug": slug, "price_prompt": p_in,
                         "price_completion": p_out,
                         "context_length": m.get("context_length")}
        return out

    def _estimate_cost(self, gen_cfg: dict, usage: Usage) -> float:
        return (usage.prompt_tokens * gen_cfg["price_prompt"]
                + usage.completion_tokens * gen_cfg["price_completion"])

    # --------------------------------------------------------------- call --
    def complete(
        self,
        gen_name: str,
        system: str,
        user: str,
        *,
        meta: dict | None = None,
        seed: int | None = None,
    ) -> Completion:
        """One chat completion with retries. Returns cleaned text only."""
        gen_cfg = self.cfg["generators"][gen_name]

        if self.dry_run:
            return self._dry_run_completion(gen_name, gen_cfg, user, meta)

        payload: dict[str, Any] = {
            "model": gen_cfg["slug"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
            "max_tokens": gen_cfg["max_tokens"],
        }
        if seed is not None:
            payload["seed"] = seed
        reasoning = gen_cfg.get("reasoning")
        if reasoning:
            # exclude=true -> traces never returned to us.
            payload["reasoning"] = dict(reasoning)
            payload["reasoning"].setdefault("exclude", True)

        url = self.base_url + self.endpoint
        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                resp = self._client.post(url, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_err = e
                self._sleep_backoff(attempt)
                continue

            if resp.status_code in self.RETRY_STATUS:
                last_err = OpenRouterError(
                    f"HTTP {resp.status_code}: {resp.text[:300]}")
                retry_after = resp.headers.get("Retry-After")
                self._sleep_backoff(attempt, retry_after)
                continue

            if resp.status_code >= 400:
                raise OpenRouterError(
                    f"HTTP {resp.status_code} (non-retryable): {resp.text[:500]}")

            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                last_err = e
                self._sleep_backoff(attempt)
                continue

            # OpenRouter can return 200 with an error body.
            if "error" in data and not data.get("choices"):
                msg = str(data["error"])[:300]
                last_err = OpenRouterError(f"API error body: {msg}")
                self._sleep_backoff(attempt)
                continue

            return self._parse(data, gen_name, gen_cfg, meta)

        raise OpenRouterError(
            f"{gen_name}: exhausted {self.max_retries} retries. Last: {last_err}")

    def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        with self._lock:
            self.n_retries += 1
        if retry_after:
            try:
                time.sleep(min(float(retry_after), self.backoff_max))
                return
            except ValueError:
                pass
        delay = min(self.backoff_base * (2 ** attempt), self.backoff_max)
        time.sleep(delay + random.uniform(0, delay * 0.25))   # full-ish jitter

    def _parse(self, data: dict, gen_name: str, gen_cfg: dict,
               meta: dict | None) -> Completion:
        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError(f"No choices in response: {str(data)[:300]}")
        msg = choices[0].get("message") or {}
        # Deliberately read ONLY `content`. `message.reasoning` is ignored so a
        # reasoning trace can never reach the dataset.
        text = (msg.get("content") or "").strip()

        u = data.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            reasoning_tokens=int(det.get("reasoning_tokens") or 0),
            cost=float(u.get("cost") or 0.0),
        )
        # Prefer OpenRouter's own cost; fall back to configured prices.
        if usage.cost <= 0:
            usage.cost = self._estimate_cost(gen_cfg, usage)

        with self._lock:
            self.n_calls += 1
            self.total_cost += usage.cost
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            running = self.total_cost

        append_jsonl(self.cost_log, {
            "ts": time.time(),
            "generator": gen_name,
            "model": gen_cfg["slug"],
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "cost_usd": round(usage.cost, 8),
            "running_total_usd": round(running, 8),
            "finish_reason": choices[0].get("finish_reason"),
            **(meta or {}),
        })

        return Completion(
            text=text,
            usage=usage,
            model=data.get("model", gen_cfg["slug"]),
            finish_reason=choices[0].get("finish_reason"),
            raw={"id": data.get("id")},
        )

    # ------------------------------------------------------------ dry run --
    def _dry_run_completion(self, gen_name: str, gen_cfg: dict, user: str,
                            meta: dict | None) -> Completion:
        """Synthetic Sinhala output shaped to satisfy the validator.

        Lets --dry-run exercise the whole generate->validate->assemble path
        with zero API calls and zero cost.
        """
        n_sent = (meta or {}).get("target_sentence_count", 3)
        n_words = (meta or {}).get("target_word_count", 40)
        per = max(4, n_words // max(1, n_sent))
        stock = ["මෙම", "කරුණ", "පිළිබඳව", "වැඩිදුර", "විස්තර", "ලිපියෙහි",
                 "සඳහන්", "වන", "අතර", "එය", "වැදගත්", "කරුණකි"]
        rng = random.Random(hash((user, gen_name)) & 0xFFFFFFFF)
        sents = []
        for _ in range(n_sent):
            words = [rng.choice(stock) for _ in range(per - 1)]
            sents.append(" ".join(words) + " වේ.")
        text = " ".join(sents)
        usage = Usage(prompt_tokens=len(user) // 4,
                      completion_tokens=len(text) // 4, cost=0.0)
        with self._lock:
            self.n_calls += 1
        return Completion(text=text, usage=usage,
                          model=gen_cfg["slug"] + " (dry-run)",
                          finish_reason="stop")

    # ------------------------------------------------------------ summary --
    def summary(self) -> dict:
        return {
            "calls": self.n_calls,
            "retries": self.n_retries,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_cost_usd": round(self.total_cost, 6),
        }
