"""
LLM gateway — multi-key, multi-provider fallback routing.

What an "LLM gateway" is
------------------------
A gateway sits between your application and the model providers and owns the
cross-cutting concerns: routing, automatic failover, retries with backoff,
response caching, and per-call observability. Hosted products (Portkey, LiteLLM,
OpenRouter) do this as a service. This module does it in-process — same
behaviour, no third-party account, nothing to pay for. See DOCS/09.

The fallback ladder
-------------------
Groq enforces rate limits per (key, model). So a 429 on the 70B model with your
primary key says nothing about the 8B model, or about your second key. The chain
exploits both axes before giving up:

    1. GROQ_API_KEY           + llama-3.3-70b-versatile   (best quality)
    2. GROQ_FALLBACK_API_KEY  + llama-3.3-70b-versatile   (second free quota)
    3. GROQ_API_KEY           + llama-3.1-8b-instant      (cheaper model, same key)
    4. GROQ_FALLBACK_API_KEY  + llama-3.1-8b-instant
    5. Gemini chat                                        (different provider entirely)

Anything below tier 1 is a degraded answer, so `fallback_used` is surfaced all
the way to the UI rather than hidden. Silent degradation is how a system looks
healthy while quietly getting worse.

The "fast" tier reverses the order (8B first). Planner and grader calls run on
every query and do not need a 70B model — spending the good quota on them is
what exhausts it before the answer is even generated.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from dataclasses import dataclass, field

import logfire

from app.config import settings


class AllTargetsFailed(RuntimeError):
    pass


_RETRYABLE_MARKERS = (
    "429", "rate limit", "ratelimit", "rate_limit", "quota", "resource_exhausted",
    "too many requests", "500", "502", "503", "504", "overloaded", "unavailable",
    "timeout", "timed out", "connection", "internal server error",
)

# Errors where retrying the same target is pointless — move on immediately.
_FATAL_MARKERS = (
    "invalid api key", "invalid_api_key", "401", "unauthorized", "authentication",
    "permission denied", "403", "model_not_found", "does not exist", "404",
    "decommissioned", "model has been deprecated",
)


def _classify(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(m in msg for m in _FATAL_MARKERS):
        return "fatal"
    if any(m in msg for m in _RETRYABLE_MARKERS):
        return "retryable"
    return "unknown"


# ── targets ────────────────────────────────────────────────────────────────────

@dataclass
class Target:
    label: str
    provider: str          # "groq" | "gemini"
    model: str
    api_key: str
    _clients: dict = field(default_factory=dict, repr=False)

    def client(self, temperature: float = 0.0, max_tokens: int | None = None):
        """
        One cached client per (temperature, max_tokens).

        Sampling parameters are set at construction rather than passed through
        `.invoke(**kwargs)`: which per-call kwargs a LangChain chat model accepts
        varies by provider and version, and a silently-ignored temperature is a
        bug you only notice in the output quality.
        """
        key = (temperature, max_tokens)
        if key in self._clients:
            return self._clients[key]

        if self.provider == "groq":
            from langchain_groq import ChatGroq
            client = ChatGroq(
                api_key=self.api_key,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=0,      # retries are this module's job, not the SDK's
                timeout=60,
            )
        elif self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            client = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
                max_retries=0,
                timeout=60,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        self._clients[key] = client
        return client


def _build_chain(tier: str, feature: str = "") -> list[Target]:
    """
    Assemble the fallback ladder, skipping targets with no credentials.

    When `feature` names a stage that owns a dedicated key (translate,
    clarifier, advisor), that key is tried **first** and the shared keys are
    appended behind it. So a busy stage cannot rate-limit the others, but a
    stage whose own key is exhausted still degrades onto the shared pool instead
    of failing outright.
    """
    primary_key = settings.GROQ_API_KEY
    fallback_key = settings.GROQ_FALLBACK_API_KEY
    big = settings.GROQ_PRIMARY_MODEL
    small = settings.GROQ_FAST_MODEL

    # A fallback key identical to the primary buys nothing — it shares the quota.
    if fallback_key and fallback_key == primary_key:
        fallback_key = ""

    order = [(small, "fast"), (big, "quality")] if tier == "fast" else [(big, "quality"), (small, "fast")]

    chain: list[Target] = []

    dedicated = settings.feature_key(feature) if feature else ""
    if dedicated:
        # Translation is mechanical, so it gets its own small model; the other
        # stages keep the tier's model choice.
        model = settings.GROQ_TRANSLATE_MODEL if feature == "translate" else order[0][0]
        chain.append(Target(f"groq-{feature}", "groq", model, dedicated))

    for model, model_tag in order:
        if primary_key:
            chain.append(Target(f"groq-primary/{model_tag}", "groq", model, primary_key))
        if fallback_key:
            chain.append(Target(f"groq-fallback/{model_tag}", "groq", model, fallback_key))

    if settings.GEMINI_API_KEY:
        chain.append(Target("gemini", "gemini", settings.GEMINI_CHAT_MODEL, settings.GEMINI_API_KEY))

    return chain


# ── response cache ─────────────────────────────────────────────────────────────

class _ResponseCache:
    """
    In-process TTL cache. Mirrors what a hosted gateway gives you: identical
    prompts (the eval harness replaying a golden set, a user re-asking) return
    instantly and cost zero quota.
    """

    def __init__(self, ttl: int, enabled: bool):
        self.ttl = ttl
        self.enabled = enabled
        self._store: dict[str, tuple[float, str, str, str]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(messages: list[dict], temperature: float, tier: str) -> str:
        payload = f"{tier}|{temperature}|" + "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str):
        if not self.enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                self.misses += 1
                return None
            stored_at, content, provider, model = entry
            if time.time() - stored_at > self.ttl:
                del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return content, provider, model

    def put(self, key: str, content: str, provider: str, model: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            # Cheap bound: this is a single-process demo cache, not a datastore.
            if len(self._store) > 512:
                oldest = sorted(self._store.items(), key=lambda kv: kv[1][0])[:128]
                for k, _ in oldest:
                    self._store.pop(k, None)
            self._store[key] = (time.time(), content, provider, model)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "enabled": self.enabled,
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


# ── response ───────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    target_label: str
    cached: bool = False
    fallback_used: bool = False
    attempts: int = 1
    latency_ms: int = 0

    def badge(self) -> str:
        if self.cached:
            return "Cache hit"
        return f"{self.target_label} · {self.model}"


# ── router ─────────────────────────────────────────────────────────────────────

class LLMRouter:
    def __init__(self):
        self._chains: dict[str, list[Target]] = {}
        self._lock = threading.Lock()
        self.cache = _ResponseCache(settings.LLM_CACHE_TTL, settings.LLM_CACHE_ENABLED)

    def chain(self, tier: str, feature: str = "") -> list[Target]:
        # Cached per (tier, feature) so a stage with its own key gets its own
        # ladder while everything else shares one.
        cache_key = f"{tier}|{settings.feature_key(feature) and feature}"
        with self._lock:
            if cache_key not in self._chains:
                built = _build_chain(tier, feature)
                if not built:
                    raise AllTargetsFailed(
                        "No LLM credentials configured. Set GROQ_API_KEY (free at "
                        "https://console.groq.com/keys) or GEMINI_API_KEY in your .env."
                    )
                self._chains[cache_key] = built
                logfire.info(
                    "LLM chain built",
                    tier=tier, feature=feature or "shared",
                    targets=[f"{t.label}:{t.model}" for t in built],
                )
            return self._chains[cache_key]

    @staticmethod
    def _to_messages(prompt: str | list[dict]) -> list[dict]:
        return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

    @staticmethod
    def _to_langchain(messages: list[dict]):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        mapping = {"system": SystemMessage, "assistant": AIMessage, "ai": AIMessage}
        return [mapping.get(m["role"], HumanMessage)(content=m["content"]) for m in messages]

    def invoke(
        self,
        prompt: str | list[dict],
        *,
        tier: str = "quality",
        temperature: float = 0.1,
        max_tokens: int | None = None,
        feature: str = "rag",
    ) -> LLMResponse:
        """Run a completion through the fallback chain. Raises AllTargetsFailed only if every target is dead."""
        messages = self._to_messages(prompt)
        # Feature is part of the cache key: the same text sent to the translator
        # and to the advisor must not share a cached reply.
        cache_key = self.cache.key(messages, temperature, f"{tier}|{feature}")

        hit = self.cache.get(cache_key)
        if hit:
            content, provider, model = hit
            logfire.info("Gateway cache hit", feature=feature, tier=tier, model=model)
            return LLMResponse(
                content=content, provider=provider, model=model,
                target_label="cache", cached=True, latency_ms=0,
            )

        chain = self.chain(tier, feature)
        lc_messages = self._to_langchain(messages)
        errors: list[str] = []
        started = time.time()
        attempts = 0

        with logfire.span("LLM call", feature=feature, tier=tier, targets=len(chain)):
            for position, target in enumerate(chain):
                for attempt in range(1, 3):  # 2 tries per target, then move on
                    attempts += 1
                    try:
                        client = target.client(temperature=temperature, max_tokens=max_tokens)
                        result = client.invoke(lc_messages)
                        content = (result.content or "").strip()

                        if not content:
                            raise ValueError("model returned empty content")

                        latency = int((time.time() - started) * 1000)
                        self.cache.put(cache_key, content, target.provider, target.model)

                        if position > 0:
                            logfire.warning(
                                "Answered by fallback target '{label}' after {n} failed attempt(s)",
                                label=target.label, n=attempts - 1, feature=feature,
                            )
                        else:
                            logfire.info(
                                "LLM ok", target=target.label, model=target.model,
                                latency_ms=latency, feature=feature,
                            )

                        return LLMResponse(
                            content=content,
                            provider=target.provider,
                            model=target.model,
                            target_label=target.label,
                            fallback_used=position > 0,
                            attempts=attempts,
                            latency_ms=latency,
                        )

                    except Exception as exc:
                        kind = _classify(exc)
                        errors.append(f"{target.label}({target.model}): {kind}: {str(exc)[:160]}")

                        if kind == "fatal":
                            logfire.warning(
                                "Target '{label}' is unusable ({err}) — skipping to next",
                                label=target.label, err=str(exc)[:160],
                            )
                            break  # next target, no point retrying

                        if attempt == 1 and kind == "retryable":
                            wait = 1.5 + random.uniform(0, 1.0)
                            logfire.warning(
                                "Target '{label}' rate-limited — retrying in {wait}s",
                                label=target.label, wait=round(wait, 1),
                            )
                            time.sleep(wait)
                            continue

                        logfire.warning(
                            "Target '{label}' failed ({err}) — falling back",
                            label=target.label, err=str(exc)[:160],
                        )
                        break  # next target

            logfire.error("All LLM targets exhausted", feature=feature, errors=errors)
            raise AllTargetsFailed(
                "Every LLM target failed. Tried:\n  " + "\n  ".join(errors)
            )

    def get_chat_model(self, tier: str = "fast"):
        """
        Raw LangChain chat model for libraries that insist on owning the call
        (NeMo Guardrails). No fallback protection — first target only.
        """
        return self.chain(tier)[0].client(temperature=0.0)

    def stats(self) -> dict:
        return {
            "cache": self.cache.stats(),
            "chains": {tier: [t.label for t in chain] for tier, chain in self._chains.items()},
        }


router = LLMRouter()
