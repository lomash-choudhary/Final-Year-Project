# 09 · LLM Gateway

You asked what Portkey is and whether you need it. Short answer: **no, and this project already
does the job**. Here is what the thing actually is, so the decision is informed rather than
inherited.

---

## What an "LLM gateway" is

A gateway sits between your application and the model providers, and owns the concerns that are
identical for every LLM call:

| Concern | What it means |
|---|---|
| **Routing** | One interface in front of several providers, so app code does not branch per vendor |
| **Fallback** | Provider A returns 429 → automatically try provider B |
| **Retries** | Transient 5xx and rate limits get retried with exponential backoff |
| **Caching** | Identical prompts return instantly without spending tokens |
| **Observability** | Latency, token counts and error rates per call, in one place |
| **Key management** | Rotate keys and budgets without touching code |

Without one, every call site grows its own try/except ladder and they drift apart.

---

## What Portkey is

A **hosted** gateway. You send requests to Portkey's endpoint, Portkey forwards them to the actual
provider, and you configure routing declaratively:

```python
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache":    {"mode": "simple"},
    "retry":    {"attempts": 2, "on_status_codes": [429, 503]},
    "targets": [
        {"override_params": {"model": "@rag/llama-3.3-70b-versatile"}},
        {"override_params": {"model": "@brag/llama-3.1-8b-instant"}},
    ],
}
```

Because it is a proxy exposing an OpenAI-compatible API, you talk to it with `ChatOpenAI` pointed
at `PORTKEY_GATEWAY_URL` — not with `ChatGroq`, which is hardwired to Groq's own endpoint and
cannot route through a proxy. The `@slug/model` syntax is Portkey-specific: the slug names a
provider configuration you set up in their dashboard.

### What it costs you

- A Portkey account, plus provider slugs configured in their dashboard
- A third-party service in the request path of every LLM call
- Semantic caching only on paid tiers; the free tier gets exact-match caching
- Another dashboard to check when something breaks

### When it is genuinely worth it

- A team sharing keys, where per-user budgets and central rotation matter
- Many providers with genuinely complex routing rules
- You want production LLM analytics without building them

None of those apply to a single-developer final-year project on free tiers.

---

## What this project does instead

`app/llm/router.py` — same behaviour, in-process, no account.

| Feature | Implementation |
|---|---|
| Fallback | Ordered target chain, tried in sequence |
| Retries | 2 attempts per target, exponential backoff with jitter |
| Error classification | `fatal` (bad key, dead model) skips the target immediately; `retryable` (429, 5xx) retries first |
| Caching | In-process TTL cache keyed by prompt + temperature + tier |
| Observability | Logfire spans on every call, with target, model, latency and fallback flag |
| Tiering | `quality` (70B first) and `fast` (8B first) chains |

### The ladder

```
quality tier                             fast tier
1. GROQ_API_KEY          · 70B           1. GROQ_API_KEY          · 8B
2. GROQ_FALLBACK_API_KEY · 70B           2. GROQ_FALLBACK_API_KEY · 8B
3. GROQ_API_KEY          · 8B            3. GROQ_API_KEY          · 70B
4. GROQ_FALLBACK_API_KEY · 8B            4. GROQ_FALLBACK_API_KEY · 70B
5. Gemini Flash                          5. Gemini Flash
```

Two axes, because Groq rate-limits per **(key, model)** pair. A 429 on the 70B model with your
primary key tells you nothing about the 8B model, or about your second key. The chain exploits both
before giving up.

Targets with no credentials are skipped at chain-build time. A `GROQ_FALLBACK_API_KEY` identical to
`GROQ_API_KEY` is detected and dropped — it shares the same quota, so it would only add latency.

### Why the fast tier exists

The planner and grader run on **every** query. Neither needs a 70B model — one classifies intent,
the other answers a yes/no question. Routing them to the 8B model is what keeps the good quota
available for the answer itself.

### Degradation is never silent

```python
LLMResponse(..., fallback_used=True, target_label="groq-fallback/quality")
```

This flows through the agent's `llm_meta` into the API response and is rendered in the UI as
`⚠︎ fallback`. A system that quietly gets worse looks healthy right up until someone reads the
output — surfacing it is what makes the degradation observable.

---

## If you do want Portkey later

1. Sign up, create provider configurations, note the slugs
2. `pip install portkey-ai`
3. `ENABLE_PORTKEY=true` and `PORTKEY_API_KEY=...` in `.env`
4. Add a Portkey target at the head of `_build_chain()` in `router.py`, constructed with
   `ChatOpenAI(base_url=PORTKEY_GATEWAY_URL, default_headers=createHeaders(...))`

Nothing else in the codebase changes — every call site goes through `router.invoke()`, which is the
point of having a gateway abstraction in the first place. The `ENABLE_PORTKEY` and
`PORTKEY_API_KEY` variables already exist in config for exactly this.

---

## Reading the gateway's state

```bash
curl -s localhost:8000/stats | python -m json.tool
```

```json
{
  "llm_gateway": {
    "cache":  { "enabled": true, "entries": 12, "hits": 5, "hit_rate": 0.294 },
    "chains": { "fast":    ["groq-primary/fast", "groq-fallback/fast", ...],
                "quality": ["groq-primary/quality", "groq-fallback/quality", ...] }
  }
}
```

If `chains` is shorter than you expect, a key is missing from `.env` — or your fallback key is a
duplicate of the primary.
