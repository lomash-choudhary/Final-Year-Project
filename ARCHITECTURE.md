# Architecture

Three views of the same system, from most detailed to most compact.

---

## 1. Query path — what happens to a question

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant UI as Streamlit UI
    participant API as FastAPI /query
    participant G as Guardrails
    participant P as Planner
    participant R as Retriever
    participant Q as Qdrant
    participant F as FlashRank
    participant GR as Grader
    participant RS as Responder
    participant GW as LLM Gateway

    U->>UI: "prevalence of theileriosis in India?"
    UI->>API: POST /query {q, thread_id}
    API->>G: fast rails (regex, 0 API calls)

    alt rail fires
        G-->>API: blocked + canned response
        API-->>UI: answer, 0 model calls spent
    else passes
        G-->>API: pass
        API->>P: invoke graph (thread state restored)
        P->>GW: classify intent + rewrite query (fast tier)
        GW-->>P: "RESEARCH" + standalone query

        P->>R: search_query
        R->>Q: embed query, top-20 cosine
        Q-->>R: 20 candidates with page payloads
        R->>F: rerank 20 against query
        F-->>R: top 5 cross-encoder ranked

        R->>GR: documents
        alt top score high
            GR-->>RS: sufficient (no model call)
        else ambiguous
            GR->>GW: grade context (fast tier)
            GW-->>GR: NO + better query
            GR->>R: re-search (once)
            R-->>GR: new documents
            GR-->>RS: sufficient
        end

        RS->>GW: synthesise with numbered context (quality tier)
        GW-->>RS: grounded answer with [n] citations
        RS-->>API: answer + plan + sources + gateway metadata
        API-->>UI: full trace
        UI-->>U: answer, reasoning steps, source passages with page numbers
    end
```

---

## 2. Ingestion path — what happens to a PDF

```mermaid
flowchart TD
    A["DATA/*.pdf"] --> B{"Manifest:<br/>bytes changed?"}
    B -->|no| SKIP["Skip — zero API cost"]
    B -->|yes| C["Loader cascade"]

    C --> C1["Tier 1 · pypdf"]
    C1 -->|blank pages| C2["Tier 2 · pdfplumber<br/>only the blank pages"]
    C2 -->|still blank| C3["Tier 3 · PyMuPDF"]
    C3 -->|still blank| WARN["Report as image-only<br/>needs OCR"]

    C1 --> D["Pages, in order"]
    C2 --> D
    C3 --> D

    D --> E["Cleaning<br/>NFKC ligatures · de-hyphenation<br/>soft-wrap join · running-head removal"]

    E --> F{"Chunker cascade"}
    F --> F1["Tier 1 · recursive splitter"]
    F1 -->|validation fails| F2["Tier 2 · paragraph packing"]
    F2 -->|validation fails| F3["Tier 3 · sliding window"]

    F1 --> G["Chunks + page attribution<br/>via offset mapping"]
    F2 --> G
    F3 --> G

    G --> H["processed_data/*.json<br/>debugging artefact"]
    G --> I{"Embedding cache<br/>sha256 hit?"}
    I -->|hit| K["Reuse vector — 0 API calls"]
    I -->|miss| J["Gemini embeddings<br/>rate-limited · retried · batch-split"]
    J -->|unreachable at init| JL["Local sentence-transformers"]

    J --> K
    JL --> K
    K --> L["Delete existing points for this source"]
    L --> M["Upsert with deterministic IDs<br/>uuid5(source:chunk_index)"]
    M --> N[("Qdrant")]
    M --> O["Manifest checkpoint<br/>after every file"]

    classDef skip fill:#6B7280,stroke:#374151,color:#fff
    classDef warn fill:#DC2626,stroke:#991B1B,color:#fff
    classDef good fill:#059669,stroke:#065F46,color:#fff
    class SKIP,K skip
    class WARN warn
    class N,O good
```

---

## 3. LLM gateway — the fallback ladder

```mermaid
flowchart LR
    REQ["invoke(prompt, tier)"] --> CACHE{"Response cache<br/>TTL 900s"}
    CACHE -->|hit| DONE["Return · 0 tokens"]
    CACHE -->|miss| T1

    T1["Groq key 1<br/>llama-3.3-70b"] -->|429 / 5xx| T1R["Retry once<br/>backoff + jitter"]
    T1R -->|still failing| T2
    T1 -->|401 / 404| T2["Groq key 2<br/>llama-3.3-70b"]
    T2 -->|fails| T3["Groq key 1<br/>llama-3.1-8b"]
    T3 -->|fails| T4["Groq key 2<br/>llama-3.1-8b"]
    T4 -->|fails| T5["Gemini Flash"]
    T5 -->|fails| ERR["AllTargetsFailed<br/>surfaced to the user"]

    T1 -->|ok| DONE
    T2 -->|ok| DEG["Return · fallback_used = true"]
    T3 -->|ok| DEG
    T4 -->|ok| DEG
    T5 -->|ok| DEG

    DEG --> UI["Shown in the UI as ⚠︎ fallback —<br/>degradation is never silent"]

    classDef ok   fill:#059669,stroke:#065F46,color:#fff
    classDef warn fill:#D97706,stroke:#92400E,color:#fff
    classDef bad  fill:#DC2626,stroke:#991B1B,color:#fff
    class DONE ok
    class DEG,UI warn
    class ERR bad
```

The `fast` tier reverses the model order (8B first). Planner and grader calls run on every query
and do not need a 70B model — spending the good quota there is what exhausts it before the answer
is even generated.

---

## 4. Agent state machine

```mermaid
stateDiagram-v2
    [*] --> Planner

    Planner --> Responder: intent = conversational
    Planner --> Retriever: intent = research

    Retriever --> Grader

    Grader --> Responder: context sufficient
    Grader --> Retriever: context weak AND refinements < MAX_REFINEMENTS

    Responder --> [*]

    note right of Grader
        The cycle is the point.
        Without it a bad first query
        produces a confident answer
        from irrelevant passages.
        Bounded by MAX_REFINEMENTS.
    end note

    note right of Planner
        Also rewrites the query into a
        standalone form. Vector search
        has no memory: "and in buffalo?"
        embeds to nothing useful.
    end note
```

---

## 5. Compact view

```mermaid
graph TB
    A["1 · Streamlit<br/>chat + eval dashboard"]
    B["2 · FastAPI + two-tier guardrails"]
    C["3 · LangGraph agent<br/>planner → retriever → grader ⟲ → responder"]
    D["4 · Qdrant + FlashRank<br/>page-tagged payloads"]
    E["5 · LLM gateway<br/>Groq ×2 keys ×2 models → Gemini"]
    F["6 · Ingestion<br/>3-tier loaders · cleaning · 3-tier chunking · cached embeddings"]
    G["7 · Evals<br/>RAGAS + zero-cost metrics + guardrail matrix"]
    H["8 · Observability<br/>Logfire · LangSmith"]

    A --> B --> C
    C <--> D
    C --> E
    F --> D
    A -.-> G
    G -.-> B
    B -.-> H
    C -.-> H
    F -.-> H

    classDef ui     fill:#2563EB,stroke:#1E40AF,color:#fff
    classDef safety fill:#DC2626,stroke:#991B1B,color:#fff
    classDef agent  fill:#7C3AED,stroke:#5B21B6,color:#fff
    classDef db     fill:#059669,stroke:#065F46,color:#fff
    classDef llm    fill:#D97706,stroke:#92400E,color:#fff
    classDef ingest fill:#4F46E5,stroke:#3730A3,color:#fff
    classDef evals  fill:#DB2777,stroke:#9D174D,color:#fff
    classDef obs    fill:#0D9488,stroke:#0F766E,color:#fff

    class A ui
    class B safety
    class C agent
    class D db
    class E llm
    class F ingest
    class G evals
    class H obs
```

---

## Design decisions worth defending

| Decision | Rationale |
|---|---|
| Vector dimension is **probed**, never hardcoded | Model availability differs per Google account and dimensions change between versions. A hardcoded 3072 is how a collection silently rejects every upsert. |
| Embedding backend is **locked after init** | Qdrant fixes the dimension at collection creation. Switching from 3072-dim Gemini to 768-dim local mid-run would corrupt the index, so a mid-run failure raises instead. |
| Deterministic point IDs + delete-before-upsert | Re-ingesting an edited document leaves no orphaned chunks from its longer previous version. |
| PDF fallbacks apply **per page** | A 124-page journal issue with four bad pages costs three page-opens, not a full re-parse — and recovered pages land back in position, so page citations stay correct. |
| Chunk strategies are **validated** before acceptance | A splitter that silently returns one 200 KB chunk embeds fine, retrieves for everything, and blows the context budget. Failing over is better than accepting it. |
| Guardrails are deterministic first | An LLM-based rail spends a model call on every greeting it exists to reject cheaply. On a free tier that is backwards. |
| Off-topic blocking requires **explicit** off-domain signals | "What did the study find?" contains no veterinary term but is a valid follow-up. False positives destroy trust faster than false negatives waste quota. |
| "No evidence" is answered **without an LLM** | A model asked to admit ignorance will sometimes answer from parametric memory instead — the exact failure a grounded system exists to prevent. |
| `plan` is not a LangGraph reducer | MemorySaver persists state per thread, so an accumulating plan would replay every previous turn's reasoning. Nodes concatenate explicitly and the planner resets it. |
| Evals hit the **live API**, not the graph | What gets measured is the system as deployed — guardrails, gateway fallback and the correction loop included. |
