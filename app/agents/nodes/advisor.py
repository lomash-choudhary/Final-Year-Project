"""
Advisor node — turns research passages into an answer a farmer can act on.

The responder node writes for a researcher: precise, cited, hedged where the
literature is uncertain. That is exactly wrong for someone standing in a shed at
6am with a sick animal. They need one thing: **can I handle this myself, or do I
call the vet?**

So this node produces a fixed, scannable shape:

    What it looks like        one or two plain sentences
    What to do now            numbered steps, or the reason to call a vet
    Watch for                 signs that mean the situation got worse

and a machine-readable `care_level` so the UI can badge urgency without parsing
prose.

Safety posture
--------------
This errs toward the vet, deliberately. A false "see a vet" costs a consultation
fee. A false "treat at home" can cost the animal. The red-flag list below forces
`vet_now` regardless of what the retrieved passages say, because a language
model reasoning over veterinary text is not a diagnostic instrument.

No citation markers. `[1]`/`[2]` in an answer with no sources panel is noise —
and sources are deliberately hidden from consumers. Set
SHOW_CITATIONS_IN_ADVICE=true to put them back.
"""

from __future__ import annotations

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.llm import AllTargetsFailed, router

VALID_CARE_LEVELS = ("home_care", "vet_soon", "vet_now", "info")

_PROMPT = """You are an experienced livestock advisor helping a small farmer in India. You are \
talking to the farmer directly.

REFERENCE MATERIAL FROM VETERINARY RESEARCH:
{context}

CONVERSATION SO FAR:
{history}

FARMER'S PROBLEM:
"{question}"

Write a short, practical answer. Ground it in the reference material above where that material is \
relevant; where it is not, rely on standard, widely accepted livestock husbandry practice and stay \
conservative.

Decide ONE care level:
- home_care : mild and clearly manageable at home
- vet_soon  : needs a vet, but within a day or two
- vet_now   : urgent, call a vet immediately

Choose vet_now whenever any of these are present, no matter what else you conclude:
severe or bloody diarrhoea; blood in milk or urine; high fever; not eating for more than 48 hours; \
laboured breathing; unable to stand; sudden collapse; bloated hard abdomen; difficult calving; \
retained placenta beyond 12 hours; convulsions; suspected poisoning; a swelling that is spreading fast.

FORMAT — follow exactly, no headings other than these:

**What this looks like**
One or two plain sentences on the most likely explanation. Say plainly if it cannot be narrowed down.

**What to do now**
If home_care: numbered steps. Give practical quantities where you can (feed, water, jaggery, \
electrolytes, warm/dry shelter, isolation from the herd).
If vet_soon or vet_now: say clearly that a vet is needed and why, then give only what is safe to do \
while waiting.

**Watch for**
2-3 warning signs that mean call the vet immediately.

Rules:
- Plain language. No medical jargon, no Latin names, no research citations, no passage numbers.
- Never name a prescription medicine, antibiotic, or injectable dose. Those are a vet's decision.
- Do not invent a diagnosis you are not confident about.
- Keep the whole answer under 200 words.

End your reply with exactly this line and nothing after it:
CARE_LEVEL: <home_care or vet_soon or vet_now>"""

_DISCLAIMER = (
    "\n\n---\n*This is general guidance based on veterinary literature, not a diagnosis. "
    "If your animal gets worse, contact a qualified veterinarian.*"
)

_FALLBACK_ANSWER = (
    "**What this looks like**\n"
    "I could not work out a reliable answer for this from what I have.\n\n"
    "**What to do now**\n"
    "Please contact your local veterinarian or the nearest veterinary hospital and describe the "
    "signs you are seeing.\n\n"
    "**Watch for**\n"
    "Any animal that stops eating for more than a day, cannot stand, or is breathing with "
    "difficulty needs a vet immediately."
)


def _format_history(messages: list[dict], limit: int = 6) -> str:
    prior = messages[:-1][-limit:]
    if not prior:
        return "(no earlier turns)"
    return "\n".join(
        f"{'Farmer' if m.get('role') == 'user' else 'Assistant'}: {str(m.get('content', ''))[:600]}"
        for m in prior
    )


def _build_context(documents: list[dict], budget: int) -> tuple[str, int]:
    """Unnumbered passages — the model must not cite them, so it must not see labels."""
    blocks: list[str] = []
    used = 0
    for doc in documents:
        block = doc.get("content", "")
        if not block.strip():
            continue
        if used + len(block) > budget and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks), len(blocks)


def _extract_care_level(text: str) -> tuple[str, str]:
    """Pull the trailing CARE_LEVEL line off the answer. Returns (clean_answer, level)."""
    level = "vet_soon"  # conservative default when the model omits the line
    lines = text.strip().splitlines()

    for index in range(len(lines) - 1, -1, -1):
        candidate = lines[index].strip()
        if candidate.upper().startswith("CARE_LEVEL:"):
            value = candidate.split(":", 1)[1].strip().lower().strip("*` ")
            if value in VALID_CARE_LEVELS:
                level = value
            lines.pop(index)
            break

    return "\n".join(lines).strip(), level


def advise_node(state: AgentState) -> dict:
    question = state.get("query_en") or state.get("original_query", "")
    messages = state.get("messages", [])
    documents = state.get("documents", [])

    context, passages_used = _build_context(documents, settings.MAX_CONTEXT_CHARS)
    if not context:
        # Husbandry advice does not strictly require the corpus, so an empty
        # retrieval is not a dead end here — unlike a research question, where
        # answering without evidence would be a hallucination.
        context = "(no directly relevant research passages were found for this problem)"

    with logfire.span("Advisor", passages=passages_used, question=question[:120]):
        try:
            response = router.invoke(
                _PROMPT.format(
                    context=context,
                    history=_format_history(messages),
                    question=question,
                ),
                tier="quality",
                temperature=0.2,
                feature="advisor",
            )
            answer, care_level = _extract_care_level(response.content)
            meta = {
                "target": response.target_label,
                "model": response.model,
                "provider": response.provider,
                "cached": response.cached,
                "fallback_used": response.fallback_used,
                "latency_ms": response.latency_ms,
                "passages_used": passages_used,
            }
        except AllTargetsFailed as exc:
            logfire.error("Advisor failed: {err}", err=str(exc)[:300])
            answer, care_level = _FALLBACK_ANSWER, "vet_soon"
            meta = {"target": "none", "error": str(exc)[:300]}

        if settings.SHOW_CITATIONS_IN_ADVICE and documents:
            sources = sorted({d["source"] for d in documents})
            answer += "\n\n*Based on: " + ", ".join(sources[:3]) + "*"

        answer += _DISCLAIMER

        logfire.info("Advice generated", care_level=care_level, passages=passages_used)

        return {
            "final_answer": answer,
            "care_level": care_level,
            "status": f"Advice generated ({care_level})",
            "plan": state.get("plan", []) + [f"Advisor: care level = {care_level}"],
            "messages": [{"role": "assistant", "content": answer}],
            "llm_meta": meta,
        }
