"""
Advisor node — turns research passages into an answer a farmer can act on.

The responder node writes for a researcher: precise, cited, hedged where the
literature is uncertain. That is exactly wrong for someone standing in a shed at
6am with a sick animal. They need one thing: **can I handle this myself, or do I
call the vet?**

So this node produces a fixed, short, scannable shape:

    Likely cause              one plain sentence
    Medicine                  what the retrieved passages name for this problem, or
                              "none found — ask a vet"; never invented
    What to do                at most three steps
    Call the vet if           at most two warning signs

and a machine-readable `care_level` so the UI can badge urgency without parsing
prose. On `vet_now` the whole answer collapses to "contact a vet now" plus at
most two things that are safe to do while waiting — a farmer facing an
emergency needs one instruction, not a page.

Safety posture
--------------
This errs toward the vet, deliberately. A false "see a vet" costs a consultation
fee. A false "treat at home" can cost the animal. The red-flag list below forces
`vet_now` regardless of what the retrieved passages say, because a language
model reasoning over veterinary text is not a diagnostic instrument.

Medicines are named **only when the reference material names them** for the
problem at hand — the corpus, not the model's memory, is the source. Antibiotics,
injectables and other prescription drugs are named but never dosed. No
prescription notes are added to the answer — the frontend carries the
disclaimer (owner decision 2026-10-02). Strengths are given only for farm-level
remedies (footbath, dressing, spray) and only when the passage states them.

The prompt deliberately has no persona and no disclaimer: every prompt token is
paid on every farmer turn, and a role-play line ("you are a livestock
advisor") added nothing the format rules did not already enforce.

No citation markers. `[1]`/`[2]` in an answer with no sources panel is noise —
and sources are deliberately hidden from consumers. Set
SHOW_CITATIONS_IN_ADVICE=true to put them back.
"""

from __future__ import annotations

import re
import unicodedata

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.llm import AllTargetsFailed, router

VALID_CARE_LEVELS = ("home_care", "vet_soon", "vet_now", "info")

_PROMPT = """Answer a farmer's question about a sick cow or buffalo, using the passages below.

PASSAGES:
{context}

CONVERSATION SO FAR:
{history}

QUESTION:
"{question}"

Care level:
- vet_now if any of: bloody diarrhoea, blood in milk or urine, high fever, off feed over 48 hours, \
laboured breathing, cannot stand, collapse, hard bloated belly, difficult calving, placenta retained \
over 12 hours, convulsions, suspected poisoning, fast-spreading swelling.
- otherwise vet_soon if it needs a vet within a day or two, else home_care.

Medicine:
- Only what the passages name for this problem. Name the exact substance (e.g. "copper sulfate \
dressing", "oxytetracycline spray"), never a vague "antiseptic". Up to 3 options.
- Generic names only, never a brand, not even in brackets: Naxcel = ceftiofur, Terramycin = \
oxytetracycline, Tramisol = levamisole.
- Only established treatments. Skip products a study was only testing as experimental.
- Antibiotics, injections and udder tubes: name only, no dose, and never tell the farmer to give them \
in "What to do". Do not add prescription notes or disclaimers.
- Footbath, dressing, wash or spray: give the strength only if the passages state it.
- If the question asks what to apply, list what to apply first. If the passages say the problem also \
needs an injection to cure it, name that too.
- If the passages name nothing for this problem, write: Ask your vet.
- Ignore outdated remedies from old texts (arsenic, mercury, turpentine or kerosene drenches, \
inflating the udder, bleeding).

Format. If vet_now, reply only:
**Contact a vet now.** <one sentence why>
**While you wait**
- <up to 2 safe steps>

Otherwise reply only:
**Likely cause**
<one sentence>
**Medicine**
<one or two lines>
**What to do**
- <up to 3 steps, each under 15 words>
**Call the vet if**
- <up to 2 signs>

Plain words: "put on the wound", not "topical"; "injection", not "systemic". Never mention passages, studies, sources, appendices, tables or pages. Under 100 words.
Last line, exactly: CARE_LEVEL: <home_care|vet_soon|vet_now>"""

_FALLBACK_ANSWER = (
    "**Contact a vet.** I can't advise on this one safely.\n\n"
    "**While you wait**\n"
    "- Keep the animal in clean, dry shade with fresh water.\n"
    "- Separate it from the rest of the herd."
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


# Passages often say "see Appendix I" or "(Table 2)"; the model copies those into
# advice for someone who has no access to the document. The prompt forbids it,
# this catches the drift.
_DOC_REFERENCE = re.compile(
    r"\s*[(\[]\s*(?:see\s+)?(?:appendix|table|fig(?:ure)?\.?|page|p\.)\s*[\w.-]*\s*[)\]]",
    re.IGNORECASE,
)


def _strip_doc_references(text: str) -> str:
    return _DOC_REFERENCE.sub("", text)


# ── medicine enforcement ──────────────────────────────────────────────────────
# The prompt asks for generic names and no outdated remedies. gpt-oss-120b still wrote "levamisole (Tramisol)",
# listed clenbuterol (banned in food animals) with no vet note, and suggested
# phenothiazine. So the rules the farmer's safety rests on are applied here, in
# code, after the model — the prompt is a request, this is the guarantee.

# Brand names found in the (largely US) corpus → generic name.
_BRANDS = {
    "naxcel": "ceftiofur", "excenel": "ceftiofur", "excede": "ceftiofur",
    "terramycin": "oxytetracycline", "liquamycin": "oxytetracycline",
    "tramisol": "levamisole", "levasole": "levamisole", "albon": "sulfadimethoxine",
    "lincomix": "lincomycin", "lincospectin": "lincomycin-spectinomycin",
    "ls-50": "lincomycin-spectinomycin", "micotil": "tilmicosin", "nuflor": "florfenicol",
    "baytril": "enrofloxacin", "draxxin": "tulathromycin", "banamine": "flunixin",
    "butalex": "buparvaquone", "berenil": "diminazene", "ivomec": "ivermectin",
    "panacur": "fenbendazole", "safe-guard": "fenbendazole", "hoof pro plus": "copper sulfate spray",
}
_BRAND_ALT = "|".join(re.escape(b) for b in sorted(_BRANDS, key=len, reverse=True))
_BRAND_IN_BRACKETS = re.compile(rf"\s*\(\s*(?:{_BRAND_ALT})\b[^)]*\)", re.IGNORECASE)
_BRAND_WORD = re.compile(rf"\b(?:{_BRAND_ALT})\b", re.IGNORECASE)

# Banned in food-producing animals, obsolete, or only experimental in the corpus.
_DISALLOWED = re.compile(
    r"clenbuterol|chloramphenicol|nitrofur\w*|furazolidone|diethylstilb\w*|phenothiazine|"
    r"arsenic\w*|mercur\w*|strychnine|turpentine|kerosene|propolis|stem cell",
    re.IGNORECASE,
)

# The model's own "(vet must prescribe)" style notes are removed: the frontend
# carries the disclaimer.
_VET_TAG = re.compile(r"\s*[(\[]\s*vet[^)\]]*[)\]]|\s*[-–,]\s*vet must prescribe", re.IGNORECASE)
_JARGON = (
    (re.compile(r"\(\W*topical[^)]*\)", re.IGNORECASE), "(put on the wound)"),
    (re.compile(r"\bsystemic\W+(?=injection)", re.IGNORECASE), ""),
    (re.compile(r"\(\W*systemic\W*\)", re.IGNORECASE), "(injection)"),
)


# gpt-oss emits non-breaking hyphens (U+2011), narrow spaces and zero-width
# characters. They render identically but defeat every regex below, so the
# answer is folded to plain characters first.
_INVISIBLE = re.compile("[\u00ad\u200b-\u200d\u2060\ufeff]")
_DASHES = re.compile("[\u2010-\u2014\u2212]")
_SPACES = re.compile("[\u00a0\u2007\u202f]")


def _normalise(text: str) -> str:
    return _SPACES.sub(" ", _DASHES.sub("-", _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))))


def _clean_brands(text: str) -> str:
    text = _BRAND_IN_BRACKETS.sub("", text)
    return _BRAND_WORD.sub(lambda m: _BRANDS[m.group(0).lower()], text)


def _plain(text: str) -> str:
    for pattern, replacement in _JARGON:
        text = pattern.sub(replacement, text)
    return text


def _medicine_items(section: list[str]) -> list[str]:
    """Filter one Medicine section: drop disallowed drugs and the model's vet notes."""
    out: list[str] = []
    for line in section:
        bullet = re.match(r"^\s*(?:[-•*]|\d+[.)])\s*", line)
        prefix = bullet.group(0) if bullet else ""
        items = []
        for item in line[len(prefix):].split(";"):
            item = _VET_TAG.sub("", item).strip(" .")
            if not item or _DISALLOWED.search(item) or item.lower().startswith("ask your vet"):
                continue
            items.append(item)
        if items:
            out.append(prefix + "; ".join(items))
    return out


def _enforce_medicine_rules(answer: str) -> str:
    """Generic names, plain words, no disallowed drugs in **Medicine**."""
    out: list[str] = []
    section: list[str] | None = None  # collecting the Medicine section's lines

    def flush() -> None:
        out.extend(_medicine_items(section) or ["Ask your vet."])
        out.append("")

    for line in _plain(_clean_brands(_normalise(answer))).splitlines():
        stripped = line.strip()
        if stripped.startswith("**"):
            if section is not None:
                flush()
            section = [] if stripped.lower().startswith("**medicine") else None
            out.append(line)
        elif section is not None:
            if stripped:
                section.append(line)
        elif not _DISALLOWED.search(line):
            # Outside the medicine list a disallowed name can only be an instruction to use it.
            out.append(line)

    if section is not None:
        flush()
    return "\n".join(out).strip()


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
            answer = _enforce_medicine_rules(_strip_doc_references(answer))
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

        logfire.info("Advice generated", care_level=care_level, passages=passages_used)

        return {
            "final_answer": answer,
            "care_level": care_level,
            "status": f"Advice generated ({care_level})",
            "plan": state.get("plan", []) + [f"Advisor: care level = {care_level}"],
            "messages": [{"role": "assistant", "content": answer}],
            "llm_meta": meta,
        }
