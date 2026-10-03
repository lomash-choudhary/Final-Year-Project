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
prescription tags or disclaimers of any kind (owner decision 2026-10-03 — the
product will handle disclaimers separately); the model's own are stripped.
Strengths are given only for farm-level remedies (footbath, dressing, spray) and
only when the passage states them. A step that has the farmer inject or dose a
prescription drug is rewritten to send them to the vet.

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

from app.agents.history import format_history
from app.agents.state import AgentState
from app.config import settings
from app.llm import AllTargetsFailed, router

VALID_CARE_LEVELS = ("home_care", "vet_soon", "vet_now", "info")

_PROMPT = """Answer a farmer's question about their cow or buffalo using only the passages below. \
The farmer may not read well: short sentences, everyday words, no medical terms.

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
- otherwise vet_soon if it needs a vet within a day or two, else home_care (prevention and \
hygiene questions are home_care).

Medicine:
- Only what the passages name for this same problem. A drug they give for a different disease does \
not count.
- Exact generic names, up to 3, never brands. Only established treatments, not products a study \
was only testing.
- Never a dose for antibiotics, injections, udder tubes, pain killers or tick-fever drugs. No \
prescription notes or disclaimers.
- Footbath, dressing, teat dip, wash or spray: give the strength only if the passages state it. \
A product described without a chemical name ("a proven germicidal teat dip") still counts.
- If asked what to apply, list that first; if the passages say the problem also needs antibiotic \
treatment ("systemic" means injection), name those drugs too.
- Passages marked [OLD BOOK] are outdated: use them only for cleaning and hygiene, never for \
medicine or surgery.
- If nothing fits, write "Ask your vet." under Medicine.

What to do: only steps the passages support, or basic care (shade, water, clean, dry, keep apart). \
The vet gives prescription medicines and decides doses, never the farmer. No cutting, burning, \
tubes, forcing anything into the mouth, or laying a cow flat on her side.

Format. If vet_now, reply only:
**Contact a vet now.** <what is likely wrong, judged only from the signs the farmer gave, in plain words \
(e.g. "a bad udder infection", "bloat: gas swelling the left side"); if unclear, say the signs are serious>
**While you wait**
- <up to 2 safe steps>

Otherwise reply only:
**Likely cause**
<one short sentence naming the problem; "Not clear from your question." only if none is named>
**Medicine**
<one or two lines>
**What to do**
- <up to 3 steps, each under 12 words>
**Call the vet if**
- <up to 2 signs>

Never mention passages, studies or pages. Under 100 words.
Last line, exactly: CARE_LEVEL: <home_care|vet_soon|vet_now>"""

# The red flags from the prompt, checked in code on the farmer's own words. The prompt alone
# let "udder swollen, clots, high fever, won't eat" through as vet_soon in the eval, and an
# under-triaged emergency is the one mistake this node must not make.
_RED_FLAGS = re.compile(
    r"blood\w*\s+(?:in\s+(?:the\s+|her\s+|his\s+)?)?(?:milk|urine|dung|stool|diarrh\w*)|bloody\s+(?:milk|urine|dung|diarrh\w*)|"
    r"red\s+urine|urine[^.]{0,15}\bred\b|high\s+fever|"
    r"(?:struggl\w*|difficult\w*|hard|laboured|labored)\s+(?:to\s+)?breath\w*|"
    r"(?:cannot|can't|can not|unable to|not able to)\s+(?:stand|get up|rise)|"
    r"collaps\w*|fell down|convuls\w*|seizure|poison\w*|"
    r"(?:left\s+side|belly|stomach)[^.]{0,30}(?:swollen|hard|bloat)|\bbloat\w*|"
    r"placenta[^.]{0,40}(?:not\s+come\s+out|retained|stuck|hanging)|"
    r"(?:not|n't|nothing)\s+(?:eaten|eating|eat)[^.]{0,25}(?:two|three|four|\d+)\s+days|"
    r"spreading\s+fast|difficult\s+calving|calf\s+(?:is\s+)?stuck",
    re.IGNORECASE,
)

_VET_NOW_ANSWER = (
    "**Contact a vet now.** These signs need a vet today.\n\n"
    "**While you wait**\n"
    "- Keep the animal in a clean, dry, shaded place with fresh water.\n"
    "- Keep it calm and away from the other animals."
)

_FALLBACK_ANSWER = (
    "**Contact a vet.** I can't advise on this one safely.\n\n"
    "**While you wait**\n"
    "- Keep the animal in clean, dry shade with fresh water.\n"
    "- Separate it from the rest of the herd."
)


def _build_context(documents: list[dict], budget: int) -> tuple[str, int]:
    """Unnumbered passages — the model must not cite them, so it must not see labels.

    The one label is [OLD BOOK]: the 1900s notes book is the corpus's only source for
    some problems, and its remedies (carbolic acid, cutting warts off) were given
    as current advice until the model could tell its passages apart.
    """
    blocks: list[str] = []
    used = 0
    historical = settings.historical_sources
    strong_modern = [
        d for d in documents if d.get("source") not in historical and (d.get("score") or 0) >= 0.3
    ]
    if strong_modern:
        # With real evidence present the old book only adds outdated remedies: the label did
        # not stop "carbolated cottonseed oil in the ears" being given for ticks. Only strong
        # modern passages count — for round bald patches the modern ones scored < 0.03 and
        # the old book's ringworm page was the only one that named the problem. One strong
        # modern passage is enough: foot rot had one (0.997) and the old book's three slots
        # supplied a mercury "bichloride footbath".
        documents = [d for d in documents if d.get("source") not in historical]
    for doc in documents:
        block = doc.get("content", "")
        if not block.strip():
            continue
        if doc.get("source") in historical:
            block = "[OLD BOOK] " + block
        if used + len(block) > budget and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks), len(blocks)


def _old_book_only_words(documents: list[dict]) -> set[str]:
    """Words that appear in [OLD BOOK] passages but in no modern one.

    The [OLD BOOK] label alone did not hold: "carbolated cottonseed oil in the ears"
    still came back as tick medicine. A medicine whose name only the 1900s book
    supplies is dropped in code instead.
    """
    historical = settings.historical_sources
    old, modern = set(), set()
    for doc in documents:
        words = set(re.findall(r"[a-z]{6,}", doc.get("content", "").lower()))
        (old if doc.get("source") in historical else modern).update(words)
    return old - modern


# Passages often say "see Appendix I" or "(Table 2)"; the model copies those into
# advice for someone who has no access to the document. The prompt forbids it,
# this catches the drift.
_DOC_REFERENCE = re.compile(
    r"\s*[(\[]\s*(?:see\s+|as\s+(?:per|in)\s+|according\s+to\s+)?(?:the\s+)?"
    r"(?:appendix|table|fig(?:ure)?\.?|page|p\.)\s*[\w.-]*\s*[)\]]"
    r"|,?\s*(?:see|as\s+(?:per|in|described\s+in)|according\s+to)\s+(?:the\s+)?(?:appendix|table)\s*[\w.-]*",
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
    "tramisol": "levamisole", "levasole": "levamisole", "tbz": "thiabendazole", "albon": "sulfadimethoxine",
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
    r"arsenic\w*|mercur\w*|strychnine|turpentine|kerosene|propolis|stem cell|"
    r"carbolic|caustic potash|silver nitrate|cauteri\w*|inflat\w* the udder|udder inflation|"
    r"bichloride|sublimate|sugar of lead|lead acetate|creolin|hot bran|"
    # Research-stage mastitis treatments from vetsci-12's review: the eval caught "Phage K
    # intramammary" named as the medicine despite the established-treatments prompt rule.
    r"phage|secretome|conditioned medium|nano\w*|baicalin|chlorogenic|probiotic\w*|carvacrol|thymol",
    re.IGNORECASE,
)

# No prescription tags or disclaimers (owner decision 2026-10-03). The model still
# adds its own ("vet only", "(needs vet's prescription)"), so they are stripped.
_VET_TAG = re.compile(
    r"\s*[(\[]\s*(?:needs?\s+|no\s+)?(?:a\s+)?(?:vet|prescription)[^)\]]*[)\]]|\s*[-–,]\s*vet must prescribe",
    re.IGNORECASE,
)
# Antibiotics, pain killers, tick-fever drugs, hormones and anything injected or put
# up the teat. "sulfa(?!te)" keeps copper sulfate — a footbath — off the list.
_RX = re.compile(
    r"\b\w*(?:cillin|mycin|micin|floxacin|cycline|fenicol|sulfa(?!te)|sulpha(?!te)|sulfonamide|"
    r"ceftiofur|cephalosporin|cefquinome|cephapirin|trimethoprim|tilmicosin|valnemulin|tiamulin|"
    r"myxin|macrolide|antibiotic|flunixin|meloxicam|ketoprofen|dexamethasone|buparvaquone|diminazene|"
    r"imidocarb|oxytocin|terbutaline|aminophylline|intramammary|udder tube|injection|injectable|"
    r"subcutaneous|intramuscular|intravenous)\w*",
    re.IGNORECASE,
)
# A step that has the farmer give the drug themselves. Steps that send them to the
# vet ("Ask the vet for…") start with other verbs and are left alone.
_SELF_DOSE = re.compile(
    r"^(\s*(?:[-•*]|\d+[.)])?\s*)(?:give|administer|inject|start|apply|use|put|insert|infuse|treat|"
    r"weigh|calculate|repeat|continue|dose)\b",
    re.IGNORECASE,
)
_JARGON = (
    (re.compile(r"\(\W*topical[^)]*\)", re.IGNORECASE), "(put on the wound)"),
    (re.compile(r"\bsystemic\W+(?=injection|antibiotic)", re.IGNORECASE), ""),
    (re.compile(r"\bsystemic\s*:\s*", re.IGNORECASE), "Injection: "),
    (re.compile(r"\(\W*systemic\W*\)", re.IGNORECASE), "(injection)"),
    (re.compile(r"\bsystemic\s+", re.IGNORECASE), ""),
    (re.compile(r"\btopical(?:ly)?\b\s*", re.IGNORECASE), ""),
    (re.compile(r"\blesions\b", re.IGNORECASE), "sores"),
    (re.compile(r"\blesion\b", re.IGNORECASE), "sore"),
    (re.compile(r"\bdebris\b", re.IGNORECASE), "dirt"),
    (re.compile(r"\banthelmintics?\b", re.IGNORECASE), "worm medicine"),
    (re.compile(r"\b(?:becomes?\s+)?non-weight-bearing\b", re.IGNORECASE), "will not stand on the foot"),
    (re.compile(r"\bha?ematuria\b", re.IGNORECASE), "red urine"),
    (re.compile(r"\binterdigital\b", re.IGNORECASE), "between the claws"),
    (re.compile(r"\ban\s+NSAID\b"), "a pain-relief medicine"),
    (re.compile(r"\bNSAIDs?\b"), "pain-relief medicine"),
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
    # "Lesions spread" → "sores spread": re-capitalise a bullet the swap lowered.
    return re.sub(r"(?m)^(\s*[-•*]\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


# One item per medicine, so a dose is cut from the prescription drug alone: "copper
# sulfate 3-5 % or oxytetracycline 5 g/L" once lost the oxytetracycline and tagged
# the footbath. "or IM" stays joined — a route is part of the item before it.
_ITEM_SPLIT = re.compile(r";|\.\s+|,\s+(?=[A-Za-z])|\s+(?:or|and)\s+(?!(?:IM|SC|IV|SQ)\b)")


def _split_items(text: str) -> list[str]:
    """Split on _ITEM_SPLIT, but never inside brackets: "levamisole (bolus, drench, feed)" was
    cut into "levamisole (bolus", "drench", "feed)"."""
    hide = {";": "\x01", ",": "\x02", ".": "\x03"}
    protected, depth = [], 0
    for char in text:
        depth += char in "(["
        depth -= char in ")]"
        protected.append(hide[char] if depth > 0 and char in hide else char)
    items: list[str] = []
    for piece in _ITEM_SPLIT.split("".join(protected)):
        for mark, char in hide.items():
            piece = piece.replace(char, mark)
        piece = re.sub(r"^(?:or|and)\s+", "", piece.strip(), flags=re.IGNORECASE)
        if items and piece and set(re.findall(r"[a-z]+", piece.lower())) <= _GENERIC:
            items[-1] += " or " + piece  # "antiseptic footbath or spray" is one item, not two
        else:
            items.append(piece)
    return items


# Substance names checked against the passages: specific drugs (by name or class suffix) and
# farm chemicals. "Iodine-based teat dip" came back although no passage mentions iodine — the
# model's memory, not the corpus. An unsupported substance is cut from its item; an item left
# with nothing is dropped.
_SUBSTANCE = re.compile(
    r"\b[a-z]*(?:cillin|mycin|micin|myxin|floxacin|cycline|fenicol|conazole|bendazole|"
    r"ceftiofur|cephapirin|cefquinome|tilmicosin|tulathromycin|valnemulin|tiamulin|levamisole|"
    r"ivermectin|flunixin|meloxicam|ketoprofen|buparvaquone|diminazene|imidocarb|oxytocin|"
    r"sulfadimethoxine|trimethoprim|iodine|chlorhexidine|formalin|formaldehyde|permanganate|"
    r"copper sulph?ate|zinc sulph?ate|lime sulfur|hydrogen peroxide|povidone|"
    r"[a-z]*methrin|amitraz|fipronil|coumaphos|diazinon)\b",
    re.IGNORECASE,
)


# A Medicine "item" that is only a timing or an instruction ("soak 5 min", "twice daily") is the
# tail of a split item whose medicine was removed; on its own it means nothing to a farmer.
_FRAGMENT = re.compile(
    r"^(?:soak|repeat|wait|leave|keep|for|then|once|twice|daily|every|\d)\b(?!.*\b(?:spray|footbath|dip|"
    r"ointment|dressing|powder|solution|injection|drench|bolus|tube|wash)\b)",
    re.IGNORECASE,
)

_GENERIC = {"footbath", "spray", "dressing", "injection", "solution", "wash", "apply", "meglumine",
            "medicine", "relief", "antibiotics", "antibiotic", "used", "with", "together"}


def _unsupported_cut(item: str, passages: str) -> str:
    """Remove substance names the passages never mention; "" if nothing specific is left."""
    cut = False
    for match in _SUBSTANCE.finditer(item):
        name = match.group(0).lower()
        if name not in passages and re.sub(r"sulph", "sulf", name) not in passages:
            item = re.sub(re.escape(match.group(0)) + r"(?:[- ]based)?\s*", "", item, flags=re.IGNORECASE)
            cut = True
    if cut:
        item = re.sub(r"\s*\(\s*(?:e\.g\.|such as|like)?[\s,;]*\)", "", item, flags=re.IGNORECASE)
    item = item.strip(" ,;:-")
    if cut:
        outside = re.sub(r"\([^)]*\)", " ", item).lower()
        if not [w for w in re.findall(r"[a-z]{3,}", outside) if w not in _GENERIC]:
            return ""
    return item


def _medicine_items(
    section: list[str], old_only: frozenset[str] = frozenset(), passages: str | None = None,
    dropped: set[str] | None = None,
) -> list[str]:
    """Filter one Medicine section: drop disallowed drugs and the model's notes, undose prescription ones."""
    out: list[str] = []
    for line in section:
        bullet = re.match(r"^\s*(?:[-•*]|\d+[.)])\s*", line)
        prefix = bullet.group(0) if bullet else ""
        items = []
        for item in _split_items(_VET_TAG.sub("", line[len(prefix):])):
            item = item.strip(" .,")
            if passages is not None:
                item = _unsupported_cut(item, passages)
                if not re.search(r"[A-Za-z]{3,}", item):
                    continue
            if _DISALLOWED.search(item) and dropped is not None:
                dropped.update(w for w in re.findall(r"[a-z]{6,}", item.lower()) if w not in _GENERIC)
            if not item or _DISALLOWED.search(item) or item.lower().startswith("ask your vet"):
                continue
            if _FRAGMENT.match(item) and not _SUBSTANCE.search(item):
                continue
            if set(re.findall(r"[a-z]{6,}", item.lower())) & old_only:
                if dropped is not None:
                    dropped.update(set(re.findall(r"[a-z]{6,}", item.lower())) & old_only)
                continue
            if _RX.search(item) and re.search(r"\b(?:give|administer|inject)\b", item, re.IGNORECASE):
                # "if swelling grows give penicillin" — the Medicine line names drugs; giving them
                # is the vet's job. Keep the names only.
                item = ", ".join(dict.fromkeys(m.group(0) for m in _SUBSTANCE.finditer(item))) or item
            if _RX.search(item):
                # No dose for a prescription drug: everything from the first number on
                # ("6.6-11 mg per kg IM daily") is the vet's decision.
                item = re.split(r"\s*[\d≈~]", item, maxsplit=1)[0].strip(" ,.-:")
            items.append(item)
        if items:
            out.append(prefix + "; ".join(items))
    return out


def _vet_step(line: str) -> str:
    """Turn "Give ceftiofur injection daily" into a step that sends the farmer to the vet."""
    match = _SELF_DOSE.match(line)
    if not match:
        return line
    drug = _RX.search(line)
    if drug:
        name = drug.group(0).lower()
        if name.startswith(("inject", "antibiotic", "intramammary", "udder tube")):
            return match.group(1) + "Ask your vet to give the medicine."
        return match.group(1) + f"Ask your vet for {name} and use it as the vet says."
    if re.search(r"\bdos(?:e|age|ing)\b", line, re.IGNORECASE):
        return match.group(1) + "Ask your vet for the right dose."
    return line


def _enforce_medicine_rules(
    answer: str, old_only: set[str] | None = None, passages: str | None = None
) -> str:
    """Generic names, plain words, no doses or tags on prescription drugs, no disallowed drugs,
    no old-book-only remedies, no self-dosing."""
    out: list[str] = []
    section: list[str] | None = None  # collecting the Medicine section's lines
    old_only = frozenset(old_only or ())
    dropped: set[str] = set()  # old-book remedy words removed from Medicine; steps naming them go too

    def flush() -> None:
        out.extend(_medicine_items(section, old_only, passages, dropped) or ["Ask your vet."])
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
        elif not _DISALLOWED.search(line) and not (
            stripped.startswith(("-", "•", "*")) and set(re.findall(r"[a-z]{6,}", line.lower())) & dropped
        ):
            # Outside the medicine list a disallowed or old-book-only name can only be an
            # instruction to use it.
            line = _vet_step(line)
            if not line.strip() or line not in out:  # rewritten steps can collapse to one sentence
                out.append(line)

    if section is not None:
        flush()
    return _fill_empty_steps("\n".join(out).strip())


_SAFE_STEPS = [
    "- Keep the animal in a clean, dry, shaded place with fresh water.",
    "- Keep it calm and away from the other animals.",
]


def _fill_empty_steps(answer: str) -> str:
    """A steps section left with no bullets (all filtered out) gets the two universal safe steps."""
    lines = answer.splitlines()
    out: list[str] = []
    for index, line in enumerate(lines):
        out.append(line)
        if re.match(r"\s*\*\*(?:while you wait|what to do)", line, re.IGNORECASE):
            rest = lines[index + 1:]
            following = next((l for l in rest if l.strip()), "")
            if not following.strip().startswith(("-", "•", "*")) or following.strip().startswith("**"):
                out.extend(_SAFE_STEPS)
    return "\n".join(out)


def _extract_care_level(text: str) -> tuple[str, str]:
    """Pull the trailing CARE_LEVEL line off the answer. Returns (clean_answer, level)."""
    level = "vet_soon"  # conservative default when the model omits the line
    lines = text.strip().splitlines()

    for index in range(len(lines) - 1, -1, -1):
        candidate = lines[index].strip()
        if re.match(r"^\W*CARE\w*[_ ]?LEVEL\W*:", candidate, re.IGNORECASE):
            value = candidate.split(":", 1)[1].strip().lower().strip("*` ")
            if value in VALID_CARE_LEVELS:
                level = value
            lines.pop(index)
            break

    return "\n".join(lines).strip(), level


def advise_node(state: AgentState) -> dict:
    question = state.get("query_en") or state.get("original_query", "")
    documents = state.get("documents", [])

    context, passages_used = _build_context(documents, settings.MAX_CONTEXT_CHARS)
    if _RED_FLAGS.search(question):
        # An emergency answer names no medicine, so the passages add nothing — and with them
        # the model named diseases the signs did not show ("lying down, high fever" became
        # metritis; "swollen left side" became pleuropneumonia). Judge from the signs alone.
        context, passages_used = "(emergency signs: answer from the farmer's signs only)", 0
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
                    history=format_history(state, user_label="Farmer", empty="(no earlier turns)"),
                    question=question,
                ),
                tier="quality",
                temperature=0.2,
                feature="advisor",
            )
            answer, care_level = _extract_care_level(response.content)
            old_only = _old_book_only_words(documents) - set(re.findall(r"[a-z]{6,}", question.lower()))
            passages = re.sub(r"sulph", "sulf", context.lower())
            answer = _enforce_medicine_rules(_strip_doc_references(answer), old_only, passages)
            if _RED_FLAGS.search(question) and care_level != "vet_now":
                logfire.warning("Red flag in question overrode care level {level}", level=care_level)
                care_level, answer = "vet_now", _VET_NOW_ANSWER
            elif care_level == "home_care" and (_RX.search(answer) or _RX.search(question)):
                # A prescription drug, asked about or named, means the vet is involved: the
                # eval's dose question ("how much oxytetracycline to inject") came back home_care.
                care_level = "vet_soon"
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
