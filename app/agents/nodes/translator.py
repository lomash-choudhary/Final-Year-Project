"""
Language nodes — the first and last steps of the graph.

Indian farmers do not type in English. This project's corpus is entirely
English, and so are the embeddings that index it, so a Hindi question has to
become English before retrieval and the answer has to go back before display.

Two nodes:
    translate_in   raw question  -> English  (runs first)
    translate_out  English answer -> user's language (runs last)

Three languages are recognised:
    en        plain English                     -> no model call at all
    hi        Devanagari ("मेरी गाय...")          -> translate
    hi-latn   Roman-script Hindi ("meri gaay...") -> translate

That last one matters more than it looks. Most Indian users type Hindi on an
English keyboard, so a detector that only checks for Devanagari would classify
the majority of real Hindi input as English and feed nonsense to the retriever.

Detection is free: a Unicode range check plus a marker-word list. A model is
only called when the text is actually not English, so English users never pay
for the translation feature.
"""

from __future__ import annotations

import re

import logfire

from app.agents.state import AgentState
from app.config import settings
from app.llm import AllTargetsFailed, router

# Devanagari block. One character is enough — mixed script still means Hindi.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Roman-script Hindi markers: function words and common cattle vocabulary that
# effectively never appear in an English sentence. Matched as whole words so
# "hai" does not fire on "hair" and "do" does not fire inside "door".
_HINGLISH_MARKERS = re.compile(
    r"\b("
    r"kya|kyu|kyun|kaise|kaisa|kaisi|kab|kahan|kaun|kitna|kitni|"
    r"hai|hain|nahi|nahin|nahi|raha|rahi|rahe|karo|kare|karna|kar|"
    r"mera|meri|mere|apna|apni|uska|uski|iska|iski|"
    r"gaay|gay|gai|bhains|bhainse|pashu|bachda|bachhda|bail|"
    r"doodh|dudh|chara|khana|khaana|paani|pani|"
    r"bimar|bimaar|bukhar|bukhaar|ilaj|ilaaj|dawa|dawai|davai|"
    r"pet|dast|sujan|sujhan|kamzor|kamzori|"
    r"acha|accha|thik|theek|bahut|bohot|zyada|jyada|kam|"
    r"please batao|batao|bataye|bataiye|madad|"
    r"ho|hua|hui|gaya|gayi|diya|liya|jata|jati"
    r")\b",
    re.IGNORECASE,
)

_LANGUAGE_NAMES = {
    "hi": "Hindi (Devanagari script)",
    "hi-latn": "Hindi written in Roman/English letters (Hinglish)",
}


def detect_language(text: str) -> str:
    """Classify input language. Costs nothing — no model call."""
    if not text or not text.strip():
        return "en"

    if _DEVANAGARI.search(text):
        return "hi"

    # Two markers, not one: a single word like "hai" can appear in an English
    # sentence by accident, but two is a reliable signal. Very short messages
    # get a lower bar since they have fewer chances to contain two.
    matches = _HINGLISH_MARKERS.findall(text)
    word_count = len(text.split())
    threshold = 1 if word_count <= 4 else 2

    if len(matches) >= threshold:
        return "hi-latn"

    return "en"


_TO_ENGLISH_PROMPT = """Translate the following text from {language} into English.

TEXT:
{text}

Rules:
- Output ONLY the English translation. No notes, no explanation, no quotes.
- Keep the meaning exact. Do not add advice, detail, or interpretation.
- Keep animal, disease and medicine names recognisable.
- If the text is already English, return it unchanged."""

_FROM_ENGLISH_PROMPT = """Translate the following English text into {language}.

TEXT:
{text}

Rules:
- Output ONLY the translation. No notes, no explanation, no preamble.
- Keep all Markdown formatting exactly as it is: **bold**, bullet points, line breaks, headings.
- Use everyday spoken language a village farmer would understand, not formal or literary vocabulary.
- Keep medicine names, dosages and numbers exactly as written in the English text.
- Do not add or remove any advice."""


def translate_in_node(state: AgentState) -> dict:
    """Detect the language and produce the English query the rest of the graph uses."""
    messages = state.get("messages", [])
    raw = str(messages[-1]["content"]) if messages else state.get("original_query", "")

    language = detect_language(raw)

    if language == "en" or not settings.ENABLE_TRANSLATION:
        # The overwhelmingly common path, and it costs nothing.
        return {
            "language": "en" if language == "en" else language,
            "original_query": raw,
            "query_en": raw,
            "plan": [],
        }

    with logfire.span("Translate to English", language=language, chars=len(raw)):
        try:
            response = router.invoke(
                _TO_ENGLISH_PROMPT.format(language=_LANGUAGE_NAMES[language], text=raw),
                tier="fast",
                temperature=0.0,
                max_tokens=400,
                feature="translate",
            )
            english = response.content.strip()
            logfire.info("Translated to English", original=raw[:80], english=english[:80])
        except AllTargetsFailed as exc:
            # Retrieval on the raw Hindi will be poor, but refusing to answer is
            # worse than answering imperfectly.
            logfire.warning("Translation unavailable ({err}) — using raw input", err=str(exc)[:200])
            english = raw

        return {
            "language": language,
            "original_query": raw,
            "query_en": english,
            "plan": [f"Language: {language} → translated to English"],
        }


def translate_out_node(state: AgentState) -> dict:
    """Put the finished answer back into the language the user wrote in."""
    language = state.get("language", "en")
    answer = state.get("final_answer", "")

    if language == "en" or not answer.strip() or not settings.ENABLE_TRANSLATION:
        return {}

    with logfire.span("Translate answer", language=language, chars=len(answer)):
        try:
            response = router.invoke(
                _FROM_ENGLISH_PROMPT.format(language=_LANGUAGE_NAMES[language], text=answer),
                tier="fast",
                temperature=0.0,
                feature="translate",
            )
            translated = response.content.strip()
        except AllTargetsFailed as exc:
            logfire.warning("Answer translation failed ({err}) — returning English", err=str(exc)[:200])
            return {"plan": state.get("plan", []) + ["Answer translation unavailable — returned in English"]}

        # Also translate the follow-up questions, which are shown separately.
        questions = state.get("follow_up_questions") or []
        translated_questions = questions
        if questions:
            try:
                joined = "\n".join(f"- {q}" for q in questions)
                q_response = router.invoke(
                    _FROM_ENGLISH_PROMPT.format(language=_LANGUAGE_NAMES[language], text=joined),
                    tier="fast",
                    temperature=0.0,
                    feature="translate",
                )
                translated_questions = [
                    line.lstrip("-•* ").strip()
                    for line in q_response.content.splitlines()
                    if line.strip()
                ] or questions
            except AllTargetsFailed:
                pass

        logfire.info("Answer translated", language=language, chars=len(translated))
        return {
            "final_answer": translated,
            "follow_up_questions": translated_questions,
            "plan": state.get("plan", []) + [f"Answer translated back to {language}"],
        }
