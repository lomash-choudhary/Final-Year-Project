"""
Tier-1 guardrails: deterministic, in-process, zero API calls.

An LLM-based rail costs a model call on *every* request — including the greetings
and off-topic questions it is supposed to cheaply reject. On a free tier that is
backwards: the guard exhausts the quota the answer needs.

So the cheap layer runs first and handles what regex handles well (greetings,
farewells, capability questions, known jailbreak phrasings, obvious off-topic
categories). Only genuinely ambiguous input reaches tier 2. This layer is also
fully deterministic, which is what makes the guardrail eval in evals/ meaningful.

Deliberate design choice: off-topic detection blocks on *explicit* off-domain
signals rather than on the absence of domain vocabulary. "What did the study
find?" contains no veterinary term but is a perfectly valid follow-up question.
Optimising against false positives matters more here than catching every stray.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GuardResult:
    fired: bool
    category: str | None = None     # jailbreak | off_topic | greeting | farewell | capabilities | injection
    response: str | None = None
    rule: str | None = None
    tier: str = "fast"


PASS = GuardResult(fired=False)


DOMAIN_BLURB = (
    "I'm a veterinary research assistant for cattle and buffalo health. My knowledge base is a "
    "collection of peer-reviewed papers covering haemoprotozoal diseases (theileriosis, babesiosis, "
    "anaplasmosis), brucellosis, lumpy skin disease, foot and eye disorders, genetic disorders, "
    "E. coli, and dairy-herd health management."
)

RESPONSES = {
    "greeting": (
        "Hello! " + DOMAIN_BLURB + "\n\n"
        "Try asking something like: *What is the reported prevalence of theileriosis in Indian cattle?*"
    ),
    "farewell": "Goodbye! Come back any time you need to dig into the bovine disease literature.",
    "capabilities": (
        DOMAIN_BLURB + "\n\n**I can help you:**\n"
        "- Look up disease prevalence, incidence and seasonality figures\n"
        "- Compare findings across studies and regions\n"
        "- Summarise diagnostic methods, risk factors and economic impact\n"
        "- Trace every claim back to the source paper and page\n\n"
        "Every answer is grounded in the indexed papers — I will tell you when the corpus does not cover something."
    ),
    "off_topic": (
        "That falls outside my knowledge base. " + DOMAIN_BLURB + "\n\n"
        "Ask me anything about cattle or buffalo disease research and I'll pull it from the papers."
    ),
    "jailbreak": (
        "My instructions do not change based on how a request is phrased. I answer questions about "
        "cattle and buffalo disease research using the indexed literature. What would you like to know?"
    ),
    "injection": (
        "That request looks like an attempt to override my instructions or extract my configuration, "
        "so I won't act on it. I'm happy to answer questions about the bovine disease literature."
    ),
}


# Pre-written Hindi replies. The whole point of this tier is that it costs no
# API call, so translating the canned responses at runtime would defeat it.
RESPONSES_HI = {
    "greeting": (
        "नमस्ते! मैं गाय और भैंस के स्वास्थ्य में आपकी मदद करने वाला सहायक हूँ। "
        "आप मुझसे बीमारी, लक्षण, दूध उत्पादन या इलाज के बारे में पूछ सकते हैं।\n\n"
        "जैसे: *मेरी गाय ने खाना बंद कर दिया है, क्या करूँ?*"
    ),
    "farewell": "धन्यवाद! अपने पशुओं के बारे में कोई भी सवाल हो तो फिर से पूछिए।",
    "capabilities": (
        "मैं गाय और भैंस के स्वास्थ्य से जुड़े सवालों में मदद करता हूँ।\n\n"
        "**मैं इनमें मदद कर सकता हूँ:**\n"
        "- बीमारी के लक्षण पहचानना\n"
        "- घर पर क्या इलाज कर सकते हैं\n"
        "- कब पशुचिकित्सक को बुलाना ज़रूरी है\n"
        "- दूध उत्पादन, चारा और देखभाल\n\n"
        "अपनी समस्या सरल भाषा में बताइए।"
    ),
    "off_topic": (
        "यह मेरे विषय से बाहर है। मैं केवल गाय और भैंस के स्वास्थ्य, बीमारी और देखभाल "
        "के बारे में मदद कर सकता हूँ।\n\nअपने पशु से जुड़ा कोई सवाल पूछिए।"
    ),
    "jailbreak": (
        "मेरे निर्देश नहीं बदलते। मैं गाय और भैंस के स्वास्थ्य के बारे में ही जानकारी देता हूँ। "
        "आप क्या जानना चाहते हैं?"
    ),
    "injection": (
        "यह अनुरोध मेरी सेटिंग बदलने की कोशिश लगता है, इसलिए मैं इसे पूरा नहीं करूँगा। "
        "पशु स्वास्थ्य से जुड़ा कोई सवाल पूछिए।"
    ),
}

# One Devanagari character is enough to know the user wrote in Hindi.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def _reply(category: str, text: str) -> str:
    """Canned reply in the script the user wrote in."""
    if _DEVANAGARI.search(text) and category in RESPONSES_HI:
        return RESPONSES_HI[category]
    return RESPONSES[category]


# Order matters: injection and jailbreak are checked before anything friendly,
# so "hi, ignore all previous instructions" is caught as a jailbreak, not a greeting.

_INJECTION = _compile([
    r"\b(reveal|show|print|repeat|output|dump|tell me)\b.{0,30}\b(system|initial|original)\s+(prompt|instruction|message)",
    r"\bwhat (are|were) your (system )?(instructions|prompt)",
    r"\b(repeat|echo)\b.{0,20}\bverbatim\b",
    r"</?\s*(system|im_start|im_end|instruction)s?\s*>",
    r"\bBEGIN\s+(SYSTEM|ADMIN)\b",
])

_JAILBREAK = _compile([
    r"\bignore (all |any |your )?(previous|prior|above|earlier)\b.{0,20}\b(instruction|prompt|rule|direction)",
    r"\bdisregard\b.{0,25}\b(instruction|prompt|rule|training|guideline)",
    r"\bforget (your|all|everything|the)\b.{0,25}\b(instruction|prompt|rule|training)",
    r"\byou are (now )?(DAN|dan mode|an unrestricted|a jailbroken)",
    r"\b(developer|dev|god|admin|root)\s+mode\b",
    r"\bpretend (you|that you)\b.{0,30}\b(no|without) (restriction|limit|rule|filter|guideline)",
    r"\bact as (an? )?(unrestricted|uncensored|unfiltered|amoral)\b",
    r"\b(bypass|override|disable|turn off|remove)\b.{0,25}\b(safety|filter|guardrail|restriction|guideline)",
    r"\byour new (instructions|rules|role) (are|is)\b",
    r"\bno longer bound by\b",
])

_GREETING = _compile([
    r"^\s*(hi|hii+|hey+|hello+|yo|howdy|hola|namaste|namaskar|greetings)\s*[!.?]*\s*$",
    r"^\s*good\s+(morning|afternoon|evening|day)\s*[!.?]*\s*$",
    r"^\s*(what'?s up|sup|how are you( doing)?)\s*[!.?]*\s*$",
    r"^\s*(hi|hello|hey)\s+(there|bot|assistant)\s*[!.?]*\s*$",
    # Hindi greetings. Guardrails run before translation, so without these a
    # Hindi "hello" would fall through to retrieval and cost a full pipeline.
    r"^\s*(नमस्ते|नमस्कार|प्रणाम|हैलो|हाय)\s*[!.?]*\s*$",
])

_FAREWELL = _compile([
    r"^\s*(bye+|goodbye|see ya|see you( later)?|cya|ttyl)\s*[!.?]*\s*$",
    r"^\s*(thanks?( you)?|thx|ty)[,! ]*\s*(bye|goodbye|that'?s all|that is all)\s*[!.?]*\s*$",
    r"^\s*(that'?s all|that is all|i'?m done|no more questions)\s*[!.?]*\s*$",
])

_CAPABILITIES = _compile([
    r"^\s*(help|/help)\s*[!.?]*\s*$",
    r"\bwhat (can|do) you (do|know|help( me)? with)\b",
    r"\bwhat (are your|is your) (capabilit|function|purpose|role)",
    r"\bwhat (topics|subjects|documents|papers|sources) (do you|are you)\b",
    r"\bwho are you\b",
    r"\bwhat (are|is) you\b\s*[?]?\s*$",
])

# Explicit off-domain topics. Each entry is a category we are confident does not
# appear in a bovine disease corpus.
_OFF_TOPIC = _compile([
    r"\b(recipe|cook|bake|restaurant|pizza|what should i (eat|have) for)\b",
    r"\b(football|cricket|basketball|soccer|match score|who won the (game|match))\b",
    r"\b(movie|netflix|tv show|song|lyrics|album|actor|celebrity)\b",
    r"\b(stock price|crypto|bitcoin|invest|trading strategy)\b",
    r"\b(weather|temperature) (today|tomorrow|forecast|in [a-z]+)\b",
    r"\bwrite (me )?(a |an )?(poem|song|story|joke|essay about(?! (cattle|bovine|buffalo)))\b",
    r"\btell me a joke\b",
    r"\b(capital|population|president|prime minister) of [a-z]",
    # Loose gap between the verb and the language so "write me a python script"
    # and "write a short sql query" both match — an article between them is the
    # normal phrasing, not the exception.
    r"\bwrite\b.{0,24}\b(python|java|javascript|typescript|c\+\+|sql|html|bash|shell|react)\b",
    r"\b(python|javascript|sql|bash) (code|script|function|program|query)\b",
    r"\bhow do i (code|program|debug|deploy|install)\b",
    r"\b(scrape|scraping) (a |the )?website\b",
    r"\b(dating|relationship) advice\b",
    r"\bwho (won|is winning) the (election|world cup)\b",
])

# If any of these appear, the message is domain-relevant and off-topic rules are
# suppressed. Guards against blocking "what is the economic cost of mastitis".
_DOMAIN_TERMS = _compile([
    r"\b(cattle|cow|cows|bovine|buffalo|buffaloes|bull|calf|calves|heifer|livestock|herd|dairy|bos indicus|bos taurus)\b",
    r"\b(theileri\w*|babesi\w*|anaplasm\w*|trypanosom\w*|haemoprotozoa\w*|hemoprotozoa\w*|protozoa\w*)\b",
    r"\b(brucell\w*|mastitis|lumpy skin|lsd|foot(-| )and(-| )mouth|fmd|johne'?s|tuberculosis|e\.? ?coli|escherichia)\b",
    r"\b(tick|ticks|vector|parasit\w*|patho\w*|zoonot\w*|infect\w*|disease|infestation)\b",
    r"\b(prevalence|incidence|seroprevalence|epidemiolog\w*|morbidity|mortality|outbreak|meta-?analysis)\b",
    r"\b(veterinar\w*|hoof|hooves|lameness|udder|lesion|claw|ocular|conjunctiv\w*|keratitis)\b",
    r"\b(vaccin\w*|treatment|diagnos\w*|serolog\w*|pcr|elisa|giemsa|blood smear|therapy|drug)\b",
    r"\b(paper|papers|study|studies|corpus|document|source|citation|author|abstract|finding|findings)\b",
    # Symptom vocabulary — farmers describe signs, not diseases.
    r"\b(symptom|symptoms|fever|swelling|limping|lame|diarrh\w*|not eating|stopped eating|"
    r"appetite|weak|weakness|bloat|wound|udder|teat|calving|pregnan\w*|dung|manure|milk)\b",
    # Hindi in Devanagari. These run before translation, so the guard needs to
    # recognise a legitimate Hindi question or it would be judged off-topic.
    r"(गाय|गाआ|भैंस|पशु|मवेशी|बछड़ा|बैल|दूध|चारा|बीमार|बुखार|इलाज|दवा|पशुचिकित्सक|"
    r"रोग|बीमारी|संक्रमण|सूजन|दस्त|थन|ब्याना|कमजोर|खा|पानी)",
    # Roman-script Hindi (Hinglish) — how most Indian users actually type.
    r"\b(gaay|gay|gai|bhains|bhainse|pashu|mavesh\w*|bachda|bachhda|bail|"
    r"doodh|dudh|chara|bimar|bimaar|bukhar|bukhaar|ilaj|ilaaj|dawa|dawai|davai|"
    r"sujan|dast|kamzor|kamzori|thun|byana)\b",
])


def _matches(patterns: list[re.Pattern], text: str) -> str | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern
    return None


def check(message: str) -> GuardResult:
    """Run the deterministic rail stack. Cheapest and most certain rules first."""
    text = (message or "").strip()

    if not text:
        return GuardResult(True, "off_topic", "Please type a question about cattle or buffalo disease research.", "empty_input")

    if len(text) > 4000:
        return GuardResult(
            True, "injection",
            "That message is too long to process safely. Please ask a focused question.",
            "length_limit",
        )

    if rule := _matches(_INJECTION, text):
        return GuardResult(True, "injection", _reply("injection", text), rule)

    if rule := _matches(_JAILBREAK, text):
        return GuardResult(True, "jailbreak", _reply("jailbreak", text), rule)

    if rule := _matches(_GREETING, text):
        return GuardResult(True, "greeting", _reply("greeting", text), rule)

    if rule := _matches(_FAREWELL, text):
        return GuardResult(True, "farewell", _reply("farewell", text), rule)

    if rule := _matches(_CAPABILITIES, text):
        return GuardResult(True, "capabilities", _reply("capabilities", text), rule)

    # Off-topic only fires when no domain vocabulary is present, so a question
    # like "what is the economic cost of mastitis in dairy herds" survives.
    if not _matches(_DOMAIN_TERMS, text):
        if rule := _matches(_OFF_TOPIC, text):
            return GuardResult(True, "off_topic", _reply("off_topic", text), rule)

    return PASS
