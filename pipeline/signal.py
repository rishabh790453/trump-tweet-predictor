"""
Structured signal extraction from post text.

This replaces dashboard/market_sim.py, which invented prices and returned them as
if they were market data. Nothing here produces a price. It produces a *claim*:
topic, polarity, intensity, and which tickers the post is directionally relevant
to. Whether that claim has any predictive value is settled by backtest.py against
real prices, not asserted here.

The same extractor runs on predicted text and on actual post text, so the two are
always directly comparable — a scoring bug cannot come from using different
feature code on each side.
"""

import json
import math
import re
from dataclasses import dataclass, asdict, field

# ── Topics ───────────────────────────────────────────────────────────────────
# Ordered by specificity: the first topic whose pattern matches wins, so
# "trade war with china" resolves to trade_tariffs rather than the broader
# foreign_policy. Keep more specific topics above more general ones.
#
# Each topic declares EXACT terms and STEM terms separately, because a blanket
# `\w*` suffix is genuinely dangerous here. `dow\w*` matches "down", `war\w*`
# matches "warm"/"warning"/"warrant", `press\w*` matches "pressure" — all common
# enough that the affected topics would fire on a large share of ordinary posts
# and emit ticker exposure off a word like "down". Exact terms match on word
# boundaries only; stems are opt-in per term.

TOPIC_TERMS: list[tuple[str, list[str], list[str]]] = [
    # (topic, exact terms, stem terms)
    ("crypto",         ["btc", "crypto", "bitcoin", "ethereum", "digital asset"],
                       ["stablecoin", "crypto"]),
    ("fed_rates",      ["fed", "the fed", "powell", "interest rate", "interest rates",
                        "rate cut", "rate cuts", "rate hike", "basis point", "basis points"],
                       ["federal reserve"]),
    # Trading partners belong here: posts about Canadian dairy or EU steel are
    # trade policy, and without these entities they fell through to "other" and
    # produced no signal at all.
    ("trade_tariffs",  ["usmca", "cusma", "import tax", "trade deficit", "trade deal",
                        "trade war", "trade talks", "canada", "canadian", "mexico",
                        "mexican", "european union", "the eu", "japan", "south korea",
                        "dairy farmers", "steel", "aluminum", "semiconductor",
                        "semiconductors", "chips act", "supply chain", "exports",
                        "imports"],
                       ["tariff"]),
    ("energy_oil",     ["oil", "opec", "gasoline", "gas price", "gas prices", "pipeline",
                        "crude", "refinery"],
                       ["drill", "energy independen"]),
    ("taxes",          ["tax", "taxes", "irs", "tax cut", "tax cuts", "no tax",
                        "tax bill", "tax reform", "income tax"],
                       []),
    ("immigration",    ["border", "migrant", "migrants", "caravan", "ice raid", "amnesty"],
                       ["immigra", "deport"]),
    ("military",       ["war", "wars", "troops", "military", "missile", "missiles", "nato",
                        "defense", "ceasefire", "peace deal", "airstrike", "invasion",
                        "strike", "strikes"],
                       []),
    ("china",          ["china", "chinese", "xi jinping", "beijing", "taiwan"], []),
    ("russia_ukraine", ["russia", "russian", "putin", "ukraine", "ukrainian", "zelensky",
                        "moscow", "kyiv"], []),
    ("middle_east",    ["iran", "iranian", "israel", "israeli", "gaza", "hamas", "hezbollah",
                        "lebanon", "netanyahu", "tehran", "hormuz"], []),
    ("markets",        ["dow", "s&p", "nasdaq", "stock market", "stocks", "wall street",
                        "record high", "market crash", "all-time high"], []),
    ("economy",        ["gdp", "jobs report", "unemployment", "prices are", "cost of living"],
                       ["econom", "inflation", "recession"]),
    ("legal",          ["judge", "judges", "court", "courts", "lawsuit", "witch hunt",
                        "appeal", "supreme court", "doj", "fbi", "prosecutor"],
                       ["indict"]),
    ("media",          ["fake news", "media", "cnn", "msnbc", "new york times", "the press",
                        "washington post", "lamestream"], []),
    ("elections",      ["election", "elections", "vote", "votes", "ballot", "ballots",
                        "poll", "polls", "campaign", "primary", "midterm", "rigged"], []),
    ("domestic_pol",   ["democrat", "democrats", "republican", "republicans", "congress",
                        "senate", "schumer", "pelosi", "radical left", "the house"], []),
]


def _build_topic_regex(exact: list[str], stems: list[str]) -> re.Pattern:
    parts = [rf"\b{re.escape(t)}\b" for t in exact]
    parts += [rf"\b{re.escape(s)}\w*" for s in stems]
    return re.compile("|".join(parts), re.I)


_TOPIC_RE = [(name, _build_topic_regex(ex, st)) for name, ex, st in TOPIC_TERMS]

# ── Polarity lexicon ─────────────────────────────────────────────────────────
# Trump-specific rather than general-purpose: generic sentiment models score
# "TOTAL WITCH HUNT" and "GREAT" poorly on this corpus because the register is
# extreme in both directions.

POSITIVE = {
    "great": 1.0, "greatest": 1.2, "tremendous": 1.0, "beautiful": 0.9, "incredible": 1.0,
    "amazing": 1.0, "strong": 0.7, "winning": 1.0, "win": 0.8, "won": 0.7,
    "historic": 0.8, "success": 0.9, "successful": 0.9, "booming": 1.1, "boom": 1.0,
    "best": 1.0, "perfect": 0.9, "wonderful": 0.9, "fantastic": 1.0, "honor": 0.6,
    "congratulations": 0.8, "thank": 0.5, "love": 0.7, "peace": 0.8, "deal": 0.5,
    "agreement": 0.5, "breakthrough": 1.0, "prosperity": 0.9, "safe": 0.5,
    # market/price direction — without these a plainly bullish post scores 0.0.
    # Bare intensifiers (massive/huge/bigly) and corpus-ambiguous tokens
    # (rally=campaign, tank=vehicle, up/down/record) are deliberately absent:
    # they cancelled real valence, e.g. "massive" neutralising "ripping us off".
    "moon": 1.0, "surge": 0.9, "surging": 0.9, "soar": 1.0, "soaring": 1.0,
    "skyrocket": 1.1, "skyrocketing": 1.1,
    "gains": 0.7, "highest": 0.8, "rising": 0.6, "climbing": 0.6,
    "thriving": 1.0, "unstoppable": 1.0,
}

NEGATIVE = {
    "disaster": -1.1, "terrible": -1.0, "horrible": -1.0, "disgrace": -1.0, "sad": -0.6,
    "weak": -0.8, "failing": -1.0, "failed": -0.9, "fail": -0.8, "crooked": -1.0,
    "corrupt": -1.0, "rigged": -1.0, "hoax": -1.0, "witch hunt": -1.1, "fake": -0.8,
    "enemy": -1.0, "destroying": -1.1, "destroy": -1.0, "disgraceful": -1.0,
    "incompetent": -1.0, "stupid": -0.9, "dumb": -0.8, "loser": -0.9, "pathetic": -0.9,
    "catastrophe": -1.2, "worst": -1.1, "ripping us off": -1.1, "unfair": -0.8,
    "threat": -0.7, "crisis": -0.8, "invasion": -0.9, "illegal": -0.7, "scam": -1.0,
    # market/price direction
    "plunge": -1.0, "plunging": -1.0, "crash": -1.1, "crashing": -1.1, "collapse": -1.1,
    "collapsing": -1.1, "plummet": -1.0,
    "losses": -0.7, "lowest": -0.7, "falling": -0.6, "slump": -0.8, "meltdown": -1.1,
}

# Divisor in the tanh squash. Chosen so ~3 strongly-valenced words reach ≈0.8
# rather than pinning at 1.0 — a plain mean-and-clip saturated on most real posts
# and threw away all resolution between "annoyed" and "incandescent".
_POLARITY_SCALE = 3.0

# Negators flip the sign of the next few sentiment terms.
_NEGATORS = {"not", "no", "never", "nobody", "nothing", "without", "isnt", "isn't",
             "wasnt", "wasn't", "dont", "don't", "doesnt", "doesn't", "cant", "can't",
             "wont", "won't", "aint", "ain't"}
_NEGATION_WINDOW = 3

# ── Ticker mapping ───────────────────────────────────────────────────────────
# topic → {ticker: sign}. Sign is *direction conditional on polarity*, applied as
# sign * polarity. Example: hostile (negative) trade_tariffs talk → SPY down.
# Magnitudes are intentionally all 1.0 / 0.5: inventing precise per-keyword basis
# points, as the old market_sim did, implies a precision nobody has measured.
# The backtest estimates real magnitudes from data.

TOPIC_TICKERS: dict[str, dict[str, float]] = {
    "trade_tariffs":  {"SPY": 1.0, "QQQ": 1.0, "FXI": 1.0},
    "china":          {"SPY": 0.5, "FXI": 1.0, "QQQ": 0.5},
    "fed_rates":      {"SPY": 1.0, "TLT": -1.0, "GLD": -0.5},
    "crypto":         {"COIN": 1.0, "MSTR": 1.0},
    "energy_oil":     {"XLE": 1.0, "USO": 1.0},
    "military":       {"ITA": -1.0, "LMT": -1.0, "GLD": -0.5, "SPY": 1.0},
    "middle_east":    {"XLE": -1.0, "USO": -1.0, "ITA": -1.0, "SPY": 0.5},
    "russia_ukraine": {"ITA": -1.0, "LMT": -1.0, "XLE": -0.5},
    "taxes":          {"SPY": 1.0, "IWM": 1.0},
    "markets":        {"SPY": 1.0, "DIA": 1.0},
    "economy":        {"SPY": 1.0, "DIA": 1.0},
    "immigration":    {"SPY": 0.0},
    "legal":          {"SPY": 0.0},
    "media":          {"SPY": 0.0},
    "elections":      {"SPY": 0.0},
    "domestic_pol":   {"SPY": 0.0},
    "other":          {},
}

ALL_TICKERS = sorted({t for m in TOPIC_TICKERS.values() for t in m})

_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
_URL_RE = re.compile(r"https?://\S+|www\.\s?\S+")
_WORD_RE = re.compile(r"[a-z']+")


@dataclass
class Signal:
    topic: str
    polarity: float          # −1 hostile … +1 favourable
    intensity: float         # 0 … 1, rhetorical force
    tickers: dict[str, float] = field(default_factory=dict)   # ticker → signed score
    matched_terms: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "topic": self.topic,
            "polarity": round(self.polarity, 4),
            "intensity": round(self.intensity, 4),
            "tickers": json.dumps(self.tickers, separators=(",", ":")),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    """Strip URLs and collapse whitespace. Applied identically to both sides."""
    return re.sub(r"\s+", " ", _URL_RE.sub(" ", text or "")).strip()


# Figurative uses of otherwise-topical words. "war on coal" is energy policy,
# not a military event; without this the post emits defence-sector exposure.
# Up to three intervening words, because real posts read "war on beautiful,
# clean coal" rather than the bare phrase.
FIGURATIVE = re.compile(
    r"\bwar\s+(?:on|of|against)\s+(?:\w+[,\s]+){0,3}"
    r"(?:coal|drugs|christmas|words|terror|poverty|cancer|crime|farmers|"
    r"women|religion|thanksgiving|energy|poor|press|media)\b"
    r"|\btrade war\b|\bculture war\b|\bprice war\b|\bbidding war\b"
    r"|\bwar\s+of\s+words\b",
    re.I,
)


def classify_topic(text: str) -> tuple[str, list[str]]:
    """First matching pattern wins — see ordering note on TOPIC_TERMS."""
    masked = FIGURATIVE.sub(" ", text)   # hide metaphors from the military matcher
    hits: list[str] = []
    topic = "other"
    for name, rx in _TOPIC_RE:
        # Match the real text for every topic except military, which is the one
        # prone to firing on figurative "war" usage.
        target = masked if name == "military" else text
        m = rx.search(target)
        if m:
            hits.append(m.group(0).lower())
            if topic == "other":
                topic = name
    return topic, hits


def score_polarity(text: str) -> tuple[float, list[str]]:
    """
    Signed lexicon sum squashed through tanh into [-1, 1].

    tanh rather than mean-and-clip: the mean pinned at exactly ±1.0 on most real
    posts (any two strong words saturate it), which collapsed polarity to a
    near-binary flag. tanh keeps resolution in the middle of the range while
    still bounding the extremes, and lets accumulated sentiment matter.

    Terms within _NEGATION_WINDOW words after a negator have their sign flipped,
    so "not a disaster" does not read as maximally hostile.
    """
    low = text.lower()
    hits: list[tuple[str, float]] = []

    # Multi-word phrases first, and remove them so their component words are not
    # double-counted (e.g. "witch hunt" should not also score "hunt").
    for phrase, w in list(POSITIVE.items()) + list(NEGATIVE.items()):
        if " " in phrase and phrase in low:
            hits.append((phrase, w))
            low = low.replace(phrase, " ")

    words = _WORD_RE.findall(low)
    for i, w in enumerate(words):
        weight = POSITIVE.get(w, NEGATIVE.get(w))
        if weight is None:
            continue
        window = words[max(0, i - _NEGATION_WINDOW):i]
        if any(n in _NEGATORS for n in window):
            weight = -weight
            w = f"NOT_{w}"
        hits.append((w, weight))

    if not hits:
        return 0.0, []

    total = sum(w for _, w in hits)
    return round(math.tanh(total / _POLARITY_SCALE), 4), [t for t, _ in hits]


def score_intensity(text: str) -> float:
    """
    Rhetorical force in [0, 1] from caps ratio, exclamations and length.

    Caps are the dominant term because in this corpus they are the clearest
    marker of an emphatic post; the weights below sum to 1.0.
    """
    if not text:
        return 0.0
    caps_words = len(_CAPS_RE.findall(text))
    total_words = max(len(text.split()), 1)
    caps_ratio = min(caps_words / total_words, 1.0)
    excl = min(text.count("!") / 3.0, 1.0)
    length = min(len(text) / 280.0, 1.0)
    return round(min(0.60 * caps_ratio + 0.25 * excl + 0.15 * length, 1.0), 4)


# A post that is only a link (typically "RT: https://...") carries no text to
# analyse. It survives a raw length filter but contains nothing once the URL is
# stripped, so it must not count as an analysable post.
MIN_ANALYSABLE_CHARS = 15


def is_analysable(text: str) -> bool:
    """True if anything remains once URLs and retweet markers are removed."""
    stripped = re.sub(r"^\s*RT[:\s]+", "", clean_text(text), flags=re.I)
    return len(stripped.strip()) >= MIN_ANALYSABLE_CHARS


def extract(text: str) -> Signal:
    """Full extraction. Safe on empty/None input — returns a neutral signal."""
    clean = clean_text(text)
    if not clean or not is_analysable(text):
        return Signal(topic="other", polarity=0.0, intensity=0.0)

    topic, topic_hits = classify_topic(clean)
    polarity, pol_hits = score_polarity(clean)
    intensity = score_intensity(clean)

    # Directional exposure = mapping sign × polarity × intensity.
    # A neutral or unemphatic post yields ~0 exposure rather than a false signal.
    tickers = {
        tk: round(sign * polarity * max(intensity, 0.1), 4)
        for tk, sign in TOPIC_TICKERS.get(topic, {}).items()
        if sign != 0.0
    }
    tickers = {k: v for k, v in tickers.items() if abs(v) >= 0.01}

    return Signal(
        topic=topic,
        polarity=round(polarity, 4),
        intensity=intensity,
        tickers=tickers,
        matched_terms=(topic_hits + pol_hits)[:12],
    )


def agreement(a: Signal, b: Signal) -> dict:
    """Compare a predicted signal against an actual one."""
    return {
        "topic_hit": int(a.topic == b.topic),
        "polarity_err": round(abs(a.polarity - b.polarity), 4),
        "direction_hit": int(
            (a.polarity > 0.05 and b.polarity > 0.05)
            or (a.polarity < -0.05 and b.polarity < -0.05)
            or (abs(a.polarity) <= 0.05 and abs(b.polarity) <= 0.05)
        ),
    }


if __name__ == "__main__":
    samples = [
        "China is RIPPING US OFF on trade. Massive TARIFFS coming soon! They have been "
        "taking advantage of our Country for DECADES. NOT ANYMORE!",
        "Congratulations to Shane Lowry on his GREAT Win at Trump International Golf Links!",
        "The Fed must CUT RATES NOW. Inflation is down, the economy is BOOMING!",
        "Judge blocks the Ballroom. TOTAL WITCH HUNT by a Radical Left lunatic!",
        "Bitcoin is going to the MOON. Crypto will make America the crypto capital!",
        "",
    ]
    for s in samples:
        sig = extract(s)
        print(f"\n{s[:70]!r}")
        print(f"  topic={sig.topic:<15} pol={sig.polarity:+.2f} int={sig.intensity:.2f}")
        print(f"  tickers={sig.tickers}")
        print(f"  terms={sig.matched_terms[:6]}")
