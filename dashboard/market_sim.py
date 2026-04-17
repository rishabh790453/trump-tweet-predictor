"""
Rule-based market impact simulation.
Classifies the generated tweet → maps keywords → % moves on tickers.
Carries simulated prices across predictions within a session.
"""

import re
from dataclasses import dataclass, field

# Base simulated prices (approx real-world anchors)
BASE_PRICES: dict[str, float] = {
    "SPY":  520.00,
    "DJI":  38500.0,
    "QQQ":  445.00,
    "XLE":   88.00,
    "LMT":  475.00,
    "RTX":  125.00,
    "MSTR": 380.00,
    "COIN": 225.00,
    "GLD":  230.00,
    "TLT":   95.00,
}

# keyword → {ticker: % impact}
RULES: dict[str, dict[str, float]] = {
    # Trade / Tariffs
    "tariff":       {"SPY": -0.35, "DJI": -0.40, "QQQ": -0.20, "XLE": -0.15},
    "tariffs":      {"SPY": -0.35, "DJI": -0.40, "QQQ": -0.20},
    "trade deal":   {"SPY": +0.30, "DJI": +0.35, "QQQ": +0.25},
    "trade war":    {"SPY": -0.50, "DJI": -0.55, "QQQ": -0.30},
    # China
    "china":        {"SPY": -0.25, "DJI": -0.30},
    # Fed / Rates
    "fed":          {"SPY": -0.40, "DJI": -0.45, "TLT": +0.30, "GLD": +0.20},
    "interest rate":{"SPY": -0.30, "DJI": -0.35, "TLT": +0.25},
    "rate cut":     {"SPY": +0.45, "DJI": +0.50, "QQQ": +0.35},
    # Taxes
    "tax cut":      {"SPY": +0.30, "DJI": +0.35},
    "no tax":       {"SPY": +0.20, "DJI": +0.25},
    "cut taxes":    {"SPY": +0.30, "DJI": +0.35},
    # Markets / Economy
    "stock market": {"SPY": +0.15, "DJI": +0.20},
    "economy":      {"SPY": +0.10, "DJI": +0.15},
    "recession":    {"SPY": -0.60, "DJI": -0.65, "GLD": +0.40},
    "inflation":    {"SPY": -0.20, "DJI": -0.25, "GLD": +0.15},
    "jobs":         {"SPY": +0.15, "DJI": +0.20},
    # Geopolitical
    "iran":         {"XLE": +1.50, "LMT": +0.80, "RTX": +0.60, "SPY": -0.20},
    "russia":       {"LMT": +0.60, "RTX": +0.50, "XLE": +0.40, "SPY": -0.15},
    "ukraine":      {"LMT": +0.70, "RTX": +0.60, "SPY": -0.10},
    "nato":         {"LMT": +0.40, "RTX": +0.35},
    "war":          {"LMT": +1.20, "RTX": +1.00, "SPY": -0.50, "GLD": +0.50},
    "peace":        {"LMT": -0.80, "RTX": -0.60, "SPY": +0.30},
    "ceasefire":    {"LMT": -0.60, "RTX": -0.50, "SPY": +0.20},
    "deal":         {"SPY": +0.25, "DJI": +0.30},
    # Crypto
    "bitcoin":      {"MSTR": +3.00, "COIN": +2.00},
    "crypto":       {"MSTR": +2.50, "COIN": +1.80},
    "ethereum":     {"COIN": +1.50, "MSTR": +0.50},
    # Oil / Energy
    "oil":          {"XLE": +0.80},
    "energy":       {"XLE": +0.40},
    # Positive/negative sentiment
    "disaster":     {"SPY": -0.20, "DJI": -0.25},
    "destroying":   {"SPY": -0.15, "DJI": -0.20},
    "winning":      {"SPY": +0.20, "DJI": +0.25},
    "great":        {"SPY": +0.08},
    "beautiful":    {"SPY": +0.05},
    "terrible":     {"SPY": -0.10},
    "horrible":     {"SPY": -0.10},
    "corrupt":      {"SPY": -0.08},
    "rigged":       {"SPY": -0.08},
}

TICKERS = list(BASE_PRICES.keys())


@dataclass
class MarketImpact:
    impacts: dict[str, float]       # ticker → % change from baseline
    prices: dict[str, float]        # ticker → new simulated price
    sentiment: str                  # "bullish" / "bearish" / "neutral"
    triggered_by: list[str] = field(default_factory=list)
    spy_delta: float = 0.0          # convenience: SPY % change


def simulate(tweet: str, current_prices: dict[str, float] | None = None) -> MarketImpact:
    base = current_prices or BASE_PRICES.copy()
    tweet_lower = tweet.lower()

    impacts: dict[str, float] = {t: 0.0 for t in TICKERS}
    triggered: list[str] = []

    for keyword, rule in RULES.items():
        if keyword in tweet_lower:
            triggered.append(keyword)
            for ticker, pct in rule.items():
                if ticker in impacts:
                    impacts[ticker] += pct

    # ALL CAPS words amplify magnitude (Trump-style emphasis = stronger signal)
    caps_count = len(re.findall(r"\b[A-Z]{3,}\b", tweet))
    amplifier = 1.0 + min(caps_count * 0.04, 0.40)
    impacts = {t: round(v * amplifier, 4) for t, v in impacts.items()}

    new_prices = {t: round(base[t] * (1 + impacts[t] / 100), 2) for t in TICKERS}

    spy_delta = impacts.get("SPY", 0.0)
    total = sum(impacts.values())
    if total > 0.15:
        sentiment = "bullish"
    elif total < -0.15:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return MarketImpact(
        impacts=impacts,
        prices=new_prices,
        sentiment=sentiment,
        triggered_by=triggered,
        spy_delta=spy_delta,
    )
