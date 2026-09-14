"""
Scoring metrics, self-contained.

BLEU/ROUGE are kept for continuity with the paper, but they are not the metric
that matters here. The paper's own result — ROUGE-2 ≤ 0.012 across every model —
says surface overlap with the exact wording is close to unrecoverable. A trading
signal does not need the wording; it needs the topic and the direction. So the
headline numbers from this module are topic accuracy and direction accuracy, with
BLEU/ROUGE reported alongside for comparability.
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# ── BLEU ─────────────────────────────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: str, reference: str, n: int = 1) -> float:
    """
    Clipped n-gram precision with brevity penalty (single reference).

    Not smoothed: for n=2 on short texts this returns exactly 0.0 fairly often,
    which is the honest answer rather than a small invented number.
    """
    cand, ref = tokenize(candidate), tokenize(reference)
    if len(cand) < n or len(ref) < n:
        return 0.0
    c_ng, r_ng = _ngrams(cand, n), _ngrams(ref, n)
    overlap = sum(min(cnt, r_ng[g]) for g, cnt in c_ng.items())
    total = max(sum(c_ng.values()), 1)
    precision = overlap / total
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / max(len(cand), 1))
    return round(bp * precision, 4)


# ── ROUGE ────────────────────────────────────────────────────────────────────

def rouge_n(candidate: str, reference: str, n: int = 1) -> float:
    """F1 over n-gram overlap."""
    cand, ref = tokenize(candidate), tokenize(reference)
    if len(cand) < n or len(ref) < n:
        return 0.0
    c_ng, r_ng = _ngrams(cand, n), _ngrams(ref, n)
    overlap = sum(min(cnt, r_ng[g]) for g, cnt in c_ng.items())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(c_ng.values())
    rec = overlap / sum(r_ng.values())
    return round(2 * prec * rec / (prec + rec), 4)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Row-rolled LCS — O(len(a)·len(b)) time, O(len(b)) space."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    cand, ref = tokenize(candidate), tokenize(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(cand), lcs / len(ref)
    return round(2 * prec * rec / (prec + rec), 4)


def text_metrics(candidate: str, reference: str) -> dict:
    return {
        "bleu1": bleu(candidate, reference, 1),
        "bleu2": bleu(candidate, reference, 2),
        "rouge1": rouge_n(candidate, reference, 1),
        "rouge2": rouge_n(candidate, reference, 2),
        "rougeL": rouge_l(candidate, reference),
    }


# ── Aggregate reporting ──────────────────────────────────────────────────────

def summarise(rows: list[dict]) -> dict:
    """
    Aggregate outcome rows for one model.

    `topic_accuracy` and `direction_accuracy` are computed over matched rows only
    — they are conditional on him having posted. `post_rate` reports how often
    that happened, so the two are always read together: a model with 90% topic
    accuracy on a 10% post rate is not a 90% model.
    """
    if not rows:
        return {"n": 0}
    matched = [r for r in rows if r.get("matched")]
    n, m = len(rows), len(matched)

    def mean(key, src):
        vals = [r[key] for r in src if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n": n,
        "matched": m,
        "post_rate": round(m / n, 4),
        "topic_accuracy": mean("topic_hit", matched),
        "direction_accuracy": mean("direction_hit", matched),
        "polarity_mae": mean("polarity_err", matched),
        "mean_similarity": mean("similarity", matched),
        "mean_lag_hours": mean("lag_hours", matched),
        "bleu1": mean("bleu1", matched),
        "rouge1": mean("rouge1", matched),
        "rougeL": mean("rougeL", matched),
    }


def baseline_topic_accuracy(topic_counts: dict[str, int]) -> float:
    """
    Accuracy of always guessing the most common topic.

    Any model's topic accuracy has to be read against this. If he posts about
    `legal` 30% of the time, a 30% score means the model has learned nothing.
    """
    total = sum(topic_counts.values())
    return round(max(topic_counts.values()) / total, 4) if total else 0.0


if __name__ == "__main__":
    cand = "China is RIPPING US OFF with unfair trade. TARIFFS are coming!"
    ref = "China has been ripping off the USA for years with unfair trade deals. Not anymore!"
    print(f"candidate: {cand}\nreference: {ref}\n")
    for k, v in text_metrics(cand, ref).items():
        print(f"  {k:>8}: {v}")
    print(f"\n  identical rougeL: {rouge_l(cand, cand)}")
    print(f"  disjoint rougeL:  {rouge_l('completely unrelated words here', ref)}")
