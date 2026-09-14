"""
Match predictions to what Trump actually posted.

Runs only on predictions whose horizon has fully elapsed. Scoring a prediction
early would count posts he had not yet had the chance to write as misses.

Every eligible prediction produces exactly one outcome row, including the ones
where nothing comparable was posted (matched=0). Those rows are the honest
denominator: a hit rate computed only over predictions that happened to land
would be conditioned on the outcome it is trying to measure.
"""

from datetime import datetime, timedelta, timezone

from . import score as sc
from . import signal as sig
from .store import (connect, insert_match, posts_between, unmatched_predictions,
                    utcnow)

# Cosine similarity above which a post is accepted as "the same story".
# The paper measured genuinely matched (headline, post) pairs at ~0.339 mean
# similarity against ~0.05 for random pairs, so 0.30 sits just under the real
# signal and well clear of noise.
MATCH_THRESHOLD = 0.30

# Media-only posts carry no text to compare. Counting them as misses would make
# the post rate a measure of his posting habits rather than model quality.
MIN_POST_CHARS = 25

_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _ENCODER


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _similarities(pred_text: str, post_texts: list[str]) -> list[float]:
    if not pred_text.strip() or not post_texts:
        return [0.0] * len(post_texts)
    enc = _encoder()
    vecs = enc.encode([pred_text] + post_texts, normalize_embeddings=True)
    pred_vec, post_vecs = vecs[0], vecs[1:]
    return [float(pred_vec @ v) for v in post_vecs]


def match_one(conn, pred) -> dict:
    """Build the outcome row for a single elapsed prediction."""
    start = _parse(pred["predicted_at"])
    end = start + timedelta(hours=float(pred["horizon_hours"]))

    candidates = [
        p for p in posts_between(conn, start.isoformat(), end.isoformat(),
                                 min_len=MIN_POST_CHARS)
        # is_analysable also rejects link-only "RT: https://..." posts, which
        # clear the length filter but contain no text once the URL is stripped.
        if not p["is_reblog"] and sig.is_analysable(p["text"])
    ]

    base = {
        "prediction_id": pred["id"],
        "matched_at": utcnow(),
        "n_candidates": len(candidates),
        "post_id": None, "matched": 0, "similarity": None, "lag_hours": None,
        "topic_hit": None, "polarity_err": None, "direction_hit": None,
        "bleu1": None, "rouge1": None, "rougeL": None,
    }

    if not candidates:
        return base   # horizon elapsed, he posted nothing with text — a real miss

    pred_text = pred["text"] or ""
    sims = _similarities(pred_text, [c["text"] for c in candidates])
    best_i = max(range(len(sims)), key=lambda i: sims[i])
    best, best_sim = candidates[best_i], sims[best_i]

    if best_sim < MATCH_THRESHOLD:
        base["similarity"] = round(best_sim, 4)
        return base

    pred_sig = sig.Signal(
        topic=pred["topic"] or "other",
        polarity=pred["polarity"] or 0.0,
        intensity=pred["intensity"] or 0.0,
    )
    post_sig = sig.extract(best["text"])
    agree = sig.agreement(pred_sig, post_sig)
    metrics = sc.text_metrics(pred_text, best["text"])

    base.update({
        "post_id": best["id"],
        "matched": 1,
        "similarity": round(best_sim, 4),
        "lag_hours": round((_parse(best["created_at"]) - start).total_seconds() / 3600, 3),
        "topic_hit": agree["topic_hit"],
        "polarity_err": agree["polarity_err"],
        "direction_hit": agree["direction_hit"],
        "bleu1": metrics["bleu1"],
        "rouge1": metrics["rouge1"],
        "rougeL": metrics["rougeL"],
    })
    return base


def match_pending(db_path=None, verbose: bool = True) -> int:
    """Score every prediction whose horizon has elapsed. Returns rows written."""
    kwargs = {"db_path": db_path} if db_path else {}
    written = 0
    with connect(**kwargs) as conn:
        pending = unmatched_predictions(conn, utcnow())
        if verbose and pending:
            print(f"  [match] {len(pending)} predictions with elapsed horizon")
        for pred in pending:
            row = match_one(conn, pred)
            insert_match(conn, row)
            written += 1
            if verbose:
                if row["matched"]:
                    print(f"    {pred['model']:<16} MATCH sim={row['similarity']:.3f} "
                          f"topic_hit={row['topic_hit']} lag={row['lag_hours']:.1f}h")
                else:
                    why = "no posts in window" if row["n_candidates"] == 0 \
                          else f"best sim {row['similarity']:.3f} < {MATCH_THRESHOLD}"
                    print(f"    {pred['model']:<16} miss  ({why})")
    return written


if __name__ == "__main__":
    n = match_pending()
    print(f"\n{n} outcome rows written")
