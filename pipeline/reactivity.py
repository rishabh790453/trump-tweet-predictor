"""
How much of his posting is reaction to news, and how much is endogenous?

The research question: a post about tariffs on Irish whiskey is a reaction to
something that happened; a post about the White House ballroom renovation is not
reacting to anything — it originates with him. No model conditioned on news
headlines can predict the second kind, however good it is.

That makes this a **ceiling on the task**, not a property of any model. If only a
minority of posts are news-reactive, the paper's low BLEU/ROUGE numbers are
largely a fact about the problem rather than a failure of Markov/RAG/LoRA, and
that reframing is more interesting than the scores themselves.

Method
------
For each sampled post at time t, take the maximum cosine similarity between the
post and any news headline in the 48h before it (the paper's window). Call that
`real_sim`.

A high similarity on its own proves nothing. Political text shares vocabulary, so
any post will match *some* headline somewhere. The comparison needs a null, and
this uses two, because the honest answer lies between them:

  * `far` placebo — the same post against a 48h window ~180 days away. This
    controls for generic vocabulary overlap. It is the LIBERAL null: it will
    credit a post as reactive even when it is about a slow, ongoing story.

  * `near` placebo — a 48h window 14 days away. Ongoing stories (a trade war, a
    war) persist for weeks, so this window often contains genuinely related
    headlines. It is the CONSERVATIVE null: it will discount posts about
    long-running topics as "not news-reactive".

Reported as a range. A single number here would be false precision.

Window size is controlled explicitly: max-similarity rises with the number of
headlines compared, so for each post the real and placebo windows are subsampled
to the SAME count. Without that, a post from 2009 (when the corpus holds ~400
NYT articles/day) would look more reactive than one from 2025 (~130/day) purely
because more headlines were available to match against.
"""

import random
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import signal as sig

REPO = Path(__file__).parent.parent
NEWS_DB = REPO / "news_data.db"
CORPUS = REPO / "trump_tweets_cleaned.csv"

WINDOW_HOURS = 48           # matches the paper's matching window
FAR_SHIFT_DAYS = 180
NEAR_SHIFT_DAYS = 14
# Cap on headlines compared per window, applied equally to real and placebo.
# Must be large enough to cover a whole 48h window: measured density is 300-600
# NYT headlines per 48h (605 in 2013, 299 in 2025). An earlier cap of 120 covered
# only 20-40% of each window, so the single headline a post was actually
# reacting to was usually not even in the comparison set — which understated
# reactivity badly. 700 covers essentially every window in the corpus.
HEADLINES_PER_WINDOW = 700
MIN_POST_CHARS = 60         # "MAKE AMERICA GREAT AGAIN!" cannot match anything
EMBED_MODEL = "all-MiniLM-L6-v2"   # same encoder as the paper

# News coverage is thin before this and the corpus has bad 0001-01-01 dates.
MIN_NEWS_YEAR = 2009


@dataclass
class PostSample:
    post_id: str
    ts: datetime
    text: str
    year: int
    topic: str
    real_sim: float = 0.0
    far_sim: float = 0.0
    near_sim: float = 0.0
    n_compared: int = 0


# ── Data ─────────────────────────────────────────────────────────────────────

def load_posts(n_per_year: int = 120, seed: int = 7) -> list[PostSample]:
    """Stratified sample by year, so no era dominates the estimate."""
    import pandas as pd

    df = pd.read_csv(CORPUS)
    df = df[df.content.notna()].copy()
    df["content"] = df.content.astype(str)
    ts = pd.to_datetime(df.timestamp, format="mixed", utc=True, errors="coerce")
    df = df[ts.notna()].copy()
    df["ts"] = ts[ts.notna()]

    # Drop retweets and link-only posts: neither is his own composition, so
    # neither speaks to whether *he* is reacting to news.
    df = df[~df.content.str.match(r"^\s*RT[\s:@]", case=False, na=False)]
    df = df[df.content.str.len() >= MIN_POST_CHARS]
    df = df[df.ts.dt.year >= MIN_NEWS_YEAR]

    rng = random.Random(seed)
    out: list[PostSample] = []
    for year, grp in df.groupby(df.ts.dt.year):
        idx = list(range(len(grp)))
        rng.shuffle(idx)
        for i in idx[:n_per_year]:
            row = grp.iloc[i]
            out.append(PostSample(
                post_id=str(row.get("id", f"{year}-{i}")),
                ts=row.ts.to_pydatetime(),
                text=row.content,
                year=int(year),
                topic=sig.extract(row.content).topic,
            ))
    return out


class NewsIndex:
    """Headline lookup by time window, straight off the indexed SQLite column."""

    def __init__(self, db_path: Path = NEWS_DB):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def window(self, end: datetime, hours: int = WINDOW_HOURS,
               limit: int = 4000) -> list[str]:
        start = end - timedelta(hours=hours)
        rows = self.conn.execute(
            """SELECT title FROM news_articles
               WHERE published_at >= ? AND published_at < ?
                 AND title IS NOT NULL AND length(title) > 20
               LIMIT ?""",
            (start.strftime("%Y-%m-%dT%H:%M:%SZ"),
             end.strftime("%Y-%m-%dT%H:%M:%SZ"), limit),
        ).fetchall()
        return [r["title"] for r in rows]


# ── Similarity ───────────────────────────────────────────────────────────────

def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def run(n_per_year: int = 120, seed: int = 7, verbose: bool = True,
        cap: int = HEADLINES_PER_WINDOW) -> list[PostSample]:
    import numpy as np

    rng = random.Random(seed)
    posts = load_posts(n_per_year, seed)
    if verbose:
        print(f"  [sample] {len(posts)} posts, {posts[0].ts.year}-{posts[-1].ts.year}")

    news = NewsIndex()
    enc = _encoder()
    if verbose:
        print(f"  [encoder] {EMBED_MODEL} loaded")

    kept: list[PostSample] = []
    for i, p in enumerate(posts):
        real = news.window(p.ts)
        far_end = p.ts - timedelta(days=FAR_SHIFT_DAYS)
        if far_end.year < MIN_NEWS_YEAR:
            far_end = p.ts + timedelta(days=FAR_SHIFT_DAYS)
        far = news.window(far_end)
        near = news.window(p.ts - timedelta(days=NEAR_SHIFT_DAYS))

        # Equal-size comparison sets — see module docstring.
        k = min(len(real), len(far), len(near), cap)
        if k < 20:
            continue
        real_s = rng.sample(real, k)
        far_s = rng.sample(far, k)
        near_s = rng.sample(near, k)

        vecs = enc.encode([p.text] + real_s + far_s + near_s,
                          normalize_embeddings=True, batch_size=256,
                          show_progress_bar=False)
        pv, rest = vecs[0], vecs[1:]
        sims = rest @ pv
        p.real_sim = float(sims[:k].max())
        p.far_sim = float(sims[k:2 * k].max())
        p.near_sim = float(sims[2 * k:].max())
        p.n_compared = k
        kept.append(p)

        if verbose and (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(posts)} processed ({len(kept)} usable)", flush=True)

    if verbose:
        print(f"  [done] {len(kept)} posts with comparable windows")
    return kept


# ── Estimation ───────────────────────────────────────────────────────────────

def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(int(q * (len(s) - 1)), len(s) - 1)
    return s[i]


def estimate(samples: list[PostSample], placebo: str = "far",
             fpr: float = 0.05) -> dict:
    """
    Fraction of posts that are news-reactive, against one placebo.

    The threshold is set at the (1-fpr) quantile of the placebo distribution, so
    by construction a non-reactive post clears it `fpr` of the time. The reactive
    fraction is then the excess of real-window exceedances above that rate.
    """
    real = [s.real_sim for s in samples]
    plac = [getattr(s, f"{placebo}_sim") for s in samples]
    if not real:
        return {"n": 0}

    thr = _quantile(plac, 1 - fpr)
    hits = sum(1 for r in real if r > thr)
    raw = hits / len(real)
    # Subtract the built-in false-positive rate and rescale, so the estimate is
    # the share genuinely above chance rather than the raw exceedance count.
    adjusted = max((raw - fpr) / (1 - fpr), 0.0)

    # Binomial standard error on the raw rate, propagated through the same scaling.
    se = (raw * (1 - raw) / len(real)) ** 0.5 / (1 - fpr)
    return {
        "n": len(real),
        "placebo": placebo,
        "threshold": round(thr, 4),
        "raw_exceedance": round(raw, 4),
        "reactive_fraction": round(adjusted, 4),
        "ci95": (round(max(adjusted - 1.96 * se, 0), 4),
                 round(min(adjusted + 1.96 * se, 1), 4)),
        "mean_real": round(sum(real) / len(real), 4),
        "mean_placebo": round(sum(plac) / len(plac), 4),
    }


def by_group(samples: list[PostSample], key: str, placebo: str = "far",
             min_n: int = 25) -> list[dict]:
    groups: dict = {}
    for s in samples:
        groups.setdefault(getattr(s, key), []).append(s)
    rows = []
    for name, grp in groups.items():
        if len(grp) < min_n:
            continue
        est = estimate(grp, placebo)
        rows.append({"group": name, **est})
    return sorted(rows, key=lambda r: -r["reactive_fraction"])


def examples(samples: list[PostSample], placebo: str = "far",
             fpr: float = 0.05, k: int = 5) -> dict:
    """Most and least news-reactive posts, for eyeballing that this is sane."""
    thr = _quantile([getattr(s, f"{placebo}_sim") for s in samples], 1 - fpr)
    by_margin = sorted(samples, key=lambda s: s.real_sim - getattr(s, f"{placebo}_sim"))
    return {
        "threshold": thr,
        "most_reactive": by_margin[-k:][::-1],
        "least_reactive": by_margin[:k],
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-year", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cap", type=int, default=HEADLINES_PER_WINDOW,
                    help="max headlines compared per window (equalised across windows)")
    args = ap.parse_args()

    print("News-reactivity study\n")
    samples = run(args.per_year, args.seed, cap=args.cap)
    print(f'  [cap] {args.cap} headlines/window; median compared = {sorted(s.n_compared for s in samples)[len(samples)//2] if samples else 0}')
    if not samples:
        raise SystemExit("no usable samples")

    print("\n" + "=" * 66)
    print("  HOW MUCH OF HIS POSTING IS REACTION TO NEWS?")
    print("=" * 66)

    for pl, desc in [("far", "180d placebo — liberal (generic vocabulary null)"),
                     ("near", "14d placebo — conservative (ongoing-story null)")]:
        e = estimate(samples, pl)
        print(f"\n  {desc}")
        print(f"    threshold (95th pct of placebo) : {e['threshold']}")
        print(f"    mean real sim / placebo sim     : {e['mean_real']} / {e['mean_placebo']}")
        print(f"    NEWS-REACTIVE FRACTION          : {e['reactive_fraction'] * 100:.1f}% "
              f"(95% CI {e['ci95'][0] * 100:.1f}-{e['ci95'][1] * 100:.1f}%)")

    lo = estimate(samples, "near")["reactive_fraction"]
    hi = estimate(samples, "far")["reactive_fraction"]
    print(f"\n  => between {lo * 100:.0f}% and {hi * 100:.0f}% of posts are news-reactive.")
    print(f"     The remaining {(1 - hi) * 100:.0f}-{(1 - lo) * 100:.0f}% originate with him and")
    print("     cannot be predicted from headlines by ANY model.")

    print("\n  BY TOPIC (far placebo)")
    print(f"    {'topic':<16}{'n':>6}{'reactive%':>11}")
    print("    " + "-" * 34)
    for r in by_group(samples, "topic"):
        print(f"    {str(r['group'])[:15]:<16}{r['n']:>6}{r['reactive_fraction'] * 100:>10.1f}%")

    print("\n  BY YEAR (far placebo)")
    print(f"    {'year':<16}{'n':>6}{'reactive%':>11}")
    print("    " + "-" * 34)
    for r in sorted(by_group(samples, "year"), key=lambda r: r["group"]):
        print(f"    {r['group']:<16}{r['n']:>6}{r['reactive_fraction'] * 100:>10.1f}%")

    ex = examples(samples)
    print("\n  MOST NEWS-REACTIVE")
    for s in ex["most_reactive"]:
        print(f"    [{s.real_sim:.2f} vs {s.far_sim:.2f}] {s.text[:88]}")
    print("\n  LEAST NEWS-REACTIVE (endogenous)")
    for s in ex["least_reactive"]:
        print(f"    [{s.real_sim:.2f} vs {s.far_sim:.2f}] {s.text[:88]}")
