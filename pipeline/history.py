"""
Historical event study over the full 2009-2026 corpus.

The live loop accumulates ~3 signal events a day, so reaching usable statistical
power takes months. The corpus already holds 86,170 posts with timestamps, and
daily bars go back decades — the 60-day retention limit applies only to intraday
data. So the question "which kinds of post actually move markets" can be answered
now, at daily resolution, over sixteen years.

The cost of daily resolution is that a single post's effect is diluted by
everything else that happened that session. That is a real loss of sensitivity,
and it is why this measures *categories* of post across thousands of sessions
rather than trying to price any individual one.

Three things this controls for, each of which would otherwise manufacture a
result:

  1. Session mapping. Every post is mapped to the next session whose OPEN is
     strictly after the post, and the return measured is that session's
     open→close. A post made mid-session is never credited with the part of the
     move that preceded it.

  2. Market drift. SPY rose most of this period, so any long-biased signal looks
     profitable. Every return is reported as EXCESS over the same ticker's
     unconditional mean open→close return across the sample.

  3. Clustering. He posts dozens of times a day. Per-post rows within one session
     are not independent observations of that session's return, so everything is
     aggregated to one observation per (session, ticker) before inference.
"""

import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import signal as sig

REPO = Path(__file__).parent.parent
CORPUS = REPO / "trump_tweets_cleaned.csv"

# Trading-day boundaries in US/Eastern. A post at 09:29 can still affect the
# open; one at 09:31 cannot affect the open it already missed.
MARKET_OPEN_ET = (9, 30)

# Political era matters more than any other split here: a 2013 post from a
# private citizen and a 2019 post from a sitting president are not the same
# instrument, even with identical text.
ERAS = [
    ("pre_potus", "2009-01-01", "2017-01-19"),
    ("term1",     "2017-01-20", "2021-01-20"),
    ("between",   "2021-01-21", "2025-01-19"),
    ("term2",     "2025-01-20", "2030-01-01"),
]

MIN_GROUP_N = 30       # below this a group's mean is not worth reporting


def era_for(date_str: str) -> str:
    for name, start, end in ERAS:
        if start <= date_str <= end:
            return name
    return "unknown"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_corpus(verbose: bool = True):
    """Corpus with parsed timestamps and an extracted signal per post."""
    import pandas as pd

    df = pd.read_csv(CORPUS)
    df = df[df.content.notna()].copy()
    df["content"] = df.content.astype(str)

    ts = pd.to_datetime(df.timestamp, format="mixed", utc=True, errors="coerce")
    df = df[ts.notna()].copy()
    df["ts_utc"] = ts[ts.notna()]
    df["ts_et"] = df.ts_utc.dt.tz_convert("America/New_York")
    df = df.sort_values("ts_utc").reset_index(drop=True)

    if verbose:
        print(f"  [corpus] {len(df):,} posts, "
              f"{df.ts_et.min().date()} .. {df.ts_et.max().date()}")
    return df


def load_daily_bars(tickers: list[str], start="2009-01-01", end=None,
                    verbose: bool = True):
    """Daily OHLC per ticker as {ticker: DataFrame indexed by session date}."""
    import pandas as pd
    import yfinance as yf

    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = yf.download(" ".join(tickers), start=start, end=end, interval="1d",
                      progress=False, auto_adjust=False, group_by="ticker")
    out = {}
    for t in tickers:
        try:
            df = raw[t].dropna(how="all").copy()
        except KeyError:
            continue
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        # Two windows, because which one is right depends on when he posted.
        #   ret      open→close  — the session after the post
        #   gap_ret  prev close→open — where an overnight post's reaction lands
        # Measuring only open→close would systematically miss the effect of
        # after-hours posting, which is most of it: by the open, the reaction has
        # already happened and we would be measuring the period after it.
        df["ret"] = (df["Close"] / df["Open"] - 1.0) * 100.0
        df["gap_ret"] = (df["Open"] / df["Close"].shift(1) - 1.0) * 100.0
        out[t] = df[["Open", "Close", "ret", "gap_ret"]].dropna()
    if verbose:
        n = len(next(iter(out.values()))) if out else 0
        print(f"  [bars] {len(out)} tickers, {n} sessions")
    return out


# ── Session mapping ──────────────────────────────────────────────────────────

def map_posts_to_sessions(df, sessions, verbose: bool = True):
    """
    Attach to each post the first session it could possibly have affected.

    `sessions` is the sorted DatetimeIndex of trading days. A post before 09:30
    ET on a trading day maps to that day; anything later maps to the next.
    """
    import numpy as np
    import pandas as pd

    post_dates = df.ts_et.dt.tz_localize(None).dt.normalize()
    before_open = (
        (df.ts_et.dt.hour < MARKET_OPEN_ET[0])
        | ((df.ts_et.dt.hour == MARKET_OPEN_ET[0])
           & (df.ts_et.dt.minute < MARKET_OPEN_ET[1]))
    )

    # searchsorted 'left' finds the same day when the post precedes the open;
    # 'right' pushes to the following session otherwise.
    idx = np.where(
        before_open,
        sessions.searchsorted(post_dates, side="left"),
        sessions.searchsorted(post_dates, side="right"),
    )
    valid = idx < len(sessions)
    out = df[valid].copy()
    out["session"] = sessions[idx[valid]]

    if verbose:
        dropped = (~valid).sum()
        print(f"  [map] {len(out):,} posts mapped"
              + (f", {dropped} after last session dropped" if dropped else ""))
    return out


# ── Event construction ───────────────────────────────────────────────────────

def build_events(df, bars, min_abs_signal: float = 0.05, verbose: bool = True):
    """
    One row per (session, ticker): the net signal from all posts mapped to that
    session, and the realised open→close return.
    """
    agg: dict[tuple, dict] = defaultdict(
        lambda: {"score": 0.0, "n_posts": 0, "topics": defaultdict(float)})

    for row in df.itertuples():
        s = sig.extract(row.content)
        if not s.tickers:
            continue
        for ticker, score in s.tickers.items():
            if abs(score) < min_abs_signal or ticker not in bars:
                continue
            cell = agg[(row.session, ticker)]
            cell["score"] += score
            cell["n_posts"] += 1
            cell["topics"][s.topic] += abs(score)

    # Unconditional mean return per ticker — the drift baseline every excess
    # return is measured against.
    baseline = {t: float(b["ret"].mean()) for t, b in bars.items()}
    baseline_gap = {t: float(b["gap_ret"].mean()) for t, b in bars.items()}

    events = []
    for (session, ticker), cell in agg.items():
        bar = bars[ticker]
        if session not in bar.index:
            continue
        ret = float(bar.loc[session, "ret"])
        gap = float(bar.loc[session, "gap_ret"])
        direction = 1 if cell["score"] > 0 else -1
        dominant = max(cell["topics"].items(), key=lambda kv: kv[1])[0]
        events.append({
            "session": session,
            "date": session.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "topic": dominant,
            "era": era_for(session.strftime("%Y-%m-%d")),
            "score": round(cell["score"], 4),
            "direction": direction,
            "n_posts": cell["n_posts"],
            "raw_return": round(ret, 5),
            "signed_return": round(direction * ret, 5),
            # Excess strips the ticker's drift, so a long-biased signal is not
            # rewarded for the market simply having gone up over sixteen years.
            "excess_return": round(direction * (ret - baseline[ticker]), 5),
            "gap_return": round(direction * gap, 5),
            "excess_gap": round(direction * (gap - baseline_gap[ticker]), 5),
        })

    if verbose:
        print(f"  [events] {len(events):,} (session, ticker) events")
    return events, baseline


# ── Statistics ───────────────────────────────────────────────────────────────

def _stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": round(mean, 5), "t": 0.0, "hit": 0.0}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    return {
        "n": n,
        "mean": round(mean, 5),
        "t": round(mean / se, 3) if se > 0 else 0.0,
        "hit": round(sum(1 for v in values if v > 0) / n, 4),
        "sd": round(math.sqrt(var), 4),
    }


def _norm_sf(z: float) -> float:
    """Two-sided normal tail probability; adequate for the n here."""
    return math.erfc(abs(z) / math.sqrt(2))


def group_by(events: list[dict], key: str, field: str = "excess_return",
             min_n: int = MIN_GROUP_N, cluster: bool = True) -> list[dict]:
    """
    Group statistics, clustered by session within each group.

    Without clustering, a day on which he posted about Russia and the market
    moved contributes one row per affected ticker — several correlated draws of
    the same underlying event. That inflates n within the group and overstates t
    exactly as it did in the intraday study.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        groups[e[key]].append(e)

    rows = []
    for name, evs in groups.items():
        if cluster:
            per: dict = defaultdict(list)
            for e in evs:
                per[e["session"]].append(e[field])
            vals = [sum(v) / len(v) for v in per.values()]
        else:
            vals = [e[field] for e in evs]
        st = _stats(vals)
        if st["n"] >= min_n:
            rows.append({"group": name, "n_raw": len(evs), **st,
                         "p": round(_norm_sf(st["t"]), 5)})
    return sorted(rows, key=lambda r: abs(r["t"]), reverse=True)


def correct_multiple(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """
    Benjamini-Hochberg across every group tested.

    This study compares ~25 groups. At alpha=0.05 roughly one will clear |t|>1.96
    by chance alone, so an uncorrected nominal hit is not evidence of anything.
    BH controls the false discovery rate while being less brutal than Bonferroni.
    """
    ranked = sorted(rows, key=lambda r: r["p"])
    m = len(ranked)
    survivors = 0
    for i, r in enumerate(ranked, 1):
        r["bh_threshold"] = round(alpha * i / m, 5)
        r["survives_bh"] = r["p"] <= r["bh_threshold"]
        if r["survives_bh"]:
            survivors = i
    # BH: everything up to the largest passing rank survives.
    for i, r in enumerate(ranked, 1):
        r["survives_bh"] = i <= survivors
    return ranked


def cluster_by_session(events: list[dict], field: str = "excess_return") -> list[float]:
    """
    One observation per session.

    Within a session the per-ticker rows share the same posts and largely the
    same market move, so they are not independent draws.
    """
    per: dict = defaultdict(list)
    for e in events:
        per[e["session"]].append(e[field])
    return [sum(v) / len(v) for v in per.values()]


def slice_report(events: list[dict], field: str = "excess_return") -> dict:
    """
    The slices that would reveal an effect hiding inside the average.

    A null overall result is only meaningful if the obvious conditional stories
    are also null: that the effect lives in the rare extreme post, in days he
    posts repeatedly on one topic, or only while he holds office.
    """
    out: dict[str, list] = {}
    srt = sorted(events, key=lambda e: abs(e["score"]))
    n = len(srt)

    out["deciles"] = [
        (f"D{i + 1}" + (" (weakest)" if i == 0 else " (strongest)" if i == 9 else ""),
         srt[i * n // 10:(i + 1) * n // 10])
        for i in range(10)
    ]
    out["tail"] = [
        ("bottom 50%", srt[:n // 2]),
        ("top 5% |score|", srt[int(n * 0.95):]),
        ("top 1% |score|", srt[int(n * 0.99):]),
    ]
    vol: dict[str, list] = defaultdict(list)
    for e in events:
        k = ("1 post" if e["n_posts"] == 1 else
             "2-4 posts" if e["n_posts"] <= 4 else
             "5-9 posts" if e["n_posts"] <= 9 else "10+ posts")
        vol[k].append(e)
    out["volume"] = sorted(vol.items(), key=lambda kv: kv[0])
    return out


def summarise_slices(buckets: list[tuple], field: str = "excess_return",
                     min_n: int = 20) -> list[dict]:
    rows = []
    for name, evs in buckets:
        vals = cluster_by_session(evs, field)
        if len(vals) < min_n:
            continue
        st = _stats(vals)
        rows.append({"group": name, **st, "p": round(_norm_sf(st["t"]), 5)})
    return rows


def _fmt(rows: list[dict], title: str, note: str = "") -> None:
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    has_bh = any("survives_bh" in r for r in rows)
    hdr = f"  {'group':<16}{'sessions':>9}{'mean%':>10}{'hit%':>8}{'t':>8}{'p':>9}"
    print(hdr + ("   verdict" if has_bh else ""))
    print("  " + "-" * (len(hdr) + (10 if has_bh else 0)))
    for r in rows:
        verdict = ""
        if has_bh:
            verdict = "   SURVIVES" if r.get("survives_bh") else (
                "   nominal" if abs(r["t"]) >= 1.96 else "   -")
        p = r.get("p", float("nan"))
        print(f"  {str(r['group'])[:15]:<16}{r['n']:>9}{r['mean']:>10.4f}"
              f"{r['hit'] * 100:>8.1f}{r['t']:>8.2f}{p:>9.4f}{verdict}")


def run(min_abs_signal: float = 0.05, tickers: list[str] | None = None,
        verbose: bool = True) -> dict:
    from .market import DEFAULT_TICKERS
    tickers = tickers or [t for t in DEFAULT_TICKERS if t not in ("COIN", "MSTR")]

    print("Historical event study — full corpus, daily open→close\n")
    df = load_corpus(verbose)
    bars = load_daily_bars(tickers, verbose=verbose)
    if not bars:
        print("  no market data"); return {}

    import pandas as pd
    sessions = pd.DatetimeIndex(sorted(next(iter(bars.values())).index))
    df = map_posts_to_sessions(df, sessions, verbose)
    events, baseline = build_events(df, bars, min_abs_signal, verbose)
    return {"events": events, "baseline": baseline}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-signal", type=float, default=0.05)
    ap.add_argument("--min-n", type=int, default=MIN_GROUP_N)
    ap.add_argument("--window", choices=["session", "gap"], default="session",
                    help="session = open→close; gap = prev close→open "
                         "(where an after-hours post's reaction lands)")
    ap.add_argument("--slices", action="store_true",
                    help="also test magnitude deciles, the extreme tail, and "
                         "posting volume — where an effect would hide")
    args = ap.parse_args()
    FIELD = "excess_return" if args.window == "session" else "excess_gap"

    res = run(min_abs_signal=args.min_signal)
    events = res.get("events", [])
    if not events:
        raise SystemExit("no events")

    all_excess = [e[FIELD] for e in events]
    clustered = cluster_by_session(events, FIELD)

    print("\n" + "=" * 62)
    print("  EXCESS RETURN (signal direction, drift removed)")
    print("=" * 62)
    _fmt([{"group": "all events", **_stats(all_excess)}], "Overall — per (session, ticker)")
    _fmt([{"group": "all sessions", **_stats(clustered)}],
         "Clustered — one per session  (READ THIS ONE)",
         "per-ticker rows within a session are not independent")

    topic_rows  = group_by(events, "topic",  field=FIELD, min_n=args.min_n)
    era_rows    = group_by(events, "era",    field=FIELD, min_n=args.min_n)
    ticker_rows = group_by(events, "ticker", field=FIELD, min_n=args.min_n)

    # One correction across every comparison made, not per-table: the false
    # positives come from the total number of groups tested.
    corrected = correct_multiple(topic_rows + era_rows + ticker_rows)
    by_name = {(r["group"], r["n"]): r for r in corrected}
    relabel = lambda rows: [by_name.get((r["group"], r["n"]), r) for r in rows]

    _fmt(relabel(topic_rows), "By topic",
         "which kinds of post actually move markets")
    _fmt(relabel(era_rows), "By era",
         "a private citizen and a sitting president are different instruments")
    _fmt(relabel(ticker_rows), "By ticker")

    n_tests = len(corrected)
    n_nominal = sum(1 for r in corrected if abs(r["t"]) >= 1.96)
    n_survive = sum(1 for r in corrected if r.get("survives_bh"))
    print(f"\n  {n_tests} groups tested. {n_nominal} nominally significant "
          f"(|t|>1.96); {n_survive} survive Benjamini-Hochberg at FDR 5%.")
    print(f"  Pure chance would produce about {n_tests * 0.05:.1f} nominal hits.")
    if args.slices:
        sl = slice_report(events, FIELD)
        _fmt(summarise_slices(sl["deciles"], FIELD), "Signal magnitude deciles",
             "if extreme posts moved markets, D10 would stand out")
        _fmt(summarise_slices(sl["tail"], FIELD), "Extreme tail",
             "the famous market-moving posts, if they exist as a class")
        _fmt(summarise_slices(sl["volume"], FIELD), "Posting volume that session")

    if n_survive == 0:
        print("\n  VERDICT: no category of post shows a daily-resolution effect that")
        print("  survives correction. Nominal hits below are consistent with noise.")
