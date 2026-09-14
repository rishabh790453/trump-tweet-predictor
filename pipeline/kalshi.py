"""
Kalshi weekly Truth Social post-count markets (series KXTRUTHSOCIAL).

Why this target rather than equities: the historical study in history.py found no
predictable price impact from post *content*, which is a hard efficient-market
problem. This market asks a different question — how many times he posts in a
week — and posting volume is far more forecastable than price direction.

Market structure (verified live 2026-09-13):
  * 10 brackets: <80, 80-99, 100-119, ... 220-240, >240
  * Week runs Sunday 00:00 ET to Saturday 23:59 ET
  * Opens Sunday 14:00 UTC, closes the following Sunday 13:59 UTC
  * Settles off Roll Call / Factbase, NOT the Truth Social API

That last point is the main model risk, so it was checked rather than assumed:
our API count for the week of 2026-09-06 was 165, and Kalshi settled that week
at 160-179. The conventions agree closely enough to model against, but this is
one observation — `calibrate_against_settled()` re-checks it every week and is
the first thing to look at if forecasts start missing.

The edge this targets is not a better prior. It is that the market stays open all
week while information arrives continuously: by Wednesday night roughly 58% of
the week's posts have already happened. A forecast conditioned on the partial
count is far sharper than the prior, and the live loop already collects exactly
that data.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from .store import connect

API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXTRUTHSOCIAL"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Bracket ladder, as (low, high_inclusive, label). Open-ended ends use ±inf.
BRACKETS: list[tuple[float, float, str]] = [
    (0, 79, "<80"), (80, 99, "80-99"), (100, 119, "100-119"),
    (120, 139, "120-139"), (140, 159, "140-159"), (160, 179, "160-179"),
    (180, 199, "180-199"), (200, 219, "200-219"), (220, 240, "220-240"),
    (241, math.inf, ">240"),
]

# Share of a week's posts falling on each day, Sunday first. Measured over the
# Truth Social era; close to uniform, which makes elapsed-fraction a good
# exposure proxy.
DAY_PROFILE = [0.132, 0.135, 0.168, 0.147, 0.145, 0.137, 0.136]


def week_start_et(dt: datetime | None = None) -> datetime:
    """Sunday 00:00 ET of the week containing dt."""
    import zoneinfo
    et = zoneinfo.ZoneInfo("America/New_York")
    dt = (dt or datetime.now(timezone.utc)).astimezone(et)
    days_since_sunday = (dt.weekday() + 1) % 7
    start = (dt - timedelta(days=days_since_sunday)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return start


def elapsed_fraction(now: datetime | None = None) -> float:
    """
    How much of the current week's posting exposure has passed.

    Uses the day profile rather than raw elapsed time, so a Tuesday evening is
    correctly treated as more than 2/7 of the week when Tuesdays are heavy.
    """
    import zoneinfo
    et = zoneinfo.ZoneInfo("America/New_York")
    now = (now or datetime.now(timezone.utc)).astimezone(et)
    start = week_start_et(now)
    day_idx = (now - start).days
    if day_idx >= 7:
        return 1.0
    within_day = ((now - start) - timedelta(days=day_idx)).total_seconds() / 86400.0
    return min(sum(DAY_PROFILE[:day_idx]) + DAY_PROFILE[day_idx] * within_day, 1.0)


# ── Market data ──────────────────────────────────────────────────────────────

def fetch_markets(series: str = SERIES, limit_pages: int = 12) -> list[dict]:
    out, cursor = [], None
    for _ in range(limit_pages):
        params = {"series_ticker": series, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{API}/markets", params=params, headers=HEADERS, timeout=25)
        r.raise_for_status()
        d = r.json()
        page = d.get("markets", [])
        out += page
        cursor = d.get("cursor")
        if not cursor or not page:
            break
    return out


def settled_weeks(markets: list[dict]) -> list[dict]:
    """One row per settled event: which bracket resolved YES."""
    by_event: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        by_event[m["event_ticker"]].append(m)

    rows = []
    for ev, ms in sorted(by_event.items()):
        win = [m for m in ms if m.get("result") == "yes"]
        if not win:
            continue
        w = win[0]
        label = w.get("subtitle") or w.get("yes_sub_title") or w["ticker"].split("-")[-1]
        rows.append({
            "event": ev,
            "label": label,
            "floor": w.get("floor_strike"),
            "cap": w.get("cap_strike"),
            "close_time": w.get("close_time", ""),
        })
    return rows


def open_week(markets: list[dict]) -> list[dict]:
    return [m for m in markets if m.get("status") == "active"]


# ── Count history ────────────────────────────────────────────────────────────

def weekly_counts_from_db(db_path=None) -> dict:
    """Exact weekly counts from collected ground truth, keyed by week-start date."""
    import pandas as pd
    kwargs = {"db_path": db_path} if db_path else {}
    with connect(**kwargs) as conn:
        rows = conn.execute("SELECT created_at FROM posts").fetchall()
    if not rows:
        return {}
    t = pd.to_datetime([r["created_at"] for r in rows], utc=True, format="mixed")
    et = t.tz_convert("America/New_York")
    wk = (et - pd.to_timedelta((et.dayofweek + 1) % 7, unit="D")).normalize()
    s = pd.Series(1, index=wk).groupby(level=0).sum()
    return {d.date().isoformat(): int(n) for d, n in s.items()}


def weekly_counts_from_corpus(since: str = "2022-02-01") -> dict:
    import pandas as pd
    from .history import CORPUS
    d = pd.read_csv(CORPUS)
    ts = pd.to_datetime(d.timestamp, format="mixed", utc=True, errors="coerce")
    d = d[ts.notna()].copy()
    et = ts[ts.notna()].dt.tz_convert("America/New_York")
    d = d[et >= since]
    et = et[et >= since]
    wk = (et - pd.to_timedelta((et.dt.dayofweek + 1) % 7, unit="D")).dt.normalize()
    s = wk.groupby(wk).size()
    return {d.date().isoformat(): int(n) for d, n in s.items()}


# ── Forecast model ───────────────────────────────────────────────────────────

@dataclass
class GammaPoisson:
    """
    Weekly count as Gamma-Poisson (negative binomial marginal).

    Poisson alone is badly wrong here: measured variance/mean is ~11 daily and
    ~17 weekly, so a Poisson model would put almost all mass in one or two
    brackets and be confidently incorrect most weeks. The Gamma prior on the rate
    absorbs that overdispersion, and — the reason for this choice — it is
    conjugate, so conditioning on a partial week is exact rather than simulated.
    """
    alpha: float        # shape
    beta: float         # rate, in units of "per week"

    @classmethod
    def fit(cls, counts: list[float], min_weeks: int = 6) -> "GammaPoisson":
        if len(counts) < min_weeks:
            raise ValueError(f"need >= {min_weeks} weeks, got {len(counts)}")
        n = len(counts)
        m = sum(counts) / n
        v = sum((c - m) ** 2 for c in counts) / (n - 1)
        # Method of moments: marginal mean = a/b, marginal var = a/b + a/b^2.
        if v <= m:
            # Underdispersed (or too few weeks to see spread). Fall back to a
            # weakly-informative prior rather than a negative beta.
            beta = 1.0
        else:
            beta = m / (v - m)
        return cls(alpha=max(m * beta, 1e-6), beta=max(beta, 1e-6))

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    def posterior(self, observed: int, exposure: float) -> "GammaPoisson":
        """Conjugate update after seeing `observed` posts over `exposure` of a week."""
        return GammaPoisson(self.alpha + observed, self.beta + exposure)

    def predictive_pmf(self, exposure: float, max_k: int = 600) -> list[float]:
        """
        Negative-binomial PMF for counts over `exposure` weeks.

        Computed in log space: r can exceed 100 here and the direct ratio form
        overflows well before the distribution's upper tail is reached.
        """
        if exposure <= 0:
            out = [0.0] * (max_k + 1)
            out[0] = 1.0
            return out
        r = self.alpha
        p = self.beta / (self.beta + exposure)     # P(success)
        log_p, log_q = math.log(p), math.log1p(-p)
        out = []
        log_pmf = r * log_p
        for k in range(max_k + 1):
            if k > 0:
                log_pmf += math.log(r + k - 1) - math.log(k) + log_q
            out.append(math.exp(log_pmf))
        total = sum(out) or 1.0
        return [v / total for v in out]


def bracket_probabilities(model: GammaPoisson, observed: int = 0,
                          exposure: float = 0.0, max_k: int = 600) -> list[dict]:
    """
    P(final weekly total falls in each bracket).

    With a partial week, `observed` posts are already banked and only the
    remaining exposure is uncertain — so the distribution is shifted by
    `observed`, not re-centred on the prior.
    """
    post = model.posterior(observed, exposure) if exposure > 0 else model
    remaining = max(1.0 - exposure, 0.0)
    pmf = post.predictive_pmf(remaining, max_k)

    out = []
    for lo, hi, label in BRACKETS:
        lo_r = max(int(math.ceil(lo - observed)), 0)
        hi_r = hi - observed
        if hi_r < 0:
            p = 0.0
        else:
            top = max_k if math.isinf(hi_r) else min(int(math.floor(hi_r)), max_k)
            p = sum(pmf[lo_r:top + 1]) if lo_r <= top else 0.0
        # `hi` is math.inf for the open-ended top bracket, and Infinity is not
        # valid JSON — it serialises to a token no parser accepts, so the whole
        # response fails rather than just that field. Emit None at the boundary
        # and keep inf only inside the maths above.
        out.append({
            "label": label,
            "low": lo,
            "high": None if math.isinf(hi) else hi,
            "prob": round(p, 5),
        })
    return out


def settled_midpoints(settled: list[dict]) -> list[float]:
    """
    Turn settled brackets into approximate weekly counts.

    Coarse by construction — a settled "140-159" only tells us the count to
    within 20 — but it is the only source of *recent* weekly counts until our own
    collection goes back far enough, and recency dominates here: the corpus
    (ending Feb 2026) puts the weekly mean near 98, while these settled weeks put
    it near 197. Fitting the prior on the corpus alone would be confidently wrong.

    Open-ended brackets are assigned a point just past the boundary rather than
    something dramatic, which biases the fitted mean slightly DOWN. That is the
    safe direction: it will not manufacture confidence in the high brackets.
    """
    out = []
    for s in settled:
        lab = s["label"]
        if lab.startswith("<"):
            out.append(float(lab[1:]) * 0.85)
        elif lab.startswith(">"):
            out.append(float(lab[1:]) * 1.1)
        elif "-" in lab:
            lo, hi = lab.split("-")
            out.append((float(lo) + float(hi)) / 2.0)
    return out


def brier_score(probs: list[dict], winning_label: str) -> float:
    """Multi-category Brier: lower is better; 0 perfect, ~0.9 for uniform-over-10."""
    return round(sum((p["prob"] - (1.0 if p["label"] == winning_label else 0.0)) ** 2
                     for p in probs), 5)


def collection_freshness(db_path=None) -> dict:
    """
    How stale the partial-week count is.

    This matters more than it looks. The forecast conditions on "posts so far",
    but that figure is only as current as the last collection cycle. If the Mac
    slept for six hours, the model sees a quiet week and shifts mass to the low
    brackets — a confident, wrong forecast produced by missing data rather than
    by his behaviour. Always check this before acting on a number.
    """
    kwargs = {"db_path": db_path} if db_path else {}
    with connect(**kwargs) as conn:
        row = conn.execute("SELECT MAX(created_at) m FROM posts").fetchone()
        run = conn.execute(
            "SELECT MAX(started_at) s FROM runs WHERE status='ok'").fetchone()
    now = datetime.now(timezone.utc)

    def age_h(ts):
        if not ts:
            return None
        d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return round((now - d).total_seconds() / 3600, 2)

    post_age = age_h(row["m"] if row else None)
    run_age = age_h(run["s"] if run else None)
    # He rarely goes a full day silent, so a >6h gap since the last successful
    # cycle means we are probably missing posts rather than observing quiet.
    return {
        "last_post_age_h": post_age,
        "last_cycle_age_h": run_age,
        "stale": (run_age is None or run_age > 6.0),
    }


def backtest_settled(settled: list[dict], window: int = 6) -> dict:
    """
    Walk-forward over settled weeks: fit on prior weeks only, predict the next.

    Uses bracket midpoints as the fitted history, so it measures whether the
    *approach* calibrates — not the precision an exact-count history would give.
    Compared against two baselines, because a Brier score alone means nothing:
      * uniform over 10 brackets
      * the empirical distribution of prior weeks (a strong, dumb baseline)
    """
    mids = settled_midpoints(settled)
    labels = [s["label"] for s in settled]
    if len(mids) < window + 2:
        return {"n": 0, "reason": f"need > {window + 1} settled weeks, have {len(mids)}"}

    rows = []
    for i in range(window, len(mids)):
        train, actual_label = mids[:i], labels[i]
        if actual_label not in {b[2] for b in BRACKETS}:
            continue                      # ladder changed; bracket no longer exists
        model = GammaPoisson.fit(train[-window:])
        probs = bracket_probabilities(model)
        model_b = brier_score(probs, actual_label)

        uniform = [{"label": b[2], "prob": 1 / len(BRACKETS)} for b in BRACKETS]
        emp_counts = {b[2]: 0 for b in BRACKETS}
        for lab in labels[:i]:
            if lab in emp_counts:
                emp_counts[lab] += 1
        tot = sum(emp_counts.values()) or 1
        empirical = [{"label": k, "prob": v / tot} for k, v in emp_counts.items()]

        rows.append({
            "week": settled[i]["event"],
            "actual": actual_label,
            "model": model_b,
            "uniform": brier_score(uniform, actual_label),
            "empirical": brier_score(empirical, actual_label),
        })

    if not rows:
        return {"n": 0, "reason": "no scorable weeks"}
    mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 4)
    return {
        "n": len(rows), "rows": rows,
        "model_brier": mean("model"),
        "uniform_brier": mean("uniform"),
        "empirical_brier": mean("empirical"),
    }


def coverage_bounds(db_path=None) -> tuple[str, str] | None:
    """Earliest and latest post we hold, as ET dates."""
    import pandas as pd
    kwargs = {"db_path": db_path} if db_path else {}
    with connect(**kwargs) as conn:
        row = conn.execute(
            "SELECT MIN(created_at) a, MAX(created_at) b FROM posts").fetchone()
    if not row or not row["a"]:
        return None
    t = pd.to_datetime([row["a"], row["b"]], utc=True, format="mixed")
    et = t.tz_convert("America/New_York")
    return et[0].date().isoformat(), et[1].date().isoformat()


def calibrate_against_settled(counts: dict, settled: list[dict],
                              db_path=None) -> list[dict]:
    """
    Compare our own weekly count to Kalshi's settled bracket for the same week.

    Settlement is Factbase, not our API, so this checks the two conventions still
    agree. A count landing outside the settled bracket means the model is being
    fit to a different quantity than the one being traded.

    Weeks our collection only partially covers are reported as `incomplete`, not
    as mismatches. Counting them as mismatches would be wrong in the most
    misleading direction: a partially-collected week always undercounts, so it
    looks exactly like a systematic convention difference.
    """
    bounds = coverage_bounds(db_path)
    out = []
    for s in settled:
        # Event ticker ends with the week's SATURDAY date, e.g. -26SEP19.
        tag = s["event"].rsplit("-", 1)[-1]
        try:
            end = datetime.strptime(tag, "%y%b%d")
        except ValueError:
            continue
        start_d = (end - timedelta(days=6)).date()
        start = start_d.isoformat()
        ours = counts.get(start)
        if ours is None:
            continue

        complete = bool(bounds) and bounds[0] <= start and bounds[1] >= end.date().isoformat()
        lo, hi = None, None
        for blo, bhi, lab in BRACKETS:
            if lab == s["label"]:
                lo, hi = blo, bhi
        agrees = lo is not None and lo <= ours <= hi
        out.append({
            "week_start": start, "settled": s["label"], "our_count": ours,
            "complete": complete,
            "agrees": agrees if complete else None,
        })
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weeks", type=int, default=16,
                    help="recent weeks used to fit the prior")
    args = ap.parse_args()

    print("Kalshi KXTRUTHSOCIAL — weekly post-count forecast\n")

    db_counts = weekly_counts_from_db()
    print(f"  collected weeks (our API): {len(db_counts)}")
    for k, v in sorted(db_counts.items())[-6:]:
        print(f"    {k}  {v:>4}")

    markets = fetch_markets()
    settled = settled_weeks(markets)
    print(f"\n  Kalshi settled weeks: {len(settled)}")
    for s in settled[-8:]:
        print(f"    {s['event']:<26} {s['label']}")

    checks = calibrate_against_settled(db_counts, settled)
    if checks:
        print("\n  Settlement-convention check (our count vs settled bracket):")
        for c in checks:
            if not c["complete"]:
                mark = "incomplete coverage — not a mismatch"
            else:
                mark = "OK" if c["agrees"] else "MISMATCH — investigate"
            print(f"    {c['week_start']}  ours={c['our_count']:<5} "
                  f"settled={c['settled']:<10} {mark}")
        usable = [c for c in checks if c["complete"]]
        if usable:
            ok = sum(1 for c in usable if c["agrees"])
            print(f"    → {ok}/{len(usable)} fully-covered weeks agree")
    else:
        print("\n  No overlapping weeks yet to check the settlement convention.")

    # Fit on complete weeks only — the current partial week would drag the mean down.
    this_week = week_start_et().date().isoformat()
    complete = [v for k, v in sorted(db_counts.items()) if k != this_week]
    # Our own exact counts are best, but there are only a few weeks of them.
    # Settled brackets supply recent (coarse) counts to fill the gap.
    mids = settled_midpoints(settled)
    if len(complete) >= 6:
        fit_on, src = complete[-args.weeks:], "our exact counts"
    elif mids:
        fit_on, src = mids[-args.weeks:], f"Kalshi settled midpoints (coarse, ±10)"
    else:
        print("\n  Not enough history to fit. Run backfill_posts.py.")
        raise SystemExit(0)

    model = GammaPoisson.fit(fit_on)
    print(f"\n  Prior fitted on {len(fit_on)} weeks from {src}")
    print(f"    mean={model.mean:.1f}  alpha={model.alpha:.2f}  beta={model.beta:.3f}")
    corpus_mean = 97.8
    print(f"    (corpus through Feb 2026 would have given ~{corpus_mean:.0f}/week — "
          f"the rate roughly doubled since, so recency matters more than sample size)")

    bt = backtest_settled(settled)
    if bt.get("n"):
        print(f"\n  WALK-FORWARD BACKTEST on {bt['n']} settled weeks (Brier, lower better)")
        print(f"    model     {bt['model_brier']:.4f}")
        print(f"    empirical {bt['empirical_brier']:.4f}   (distribution of past weeks)")
        print(f"    uniform   {bt['uniform_brier']:.4f}   (1/10 each)")
        best = min(bt["model_brier"], bt["empirical_brier"], bt["uniform_brier"])
        if bt["model_brier"] > best:
            print("    → the model does NOT beat the naive baseline on this sample")
        print(f"    (n={bt['n']} weeks — far too few to conclude; indicative only)")
    else:
        print(f"\n  Backtest unavailable: {bt.get('reason')}")

    fresh = collection_freshness()
    frac = elapsed_fraction()
    observed = db_counts.get(this_week, 0)
    print(f"\n  Current week starting {this_week}: {observed} posts so far, "
          f"{frac * 100:.0f}% of week elapsed")
    print(f"  Collection freshness: last post {fresh['last_post_age_h']}h ago, "
          f"last cycle {fresh['last_cycle_age_h']}h ago")
    if fresh["stale"]:
        print("  ⚠️  STALE COLLECTION — the partial count is probably missing posts,")
        print("      which biases the forecast toward the LOW brackets. Run")
        print("      run_cycle.py before acting on these numbers.")

    print("\n  BRACKET PROBABILITIES")
    print(f"    {'bracket':<12}{'prior':>9}{'now':>9}")
    print("    " + "-" * 30)
    prior = bracket_probabilities(model)
    now = bracket_probabilities(model, observed, frac)
    for a, b in zip(prior, now):
        print(f"    {a['label']:<12}{a['prob'] * 100:>8.1f}%{b['prob'] * 100:>8.1f}%")
