"""
Event study and paper trading.

Two questions, in the order they have to be answered:

  1. Do his ACTUAL posts move prices in the direction the signal claims?
     If not, nothing downstream can work, because a perfect prediction of a post
     that moves nothing is worth nothing. This is the prerequisite.

  2. Do PREDICTED posts carry the same information ahead of time?
     Only worth asking once (1) is answered yes.

Both are run against real bars from market.py. No simulated prices anywhere.

Where look-ahead is avoided:
  * entry is the first bar strictly at or after the event (market.event_return)
  * a post outside market hours is entered at the next session, not backdated
  * the signal for a post is computed from that post's own text only
  * for predictions, the signal comes from the immutable prediction row written
    before the outcome existed

What this cannot rule out: the news that triggered the post was already public,
so any measured move may be the market reacting to the news rather than to him.
Interpreting a positive result as tradeable requires a news-only control, which
`--control` reports alongside.
"""

import math
import uuid
from datetime import datetime, timedelta, timezone

from . import signal as sig
from .store import connect, utcnow

DEFAULT_WINDOW_MIN = 60
MIN_ABS_SIGNAL = 0.05     # below this the signal is noise; no position taken

# Rough floor for saying anything about a per-event edge of the size plausible
# here (a few basis points against intraday vol). Below this the result is a
# coin-flip dressed as a finding — the CLI says so rather than printing a
# confident-looking table.
MIN_EVENTS_FOR_INFERENCE = 200


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _tstat(xs: list[float]) -> float:
    """One-sample t against zero mean."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return 0.0
    return round(mean / math.sqrt(var / n), 3)


def _summarise(trades: list[dict], label: str) -> dict:
    """Aggregate signed returns. `pnl_pct` is the return as the signal directed."""
    if not trades:
        return {"label": label, "n": 0}
    signed = [t["signed_return"] for t in trades]
    wins = [s for s in signed if s > 0]
    mean = sum(signed) / len(signed)
    return {
        "label": label,
        "n": len(signed),
        "hit_rate": round(len(wins) / len(signed), 4),
        "mean_return_pct": round(mean, 5),
        "median_return_pct": round(sorted(signed)[len(signed) // 2], 5),
        "total_return_pct": round(sum(signed), 4),
        "t_stat": _tstat(signed),
        "best": round(max(signed), 4),
        "worst": round(min(signed), 4),
    }


# ── 1. Do actual posts move prices? ──────────────────────────────────────────

def event_study(window_min: int = DEFAULT_WINDOW_MIN, min_abs_signal: float = MIN_ABS_SIGNAL,
                db_path=None, verbose: bool = True) -> dict:
    """
    For every stored post with text, take the signal's directional call on each
    mapped ticker and measure the realised return over the following window.
    """
    from .market import event_return

    kwargs = {"db_path": db_path} if db_path else {}
    trades: list[dict] = []
    skipped = {"not_analysable": 0, "no_signal": 0, "no_bars": 0}

    with connect(**kwargs) as conn:
        posts = conn.execute(
            "SELECT * FROM posts WHERE length(text) >= 25 AND is_reblog = 0 "
            "ORDER BY created_at"
        ).fetchall()

        for p in posts:
            if not sig.is_analysable(p["text"]):
                skipped["not_analysable"] = skipped.get("not_analysable", 0) + 1
                continue
            s = sig.extract(p["text"])
            active = {t: v for t, v in s.tickers.items() if abs(v) >= min_abs_signal}
            if not active:
                skipped["no_signal"] += 1
                continue
            evt = _parse(p["created_at"])
            for ticker, score in active.items():
                r = event_return(conn, ticker, evt, window_min)
                if r is None:
                    skipped["no_bars"] += 1
                    continue
                direction = 1 if score > 0 else -1
                trades.append({
                    "post_id": p["id"],
                    "ticker": ticker,
                    "topic": s.topic,
                    "signal": round(score, 4),
                    "direction": direction,
                    "raw_return": r["return_pct"],
                    "signed_return": round(direction * r["return_pct"], 5),
                    "entry_ts": r["entry_ts"],
                    "event_ts": p["created_at"],
                })

    if verbose:
        print(f"  [event] {len(trades)} events, skipped: {skipped}")
    return {"trades": trades, "skipped": skipped,
            "overall": _summarise(trades, f"actual posts @{window_min}m")}


def cluster_by_event(trades: list[dict], key: str = "post_id") -> list[dict]:
    """
    Collapse to one observation per post.

    The per-ticker rows are NOT independent: one post emits a position in several
    tickers at once, and those tickers co-move (SPY/QQQ/DIA/IWM are largely the
    same bet). Treating them as separate observations inflates n several-fold and
    makes every t-stat and p-value look stronger than the evidence supports.
    Averaging within a post gives one genuinely independent observation per event,
    which is what inference should run on.
    """
    groups: dict[str, list[dict]] = {}
    for t in trades:
        groups.setdefault(t.get(key) or t.get("prediction_id", "?"), []).append(t)
    out = []
    for gid, rows in groups.items():
        out.append({
            **rows[0],
            "signed_return": sum(r["signed_return"] for r in rows) / len(rows),
            "raw_return": sum(abs(r["raw_return"]) for r in rows) / len(rows),
            "n_legs": len(rows),
        })
    return out


def by_topic(trades: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for t in trades:
        groups.setdefault(t["topic"], []).append(t)
    out = [_summarise(v, k) for k, v in groups.items()]
    return sorted(out, key=lambda r: r.get("n", 0), reverse=True)


def by_ticker(trades: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for t in trades:
        groups.setdefault(t["ticker"], []).append(t)
    out = [_summarise(v, k) for k, v in groups.items()]
    return sorted(out, key=lambda r: r.get("n", 0), reverse=True)


# ── 2. Do predictions carry the signal ahead of time? ────────────────────────

def prediction_backtest(window_min: int = DEFAULT_WINDOW_MIN,
                        min_abs_signal: float = MIN_ABS_SIGNAL,
                        db_path=None, verbose: bool = True) -> dict:
    """
    Same measurement, but the position is taken at the PREDICTION timestamp using
    only the signal stored in the immutable prediction row.
    """
    from .market import event_return

    kwargs = {"db_path": db_path} if db_path else {}
    trades: list[dict] = []

    with connect(**kwargs) as conn:
        preds = conn.execute(
            "SELECT * FROM predictions WHERE text IS NOT NULL AND text != '' "
            "ORDER BY predicted_at"
        ).fetchall()

        for p in preds:
            s = sig.extract(p["text"])
            active = {t: v for t, v in s.tickers.items() if abs(v) >= min_abs_signal}
            if not active:
                continue
            evt = _parse(p["predicted_at"])
            for ticker, score in active.items():
                r = event_return(conn, ticker, evt, window_min)
                if r is None:
                    continue
                direction = 1 if score > 0 else -1
                trades.append({
                    "prediction_id": p["id"],
                    "model": p["model"],
                    "ticker": ticker,
                    "topic": s.topic,
                    "signal": round(score, 4),
                    "direction": direction,
                    "raw_return": r["return_pct"],
                    "signed_return": round(direction * r["return_pct"], 5),
                    "entry_ts": r["entry_ts"],
                })

    by_model: dict[str, list[dict]] = {}
    for t in trades:
        by_model.setdefault(t["model"], []).append(t)

    if verbose:
        print(f"  [pred-bt] {len(trades)} prediction-events")
    return {
        "trades": trades,
        "overall": _summarise(trades, f"predictions @{window_min}m"),
        "by_model": sorted((_summarise(v, k) for k, v in by_model.items()),
                           key=lambda r: r.get("n", 0), reverse=True),
    }


# ── Control: random direction on the same events ─────────────────────────────

def random_control(trades: list[dict], trials: int = 2000, seed: int = 0) -> dict:
    """
    Same events and windows, directions assigned at random, averaged over many
    trials.

    A single coin-flip draw is not a baseline — at small n its hit rate swings
    wildly and can look far better or worse than the real strategy purely by
    chance. Averaging over `trials` draws gives the expected random outcome, and
    `p_value` reports the fraction of random draws that beat the real strategy's
    mean return, which is the number worth reading.
    """
    import random
    if not trades:
        return {"label": "random control", "n": 0}

    rng = random.Random(seed)
    actual_mean = sum(t["signed_return"] for t in trades) / len(trades)
    means, hits = [], []
    for _ in range(trials):
        signed = [abs(t["raw_return"]) * rng.choice([1, -1]) for t in trades]
        means.append(sum(signed) / len(signed))
        hits.append(sum(1 for s in signed if s > 0) / len(signed))

    beat = sum(1 for m in means if m >= actual_mean)
    return {
        "label": f"random x{trials}",
        "n": len(trades),
        "hit_rate": round(sum(hits) / trials, 4),
        "mean_return_pct": round(sum(means) / trials, 5),
        "median_return_pct": round(sorted(means)[trials // 2], 5),
        "total_return_pct": round(sum(means) / trials * len(trades), 4),
        "t_stat": 0.0,
        "best": round(max(means), 4),
        "worst": round(min(means), 4),
        "p_value": round(beat / trials, 4),
    }


# ── Paper trading ledger ─────────────────────────────────────────────────────

def open_paper_trades(window_min: int = DEFAULT_WINDOW_MIN, notional: float = 1000.0,
                      min_abs_signal: float = MIN_ABS_SIGNAL, db_path=None) -> int:
    """
    Record intended positions for predictions that have no ledger entry yet.

    Nothing here contacts a broker. Entry price is filled in only when a real bar
    at or after the prediction exists, so a position opened outside market hours
    stays pending rather than being filled at a fabricated price.
    """
    from .market import event_return

    kwargs = {"db_path": db_path} if db_path else {}
    opened = 0
    with connect(**kwargs) as conn:
        preds = conn.execute(
            """SELECT p.* FROM predictions p
               LEFT JOIN paper_trades t ON t.prediction_id = p.id
               WHERE t.prediction_id IS NULL AND p.text IS NOT NULL AND p.text != ''"""
        ).fetchall()

        for p in preds:
            s = sig.extract(p["text"])
            active = {t: v for t, v in s.tickers.items() if abs(v) >= min_abs_signal}
            evt = _parse(p["predicted_at"])
            for ticker, score in active.items():
                r = event_return(conn, ticker, evt, window_min)
                direction = 1 if score > 0 else -1
                conn.execute(
                    """INSERT INTO paper_trades
                       (id, prediction_id, ticker, direction, opened_at, closed_at,
                        entry_price, exit_price, return_pct, notional, status, note)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        uuid.uuid4().hex[:16], p["id"], ticker, direction,
                        p["predicted_at"],
                        r["exit_ts"] if r else None,
                        r["entry_price"] if r else None,
                        r["exit_price"] if r else None,
                        round(direction * r["return_pct"], 5) if r else None,
                        notional,
                        "closed" if r else "open",
                        f"signal={score:+.3f} topic={s.topic}",
                    ),
                )
                opened += 1
    return opened


def ledger_summary(db_path=None) -> dict:
    kwargs = {"db_path": db_path} if db_path else {}
    with connect(**kwargs) as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE status='closed' AND return_pct IS NOT NULL"
        ).fetchall()
        openn = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status='open'").fetchone()[0]
    trades = [{"signed_return": r["return_pct"], "raw_return": r["return_pct"]} for r in rows]
    out = _summarise(trades, "paper ledger")
    out["open_positions"] = openn
    if trades:
        out["pnl_dollars"] = round(
            sum(r["return_pct"] / 100 * r["notional"] for r in rows), 2)
    return out


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"  {'group':<18}{'n':>6}{'hit%':>8}{'mean%':>10}{'total%':>10}{'t':>8}")
    print("  " + "-" * 60)
    for r in rows:
        if not r.get("n"):
            continue
        print(f"  {r['label'][:17]:<18}{r['n']:>6}{r['hit_rate'] * 100:>8.1f}"
              f"{r['mean_return_pct']:>10.4f}{r['total_return_pct']:>10.3f}"
              f"{r['t_stat']:>8.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Event study / backtest on real prices")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW_MIN,
                    help="holding window in minutes")
    ap.add_argument("--control", action="store_true", help="include random-direction control")
    ap.add_argument("--predictions", action="store_true", help="also backtest predictions")
    args = ap.parse_args()

    print(f"Event study — actual posts, {args.window}-minute window\n")
    res = event_study(window_min=args.window)
    o = res["overall"]

    clustered = cluster_by_event(res["trades"]) if res["trades"] else []

    if not o.get("n"):
        print("\n  No events with both a signal and overlapping market bars yet.")
        print("  Collect posts (run_cycle.py) and backfill bars (pipeline.market).")
    else:
        _print_table("Overall", [o])
        _print_table("By topic", by_topic(res["trades"]))
        _print_table("By ticker", by_ticker(res["trades"]))
        # Inference runs on the clustered sample, not the per-ticker rows.
        co = _summarise(clustered, f"clustered by post @{args.window}m")
        _print_table("Clustered — one observation per post (use THIS for inference)",
                     [co])
        print(f"  {len(res['trades'])} ticker-legs collapse to {co['n']} "
              f"independent events")

        if args.control:
            ctl_raw = random_control(res["trades"])
            ctl_cl = random_control(clustered)
            _print_table("Control", [ctl_raw, ctl_cl])
            print(f"\n  p(random >= actual), per-leg   = {ctl_raw['p_value']}  "
                  f"(OVERSTATED — legs are correlated)")
            print(f"  p(random >= actual), clustered = {ctl_cl['p_value']}  "
                  f"(the one to read)")

        n = co["n"]
        if n < MIN_EVENTS_FOR_INFERENCE:
            print(f"\n  ⚠️  n={n} is far below the ~{MIN_EVENTS_FOR_INFERENCE} events "
                  f"needed to distinguish a real effect from noise.\n"
                  f"      At this sample size neither a positive nor a negative result "
                  f"means anything.\n"
                  f"      Keep run_cycle.py collecting and re-run in a few weeks.")

    if args.predictions:
        pres = prediction_backtest(window_min=args.window)
        if pres["overall"].get("n"):
            _print_table("Predictions overall", [pres["overall"]])
            _print_table("Predictions by model", pres["by_model"])
        else:
            print("\n  No prediction-events with market bars yet.")
