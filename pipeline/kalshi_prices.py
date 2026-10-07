"""
Historic Kalshi price harvester.

This unblocks the question everything else waited on: *is the market already
priced correctly, and when?* Without prices we could measure our own calibration
but never edge.

Two endpoints return real prices with NO authentication. The one that matters:

    GET /series/{series}/markets/{ticker}/candlesticks
        ?start_ts&end_ts&period_interval={1|60|1440}

returning per-period OHLC plus yes_bid / yes_ask / volume / open_interest. Note
the path nests the market under its series — the flat /markets/{t}/candlesticks
form 404s, which is what made this look unavailable earlier. Prices live under
`*_dollars` keys; the legacy top-level fields (yes_bid, last_price) on the
/markets response are null unauthenticated, which is why the first survey
concluded there was no price data at all. There is.

Harvest urgently rather than carefully: Kalshi retains only ~11 events per
weekly series, so every week that passes without harvesting is settled price
history permanently lost.
"""

import argparse
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
DB = Path(__file__).parent.parent / "kalshi_prices.db"
PAUSE = 0.35

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    ticker        TEXT NOT NULL,
    series        TEXT NOT NULL,
    event         TEXT NOT NULL,
    ts            INTEGER NOT NULL,     -- end_period_ts, unix UTC
    interval_min  INTEGER NOT NULL,
    close         REAL, high REAL, low REAL, mean REAL, open REAL,
    yes_bid       REAL, yes_ask REAL,
    volume        REAL, open_interest REAL,
    PRIMARY KEY (ticker, ts, interval_min)
);
CREATE INDEX IF NOT EXISTS idx_candles_event ON candles(event, ts);

CREATE TABLE IF NOT EXISTS market_meta (
    ticker TEXT PRIMARY KEY, series TEXT, event TEXT,
    label TEXT, floor_strike REAL, cap_strike REAL,
    status TEXT, result TEXT, open_time TEXT, close_time TEXT
);
"""


def _conn():
    c = sqlite3.connect(str(DB), timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    return c


def _f(d, *path):
    """Nested lookup returning float or None — candles omit keys when untraded."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def fetch_markets(series: str) -> list[dict]:
    out, cursor = [], None
    for _ in range(14):
        p = {"series_ticker": series, "limit": 200}
        if cursor:
            p["cursor"] = cursor
        r = requests.get(f"{API}/markets", params=p, headers=HEADERS, timeout=30)
        r.raise_for_status()
        d = r.json()
        page = d.get("markets", [])
        out += page
        cursor = d.get("cursor")
        if not cursor or not page:
            break
    return out


def store_meta(markets: list[dict], series: str) -> None:
    rows = [{
        "ticker": m["ticker"], "series": series, "event": m["event_ticker"],
        "label": m.get("yes_sub_title") or m.get("subtitle") or "",
        "floor_strike": m.get("floor_strike"), "cap_strike": m.get("cap_strike"),
        "status": m.get("status"), "result": m.get("result"),
        "open_time": m.get("open_time"), "close_time": m.get("close_time"),
    } for m in markets]
    with _conn() as c:
        c.executemany("""INSERT OR REPLACE INTO market_meta
            (ticker,series,event,label,floor_strike,cap_strike,status,result,open_time,close_time)
            VALUES (:ticker,:series,:event,:label,:floor_strike,:cap_strike,
                    :status,:result,:open_time,:close_time)""", rows)


def fetch_candles(series: str, ticker: str, start_ts: int, end_ts: int,
                  interval: int = 60) -> list[dict]:
    r = requests.get(f"{API}/series/{series}/markets/{ticker}/candlesticks",
                     params={"start_ts": start_ts, "end_ts": end_ts,
                             "period_interval": interval},
                     headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return []
    return r.json().get("candlesticks", [])


def harvest(series: str, lookback_days: int = 75, interval: int = 60,
            verbose: bool = True) -> int:
    markets = fetch_markets(series)
    store_meta(markets, series)
    if verbose:
        ev = len({m["event_ticker"] for m in markets})
        print(f"{len(markets)} markets across {ev} events in {series}", flush=True)

    now = int(time.time())
    start = now - lookback_days * 86400
    total = 0
    for i, m in enumerate(markets, 1):
        tk = m["ticker"]
        cs = fetch_candles(series, tk, start, now, interval)
        rows = []
        for c in cs:
            rows.append({
                "ticker": tk, "series": series, "event": m["event_ticker"],
                "ts": c.get("end_period_ts"), "interval_min": interval,
                "close": _f(c, "price", "close_dollars"),
                "high": _f(c, "price", "high_dollars"),
                "low": _f(c, "price", "low_dollars"),
                "mean": _f(c, "price", "mean_dollars"),
                "open": _f(c, "price", "open_dollars"),
                "yes_bid": _f(c, "yes_bid", "close_dollars"),
                "yes_ask": _f(c, "yes_ask", "close_dollars"),
                "volume": _f(c, "volume_fp"),
                "open_interest": _f(c, "open_interest_fp"),
            })
        rows = [r for r in rows if r["ts"]]
        if rows:
            with _conn() as c:
                c.executemany("""INSERT OR REPLACE INTO candles
                   (ticker,series,event,ts,interval_min,close,high,low,mean,open,
                    yes_bid,yes_ask,volume,open_interest)
                   VALUES (:ticker,:series,:event,:ts,:interval_min,:close,:high,
                           :low,:mean,:open,:yes_bid,:yes_ask,:volume,:open_interest)""", rows)
            total += len(rows)
        if verbose and (i % 20 == 0 or i == len(markets)):
            print(f"  [{i}/{len(markets)}] {tk} +{len(rows)} candles "
                  f"(total {total:,})", flush=True)
        time.sleep(PAUSE)
    return total


def summary() -> None:
    with _conn() as c:
        n, lo, hi = c.execute(
            "SELECT COUNT(*),MIN(ts),MAX(ts) FROM candles").fetchone()
        print(f"\n{n:,} candles stored")
        if n:
            f = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"  range {f(lo)} .. {f(hi)} UTC")
        print(f"\n  {'event':<28}{'mkts':>6}{'candles':>9}{'priced':>8}")
        print("  " + "-" * 52)
        for ev, nm, nc, pr in c.execute(
            """SELECT event, COUNT(DISTINCT ticker), COUNT(*),
                      SUM(CASE WHEN close IS NOT NULL THEN 1 ELSE 0 END)
               FROM candles GROUP BY event ORDER BY event"""):
            print(f"  {ev:<28}{nm:>6}{nc:>9}{pr:>8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="KXTRUTHSOCIAL")
    ap.add_argument("--days", type=int, default=75)
    ap.add_argument("--interval", type=int, default=60, choices=[1, 60, 1440])
    args = ap.parse_args()
    n = harvest(args.series, args.days, args.interval)
    print(f"\nharvested {n:,} candle rows -> {DB}")
    summary()
