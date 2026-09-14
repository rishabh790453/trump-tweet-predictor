"""
Real market data.

Everything here comes from yfinance. Nothing is simulated — which is the whole
point of replacing dashboard/market_sim.py, whose `BASE_PRICES` and per-keyword
percentage moves were invented and then displayed as if they were market data.

Intraday history is the binding constraint: Yahoo serves 1-minute bars for only
about the last 30 days and 5-minute bars for about 60. Event studies on older
posts therefore have to fall back to daily bars, which is why fetch_bars() takes
an explicit interval and event_return() reports which interval it actually used.
"""

from datetime import datetime, timedelta, timezone

from .store import connect, insert_market_bars

# Tickers reachable from signal.TOPIC_TICKERS, plus broad-market references.
DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "TLT", "GLD",
                   "XLE", "USO", "FXI", "ITA", "LMT", "COIN", "MSTR"]

# Yahoo's practical intraday retention.
INTRADAY_LIMITS = {"1m": 30, "5m": 60, "15m": 60, "1h": 730}

US_MARKET_OPEN = (13, 30)    # 09:30 ET in UTC (EDT); see _is_market_hours caveat
US_MARKET_CLOSE = (20, 0)    # 16:00 ET in UTC (EDT)


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return _utc(dt)


def fetch_bars(tickers: list[str], start: datetime, end: datetime,
               interval: str = "5m", verbose: bool = True) -> list[dict]:
    """
    Fetch OHLCV bars. Returns rows ready for the `market` table.

    Silently returning nothing when a window predates Yahoo's intraday retention
    would look identical to "the market did not move", so the retention check is
    explicit and loud.
    """
    import yfinance as yf

    start, end = _utc(start), _utc(end)
    if interval in INTRADAY_LIMITS:
        oldest = datetime.now(timezone.utc) - timedelta(days=INTRADAY_LIMITS[interval])
        if start < oldest:
            if verbose:
                print(f"  [market] {interval} bars unavailable before "
                      f"{oldest.date()} (Yahoo retention); falling back to 1d")
            interval = "1d"

    rows: list[dict] = []
    try:
        data = yf.download(
            tickers=" ".join(tickers), start=start, end=end, interval=interval,
            progress=False, auto_adjust=False, group_by="ticker", threads=True,
        )
    except Exception as exc:
        print(f"  [market] download failed: {type(exc).__name__}: {exc}")
        return rows

    if data is None or data.empty:
        if verbose:
            print(f"  [market] no {interval} data for {start:%Y-%m-%d}..{end:%Y-%m-%d}")
        return rows

    for ticker in tickers:
        try:
            df = data[ticker] if len(tickers) > 1 else data
        except KeyError:
            continue
        df = df.dropna(how="all")
        for ts, row in df.iterrows():
            ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            close = row.get("Close")
            if close is None or close != close:      # NaN check
                continue
            rows.append({
                "ticker": ticker,
                "ts": ts_utc.isoformat(),
                "open": float(row.get("Open", close)),
                "high": float(row.get("High", close)),
                "low": float(row.get("Low", close)),
                "close": float(close),
                "volume": float(row.get("Volume", 0) or 0),
                "interval": interval,
            })
    if verbose:
        print(f"  [market] {len(rows)} {interval} bars for {len(tickers)} tickers")
    return rows


def store_bars(rows: list[dict], db_path=None) -> int:
    if not rows:
        return 0
    kwargs = {"db_path": db_path} if db_path else {}
    with connect(**kwargs) as conn:
        return insert_market_bars(conn, rows)


def is_market_hours(dt: datetime) -> bool:
    """
    Rough US equity session check.

    Uses fixed UTC offsets for EDT, so it is off by an hour during EST and does
    not know about holidays. Good enough to flag "this post landed overnight",
    not good enough to time an order against.
    """
    dt = _utc(dt)
    if dt.weekday() >= 5:
        return False
    hm = (dt.hour, dt.minute)
    return US_MARKET_OPEN <= hm < US_MARKET_CLOSE


def event_return(conn, ticker: str, event_time: datetime,
                 window_minutes: int = 60) -> dict | None:
    """
    Return over [event_time, event_time + window] from stored bars.

    Entry is the first bar at or after the event, never the bar containing it:
    that bar opened before the event and using it would embed a few minutes of
    hindsight in every trade — small per event, decisive over a backtest.
    """
    event_time = _utc(event_time)

    entry = conn.execute(
        """SELECT ts, close, interval FROM market
           WHERE ticker = ? AND ts >= ? ORDER BY ts LIMIT 1""",
        (ticker, event_time.isoformat()),
    ).fetchone()
    if not entry:
        return None

    # The holding window runs from the ENTRY bar, not from the event. Measuring
    # it from the event drops every post made outside market hours: entry lands
    # at the next open, by which time an event-anchored window has already
    # closed, and the function returns None as if no data existed. Since most of
    # his posting is overnight, that silently discarded most of the sample.
    end = _parse_ts(entry["ts"]) + timedelta(minutes=window_minutes)

    exit_row = conn.execute(
        """SELECT ts, close FROM market
           WHERE ticker = ? AND ts >= ? AND ts <= ? ORDER BY ts DESC LIMIT 1""",
        (ticker, entry["ts"], end.isoformat()),
    ).fetchone()
    if not exit_row or exit_row["ts"] == entry["ts"]:
        return None

    entry_px, exit_px = float(entry["close"]), float(exit_row["close"])
    if entry_px == 0:
        return None
    return {
        "ticker": ticker,
        "entry_ts": entry["ts"],
        "exit_ts": exit_row["ts"],
        "entry_price": entry_px,
        "exit_price": exit_px,
        "return_pct": round((exit_px / entry_px - 1) * 100, 4),
        "interval": entry["interval"],
        # Minutes between the post and the first tradeable bar. Large values mean
        # an overnight post filled at the next open — by which time the news has
        # been public for hours and any reaction is likely already in the price.
        "entry_delay_min": round(
            (_parse_ts(entry["ts"]) - event_time).total_seconds() / 60, 1),
    }


def backfill(days: int = 25, interval: str = "5m",
             tickers: list[str] | None = None, db_path=None) -> int:
    """Populate recent bars so event studies have something to read."""
    tickers = tickers or DEFAULT_TICKERS
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    rows = fetch_bars(tickers, start, end, interval)
    return store_bars(rows, db_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch real market bars")
    ap.add_argument("--days", type=int, default=25)
    ap.add_argument("--interval", default="5m")
    args = ap.parse_args()

    print(f"Backfilling {args.interval} bars for {args.days} days...")
    n = backfill(days=args.days, interval=args.interval)
    print(f"stored {n} bars\n")

    with connect() as conn:
        for row in conn.execute(
            """SELECT ticker, COUNT(*) n, MIN(ts) first, MAX(ts) last
               FROM market GROUP BY ticker ORDER BY ticker"""):
            print(f"  {row['ticker']:<6} {row['n']:>5} bars  "
                  f"{row['first'][:16]} .. {row['last'][:16]}")
