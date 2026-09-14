"""
SQLite store for the live prediction loop.

Design rule that everything else depends on: a prediction row is written at time T,
before the outcome exists, and is never updated afterwards. Matching, scoring and
market outcomes all live in separate tables keyed by prediction id. That is what
makes a later backtest credible — there is no way to quietly improve a prediction
after seeing what Trump actually posted.

Ground truth (`posts`) accepts rows from the Truth Social API only. News-derived
quotes are deliberately excluded: they come from the same feeds used as prediction
input, so scoring against them would be circular.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Overridable so a container can point at a mounted volume. Without this the
# store lands on the container filesystem and every redeploy wipes the collected
# history — the one asset here that cannot be re-fetched, since news scrolls off
# RSS and Truth Social paging is rate-limited.
DB_PATH = Path(os.environ.get("PIPELINE_DB")
               or Path(__file__).parent.parent / "live_pipeline.db")

# Only source accepted into the `posts` (ground truth) table.
TRUSTED_POST_SOURCE = "truthsocial_api"


def utcnow() -> str:
    """Single definition of 'now' — always UTC, always ISO with offset."""
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
-- ── News headlines, as observed at collection time ──────────────────────────
CREATE TABLE IF NOT EXISTS news (
    id            TEXT PRIMARY KEY,   -- sha1 of normalised title
    title         TEXT NOT NULL,
    source        TEXT NOT NULL,
    url           TEXT,
    published_at  TEXT,               -- feed-reported; unreliable, some feeds lie
    collected_at  TEXT NOT NULL,      -- when WE saw it; trustworthy, use this
    relevance     REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_news_collected ON news(collected_at);

-- ── Predictions: immutable, written at T before the outcome exists ──────────
CREATE TABLE IF NOT EXISTS predictions (
    id                TEXT PRIMARY KEY,
    predicted_at      TEXT NOT NULL,   -- T
    model             TEXT NOT NULL,   -- finetuned_llama | rag | claude | markov | ngram | template
    horizon_hours     REAL NOT NULL,   -- prediction claims: he posts about this within T..T+h
    news_window_start TEXT NOT NULL,
    news_window_end   TEXT NOT NULL,
    news_ids          TEXT NOT NULL,   -- json list of news.id
    headlines         TEXT NOT NULL,   -- json list, snapshotted so later feed edits can't alter history
    text              TEXT,            -- generated post text
    -- structured signal derived from `text` at prediction time
    topic             TEXT,
    polarity          REAL,
    intensity         REAL,
    tickers           TEXT,            -- json {ticker: signed weight}
    run_id            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_at    ON predictions(predicted_at);
CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model);

-- ── Ground truth: actual posts, Truth Social API only ───────────────────────
CREATE TABLE IF NOT EXISTS posts (
    id            TEXT PRIMARY KEY,   -- Truth Social status id
    created_at    TEXT NOT NULL,      -- when he posted
    collected_at  TEXT NOT NULL,      -- when we saw it
    text          TEXT NOT NULL,
    has_media     INTEGER NOT NULL DEFAULT 0,
    is_reblog     INTEGER NOT NULL DEFAULT 0,
    favourites    INTEGER,
    reblogs       INTEGER,
    url           TEXT,
    source        TEXT NOT NULL,
    topic         TEXT,
    polarity      REAL,
    intensity     REAL,
    tickers       TEXT,
    CHECK (source = 'truthsocial_api')
);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at);

-- ── Outcomes: one row per prediction once its horizon has elapsed ───────────
-- Keyed on prediction_id alone, and post_id is nullable, because "the horizon
-- passed and he posted nothing comparable" is a real outcome that has to be
-- recordable. Dropping those rows would silently condition every hit rate on
-- the subset where he happened to post — the exact selection bias that makes a
-- backtest look better than the strategy is.
CREATE TABLE IF NOT EXISTS matches (
    prediction_id TEXT PRIMARY KEY,
    post_id       TEXT,              -- NULL = no qualifying post in the horizon
    matched       INTEGER NOT NULL,  -- 1 matched, 0 no post / below threshold
    matched_at    TEXT NOT NULL,
    similarity    REAL,              -- semantic sim, prediction text vs actual
    lag_hours     REAL,              -- post.created_at - prediction.predicted_at
    topic_hit     INTEGER,
    polarity_err  REAL,
    direction_hit INTEGER,
    bleu1         REAL,
    rouge1        REAL,
    rougeL        REAL,
    n_candidates  INTEGER,           -- posts available in the window
    FOREIGN KEY (prediction_id) REFERENCES predictions(id),
    FOREIGN KEY (post_id)       REFERENCES posts(id)
);

-- ── Market prices, real data only ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market (
    ticker TEXT NOT NULL,
    ts     TEXT NOT NULL,            -- bar timestamp, UTC
    open   REAL, high REAL, low REAL, close REAL, volume REAL,
    interval TEXT NOT NULL,          -- '1m' | '5m' | '1d'
    PRIMARY KEY (ticker, ts, interval)
);
CREATE INDEX IF NOT EXISTS idx_market_ts ON market(ts);

-- ── Paper trades: no real money, ever, from this table ──────────────────────
CREATE TABLE IF NOT EXISTS paper_trades (
    id            TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    direction     INTEGER NOT NULL,   -- +1 long, -1 short
    opened_at     TEXT NOT NULL,
    closed_at     TEXT,
    entry_price   REAL,
    exit_price    REAL,
    return_pct    REAL,
    notional      REAL NOT NULL,
    status        TEXT NOT NULL,      -- open | closed | abandoned
    note          TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

-- ── Run log: one row per cycle, for debugging gaps in collection ────────────
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    news_added   INTEGER DEFAULT 0,
    posts_added  INTEGER DEFAULT 0,
    preds_added  INTEGER DEFAULT 0,
    matches_added INTEGER DEFAULT 0,
    status       TEXT,
    error        TEXT
);
"""


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


# ── Writers ──────────────────────────────────────────────────────────────────

def insert_news(conn, items: list[dict]) -> int:
    """Insert news items, ignoring ones already seen. Returns count actually added."""
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO news
           (id, title, source, url, published_at, collected_at, relevance)
           VALUES (:id, :title, :source, :url, :published_at, :collected_at, :relevance)""",
        items,
    )
    return conn.total_changes - before


def insert_posts(conn, items: list[dict]) -> int:
    """
    Insert ground-truth posts. The CHECK constraint rejects anything not sourced
    from the Truth Social API, so a future contributor cannot accidentally feed
    news-scraped quotes into ground truth.
    """
    for it in items:
        if it.get("source") != TRUSTED_POST_SOURCE:
            raise ValueError(
                f"refusing post from untrusted source {it.get('source')!r}; "
                f"ground truth accepts only {TRUSTED_POST_SOURCE!r}"
            )
    before = conn.total_changes
    conn.executemany(
        """INSERT OR IGNORE INTO posts
           (id, created_at, collected_at, text, has_media, is_reblog,
            favourites, reblogs, url, source, topic, polarity, intensity, tickers)
           VALUES (:id, :created_at, :collected_at, :text, :has_media, :is_reblog,
                   :favourites, :reblogs, :url, :source, :topic, :polarity,
                   :intensity, :tickers)""",
        items,
    )
    return conn.total_changes - before


def insert_prediction(conn, pred: dict) -> None:
    """
    Write one prediction. Deliberately INSERT (not UPSERT): if the id already
    exists this raises, because silently overwriting a prediction after the fact
    is exactly the failure mode this schema exists to prevent.
    """
    conn.execute(
        """INSERT INTO predictions
           (id, predicted_at, model, horizon_hours, news_window_start, news_window_end,
            news_ids, headlines, text, topic, polarity, intensity, tickers, run_id)
           VALUES (:id, :predicted_at, :model, :horizon_hours, :news_window_start,
                   :news_window_end, :news_ids, :headlines, :text, :topic, :polarity,
                   :intensity, :tickers, :run_id)""",
        pred,
    )


def insert_match(conn, match: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO matches
           (prediction_id, post_id, matched, matched_at, similarity, lag_hours,
            topic_hit, polarity_err, direction_hit, bleu1, rouge1, rougeL,
            n_candidates)
           VALUES (:prediction_id, :post_id, :matched, :matched_at, :similarity,
                   :lag_hours, :topic_hit, :polarity_err, :direction_hit,
                   :bleu1, :rouge1, :rougeL, :n_candidates)""",
        match,
    )


def insert_market_bars(conn, bars: list[dict]) -> int:
    before = conn.total_changes
    conn.executemany(
        """INSERT OR REPLACE INTO market
           (ticker, ts, open, high, low, close, volume, interval)
           VALUES (:ticker, :ts, :open, :high, :low, :close, :volume, :interval)""",
        bars,
    )
    return conn.total_changes - before


def start_run(conn, run_id: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, started_at, status) VALUES (?, ?, 'running')",
        (run_id, utcnow()),
    )


def finish_run(conn, run_id: str, **counts) -> None:
    conn.execute(
        """UPDATE runs SET finished_at=?, news_added=?, posts_added=?, preds_added=?,
                           matches_added=?, status=?, error=?
           WHERE run_id=?""",
        (
            utcnow(),
            counts.get("news_added", 0),
            counts.get("posts_added", 0),
            counts.get("preds_added", 0),
            counts.get("matches_added", 0),
            counts.get("status", "ok"),
            counts.get("error"),
            run_id,
        ),
    )


# ── Readers ──────────────────────────────────────────────────────────────────

def recent_news(conn, since_iso: str, min_relevance: float = 0.0) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM news
           WHERE collected_at >= ? AND relevance >= ?
           ORDER BY relevance DESC, collected_at DESC""",
        (since_iso, min_relevance),
    ).fetchall()


def unmatched_predictions(conn, now_iso: str) -> list[sqlite3.Row]:
    """
    Predictions whose horizon has fully elapsed and which have no match row yet.
    Waiting for the horizon to close avoids scoring a prediction before Trump has
    had the full window to post.
    """
    return conn.execute(
        """SELECT p.* FROM predictions p
           LEFT JOIN matches m ON m.prediction_id = p.id
           WHERE m.prediction_id IS NULL
             AND datetime(p.predicted_at, '+' || p.horizon_hours || ' hours') <= datetime(?)
           ORDER BY p.predicted_at""",
        (now_iso,),
    ).fetchall()


def posts_between(conn, start_iso: str, end_iso: str, min_len: int = 1) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM posts
           WHERE created_at >= ? AND created_at <= ? AND length(text) >= ?
           ORDER BY created_at""",
        (start_iso, end_iso, min_len),
    ).fetchall()


def latest_post_id(conn) -> str | None:
    row = conn.execute("SELECT id FROM posts ORDER BY created_at DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def stats(conn) -> dict:
    q = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "news": q("SELECT COUNT(*) FROM news"),
        "posts": q("SELECT COUNT(*) FROM posts"),
        "posts_with_text": q("SELECT COUNT(*) FROM posts WHERE length(text) > 0"),
        "predictions": q("SELECT COUNT(*) FROM predictions"),
        "matches": q("SELECT COUNT(*) FROM matches"),
        "paper_trades": q("SELECT COUNT(*) FROM paper_trades"),
        "runs": q("SELECT COUNT(*) FROM runs"),
        "oldest_post": q("SELECT COALESCE(MIN(created_at),'-') FROM posts"),
        "newest_post": q("SELECT COALESCE(MAX(created_at),'-') FROM posts"),
    }


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    init_db()
    with connect() as c:
        print(f"initialised {DB_PATH}")
        for k, v in stats(c).items():
            print(f"  {k:>16}: {v}")
