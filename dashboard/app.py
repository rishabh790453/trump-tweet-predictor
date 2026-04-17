"""
FastAPI backend — Trump Tweet Live Predictor dashboard.
Run: uvicorn app:app --port 8000
"""

import asyncio
import csv
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))
from market_sim import BASE_PRICES, simulate
from news_etl import fetch_all_news
from predictor import TweetPredictor

app = FastAPI(title="Trump Tweet Live Predictor")

REPO_ROOT = Path(__file__).parent.parent

# ── Global state ──────────────────────────────────────────────────────────────
predictor: TweetPredictor | None = None
seen_ids: set[str] = set()
feed: list[dict] = []
session_prices: dict[str, float] = BASE_PRICES.copy()
spy_history: list[float] = [0.0]
connected: list[WebSocket] = []

POLL_INTERVAL = 300
MAX_NEW_PER_POLL = 3


# ── Load historical tweet+market data ────────────────────────────────────────

def load_real_tweets(n: int = 200) -> list[dict]:
    """Load recent tweets that have market data attached."""
    path = REPO_ROOT / "trump_tweets_finance_with_market_data.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row.get("SP500_pct", "").strip()
            dw = row.get("Dow_pct", "").strip()
            content = row.get("content", "").strip()
            if not content or content.startswith("RT "):
                continue
            rows.append({
                "ts": row.get("timestamp", ""),
                "platform": row.get("platform", ""),
                "content": content[:280],
                "sp500": float(sp) if sp else None,
                "dow": float(dw) if dw else None,
                "likes": row.get("likes", "0"),
            })
    # Most recent first, keep rows with market data near top
    with_data = [r for r in rows if r["sp500"] is not None]
    without = [r for r in rows if r["sp500"] is None]
    combined = (with_data + without)[:n]
    return combined


REAL_TWEETS: list[dict] = []


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global predictor, REAL_TWEETS
    REAL_TWEETS = load_real_tweets(300)
    loop = asyncio.get_event_loop()
    predictor = await loop.run_in_executor(None, TweetPredictor)
    asyncio.create_task(_poll_loop())


# ── WebSocket broadcast ───────────────────────────────────────────────────────

async def broadcast(data: dict):
    dead = []
    for ws in connected:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            connected.remove(ws)
        except ValueError:
            pass


# ── Prediction + market pipeline ──────────────────────────────────────────────

async def _process_item(item) -> dict:
    global session_prices, spy_history

    loop = asyncio.get_event_loop()
    prediction = await loop.run_in_executor(None, predictor.predict, [item.title])

    impact = simulate(prediction, session_prices)
    session_prices = impact.prices

    cumulative_spy = spy_history[-1] + impact.spy_delta
    spy_history.append(round(cumulative_spy, 4))
    if len(spy_history) > 200:
        spy_history = spy_history[-200:]

    return {
        "type": "update",
        "ts": datetime.now(timezone.utc).isoformat(),
        "news": {
            "id": item.id,
            "title": item.title,
            "source": item.source,
            "url": item.url,
            "relevance": item.relevance,
        },
        "prediction": prediction,
        "market": {
            "sentiment": impact.sentiment,
            "impacts": impact.impacts,
            "prices": impact.prices,
            "triggered_by": impact.triggered_by,
            "spy_delta": impact.spy_delta,
            "spy_history": spy_history[-60:],
        },
        "mode": predictor.mode,
    }


async def _poll_loop():
    await asyncio.sleep(2)
    while True:
        try:
            items = await asyncio.get_event_loop().run_in_executor(None, fetch_all_news, 8)
            new_items = [i for i in items if i.id not in seen_ids]
            count = 0
            for item in new_items:
                if count >= MAX_NEW_PER_POLL:
                    break
                seen_ids.add(item.id)
                try:
                    event = await _process_item(item)
                    feed.insert(0, event)
                    if len(feed) > 100:
                        feed.pop()
                    await broadcast(event)
                    count += 1
                    await asyncio.sleep(1)
                except Exception as e:
                    print(f"[App] Prediction error: {e}")
        except Exception as e:
            print(f"[App] Poll error: {e}")
        await asyncio.sleep(POLL_INTERVAL)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected.append(ws)
    await ws.send_json({
        "type": "init",
        "feed": feed[:30],
        "spy_history": spy_history[-60:],
        "mode": predictor.mode if predictor else "loading",
        "prices": session_prices,
        "real_tweets": REAL_TWEETS[:50],
    })
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        try:
            connected.remove(ws)
        except ValueError:
            pass


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/api/predict")
async def predict_custom(headline: str):
    loop = asyncio.get_event_loop()
    pred = await loop.run_in_executor(None, predictor.predict, [headline])
    impact = simulate(pred)
    return {
        "headline": headline,
        "prediction": pred,
        "market": {
            "sentiment": impact.sentiment,
            "impacts": impact.impacts,
            "triggered_by": impact.triggered_by,
            "spy_delta": impact.spy_delta,
        },
        "mode": predictor.mode,
    }


@app.get("/api/news")
async def get_news():
    loop = asyncio.get_event_loop()
    items = await loop.run_in_executor(None, fetch_all_news, 8)
    return [
        {"id": i.id, "title": i.title, "source": i.source,
         "url": i.url, "relevance": i.relevance}
        for i in items
    ]


@app.get("/api/real-tweets")
async def get_real_tweets(n: int = 50):
    return REAL_TWEETS[:n]


@app.get("/api/status")
async def status():
    return {
        "mode": predictor.mode if predictor else "loading",
        "predictions_generated": len(feed),
        "spy_history": spy_history[-20:],
        "real_tweets_loaded": len(REAL_TWEETS),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
