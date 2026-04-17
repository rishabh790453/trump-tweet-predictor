"""
FastAPI backend for Trump Tweet Live Predictor dashboard.
Run: uvicorn app:app --reload --port 8000
"""

import asyncio
import json
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

# ── Global state ──────────────────────────────────────────────────────────────
predictor: TweetPredictor | None = None
seen_ids: set[str] = set()
feed: list[dict] = []                   # most-recent-first event log
session_prices: dict[str, float] = BASE_PRICES.copy()
spy_history: list[float] = [0.0]        # cumulative SPY % changes
connected: list[WebSocket] = []

POLL_INTERVAL = 300   # seconds between RSS polls
MAX_NEW_PER_POLL = 3  # cap predictions per cycle


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global predictor
    # Load model in background so server responds immediately
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


# ── Background news → prediction loop ────────────────────────────────────────

async def _process_item(item) -> dict | None:
    global session_prices, spy_history

    loop = asyncio.get_event_loop()
    prediction = await loop.run_in_executor(None, predictor.predict, [item.title])

    impact = simulate(prediction, session_prices)
    session_prices = impact.prices

    cumulative_spy = spy_history[-1] + impact.spy_delta
    spy_history.append(round(cumulative_spy, 4))
    if len(spy_history) > 200:
        spy_history = spy_history[-200:]

    event = {
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
            "spy_history": spy_history[-50:],
        },
        "mode": predictor.mode,
    }
    return event


async def _poll_loop():
    await asyncio.sleep(2)  # give server a moment to start
    while True:
        try:
            items = await asyncio.get_event_loop().run_in_executor(None, fetch_all_news, 10)
            new_items = [i for i in items if i.id not in seen_ids]

            count = 0
            for item in new_items:
                if count >= MAX_NEW_PER_POLL:
                    break
                seen_ids.add(item.id)
                try:
                    event = await _process_item(item)
                    if event:
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


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    connected.append(ws)
    # Send current state immediately
    await ws.send_json({
        "type": "init",
        "feed": feed[:30],
        "spy_history": spy_history[-50:],
        "mode": predictor.mode if predictor else "loading",
        "prices": session_prices,
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
        },
        "mode": predictor.mode,
    }


@app.get("/api/news")
async def get_news():
    loop = asyncio.get_event_loop()
    items = await loop.run_in_executor(None, fetch_all_news, 10)
    return [
        {"id": i.id, "title": i.title, "source": i.source, "url": i.url, "relevance": i.relevance}
        for i in items
    ]


@app.get("/api/status")
async def status():
    return {
        "mode": predictor.mode if predictor else "loading",
        "predictions_generated": len(feed),
        "spy_history": spy_history[-20:],
    }


# ── Serve dashboard ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
