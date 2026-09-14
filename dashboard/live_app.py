"""
Live dashboard — FastAPI + WebSocket over the pipeline store.

    venv_mac/bin/python3 -m uvicorn dashboard.live_app:app --port 8000
    open http://localhost:8000

The watcher runs as a background thread INSIDE this process, which is the point:
Truth Social enforces one rate-limit budget per client, so collection has to have
exactly one owner. Running the dashboard and a separate collector meant both got
429'd. Here the web server and the collector are the same process, and the
watcher pushes straight to connected browsers as posts land.

Deliberately not reusing the old dashboard/app.py:
  * it imports market_sim, which invents prices and renders them as market data
  * it imports live_tweets_etl, whose primary source (thetrumparchive) is dead and
    whose fallback scrapes Trump quotes out of the same news feeds used as
    prediction input — circular, and unusable as ground truth
This one reads only the pipeline store, so everything on screen traces back to a
row that scoring and the Kalshi forecast also use.
"""

import asyncio
import json
import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import kalshi
from pipeline.store import connect, init_db, stats
from pipeline.watcher import Watcher

app = FastAPI(title="Trump Post Pipeline — Live")

STATIC = Path(__file__).parent / "static"

_clients: list[WebSocket] = []
_watcher: Watcher | None = None
_loop: asyncio.AbstractEventLoop | None = None


# ── WebSocket plumbing ───────────────────────────────────────────────────────

async def _broadcast(kind: str, payload: dict):
    msg = json.dumps({"kind": kind, "payload": payload, "at": datetime.now(timezone.utc).isoformat()})
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _clients:
            _clients.remove(ws)


def _on_event(kind: str, payload: dict):
    """
    Called from the watcher thread, so hop onto the event loop rather than
    touching WebSockets directly — sending from another thread corrupts the
    connection state.
    """
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(kind, payload), _loop)


# ── Data accessors ───────────────────────────────────────────────────────────

def _rows(sql: str, args=()) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def recent_posts(limit: int = 30) -> list[dict]:
    return _rows(
        """SELECT id, created_at, text, has_media, is_reblog, topic, polarity,
                  intensity, favourites, url
           FROM posts ORDER BY created_at DESC LIMIT ?""", (limit,))


def recent_predictions(limit: int = 20) -> list[dict]:
    return _rows(
        """SELECT p.id, p.predicted_at, p.model, p.text, p.topic, p.polarity,
                  p.intensity, p.horizon_hours,
                  m.matched, m.similarity, m.topic_hit
           FROM predictions p LEFT JOIN matches m ON m.prediction_id = p.id
           ORDER BY p.predicted_at DESC LIMIT ?""", (limit,))


def recent_news(limit: int = 25) -> list[dict]:
    return _rows(
        """SELECT id, title, source, url, published_at, collected_at, relevance
           FROM news ORDER BY collected_at DESC, relevance DESC LIMIT ?""", (limit,))


def week_tracker() -> dict:
    """Current Kalshi week: posts so far, pace, and bracket probabilities."""
    counts = kalshi.weekly_counts_from_db()
    this_week = kalshi.week_start_et().date().isoformat()
    observed = counts.get(this_week, 0)
    frac = kalshi.elapsed_fraction()
    fresh = kalshi.collection_freshness()

    complete = [v for k, v in sorted(counts.items()) if k != this_week]
    model = None
    source = None
    try:
        if len(complete) >= 6:
            model = kalshi.GammaPoisson.fit(complete[-16:])
            source = f"{len(complete[-16:])} collected weeks"
        else:
            settled = kalshi.settled_weeks(kalshi.fetch_markets())
            mids = kalshi.settled_midpoints(settled)
            if len(mids) >= 6:
                model = kalshi.GammaPoisson.fit(mids[-16:])
                source = f"{len(mids[-16:])} Kalshi settled weeks (coarse)"
    except Exception as exc:
        source = f"model unavailable: {type(exc).__name__}"

    brackets = []
    if model:
        brackets = kalshi.bracket_probabilities(model, observed, frac)

    return {
        "week_start": this_week,
        "observed": observed,
        "elapsed_pct": round(frac * 100, 1),
        # Simple pace extrapolation, shown next to the model so an obviously
        # divergent model output is easy to spot.
        "pace_projection": round(observed / frac) if frac > 0.02 else None,
        "model_mean": round(model.mean, 1) if model else None,
        "model_source": source,
        "brackets": brackets,
        "freshness": fresh,
    }


def health() -> dict:
    with connect() as conn:
        st = stats(conn)
    w = _watcher.stats if _watcher else {}
    return {"store": st, "watcher": w,
            "clients": len(_clients),
            "now": datetime.now(timezone.utc).isoformat()}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/posts")
async def api_posts(limit: int = 30):
    return JSONResponse(recent_posts(limit))


@app.get("/api/predictions")
async def api_predictions(limit: int = 20):
    return JSONResponse(recent_predictions(limit))


@app.get("/api/news")
async def api_news(limit: int = 25):
    return JSONResponse(recent_news(limit))


@app.get("/api/week")
async def api_week():
    return JSONResponse(week_tracker())


@app.get("/api/health")
async def api_health():
    return JSONResponse(health())


@app.get("/api/bootstrap")
async def api_bootstrap():
    """Everything the page needs on load, in one round trip."""
    return JSONResponse({
        "posts": recent_posts(30),
        "predictions": recent_predictions(20),
        "news": recent_news(25),
        "week": week_tracker(),
        "health": health(),
    })


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    _clients.append(websocket)
    try:
        await websocket.send_text(json.dumps({"kind": "hello", "payload": health()}))
        while True:
            # No client->server protocol; this just keeps the socket open and
            # detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _clients:
            _clients.remove(websocket)


@app.get("/", response_class=HTMLResponse)
async def index():
    page = STATIC / "live.html"
    if not page.exists():
        return HTMLResponse("<h1>live.html missing</h1>", status_code=500)
    return HTMLResponse(page.read_text())


# ── Lifecycle ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _watcher, _loop
    init_db()
    _loop = asyncio.get_running_loop()
    # Hosted containers omit torch/faiss, so the model set must be configurable
    # rather than defaulting to ones whose imports are not installed there.
    models = [m.strip() for m in
              os.environ.get("WATCHER_MODELS", "markov,ngram,rag_retrieve").split(",")
              if m.strip()]
    _watcher = Watcher(models=models, on_event=_on_event, verbose=True)
    threading.Thread(target=_watcher.run, daemon=True, name="watcher").start()

    async def heartbeat():
        # Periodic push so the week tracker and health panel stay current even
        # during a long quiet stretch with no posts.
        while True:
            await asyncio.sleep(30)
            try:
                await _broadcast("tick", {"health": health(), "week": week_tracker()})
            except Exception:
                pass

    asyncio.create_task(heartbeat())


@app.on_event("shutdown")
async def shutdown():
    if _watcher:
        _watcher.stop()
