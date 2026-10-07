"""
Serverless live endpoint.

Vercel cannot run the full pipeline: functions are stateless, the filesystem is
ephemeral and there is no always-on process, so the 30s watcher, the SQLite
store and the WebSocket push all have nowhere to live. What IS possible — and
what this does — is fetch everything fresh on each request, because every source
serves current state on demand:

    Truth Social   last 20 posts
    News RSS       current headlines
    Kalshi         open brackets + settled weeks

So this is a live *view*, not a collector. It cannot show accumulated history,
prediction scoring, or an exact weekly post count, because those require having
been watching continuously. Those live in the Render/local deployment.

Everything is fetched in parallel with short timeouts: Vercel's Hobby plan kills
a function at 10s, and ten sequential RSS fetches would blow through that.
Responses carry s-maxage so repeat visitors hit Vercel's edge cache instead of
re-triggering every upstream fetch.
"""

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler

import feedparser
import requests

# Vercel executes this from /var/task, not from the file's own directory, so a
# sibling module is not importable by default (ModuleNotFoundError at cold
# start). Put this file's directory on the path first.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import postsignal as sig

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept": "application/json"}

TRUTH_ACCOUNT = "107780257626128497"
TRUTH_URL = f"https://truthsocial.com/api/v1/accounts/{TRUTH_ACCOUNT}/statuses"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# Truth Social sits behind Cloudflare and returns HTTP 403 ("Just a moment...")
# to datacenter IPs, so this function cannot fetch posts directly no matter how
# it retries. The collector running on a residential IP publishes what it has to
# this gist, and we read that instead. News and Kalshi are NOT blocked and are
# still fetched live on every request.
SNAPSHOT_URL = ("https://gist.githubusercontent.com/rishabh790453/"
                "00d38fd5b9d11b73b29ccf05da6f3b48/raw/trump_pipeline_snapshot.json")

# A subset of the pipeline's feeds — the fastest and most reliable ones. The
# full list is fine for a background poller with no deadline; here every extra
# feed is latency inside a 10s budget.
FEEDS = [
    ("NYT", "https://rss.nytimes.com/services/xml/rss/nyt/US.xml"),
    ("NYT-Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ("Guardian", "https://www.theguardian.com/us-news/rss"),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Fox News", "https://moxie.foxnews.com/google-publisher/politics.xml"),
    ("Axios", "https://api.axios.com/feed/"),
]

KEYWORDS = {
    "trump": 3.0, "white house": 2.0, "tariff": 2.5, "tariffs": 2.5, "fed": 2.0,
    "powell": 2.5, "inflation": 1.5, "china": 2.0, "russia": 1.8, "iran": 1.8,
    "israel": 1.8, "ukraine": 1.8, "tax": 1.5, "immigration": 1.5, "border": 1.5,
    "election": 1.5, "congress": 1.2, "senate": 1.2, "supreme court": 1.5,
    "stock": 1.5, "market": 1.0, "economy": 1.0, "crypto": 1.5, "bitcoin": 1.5,
}
_KW = {k: re.compile(rf"\b{re.escape(k)}", re.I) for k in KEYWORDS}

BRACKETS = [(0, 79, "<80"), (80, 99, "80-99"), (100, 119, "100-119"),
            (120, 139, "120-139"), (140, 159, "140-159"), (160, 179, "160-179"),
            (180, 199, "180-199"), (200, 219, "200-219"), (220, 240, "220-240"),
            (241, math.inf, ">240")]


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    t = re.sub(r"<br\s*/?>", " ", raw, flags=re.I)
    t = re.sub(r"</p>\s*<p>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        t = t.replace(a, b)
    # The API injects a space inside URLs ("https://www. wsj.com"); rejoin so the
    # signal extractor's URL stripper removes them cleanly.
    t = re.sub(r"(https?://\S*?)\s+(\S)", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip()


# ── Fetchers, each defensive: one failing source must not empty the page ──────

def fetch_snapshot() -> dict:
    """Posts as collected by the residential-IP collector."""
    try:
        # GitHub's raw CDN pins /raw/<file> to a revision and kept serving a
        # snapshot ~20 minutes stale even with Cache-Control: no-cache. A
        # per-minute cache-bust param defeats it, and lines up with the
        # s-maxage=60 on our own response so it costs nothing extra.
        bust = int(time.time() // 60)
        r = requests.get(f"{SNAPSHOT_URL}?cb={bust}", timeout=6,
                         headers={"User-Agent": UA, "Cache-Control": "no-cache"})
        r.raise_for_status()
        d = r.json()
        posts = []
        for p in d.get("posts", []):
            s = sig.extract(p.get("text") or "")
            posts.append({**p, "topic": s.topic, "polarity": s.polarity,
                          "intensity": s.intensity, "tickers": s.tickers})
        return {"ok": True, "via": "snapshot", "posts": posts,
                "snapshot_at": d.get("generated_at"), "week": d.get("week"),
                "predictions": d.get("predictions", []),
                "totals": d.get("totals", {})}
    except Exception as e:
        return {"ok": False, "via": "snapshot", "error": type(e).__name__, "posts": []}


def fetch_posts() -> dict:
    """Try Truth Social directly, then fall back to the collector's snapshot."""
    try:
        r = requests.get(TRUTH_URL, headers=HEADERS, timeout=6,
                         params={"limit": 20, "exclude_replies": "true"})
        if r.status_code != 200:
            # 403 = Cloudflare blocks this datacenter range; no retry helps.
            snap = fetch_snapshot()
            snap["direct_error"] = f"HTTP {r.status_code}"
            return snap
        out = []
        for p in r.json():
            text = _strip_html(p.get("content", ""))
            s = sig.extract(text)
            out.append({
                "id": str(p["id"]), "created_at": p.get("created_at", ""),
                "text": text, "has_media": bool(p.get("media_attachments")),
                "is_reblog": bool(p.get("reblog")),
                "favourites": p.get("favourites_count"),
                "url": p.get("url") or p.get("uri") or "",
                "topic": s.topic, "polarity": s.polarity,
                "intensity": s.intensity, "tickers": s.tickers,
            })
        return {"ok": True, "via": "direct", "posts": out}
    except Exception as e:
        snap = fetch_snapshot()
        snap["direct_error"] = type(e).__name__
        return snap


def _one_feed(name: str, url: str) -> list:
    try:
        f = feedparser.parse(url)
        now = datetime.now(timezone.utc)
        out = []
        for e in f.entries[:10]:
            title = (e.get("title") or "").strip()
            if not title:
                continue
            score = sum(w for k, w in KEYWORDS.items() if _KW[k].search(title))
            out.append({"title": title, "source": name, "url": e.get("link", ""),
                        "published": e.get("published", ""),
                        "relevance": round(min(score / 6.0, 1.0), 3)})
        return out
    except Exception:
        return []


def fetch_news() -> dict:
    items = []
    with ThreadPoolExecutor(max_workers=len(FEEDS)) as ex:
        futs = {ex.submit(_one_feed, n, u): n for n, u in FEEDS}
        for f in as_completed(futs, timeout=7):
            try:
                items += f.result()
            except Exception:
                pass
    seen, uniq = set(), []
    for it in sorted(items, key=lambda x: -x["relevance"]):
        k = it["title"].lower()[:70]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    return {"ok": bool(uniq), "news": uniq[:25]}


def _gamma_poisson_fit(counts):
    n = len(counts)
    m = sum(counts) / n
    v = sum((c - m) ** 2 for c in counts) / (n - 1)
    beta = 1.0 if v <= m else m / (v - m)
    return max(m * beta, 1e-6), max(beta, 1e-6)


def _pmf(alpha, beta, exposure, max_k=600):
    """Negative-binomial PMF in log space — alpha exceeds 100 here and the
    direct ratio form overflows before reaching the upper tail."""
    r, p = alpha, beta / (beta + exposure)
    lp, lq = math.log(p), math.log1p(-p)
    out, lg = [], r * lp
    for k in range(max_k + 1):
        if k:
            lg += math.log(r + k - 1) - math.log(k) + lq
        out.append(math.exp(lg))
    tot = sum(out) or 1.0
    return [v / tot for v in out]


def fetch_kalshi() -> dict:
    try:
        r = requests.get(f"{KALSHI}/markets", headers=HEADERS, timeout=7,
                         params={"series_ticker": "KXTRUTHSOCIAL", "limit": 200})
        r.raise_for_status()
        markets = r.json().get("markets", [])

        by_event = {}
        for m in markets:
            by_event.setdefault(m["event_ticker"], []).append(m)

        settled, mids = [], []
        for ev, ms in sorted(by_event.items()):
            win = [m for m in ms if m.get("result") == "yes"]
            if not win:
                continue
            lab = (win[0].get("subtitle") or win[0].get("yes_sub_title")
                   or win[0]["ticker"].split("-")[-1])
            settled.append({"event": ev, "label": lab})
            if lab.startswith("<"):
                mids.append(float(lab[1:]) * 0.85)
            elif lab.startswith(">"):
                mids.append(float(lab[1:]) * 1.1)
            elif "-" in lab:
                lo, hi = lab.split("-")
                mids.append((float(lo) + float(hi)) / 2)

        brackets = []
        if len(mids) >= 6:
            a, b = _gamma_poisson_fit(mids[-16:])
            pmf = _pmf(a, b, 1.0)
            for lo, hi, lab in BRACKETS:
                top = 600 if math.isinf(hi) else min(int(hi), 600)
                brackets.append({"label": lab,
                                 "prob": round(sum(pmf[int(lo):top + 1]), 5)})

        open_ev = sorted({m["event_ticker"] for m in markets
                          if m.get("status") == "active"})
        return {"ok": True, "settled": settled[-10:], "brackets": brackets,
                "open_event": open_ev[-1] if open_ev else None,
                "prior_mean": round(sum(mids[-16:]) / len(mids[-16:]), 1) if mids else None}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "settled": [], "brackets": []}


def build() -> dict:
    with ThreadPoolExecutor(max_workers=3) as ex:
        fp, fn, fk = ex.submit(fetch_posts), ex.submit(fetch_news), ex.submit(fetch_kalshi)
        posts, news, kal = fp.result(), fn.result(), fk.result()

    texted = [p for p in posts["posts"] if p["text"]]
    with_signal = [p for p in texted if p["tickers"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "posts": posts, "news": news, "kalshi": kal,
        "via": posts.get("via", "direct"),
        "snapshot_at": posts.get("snapshot_at"),
        "week": posts.get("week"),
        # Predictions only exist when a collector made them, so they ride along
        # with the snapshot. Surface them at top level rather than buried under
        # `posts`, where the frontend was never looking.
        "predictions": posts.get("predictions", []),
        "totals": posts.get("totals", {}),
        "summary": {
            "n_posts": len(posts["posts"]),
            "n_with_text": len(texted),
            "n_media_only": len(posts["posts"]) - len(texted),
            "n_with_signal": len(with_signal),
            "newest_at": posts["posts"][0]["created_at"] if posts["posts"] else None,
        },
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            payload, status = build(), 200
        except Exception as e:
            payload, status = {"error": f"{type(e).__name__}: {e}"}, 500
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        # Edge-cache for a minute: he does not post more than once a minute, and
        # this keeps a traffic spike from hammering Truth Social from every region.
        self.send_header("Cache-Control", "public, s-maxage=60, stale-while-revalidate=300")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
