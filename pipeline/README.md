# Live prediction pipeline

Collects live news, predicts what Trump will post, records what he actually
posts, scores the two against each other, and tests whether any of it is
tradeable against real market data.

Separate from the paper's offline evaluation (`evaluate_all.py`), which scores
five fixed examples. This runs continuously and accumulates its own record.

---

## Quick start

```bash
# LIVE: collector + dashboard in one process, http://localhost:8000
venv_mac/bin/python3 -m uvicorn dashboard.live_app:app --port 8000

# one cycle: news → predict → ground truth → score
venv_mac/bin/python3 run_cycle.py

# what's been collected and whether it means anything yet
venv_mac/bin/python3 status.py

# does any of this move markets?
venv_mac/bin/python3 -m pipeline.backtest --window 60 --control
```

Run everything with `venv_mac/bin/python3`. Plain `python3` lacks torch, and the
prediction scripts catch that failure per-model, so a run started with the wrong
interpreter completes "successfully" having silently dropped the Llama and RAG
models.

---

## The integrity rule

**A prediction is written at time T, before the outcome exists, and is never
updated.** `store.insert_prediction` deliberately uses `INSERT`, not upsert — a
duplicate id raises rather than overwriting.

Everything downstream depends on this. Without it there is no way, months later,
to show that a good-looking result was not quietly improved after the fact.

Three related guards:

| Guard | Where | Why |
|---|---|---|
| Ground truth is Truth Social API only | `store.py` CHECK constraint | The old RSS quote-scraper pulled Trump quotes out of the same news feeds used as prediction *input* — scoring against those is circular |
| Outcomes recorded even when he posts nothing | `matches.matched = 0` | Dropping those would condition every hit rate on the outcome being measured |
| Scoring waits for the full horizon | `unmatched_predictions` | Scoring early counts posts he hadn't had the chance to write as misses |

---

## Modules

| Module | Does |
|---|---|
| `store.py` | SQLite schema and accessors. The integrity rules live here |
| `news_feed.py` | 10 live RSS feeds, with a staleness guard |
| `truth_feed.py` | Truth Social API — the only ground-truth source |
| `signal.py` | Text → `{topic, polarity, intensity, tickers}` |
| `predict.py` | Timestamped, immutable predictions from a news window |
| `match.py` | Links elapsed predictions to actual posts |
| `score.py` | BLEU/ROUGE plus topic and direction accuracy |
| `market.py` | Real OHLCV bars via yfinance. Nothing simulated |
| `backtest.py` | Event study, clustering, controls, paper ledger |
| `history.py` | Full-corpus event study, 2009-2026 |
| `kalshi.py` | KXTRUTHSOCIAL weekly post-count forecast |
| `watcher.py` | Near-real-time collector (30s posts, 10m news) |

`../run_cycle.py` orchestrates one cycle; `../status.py` reports.

---

## Headline finding: no daily-resolution effect

`pipeline/history.py` runs the event study over the **whole corpus** — 86,170
posts, 2009-2026, 5,624 (session, ticker) events against real daily bars. This
answers "which kinds of post move markets" now, rather than after two months of
live collection.

```bash
venv_mac/bin/python3 -m pipeline.history --slices
venv_mac/bin/python3 -m pipeline.history --window gap   # overnight reaction
```

**Result: nothing survives correction, anywhere.**

| Slice | Best result | Verdict |
|---|---|---|
| Overall (session-clustered) | t=+0.70, hit 51.0% | null |
| Overnight gap | t=−1.13, hit 49.8% | null |
| By topic (10) | middle_east t=1.68, p=0.093 | null after BH |
| By era (4) | term1 t=0.51 | null |
| By ticker (11) | USO t=1.40 | null |
| Signal magnitude deciles | no monotonic pattern | null |
| Extreme tail (top 1%) | t=−0.31, hit 37% | null |
| Posting volume | t≤0.56 | null |

25 groups tested; 0 nominally significant after session-clustering; chance alone
would give ~1.2. Both return windows are null, so this is not an artifact of
measuring the wrong one.

Two earlier nominal hits — `russia_ukraine` t=2.28 and `middle_east` t=1.97 —
**disappeared once clustered by session** (to 1.67 and 1.68). They were the same
correlated-legs artifact described below, not findings.

### What this rules out, and what it doesn't

Ruled out: a persistent, daily-scale, direction-predictable effect from any
category of post, at this signal definition. That is the strategy most people
would build, and the data says don't.

Not ruled out:

1. **Fast intraday decay.** A reaction completing in 5-15 minutes is invisible
   in both a 6.5-hour session return and an overnight gap. This is the one live
   surviving hypothesis, and measuring it is exactly what the live loop does.
2. **A better signal.** The extractor is a keyword lexicon. An LLM scoring market
   relevance directly might separate what keywords cannot.
3. **Individual megaton posts.** A specific tariff announcement may move billions
   without the *class* "trade posts" being predictable — which is what was tested.
   Trading the class is what fails here.

---

## What the data actually looks like

Measured on a 13-day sample (Sept 2026), and these ratios drive everything:

- **~40% of posts have no usable text.** Pure media, reblogs, or a bare link.
- **Of those that do, ~75% carry no market-relevant signal.** Birthdays, golf,
  media attacks, endorsements.
- **Net: roughly 1 in 6 posts produces a tradeable signal**, about 3/day.

That event rate is the binding constraint on this whole idea. At ~3 events/day,
reaching the ~200 independent events needed for inference takes about **two
months of continuous collection**. There is no way to shortcut it: Yahoo only
serves 5-minute bars for ~60 days, so older posts can only be studied at daily
resolution, where a single post's effect is invisible.

---

## Reading a backtest honestly

Two traps the code handles explicitly, both of which make results look better
than they are:

**1. Correlated legs.** One post opens positions in several tickers, and
SPY/QQQ/DIA/IWM are largely the same bet. Counting each leg as an observation
inflated n from 38 to 99 and moved the p-value from 0.053 to 0.0105. Always read
the **clustered** row — `backtest.py` prints both and labels the per-leg one as
overstated.

**2. A single random draw is not a control.** At small n a one-seed coin flip
swings wildly. `random_control` averages 2,000 draws and reports the fraction
that beat the real strategy.

Still not controlled for, and the reason a positive result would not yet justify
trading: **the news that triggered the post was already public.** Any measured
move may be the market reacting to the news rather than to him. Separating those
needs a news-only control — matched windows following the same headlines with no
post. That is the next piece of work, and it should be built before any money is
involved.

---

## One collector, always

**Never run `backfill_posts.py` and the watcher/dashboard at the same time.**

Truth Social enforces a single rate-limit budget per client, not per process.
Measured on 2026-09-13: with a backfill running in the background, a 5-second
poller got HTTP 429 on **11 of 12** requests. With nothing else competing, 30-second
polling ran **8/8 clean**. Two collectors do not collect twice as fast — they
starve each other, and the failure looks like "he stopped posting" rather than
like an error.

This is why `dashboard/live_app.py` runs the watcher as a thread *inside* the web
server process rather than as a separate service: one process, one budget, one
owner of the API.

Cadences, and why they differ:

| Source | Interval | Reason |
|---|---|---|
| Truth Social | 30s | Prompt detection is the point; one 20-post page can't overflow in 30s |
| News RSS | 10 min | Feeds update in minutes; hitting 10 publishers every 30s is pointless |
| Predictions | 30 min | Only meaningful once the news window has actually changed |

---

## Scheduling

`deploy/com.rishabh.trumppipeline.plist` runs a cycle every 30 minutes. Not
installed automatically:

```bash
cp deploy/com.rishabh.trumppipeline.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.rishabh.trumppipeline.plist
```

`StartInterval` fires only while the Mac is awake. Sleep gaps are expected and
`status.py` reports the largest gap. Ground truth mostly self-heals: each cycle
pages back until it reaches a post already stored, so a gap under ~20 posts is
backfilled on the next wake. News is not recoverable — a headline that has
scrolled off an RSS feed is gone.

For the fine-tuned Llama, use the daemon instead. The 16GB model takes minutes to
load on MPS, so a cron job that reloads it every cycle spends most of its life
loading:

```bash
venv_mac/bin/python3 run_cycle.py --daemon --interval 1800 \
    --models finetuned_llama,rag_retrieve
```

---

## Operational notes

- **Truth Social rate limits** at roughly 4 rapid pages, then HTTP 429 with no
  `Retry-After`. `truth_feed` backs off exponentially. Normal cycles fetch 1–2
  pages and never approach it; only bulk backfill does.
- **Dead feeds.** Reuters, AP, Politico and BBC were dropped — see `KNOWN_BROKEN`
  in `news_feed.py`. CNN is listed there too for a worse reason: it parses fine
  and returns 30 entries whose newest is from December 2022. Every feed is
  recency-checked at runtime because a stale feed is invisible where a dead one
  is obvious.
- **`market_sim.py` in `dashboard/` is superseded.** It invented base prices and
  per-keyword percentage moves and rendered them as market data. Nothing in this
  package simulates a price.
