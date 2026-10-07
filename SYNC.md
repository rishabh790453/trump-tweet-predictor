# Syncing this repo to a second machine

The repo is 20GB on disk but only ~40MB of that belongs in git. The split is not
arbitrary — it follows what can be regenerated and what cannot.

| Tier | What | Size | How it syncs |
|---|---|---|---|
| 1 | Code, corpus, paper | ~40MB | **git** |
| 2 | `kalshi_prices.db` | 4MB, +0.4MB/wk | **git** — cannot be re-fetched, see below |
| 3 | Model weights, news DB, venv | ~19GB | **not git** — see Tier 3 |
| 4 | API keys | tiny | **never git** — copy by hand |

---

## One-time setup on the second machine

```bash
git clone https://github.com/rishabh790453/trump_tweets_combined.git
cd trump_tweets_combined
git checkout live-dashboard

python3.11 -m venv venv_mac
venv_mac/bin/pip install -r requirements.txt
venv_mac/bin/pip install yfinance fastapi 'uvicorn[standard]' feedparser pandas numpy

cp /path/from/other/machine/config/news_api_config.json config/
```

Then verify:

```bash
venv_mac/bin/python3 status.py
venv_mac/bin/python3 -c "import sqlite3;print(sqlite3.connect('kalshi_prices.db').execute('SELECT COUNT(*) FROM candles').fetchone())"
```

### If git refuses to run on macOS

```
You have not agreed to the Xcode license agreements.
```

`/usr/bin/git` is Apple's and is gated behind the Xcode licence. Either accept it
once (`sudo xcodebuild -license`), or point the toolchain at CommandLineTools,
whose git is not gated:

```bash
sudo xcode-select -s /Library/Developer/CommandLineTools
```

Without sudo, call it directly: `/Library/Developer/CommandLineTools/usr/bin/git`

---

## Tier 2: why the price database is in git

`kalshi_prices.db` is committed even though binaries in git are normally a
mistake. The reason is that **Kalshi deletes its own history.** The API retains
only ~11 weekly events and drops the oldest as new ones open — verified by
watching three events (26JUL11, 26JUL18, 26JUL25) go from *listed* to *HTTP 404*
inside a single working session.

So this file is not a cache. It is the only copy of settled price history that
will ever exist, it grows ~0.4MB/week (~19MB/year), and re-deriving it is
impossible at any price.

**Rule: exactly one machine harvests and commits it.** The other pulls. A binary
SQLite file cannot be merged, so two machines both writing it produces a conflict
that can only be resolved by throwing one side away.

```bash
# on the designated harvester, weekly:
venv_mac/bin/python3 -m pipeline.kalshi_prices --series KXTRUTHSOCIAL --days 75
git add kalshi_prices.db && git commit -m "Harvest Kalshi prices $(date +%F)" && git push
```

`gdelt_volume.db` is deliberately NOT committed — GDELT's archive is permanent,
so it re-downloads at will.

---

## Tier 3: the large assets

Not in git, listed in `.gitignore`. Decide per asset — most are not needed for
the Kalshi research at all.

| Asset | Size | Needed for | How to get it on machine 2 |
|---|---|---|---|
| `trump_llama_v3/` | 16GB | Generating predictions with the fine-tuned model | Upload to HuggingFace once (`rishabh790453/trump-llama-v3`), then `huggingface-cli download`. Or external SSD. |
| `news_data.db` | 1.6GB | Nothing current — it is 97.5% NYT and unusable for news-volume work | Skip. Use `pipeline/gdelt.py` instead. |
| `rag_index.faiss` + `rag_data.jsonl` | 130MB | RAG retrieval model | HuggingFace Datasets, or rebuild from the corpus |
| `news_2010_2026_full.csv` | 153MB | Nothing current | Skip |
| `venv_mac/` | 1.2GB | — | Never sync. Recreate from requirements. |

Direct transfer, if both machines are on the same network, is usually simpler
than any cloud step:

```bash
rsync -avP --info=progress2 \
  trump_llama_v3/ user@other-machine:~/trump_tweets_combined/trump_llama_v3/
```

---

## The collector: only one machine, ever

This is not a sync preference, it is a hard constraint. Truth Social rate-limits
**per client, not per process**. Two collectors do not collect twice as fast —
they starve each other, and the failure is silent: a starved poller sees no new
posts, which looks exactly like "he stopped posting" and biases the weekly count
forecast toward the low brackets.

So `live_pipeline.db` stays gitignored. It is live, machine-specific state, it is
written continuously in WAL mode, and committing it would conflict constantly.

If the second machine needs to *see* collected state, it already can —
`publish_snapshot.py` pushes recent posts and the week-to-date count to a public
gist, which any machine can read:

```bash
venv_mac/bin/python3 publish_snapshot.py     # on the collector
```

---

## Day to day

```bash
git pull                     # start of a session
# ... work ...
git add -A && git commit -m "..." && git push
```

Keep `.gitignore` honest. Before committing anything large:

```bash
git status --porcelain | grep '^??' | awk '{print $2}' | xargs du -sh 2>/dev/null | sort -rh | head
```
