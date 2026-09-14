"""
Prediction at time T.

Every prediction row is written before the outcome exists and is never revised.
The headline list is snapshotted into the row rather than referenced by id alone,
so if a publisher later edits or pulls a story the historical record of what the
model actually saw stays intact.

Model loading is lazy and cached per process. The fine-tuned Llama is 16GB and
takes minutes to load on MPS, so a long-running daemon amortises it while a
one-shot cron run should stick to the cheap models.
"""

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import signal as sig
from .store import jdump, utcnow

REPO = Path(__file__).parent.parent

SYSTEM_PROMPT = (
    "You are Donald Trump writing a Truth Social post. "
    "Write 1-3 punchy sentences reacting to the given news headlines. "
    "Match Trump's voice exactly: emphatic, direct, ALL CAPS for key words, "
    "attacks opponents by name, ends with a slogan or exclamation. "
    "Do NOT include URLs or hashtags. Output only the post text."
)

# Models that need no heavyweight dependencies — safe for a cron one-shot.
CHEAP_MODELS = {"markov", "ngram", "rag_retrieve"}
# Models that load multi-GB weights — use in a warm daemon.
HEAVY_MODELS = {"finetuned_llama", "rag_llm"}

DEFAULT_HORIZON_HOURS = 6.0
DEFAULT_NEWS_WINDOW_HOURS = 6.0
MAX_HEADLINES = 8


@dataclass
class Prediction:
    id: str
    predicted_at: str
    model: str
    horizon_hours: float
    news_window_start: str
    news_window_end: str
    news_ids: list[str]
    headlines: list[str]
    text: str
    signal: sig.Signal

    def as_row(self, run_id: str) -> dict:
        row = {
            "id": self.id,
            "predicted_at": self.predicted_at,
            "model": self.model,
            "horizon_hours": self.horizon_hours,
            "news_window_start": self.news_window_start,
            "news_window_end": self.news_window_end,
            "news_ids": jdump(self.news_ids),
            "headlines": jdump(self.headlines),
            "text": self.text,
            "run_id": run_id,
        }
        row.update(self.signal.as_row())
        return row


# The corpus stores URLs with spaces injected by earlier scraping
# ("https: t. co 5jiSyvQiNj", "www. donaldjtrump. com J777QHdYa3"), so a plain
# https?://\S+ pattern leaves most of the URL behind as word salad.
_URL_FRAGMENT = re.compile(
    r"(?:https?\s*:\s*(?://)?\s*(?:\S+?\s*\.\s*)+\S+"
    r"|www\s*\.\s*(?:\S+?\s*\.\s*)+\S+"
    r"|\bt\s*\.\s*co\b\s*\S*)"
    # Trailing path slug, split off by the same scraping that spaced the domain.
    # Requires a digit so ordinary following words are not swallowed.
    r"(?:\s+(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{5,})?",
    re.I,
)


def _clean_generation(text: str) -> str:
    """Strip URLs, retweet markers, hashtags and leaked instruction framing."""
    text = _URL_FRAGMENT.sub(" ", text or "")
    text = re.sub(r"#\w+", "", text)
    # A retweet is not an original post. Only strip an explicit @handle after the
    # marker — a bare \w+ would eat the first real word ("RT: Today we..." → "we").
    text = re.sub(r"^\s*RT[:\s]+(?:@\w+[:\s]+)?", "", text, flags=re.I)
    text = re.sub(r"^\s*(post|tweet|response|answer)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r'^\s*["“]|["”]\s*$', "", text.strip())
    return re.sub(r"\s+", " ", text).strip()[:600]


# ── Model backends ───────────────────────────────────────────────────────────

class _Backends:
    """Lazily-constructed, process-cached model backends."""

    def __init__(self):
        self._markov = None
        self._ngram = None
        self._rag = None          # (index, records, encoder)
        self._llama = None        # (model, tokenizer, device)

    # -- statistical baselines ------------------------------------------------

    def markov(self):
        if self._markov is None:
            import sys
            sys.path.insert(0, str(REPO))
            from markov_predictor import MarkovTweetGenerator
            # __init__ already calls build_chain(); calling it again appends a
            # second copy of every transition and doubles start_words.
            self._markov = MarkovTweetGenerator(
                csv_file=str(REPO / "trump_tweets_cleaned.csv"), order=2)
        return self._markov

    def ngram(self):
        if self._ngram is None:
            import sys
            sys.path.insert(0, str(REPO))
            from ngram_predictor import NgramTweetGenerator
            # __init__ already calls build_models() — see markov note above.
            self._ngram = NgramTweetGenerator(
                csv_file=str(REPO / "trump_tweets_cleaned.csv"), n=3)
        return self._ngram

    # -- retrieval ------------------------------------------------------------

    def rag(self):
        if self._rag is None:
            import faiss
            from sentence_transformers import SentenceTransformer
            index = faiss.read_index(str(REPO / "rag_index.faiss"))
            records = []
            with open(REPO / "rag_data.jsonl") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self._rag = (index, records, encoder)
        return self._rag

    def retrieve(self, headlines: list[str], k: int = 5) -> list[dict]:
        index, records, encoder = self.rag()
        emb = encoder.encode([" ".join(headlines)], normalize_embeddings=True)
        _, idxs = index.search(emb.astype("float32"), k)
        return [records[i] for i in idxs[0] if 0 <= i < len(records)]

    # -- fine-tuned Llama -----------------------------------------------------

    def llama(self):
        if self._llama is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            path = REPO / "trump_llama_v3"
            if not path.exists():
                raise FileNotFoundError(f"fine-tuned model not found at {path}")
            tok = AutoTokenizer.from_pretrained(str(path))
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            # bfloat16 only — float16 segfaults on Metal (see README lessons).
            model = AutoModelForCausalLM.from_pretrained(
                str(path), dtype=torch.bfloat16).to(device)
            model.eval()
            self._llama = (model, tok, device)
        return self._llama

    def _generate_llm(self, messages, max_new_tokens=120, temperature=0.85) -> str:
        import torch
        model, tok, device = self.llama()
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=True, top_p=0.9, pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)


_BACKENDS = _Backends()


# ── Per-model prediction ─────────────────────────────────────────────────────

def _unwrap(out) -> str:
    """generate_from_news returns list[dict] with a 'tweet' key, not list[str]."""
    if not out:
        return ""
    first = out[0]
    return first.get("tweet", "") if isinstance(first, dict) else str(first)


def _predict_markov(headlines: list[str]) -> str:
    gen = _BACKENDS.markov()
    out = gen.generate_from_news([{"title": h} for h in headlines], num_tweets=1)
    return _clean_generation(_unwrap(out))


def _predict_ngram(headlines: list[str]) -> str:
    gen = _BACKENDS.ngram()
    out = gen.generate_from_news([{"title": h} for h in headlines], num_tweets=1)
    return _clean_generation(_unwrap(out))


def _predict_rag_retrieve(headlines: list[str]) -> str:
    """Nearest historical post. No generation — a strong, honest baseline."""
    hits = _BACKENDS.retrieve(headlines, k=3)
    for h in hits:
        txt = h.get("tweet") or h.get("completion") or ""
        if txt:
            return _clean_generation(txt)
    return ""


def _predict_finetuned(headlines: list[str]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(headlines)},
    ]
    return _clean_generation(_BACKENDS._generate_llm(messages))


def _predict_rag_llm(headlines: list[str]) -> str:
    examples = _BACKENDS.retrieve(headlines, k=5)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples[:3]:
        hl = ex.get("news") or ex.get("headline") or ""
        tw = ex.get("tweet") or ex.get("completion") or ""
        if hl and tw:
            messages.append({"role": "user", "content": hl})
            messages.append({"role": "assistant", "content": tw})
    messages.append({"role": "user", "content": "\n".join(headlines)})
    return _clean_generation(_BACKENDS._generate_llm(messages))


PREDICTORS = {
    "markov": _predict_markov,
    "ngram": _predict_ngram,
    "rag_retrieve": _predict_rag_retrieve,
    "finetuned_llama": _predict_finetuned,
    "rag_llm": _predict_rag_llm,
}


# ── Public API ───────────────────────────────────────────────────────────────

def select_headlines(news_rows, limit: int = MAX_HEADLINES,
                     min_relevance: float = 0.3) -> list[dict]:
    """
    Pick the headlines the models will condition on.

    Deduplicates near-identical stories syndicated across outlets — without this
    a single big story crowds out the whole window and the prediction collapses
    onto one topic.
    """
    chosen: list[dict] = []
    seen_sigs: list[set[str]] = []
    for row in news_rows:
        if row["relevance"] < min_relevance:
            continue
        words = {w for w in re.findall(r"[a-z]{4,}", row["title"].lower())}
        if any(len(words & s) / max(len(words | s), 1) > 0.5 for s in seen_sigs):
            continue
        seen_sigs.append(words)
        chosen.append(dict(row))
        if len(chosen) >= limit:
            break
    return chosen


def make_prediction(model: str, headlines: list[dict],
                    horizon_hours: float = DEFAULT_HORIZON_HOURS,
                    window_hours: float = DEFAULT_NEWS_WINDOW_HOURS) -> Prediction:
    """Generate one prediction. Raises if the model name is unknown."""
    if model not in PREDICTORS:
        raise ValueError(f"unknown model {model!r}; known: {sorted(PREDICTORS)}")

    now = datetime.now(timezone.utc)
    titles = [h["title"] for h in headlines]
    text = PREDICTORS[model](titles)

    return Prediction(
        id=f"{model}-{uuid.uuid4().hex[:12]}",
        predicted_at=now.isoformat(),
        model=model,
        horizon_hours=horizon_hours,
        news_window_start=(now - timedelta(hours=window_hours)).isoformat(),
        news_window_end=now.isoformat(),
        news_ids=[h["id"] for h in headlines],
        headlines=titles,
        text=text,
        signal=sig.extract(text),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="One-off prediction from live news")
    ap.add_argument("--models", default="markov,ngram,rag_retrieve",
                    help="comma-separated: " + ",".join(sorted(PREDICTORS)))
    ap.add_argument("--headlines", nargs="*", help="override with explicit headlines")
    args = ap.parse_args()

    if args.headlines:
        rows = [{"id": f"manual-{i}", "title": h, "relevance": 1.0}
                for i, h in enumerate(args.headlines)]
    else:
        from .news_feed import fetch_all
        print("Fetching live news...\n")
        news, _ = fetch_all(verbose=False)
        rows = select_headlines(news)

    print(f"\nConditioning on {len(rows)} headlines:")
    for r in rows:
        print(f"  • {r['title'][:84]}")

    for m in args.models.split(","):
        m = m.strip()
        print(f"\n── {m} " + "─" * (66 - len(m)))
        try:
            p = make_prediction(m, rows)
            print(f"  {p.text[:300]}")
            print(f"  → topic={p.signal.topic}  polarity={p.signal.polarity:+.2f}  "
                  f"intensity={p.signal.intensity:.2f}")
            print(f"  → tickers={p.signal.tickers}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
