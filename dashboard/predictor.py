"""
Tweet prediction engine — three modes, auto-detected:
  1. claude_api  : Anthropic Claude API + few-shot RAG examples (cloud-friendly)
  2. rag_llm     : Local fine-tuned Llama 3 8B + FAISS retrieval (full local)
  3. template    : Pure keyword template fallback (no deps)

Priority: claude_api (if ANTHROPIC_API_KEY set) > rag_llm (if local model exists) > template
"""

import json
import os
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RAG_DATA_PATH   = REPO_ROOT / "rag_data.jsonl"
RAG_INDEX_PATH  = REPO_ROOT / "rag_index.faiss"
LORA_MODEL_PATH = REPO_ROOT / "trump_llama_v3"
SAMPLE_EXAMPLES = Path(__file__).parent / "sample_examples.jsonl"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

SYSTEM_PROMPT = (
    "You are Donald Trump writing a Truth Social post. "
    "Write 1-3 punchy sentences reacting to the given news headline. "
    "Match Trump's voice exactly: emphatic, direct, uses ALL CAPS for key words, "
    "attacks opponents by name, ends with a slogan or exclamation. "
    "Do NOT include URLs or hashtags. Just the post text."
)

TEMPLATES = [
    "TERRIBLE! {kw} is a DISASTER for America. We need STRONG leadership NOW!",
    "The Fake News media won't tell you the truth about {kw}. SAD!",
    "{kw} is happening because of the RADICAL LEFT. They are DESTROYING our great country!",
    "Nobody knows more about {kw} than me, believe me. We will WIN, WIN, WIN!",
    "Can you believe what they're doing with {kw}? MAKE AMERICA GREAT AGAIN!",
    "{kw} — another DISASTER created by Crooked politicians. When I was president we had the GREATEST economy EVER!",
    "The Lamestream Media is covering up the truth about {kw}. ENEMY OF THE PEOPLE!",
    "BREAKING: {kw}. The Deep State is at it again. TOTAL WITCH HUNT!",
]


def _extract_kw(headlines: list[str]) -> str:
    stop = {"the","a","an","is","are","was","were","to","of","in","for","on",
            "with","as","at","by","from","into","after","says","will","has","have"}
    words = [w for h in headlines for w in re.split(r"\W+", h)
             if len(w) > 3 and w.lower() not in stop]
    return " ".join(words[:3]) if words else "this situation"


def _load_sample_examples(n: int = 30) -> list[dict]:
    path = SAMPLE_EXAMPLES if SAMPLE_EXAMPLES.exists() else RAG_DATA_PATH
    if not path.exists():
        return []
    with open(path) as f:
        lines = [l for l in f.readlines() if l.strip()]
    sample = random.sample(lines, min(n, len(lines)))
    out = []
    for line in sample:
        d = json.loads(line)
        headline = d.get("news") or d.get("headline") or ""
        tweet = d.get("tweet") or d.get("completion") or ""
        if headline and tweet:
            out.append({"headline": headline[:300], "tweet": tweet[:280]})
    return out


class TweetPredictor:
    def __init__(self):
        self.mode = "template"
        self.rag_data: list[dict] = []
        self.index = None
        self.embedder = None
        self.model = None
        self.tokenizer = None
        self._claude = None
        self._sample_examples: list[dict] = []
        self._init()

    def _init(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            self._init_claude(api_key)
        else:
            self._init_local_rag()

    # ── Claude API mode ───────────────────────────────────────────────────────

    def _init_claude(self, api_key: str):
        try:
            import anthropic
            self._claude = anthropic.Anthropic(api_key=api_key)
            self._sample_examples = _load_sample_examples(40)
            self.mode = "claude_api"
            print(f"[Predictor] Claude API ready ({len(self._sample_examples)} few-shot examples)")
        except ImportError:
            print("[Predictor] anthropic package not installed — falling back to local")
            self._init_local_rag()
        except Exception as e:
            print(f"[Predictor] Claude init error: {e} — falling back to local")
            self._init_local_rag()

    def _predict_claude(self, headlines: list[str]) -> str:
        # Build few-shot prompt from sample examples
        examples = random.sample(self._sample_examples, min(5, len(self._sample_examples)))
        messages = []
        for ex in examples:
            messages.append({"role": "user",    "content": ex["headline"]})
            messages.append({"role": "assistant","content": ex["tweet"]})
        messages.append({"role": "user", "content": "\n".join(headlines)})

        resp = self._claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        result = resp.content[0].text.strip()
        result = re.sub(r"https?://\S+", "", result).strip()
        return result

    # ── Local RAG + Llama mode ────────────────────────────────────────────────

    def _init_local_rag(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            if not RAG_DATA_PATH.exists() or not RAG_INDEX_PATH.exists():
                print("[Predictor] RAG files not found — using template mode")
                return

            print("[Predictor] Loading RAG data...", flush=True)
            with open(RAG_DATA_PATH) as f:
                self.rag_data = [json.loads(l) for l in f]
            self.index = faiss.read_index(str(RAG_INDEX_PATH))
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.mode = "rag_retrieve"
            print(f"[Predictor] RAG ready — {len(self.rag_data):,} examples", flush=True)
            self._init_llm()
        except ImportError as e:
            print(f"[Predictor] Missing dep ({e}) — template mode")
        except Exception as e:
            print(f"[Predictor] RAG error: {e} — template mode")

    def _init_llm(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_token = os.environ.get("HF_TOKEN", "")
            model_path = str(LORA_MODEL_PATH) if LORA_MODEL_PATH.exists() else BASE_MODEL

            print(f"[Predictor] Loading LLM from {model_path}...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, token=hf_token
            ).to(device)
            self.model.eval()
            self.mode = "rag_llm"
            print(f"[Predictor] LLM ready on {device}", flush=True)
        except Exception as e:
            print(f"[Predictor] LLM not available ({e}) — staying in rag_retrieve")

    def _retrieve(self, headlines: list[str], k: int = 5) -> list[dict]:
        query = " ".join(headlines)
        emb = self.embedder.encode([query], normalize_embeddings=True)
        import numpy as np
        _, idxs = self.index.search(emb.astype("float32"), k)
        return [self.rag_data[i] for i in idxs[0] if 0 <= i < len(self.rag_data)]

    def _predict_rag_llm(self, headlines: list[str]) -> str:
        import torch
        examples = self._retrieve(headlines, k=5)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in examples[:3]:
            hl = ex.get("news") or ex.get("headline") or ""
            tw = ex.get("tweet") or ex.get("completion") or ""
            if hl and tw:
                messages.append({"role": "user",      "content": hl})
                messages.append({"role": "assistant", "content": tw})
        messages.append({"role": "user", "content": "\n".join(headlines)})

        device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=120, temperature=0.85,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return re.sub(r"https?://\S+", "", result).strip()

    def _predict_rag_retrieve(self, headlines: list[str]) -> str:
        examples = self._retrieve(headlines, k=3)
        if examples:
            tw = examples[0].get("tweet") or examples[0].get("completion") or ""
            if tw:
                return re.sub(r"https?://\S+", "", tw).strip()
        return self._predict_template(headlines)

    def _predict_template(self, headlines: list[str]) -> str:
        kw = _extract_kw(headlines)
        return random.choice(TEMPLATES).format(kw=kw)

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, headlines: list[str]) -> str:
        if self.mode == "claude_api":
            return self._predict_claude(headlines)
        elif self.mode == "rag_llm":
            return self._predict_rag_llm(headlines)
        elif self.mode == "rag_retrieve":
            return self._predict_rag_retrieve(headlines)
        else:
            return self._predict_template(headlines)
