"""
Tweet prediction engine.
Tries to load the RAG pipeline (FAISS index + sentence-transformers + Llama 3).
Falls back gracefully through: rag_llm → rag_retrieve → template
"""

import json
import os
import random
import re
from pathlib import Path

# Paths relative to repo root (one level up from dashboard/)
REPO_ROOT = Path(__file__).parent.parent
RAG_DATA_PATH = REPO_ROOT / "rag_data.jsonl"
RAG_INDEX_PATH = REPO_ROOT / "rag_index.faiss"
LORA_MODEL_PATH = REPO_ROOT / "trump_llama_v3"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

SYSTEM_PROMPT = (
    "You are Donald Trump. Write a short social media post (1-3 sentences) "
    "reacting to the following news headline. Use Trump's voice: direct, "
    "emphatic, occasionally ALL CAPS for key words, punchy."
)

TEMPLATES = [
    "TERRIBLE! {kw} is a DISASTER for America. We need STRONG leadership NOW!",
    "The Fake News media won't tell you the truth about {kw}. SAD!",
    "{kw} is happening because of the RADICAL LEFT. They are DESTROYING our great country!",
    "Nobody knows more about {kw} than me, believe me. We will WIN!",
    "BREAKING: {kw}. The Deep State doesn't want you to know this. TOTAL WITCH HUNT!",
    "Can you believe what they're doing with {kw}? In the history of our country, "
    "there has NEVER been anything like this. MAKE AMERICA GREAT AGAIN!",
    "{kw} — another DISASTER created by Crooked politicians. When I was president, "
    "we had the GREATEST economy EVER. Now look at this mess!",
    "The Lamestream Media is covering up the truth about {kw}. "
    "They are truly the ENEMY OF THE PEOPLE!",
]


def _extract_keywords(headlines: list[str]) -> str:
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "for", "on", "with", "as", "at", "by",
        "from", "up", "about", "into", "through", "after",
    }
    words = []
    for h in headlines:
        for w in re.split(r"\W+", h):
            if len(w) > 3 and w.lower() not in stopwords:
                words.append(w)
    return " ".join(words[:3]) if words else "this situation"


class TweetPredictor:
    def __init__(self):
        self.mode = "template"
        self.rag_data: list[dict] = []
        self.index = None
        self.embedder = None
        self.model = None
        self.tokenizer = None
        self._load_rag()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_rag(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss  # noqa: F401

            if not RAG_DATA_PATH.exists():
                print(f"[Predictor] rag_data.jsonl not found at {RAG_DATA_PATH}, using template mode")
                return
            if not RAG_INDEX_PATH.exists():
                print(f"[Predictor] rag_index.faiss not found at {RAG_INDEX_PATH}, using template mode")
                return

            print("[Predictor] Loading RAG data...", flush=True)
            with open(RAG_DATA_PATH) as f:
                self.rag_data = [json.loads(line) for line in f]
            import faiss as _faiss
            self.index = _faiss.read_index(str(RAG_INDEX_PATH))
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.mode = "rag_retrieve"
            print(f"[Predictor] RAG ready — {len(self.rag_data):,} examples", flush=True)
            self._load_llm()

        except ImportError as e:
            print(f"[Predictor] Missing dependency ({e}), using template mode")
        except Exception as e:
            print(f"[Predictor] RAG load error: {e}, using template mode")

    def _load_llm(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_token = os.environ.get("HF_TOKEN", "")

            # Prefer local LoRA-merged model if it exists
            model_path = str(LORA_MODEL_PATH) if LORA_MODEL_PATH.exists() else BASE_MODEL

            print(f"[Predictor] Loading LLM from {model_path}...", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                token=hf_token,
            ).to(device)
            self.model.eval()
            self.mode = "rag_llm"
            print(f"[Predictor] LLM ready on {device}", flush=True)

        except Exception as e:
            print(f"[Predictor] LLM not available ({e}), staying in rag_retrieve mode")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, headlines: list[str], k: int = 5) -> list[dict]:
        query = " ".join(headlines)
        emb = self.embedder.encode([query], normalize_embeddings=True)
        import numpy as np
        _, idxs = self.index.search(emb.astype("float32"), k)
        return [self.rag_data[i] for i in idxs[0] if 0 <= i < len(self.rag_data)]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def predict(self, headlines: list[str]) -> str:
        if self.mode == "rag_llm":
            return self._predict_rag_llm(headlines)
        elif self.mode == "rag_retrieve":
            return self._predict_rag_retrieve(headlines)
        else:
            return self._predict_template(headlines)

    def _predict_rag_llm(self, headlines: list[str]) -> str:
        import torch

        examples = self._retrieve(headlines, k=5)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in examples[:3]:
            hl = ex.get("headline") or ex.get("headlines") or ""
            tw = ex.get("tweet") or ex.get("completion") or ""
            if hl and tw:
                messages.append({"role": "user", "content": hl})
                messages.append({"role": "assistant", "content": tw})
        messages.append({"role": "user", "content": "\n".join(headlines)})

        device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.85,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs.input_ids.shape[1]:]
        result = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # Strip URL artifacts
        result = re.sub(r"https?://\S+", "", result).strip()
        return result

    def _predict_rag_retrieve(self, headlines: list[str]) -> str:
        """Return the closest real Trump tweet as a proxy (no LLM needed)."""
        examples = self._retrieve(headlines, k=3)
        if examples:
            tw = examples[0].get("tweet") or examples[0].get("completion") or ""
            if tw:
                return tw
        return self._predict_template(headlines)

    def _predict_template(self, headlines: list[str]) -> str:
        kw = _extract_keywords(headlines)
        return random.choice(TEMPLATES).format(kw=kw)
