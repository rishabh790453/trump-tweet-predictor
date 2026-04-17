"""
make_interview_pdf.py  -- generates interview_prep.pdf using fpdf2 + Arial Unicode TTF
"""
from fpdf import FPDF, XPos, YPos
import os

FONT_REG  = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_ITAL = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
FONT_BI   = "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"

BLUE      = (0,   40,  100)
DKGRAY    = (50,  50,   50)
GRAY      = (110, 110, 110)
LTGRAY    = (240, 240, 240)
WARN_BG   = (255, 248, 220)
WARN_BOR  = (180,  90,   0)
Q_BG      = (230, 240, 255)
CODE_BG   = (245, 245, 245)
WHITE     = (255, 255, 255)
BLACK     = (0,   0,    0)


class PDF(FPDF):
    def header(self):
        self.set_font("ArialItal", size=8)
        self.set_text_color(*GRAY)
        self.cell(0, 6, "Trump Tweet Predictor -- Interview Preparation Guide | Rishabh Gulecha",
                  align="R")
        self.ln(2)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", size=8)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def h1(self, text):
        self.ln(4)
        self.set_font("ArialBold", size=13)
        self.set_text_color(*BLUE)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(*BLACK)

    def h2(self, text):
        self.ln(3)
        self.set_font("ArialBold", size=11)
        self.set_text_color(*BLUE)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*BLACK)
        self.ln(1)

    def body(self, text, size=10):
        self.set_font("Arial", size=size)
        self.set_text_color(*DKGRAY)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, items, size=10):
        self.set_font("Arial", size=size)
        self.set_text_color(*DKGRAY)
        for item in items:
            self.set_x(self.l_margin + 4)
            self.multi_cell(0, 5.5, f"*  {item}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def numbered(self, items, size=10):
        self.set_font("Arial", size=size)
        self.set_text_color(*DKGRAY)
        for i, item in enumerate(items, 1):
            self.set_x(self.l_margin + 4)
            self.multi_cell(0, 5.5, f"{i}.  {item}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def q_box(self, question):
        self.ln(2)
        self.set_fill_color(*Q_BG)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.6)
        w = self.w - self.l_margin - self.r_margin
        x0, y0 = self.l_margin, self.get_y()
        self.set_font("ArialBold", size=10)
        self.set_text_color(*BLUE)
        # estimate height
        lines = len(question) / 90 + 1
        h = max(10, int(lines) * 6 + 6)
        self.rect(x0, y0, w, h, style="FD")
        self.set_xy(x0 + 3, y0 + 2)
        self.multi_cell(w - 6, 5.5, question, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(max(self.get_y(), y0 + h) + 2)
        self.set_text_color(*BLACK)

    def callout(self, text, bg, border, label=""):
        self.ln(2)
        self.set_fill_color(*bg)
        self.set_draw_color(*border)
        self.set_line_width(0.6)
        w = self.w - self.l_margin - self.r_margin
        x0, y0 = self.l_margin, self.get_y()
        self.set_font("Arial", size=9.5)
        char_per_line = int(w / 4.8)
        lines = sum(max(1, len(item) // char_per_line + 1) for item in text.split("\n"))
        h = lines * 5.5 + (8 if label else 4)
        self.rect(x0, y0, w, h, style="FD")
        cy = y0 + 3
        self.set_xy(x0 + 4, cy)
        if label:
            self.set_font("ArialBold", size=9.5)
            self.set_text_color(*border)
            self.cell(0, 5.5, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_x(x0 + 4)
        self.set_font("Arial", size=9.5)
        self.set_text_color(*DKGRAY)
        self.multi_cell(w - 8, 5.2, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(max(self.get_y(), y0 + h) + 2)
        self.set_text_color(*BLACK)

    def code(self, text):
        self.ln(1)
        self.set_fill_color(*CODE_BG)
        self.set_draw_color(*GRAY)
        self.set_line_width(0.3)
        self.set_font("Mono", size=8.5)
        self.set_text_color(*DKGRAY)
        w = self.w - self.l_margin - self.r_margin
        lines = text.strip().split("\n")
        h = len(lines) * 5 + 6
        x0, y0 = self.l_margin, self.get_y()
        self.rect(x0, y0, w, h, style="FD")
        self.set_xy(x0 + 3, y0 + 3)
        for line in lines:
            self.set_x(x0 + 3)
            self.cell(0, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_y(y0 + h + 2)
        self.set_text_color(*BLACK)
        self.ln(1)

    def table(self, headers, rows, col_widths=None):
        self.ln(2)
        w = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [w / len(headers)] * len(headers)
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        self.set_font("ArialBold", size=9)
        self.set_draw_color(*BLUE)
        for h, cw in zip(headers, col_widths):
            self.cell(cw, 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(*DKGRAY)
        self.set_font("Arial", size=9)
        for ri, row in enumerate(rows):
            self.set_fill_color(*(LTGRAY if ri % 2 == 0 else WHITE))
            for val, cw in zip(row, col_widths):
                self.cell(cw, 6.5, str(val), border=1, fill=True, align="C")
            self.ln()
        self.set_text_color(*BLACK)
        self.ln(2)


# ── instantiate ────────────────────────────────────────────────────────────────
pdf = PDF()
pdf.set_margins(18, 18, 18)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_font("Arial",     fname=FONT_REG)
pdf.add_font("ArialBold", fname=FONT_BOLD)
pdf.add_font("ArialItal", fname=FONT_ITAL)
pdf.add_font("ArialBI",   fname=FONT_BI)
pdf.add_font("Mono",      fname=FONT_MONO)
pdf.add_page()

# ── Cover ──────────────────────────────────────────────────────────────────────
pdf.ln(8)
pdf.set_font("ArialBold", size=22)
pdf.set_text_color(*BLUE)
pdf.cell(0, 12, "Presidential Tweet Prediction Project", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Arial", size=13)
pdf.set_text_color(*DKGRAY)
pdf.cell(0, 8, "Complete SWE / ML / AI Interview Preparation Guide", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)
pdf.set_font("ArialItal", size=11)
pdf.set_text_color(*GRAY)
pdf.cell(0, 7, "Everything you built, every mistake, every concept -- ready to explain in an interview",
         align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(3)
pdf.set_font("Arial", size=10)
pdf.cell(0, 6, "Rishabh Gulecha   |   rishabhgulecha28@gmail.com   |   April 2026", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(6)
pdf.set_draw_color(*BLUE); pdf.set_line_width(0.8)
pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
pdf.ln(6)

# ── 1. The Pitch ───────────────────────────────────────────────────────────────
pdf.h1("1.  The 30-Second Pitch")
pdf.callout(
    '"I built an end-to-end NLP system that predicts what Donald Trump would tweet in response '
    'to a given news headline. I assembled a dataset of 86,000 posts and 1.3 million news articles, '
    'semantically matched them using sentence embeddings and FAISS, then compared four generation '
    'systems: a Markov chain baseline, an N-gram baseline, a LoRA fine-tuned Llama 3 8B, and a RAG '
    'pipeline. I ran full BLEU/ROUGE evaluation across five real held-out examples and wrote up the '
    'results as an academic paper targeting arXiv. The whole thing ran locally on an Apple M5 MacBook '
    '-- no cloud GPUs."',
    bg=Q_BG, border=BLUE, label='What to say when asked "tell me about this project":')
pdf.h2("Scale at a glance")
pdf.bullet([
    "86,170 tweets  (Twitter 2009-2021 + Truth Social 2022-2026)",
    "1.3M news articles  (NYT Archive API + Guardian Open Platform)",
    "65,733 semantically matched news-tweet training pairs",
    "Llama 3 8B Instruct -- 8 billion parameters, LoRA fine-tuned (16M trainable params, 0.2%)",
    "FAISS index of 58,229 dense 384-dim vectors for RAG retrieval",
    "17 hours of training on Apple Silicon M5  (MPS backend, no cloud)",
])

# ── 2. What You Built ──────────────────────────────────────────────────────────
pdf.h1("2.  What You Built -- Technical Deep Dive")

pdf.h2("2.1  Data Pipeline")
pdf.body("Step 1 -- Tweet cleaning: Loaded raw CSV of 86,170 posts. Filtered out retweets (RT prefix), "
         "replies (leading @), posts under 30 chars, posts over 400 chars. Stripped all URLs. Result: 65,733 usable posts.")
pdf.body("Step 2 -- News corpus: NYT Archive API month-by-month 2009-2026 (rate-limited, wrote backfill jobs). "
         "Added 32K Guardian articles. Stored in SQLite (news_data.db): title, published_at, source, section.")
pdf.body("Step 3 -- Semantic matching: For each post on date d, retrieved all headlines in the 48h window before d. "
         "Embedded with all-MiniLM-L6-v2 (384-dim). Cosine similarity >= 0.20 threshold, top-10 per post. "
         "Discarded posts with <3 qualifying headlines. Average kept similarity: 0.339 vs ~0.05 random (7x signal-to-noise).")

pdf.h2("2.2  Markov Chain and N-gram Baselines")
pdf.body("Second-order Markov chain: 607,851 unique bigrams from 84,327 tweets. Keyword-seeded from headlines at inference. "
         "Trigram model with similar approach. Neither has real news awareness -- they are tweet-corpus statistics only. "
         "They serve as lower bounds: the minimum score a news-conditioned model must beat.")

pdf.h2("2.3  LoRA Fine-Tuned Llama 3 8B")
pdf.body("Base model: meta-llama/Meta-Llama-3-8B-Instruct. LoRA injects low-rank adapters into attention + MLP layers. "
         "With rank r=16: only 16M of 8B params are trainable (0.2%). Fits in 32GB unified memory without quantization.")
pdf.body("Training format -- prompt/completion dict for completion-only loss:")
pdf.code('{\n'
         '  "prompt": [\n'
         '    {"role": "system",    "content": "You are Donald Trump..."},\n'
         '    {"role": "user",      "content": "Today\'s news headlines:\\n- ..."},\n'
         '  ],\n'
         '  "completion": [\n'
         '    {"role": "assistant", "content": "<actual tweet>"},\n'
         '  ]\n'
         '}')
pdf.body("Key config: completion_only_loss=True | 25K examples, 1 epoch | eff. batch=32 | lr=2e-4 cosine | bfloat16 MPS")
pdf.body("Result: validation loss 2.866 | token accuracy 45.4% on tweet tokens only (honest metric -- prompt tokens excluded)")

pdf.h2("2.4  RAG Pipeline")
pdf.body("Index: all 65,733 pairs embedded (384-dim), stored in FAISS IndexFlatIP (exact cosine for normalized vectors). "
         "Inference: embed query headlines, retrieve top-5 nearest (headline, tweet) pairs, inject as few-shot examples, "
         "generate with BASE Llama 3 8B Instruct (not the fine-tuned version).")
pdf.body("Why base model? The retrieved examples carry the stylistic signal. Using the fine-tuned model would conflate "
         "two variables. Clean ablation requires one variable changed at a time.")

pdf.h2("2.5  Evaluation Results (5 held-out examples, avg over 5 candidates each)")
pdf.table(
    ["Model", "BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
    [
        ["Markov Chain",       "0.032", "0.004", "0.090", "0.008", "0.069"],
        ["N-gram (trigram)",   "0.047", "0.008", "0.123", "0.012", "0.086"],
        ["Fine-tuned Llama 3", "0.055", "0.008", "0.154", "0.009", "0.095"],
        ["RAG",                "0.056", "0.012", "0.153", "0.010", "0.097"],
    ],
    col_widths=[48, 22, 22, 22, 22, 22])
pdf.body("Key finding: Fine-tune and RAG TIE on average (ROUGE-1: 0.154 vs 0.153). A single example suggested RAG won -- "
         "five diverse examples reveal parity. ROUGE-2 <= 0.012 across ALL models. Trump's exact phrases are unrecoverable by surface metrics.")

# ── 3. What Went Wrong ─────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("3.  What Went Wrong -- The Real Learning")

pdf.h2("3.1  Prompt Leakage (v1 and v2)")
pdf.callout(
    'Bug: Fine-tuned Llama repeated the system prompt verbatim -- "You are Donald Trump. '
    'Given today\'s news headlines..." -- instead of generating tweets.',
    bg=WARN_BG, border=WARN_BOR, label="Failure:")
pdf.body("Root cause: Standard SFT computes loss over the entire sequence (system + user + assistant). "
         "The system prompt appears thousands of times with a gradient signal, so the model learns to predict it. "
         "This is called prompt leakage.")
pdf.body("Fix: completion_only_loss=True in TRL SFTConfig. Tokenizes prompt and completion separately. "
         "Gradients flow ONLY from assistant (tweet) tokens. The model sees the prompt as context but is never "
         "penalized for not predicting it.")
pdf.body("Why it matters: This is one of the most common mistakes in instruction fine-tuning. Every production "
         "SFT system (InstructGPT, Llama 2 Chat) uses this masking. It is not optional.")

pdf.h2("3.2  TRL 1.0 API Breaking Changes")
pdf.callout(
    "ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'",
    bg=WARN_BG, border=WARN_BOR, label="Failure:")
pdf.body("TRL 0.x had DataCollatorForCompletionOnlyLM. TRL 1.0 removed it. Also tried assistant_only_loss=True -- "
         "requires {% generation %} in the model's Jinja chat template, which Llama 3 does not have. "
         "Fix: use prompt/completion dict format + completion_only_loss=True.")
pdf.body("Lesson: ML libraries (TRL, PEFT, transformers) break APIs constantly. Always check the changelog "
         "before copying tutorial code. Version mismatches are the #1 source of wasted hours in ML projects.")

pdf.h2("3.3  Mac Sleep During Training (Twice)")
pdf.body("Training Llama 3 8B for 17 hours on a laptop is risky. The Mac slept twice:")
pdf.bullet([
    "Sleep 1 at step 672: step time spiked from 49s to 295s. Recovered within ~10 steps.",
    "Sleep 2 at step 725: step time spiked to 3,920s (65 minutes for ONE step). Also recovered.",
])
pdf.body("Fix: nohup caffeinate -i &  -- caffeinate prevents macOS sleep, nohup ensures it survives shell "
         "closure, & runs in background. Always run this before starting a long job and closing the lid.")
pdf.body("Why it recovered: PyTorch/MPS pauses on sleep but does not crash. Gradient state is preserved "
         "in RAM. Training resumes from exactly where it left off when the Mac wakes.")

pdf.h2("3.4  Log File Appeared Frozen")
pdf.body("Redirected training output to a file. The file appeared stuck at 23KB for hours.")
pdf.body("Root cause: Python's stdout is block-buffered when writing to a file (vs. line-buffered to a terminal). "
         "Buffers flush every ~4-8KB. Training was running fine the whole time.")
pdf.body("Confirmed with: lsof -p {pid} showed file descriptor 1 (stdout) pointing to the log file in write mode.")
pdf.body("Fix: python -u finetune_llama.py  (unbuffered stdout) or set PYTHONUNBUFFERED=1.")

pdf.h2("3.5  Single Example Misled Model Ranking")
pdf.body("Running on one example (NYC Tax) showed RAG > Fine-tune (ROUGE-1: 0.187 vs 0.152). "
         "Across five diverse examples they tie (0.154 vs 0.153). The single example happened to match "
         "RAG's retrieval index well. Lesson: always evaluate on multiple diverse examples before "
         "drawing conclusions about model ordering.")

# ── 4. Core Concepts ───────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("4.  Core ML Concepts -- Explained With Your Project")

pdf.h2("4.1  Attention Mechanism")
pdf.q_box("Q: Explain how attention works.")
pdf.body("Each token produces Q, K, V vectors. Attention score between positions:")
pdf.code("Attention(Q,K,V) = softmax( Q*K^T / sqrt(d_k) ) * V")
pdf.body("sqrt(d_k) scaling prevents softmax saturation (vanishing gradients) at high dimensions. "
         "Causal masking (lower-triangular) means position i cannot attend to j > i -- this makes "
         "generation autoregressive. In Llama 3: Grouped Query Attention (GQA) lets multiple query heads "
         "share key/value heads, reducing memory bandwidth so 8B params fits in 32GB.")

pdf.h2("4.2  LoRA -- Low-Rank Adaptation")
pdf.q_box("Q: What is LoRA and why did you use it?")
pdf.body("Standard fine-tuning updates all weight matrices W. For Llama 3 8B: ~32GB just for gradients. "
         "LoRA freezes W and injects two small matrices:")
pdf.code("W' = W + B*A\nB in R^(d x r),  A in R^(r x d),  r << d\n\n"
         "With r=16, d=4096:\n"
         "  LoRA params per matrix:  2 * 4096 * 16 = 131K\n"
         "  Full params per matrix:  4096^2       = 16.8M   (128x smaller)\n\n"
         "Applied to: q, k, v, o, gate, up, down  -- 16M total trainable of 8B (0.2%)\n"
         "At inference: merge W' = W + B*A once. Zero added latency.")

pdf.h2("4.3  RAG vs Fine-Tuning")
pdf.q_box("Q: When would you use RAG over fine-tuning?")
pdf.table(
    ["", "RAG", "Fine-Tuning"],
    [
        ["Training cost",        "None",                    "17h on M5"],
        ["Update knowledge",     "Yes (swap index)",        "No (retrain)"],
        ["Novel entities",       "Needs relevant docs",     "Generalizes from patterns"],
        ["Prompt length",        "Long (k examples)",       "Short"],
        ["Stylistic consistency","From retrieved examples",  "Baked into weights"],
    ],
    col_widths=[52, 62, 62])
pdf.body("Your result: 5 diverse topics -> they TIE (ROUGE-1: 0.154 vs 0.153). RAG wins when topics "
         "closely match retrieval index; fine-tune wins on novel geopolitical entities. "
         "Hybrid (retrieve then refine with fine-tuned model) is the natural next step.")

pdf.h2("4.4  FAISS")
pdf.q_box("Q: How does FAISS work and why is exact search OK at your scale?")
pdf.code("import faiss, numpy as np\n\n"
         "d = 384  # embedding dim\n"
         "index = faiss.IndexFlatIP(d)  # exact inner product = cosine for normalized vecs\n"
         "vecs = encoder.encode(all_headlines, normalize_embeddings=True).astype('float32')\n"
         "index.add(vecs)\n\n"
         "q = encoder.encode([query], normalize_embeddings=True).astype('float32')\n"
         "scores, indices = index.search(q, k=5)")
pdf.body("IndexFlatIP: O(n*d) exact search. n=58,229, d=384 -> ~22M multiply-adds -> <10ms on CPU. "
         "Approximate indexes (HNSW, IVF) only needed at millions+ vectors. "
         "normalize_embeddings=True is essential: without it, longer texts score higher from embedding magnitude alone.")

pdf.h2("4.5  BLEU and ROUGE")
pdf.q_box("Q: What are BLEU and ROUGE and what are their limitations?")
pdf.body("BLEU measures n-gram PRECISION of generated text vs reference + brevity penalty for short outputs. "
         "ROUGE measures n-gram RECALL (how much of the reference is covered by the generated text). "
         "ROUGE-L uses longest common subsequence (order-sensitive).")
pdf.body("Limitation from your project: ROUGE-2 <= 0.012 across ALL models. Trump's specific bigrams "
         "('TAX, TAX, TAX', 'DESTROYING New York') appear only in that exact context -- even a stylistically "
         "perfect match scores near zero. Surface metrics systematically underestimate stylistic generation quality.")

pdf.h2("4.6  bfloat16 vs float16")
pdf.q_box("Q: Why bfloat16 and not float16 on Apple Silicon?")
pdf.table(
    ["Format", "Exponent bits", "Mantissa bits", "Max value"],
    [
        ["float32",  "8", "23", "3.4 x 10^38"],
        ["float16",  "5", "10", "65,504"],
        ["bfloat16", "8", "7",  "3.4 x 10^38"],
    ],
    col_widths=[38, 38, 38, 62])
pdf.body("bfloat16 keeps float32's 8-bit exponent (same dynamic range). float16 has a smaller exponent -- "
         "gradients can overflow to inf. On Apple MPS, float16 causes segfaults (Metal shader compilation fails). "
         "Always use bfloat16 on Apple Silicon.")

# ── 5. Interview Q&A ───────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("5.  Common Interview Questions -- With Your Answers")

pdf.h2("5.1  System Design")
pdf.q_box("Q: How would you scale this to production?")
pdf.numbered([
    "Serving: Export merged LoRA weights, serve with vLLM (continuous batching, PagedAttention) -- handles 100s req/s on a single A100.",
    "Quantization: Apply 4-bit GPTQ or AWQ to reduce to ~4GB -- one GPU, lower latency.",
    "RAG scaling: Replace IndexFlatIP with HNSW (O(log n) approximate) for million-scale indexes. Add a cross-encoder re-ranker.",
    "Monitoring: Track generation latency, cache hit rate, output quality metrics. A/B test fine-tune vs RAG on live traffic.",
    "Data freshness: Automated daily pipeline to append new tweet-headline pairs. Periodic LoRA re-training. FAISS index rebuilt nightly.",
])

pdf.q_box("Q: How did you guarantee dataset quality -- how do you know pairs are actually related?")
pdf.numbered([
    "Temporal proximity: Only news published in 48h before a tweet. Trump responds quickly to the news cycle.",
    "Semantic threshold: cosine >= 0.20 via all-MiniLM-L6-v2. Average kept similarity 0.339 vs ~0.05 random = 7x signal-to-noise ratio.",
    "Minimum coverage: Required >= 3 qualifying headlines per tweet to ensure a genuine news cluster response.",
])

pdf.q_box("Q: What would you do differently with more time/compute?")
pdf.bullet([
    "Train 2-3 epochs (stopped at 1 due to M5 time constraint)",
    "Increase LoRA rank from 16 to 32-64 for more model capacity",
    "Try DPO (Direct Preference Optimization) -- train on human preferences for stylistic authenticity",
    "Human evaluation: A/B test generated vs real tweets with political scientists",
    "Style metrics: capitalization rate, exclamation density, all-caps token ratio (better than ROUGE-2 for this task)",
    "Hybrid model: RAG retrieval followed by fine-tuned model generation",
])

pdf.h2("5.2  Behavioral / STAR Answers")
pdf.q_box("Q: Tell me about a time you debugged a difficult problem.")
pdf.callout(
    "Situation: Fine-tuned Llama repeated the system prompt instead of generating tweets.\n"
    "Task: Diagnose why a 17-hour training run produced useless outputs.\n"
    "Action: Inspected outputs character-by-character. Recognized the exact system prompt text. "
    "Traced to loss computation -- the model was optimizing to predict prompt tokens too. "
    "Researched TRL 1.0 API. Found completion_only_loss as the correct fix. Retrained.\n"
    "Result: Token accuracy went from artificially inflated to honest 45.4% on tweet tokens only. "
    "All subsequent outputs were grounded and stylistically coherent.",
    bg=Q_BG, border=BLUE, label="STAR -- Prompt Leakage Bug:")

pdf.q_box("Q: Tell me about handling unexpected technical challenges.")
pdf.callout(
    "Situation: 17-hour training run. Log appeared frozen AND Mac was sleeping.\n"
    "Task: Determine if training had crashed or was still running.\n"
    "Action: Ran 'lsof -p {pid}' to verify the process was still writing to the log. "
    "Checked CPU/memory to confirm GPU activity. Identified Python output buffering as a separate issue from the sleep. "
    "Armed 'caffeinate -i' to prevent future sleep events.\n"
    "Result: Both issues diagnosed and resolved without restarting the 17-hour training run. "
    "Training completed successfully.",
    bg=Q_BG, border=BLUE, label="STAR -- Sleep + Log Buffering Double Failure:")

# ── 6. Numbers ─────────────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("6.  Numbers to Memorize -- Say These Without Hesitation")
pdf.table(
    ["Fact", "Value"],
    [
        ["Total posts (Twitter + Truth Social)", "86,170"],
        ["Training pairs after semantic matching", "65,733"],
        ["Validation set size", "6,574"],
        ["News articles (NYT + Guardian)", "~1.3 million"],
        ["FAISS index size", "58,229 vectors"],
        ["Embedding dimension (all-MiniLM-L6-v2)", "384"],
        ["Llama 3 8B total parameters", "8 billion"],
        ["LoRA rank r", "16"],
        ["LoRA alpha", "32"],
        ["Trainable parameters", "16M  (0.2% of 8B)"],
        ["Training examples used", "25,000"],
        ["Effective batch size", "32  (batch 2 x grad_accum 16)"],
        ["Learning rate", "2e-4 with cosine decay"],
        ["Warmup steps", "40"],
        ["Training duration", "~17 hours, 782 optimizer steps"],
        ["Step time on M5 MPS", "~50 seconds/step"],
        ["Final validation loss", "2.866"],
        ["Final token accuracy (tweet tokens only)", "45.4%"],
        ["ROUGE-1 Fine-tune (avg 5 examples)", "0.154"],
        ["ROUGE-1 RAG (avg 5 examples)", "0.153"],
        ["Max ROUGE-2 (any model)", "0.012"],
        ["Semantic similarity threshold", ">= 0.20"],
        ["Average kept pair similarity", "0.339"],
        ["Number of evaluation examples", "5"],
    ],
    col_widths=[112, 64])

# ── 7. Vocabulary ──────────────────────────────────────────────────────────────
pdf.h1("7.  Vocabulary Cheat Sheet -- One-Sentence Definitions")
terms = [
    ("LoRA", "Parameter-efficient fine-tuning that injects low-rank adapter matrices (B*A) into frozen weights, training only 0.2% of parameters."),
    ("SFT", "Supervised Fine-Tuning -- training a model on labeled (input, output) pairs with cross-entropy loss."),
    ("Completion-only loss", "Loss masking that applies cross-entropy only to the assistant's response tokens, not to the prompt or system message."),
    ("RAG", "Retrieval-Augmented Generation -- augmenting an LLM with documents retrieved at inference time to provide factual grounding without fine-tuning."),
    ("FAISS", "Facebook AI Similarity Search -- library for fast exact and approximate nearest-neighbor search over dense vectors."),
    ("Causal LM", "A language model with left-to-right attention masking that can only attend to previous tokens -- the standard autoregressive generation architecture."),
    ("BPE", "Byte-Pair Encoding -- tokenization that merges frequent byte pairs iteratively to build a vocabulary of subword units."),
    ("bfloat16", "16-bit float with float32's exponent range but reduced mantissa; avoids gradient overflow on Apple MPS."),
    ("GQA", "Grouped Query Attention -- Llama 3's attention variant where multiple query heads share key/value heads to reduce memory bandwidth."),
    ("KV cache", "Cache of past key/value attention tensors that avoids recomputing them at each generation step."),
    ("Perplexity", "e^loss -- geometric mean of inverse probability per token; measures how surprised the model is by held-out data."),
    ("BLEU", "Precision-based n-gram overlap metric with brevity penalty; measures how much of generated text appears in the reference."),
    ("ROUGE-L", "Recall-based metric using longest common subsequence; sensitive to word ordering unlike ROUGE-1/2."),
    ("Prompt leakage", "Fine-tuning artifact where the model learns to output the system prompt verbatim because prompt tokens appear in the loss computation."),
    ("Semantic matching", "Using sentence embeddings and cosine similarity to pair documents by meaning rather than exact keyword overlap."),
]
pdf.set_font("Arial", size=9.5)
for term, defn in terms:
    pdf.set_font("ArialBold", size=9.5)
    pdf.set_text_color(*BLUE)
    pdf.cell(44, 5.5, term + ":", new_x=XPos.END, new_y=YPos.TOP)
    pdf.set_font("Arial", size=9.5)
    pdf.set_text_color(*DKGRAY)
    pdf.multi_cell(0, 5.5, defn, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(0.5)

# ── 8. Full Story ──────────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("8.  The Full Story -- Walk Me Through the Project")
pdf.body("Use this for long-form behavioral interviews.")
pdf.numbered([
    "Idea: Framed tweet prediction as a news-conditioned generation task. Can an LLM learn someone's reaction style from news headlines?",
    "Data collection: Scraped 86K Trump posts from two platforms. Hit NYT Archive API month-by-month (rate-limited, wrote incremental backfill jobs). 1.3M articles after weeks of incremental fetching.",
    "Semantic matching: Embedded news headlines and tweets with all-MiniLM-L6-v2, computed cosine similarity, filtered to >= 0.20. 65,733 training pairs with verified semantic grounding.",
    "GPT-2 baseline: Fine-tuned GPT-2 first (simpler, faster). Showed the task was learnable (val loss 2.18, perplexity 8.87). Deleted after validating the pipeline -- not in final paper.",
    "Llama v1 (failure): Fine-tuned Llama 3 8B without prompt masking. Outputs repeated the system prompt. Did not immediately recognize the root cause.",
    "Llama v2 (failure): Tried DataCollatorForCompletionOnlyLM -- removed in TRL 1.0. Tried assistant_only_loss -- failed because Llama 3's chat template lacks {% generation %}.",
    "Root cause diagnosis: Loss on prompt tokens causes the model to learn to predict them. Found completion_only_loss in TRL 1.0.0 SFTConfig using prompt/completion dict format.",
    "Llama v3 (success): 17-hour training run. Two Mac sleep interruptions at steps 672 and 725. Both recovered. Used nohup caffeinate -i & going forward. Final: loss 2.866, accuracy 45.4%.",
    "RAG pipeline: Built FAISS IndexFlatIP over all 65,733 pairs. Top-5 retrieval + base Llama 3 8B generation. No training needed.",
    "Evaluation: 5 real Trump posts from April 2026 as held-out examples. All 4 models on each. Scored BLEU+ROUGE. Generated 4 publication-quality figures.",
    "Paper: Full LaTeX academic paper (~3,500 words, 4 figures, 4 tables, 14 citations). Key finding: fine-tune and RAG tie on average but with complementary strengths. Targeting arXiv.",
])

out = "/Users/rishabh/Downloads/trump_tweets_overleaf/interview_prep.pdf"
pdf.output(out)
print(f"Saved: {out}")
print(f"Size: {os.path.getsize(out) // 1024} KB")
