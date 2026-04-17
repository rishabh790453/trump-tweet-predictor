"""
evaluate_all.py
===============
Full evaluation pipeline:
  - Run all 4 models on 5 real Trump tweet examples
  - Score with BLEU / ROUGE / BERTScore
  - Generate all paper figures
"""

import json, re, random, warnings, os
warnings.filterwarnings("ignore")
random.seed(42)

import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("punkt",     quiet=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_lib

# ── 5 evaluation examples ─────────────────────────────────────────────────────
EXAMPLES = [
    {
        "id": "nyc_tax",
        "label": "NYC Luxury Tax",
        "headlines": [
            "Fox News: 'We're taxing the rich' — NYC Mayor Mamdani touts new $500M-a-year tax on luxury second homes",
            "New York City Mayor proposes half-billion dollar annual tax on high-end second properties",
            "Democrats push wealth tax as NYC housing crisis deepens",
        ],
        "ground_truth": (
            "Sadly, Mayor Mamdani is DESTROYING New York! It has no chance! "
            "The United States of America should not contribute to its failure. "
            "It will only get WORSE. The TAX, TAX, TAX Policies are SO WRONG. "
            "People are fleeing. They must change their ways, AND FAST. "
            'History has proven, THIS "STUFF" JUST DOESN\'T WORK. '
            "Thank you for your attention to this matter! President DJT"
        ),
    },
    {
        "id": "lebanon_ceasefire",
        "label": "Lebanon–Israel Ceasefire",
        "headlines": [
            "Israel and Lebanon hold first direct talks in 34 years in Washington",
            "Netanyahu and Lebanese President Aoun meet with Rubio at State Department",
            "US brokers deal as Israel-Lebanon border tensions escalate",
            "Hezbollah warns of retaliation as IDF operations continue in south Lebanon",
        ],
        "ground_truth": (
            "I just had excellent conversations with the Highly Respected President "
            "Joseph Aoun, of Lebanon, and Prime Minister Bibi Netanyahu, of Israel. "
            "These two Leaders have agreed that in order to achieve PEACE between "
            "their Countries, they will formally begin a 10 Day CEASEFIRE at 5 P.M. EST. "
            "I have directed Vice President JD Vance and Secretary of State Rubio, "
            "together with the Chairman of the Joint Chiefs of Staff, Dan Razin' Caine, "
            "to work with Israel and Lebanon to achieve a Lasting PEACE. "
            "It has been my Honor to solve 9 Wars across the World, and this will be "
            "my 10th, so let's, GET IT DONE! President DONALD J. TRUMP"
        ),
    },
    {
        "id": "hormuz_china",
        "label": "Strait of Hormuz / China",
        "headlines": [
            "US military blockades Strait of Hormuz amid Iran nuclear standoff",
            "China urges restraint as Trump threatens to close Persian Gulf shipping lanes",
            "Iran oil exports at risk as Trump administration tightens sanctions enforcement",
            "Beijing and Washington reach back-channel agreement on Iran weapons shipments",
        ],
        "ground_truth": (
            "China is very happy that I am permanently opening the Strait of Hormuz. "
            "I am doing it for them, also — And the World. This situation will never happen again. "
            "They have agreed not to send weapons to Iran. President Xi will give me a big, fat, hug "
            "when I get there in a few weeks. We are working together smartly, and very well! "
            "Doesn't that beat fighting??? BUT REMEMBER, we are very good at fighting, if we have to "
            "— far better than anyone else!!! President DJT"
        ),
    },
    {
        "id": "ballroom_judge",
        "label": "White House Ballroom / Judge",
        "headlines": [
            "Federal judge blocks construction of White House ballroom addition, citing permit violations",
            "DC District Court halts Trump's $400 million White House renovation project",
            "Citizen lawsuit challenges White House ballroom construction funded by private donors",
            "Trump White House ballroom project faces legal challenge from DC judge",
        ],
        "ground_truth": (
            "A Trump Hating Judge, for the first time in History, wants Congress to pay "
            "Hundreds of Millions of Dollars for a Glorious Ballroom, instead of accepting "
            "Donations from Great American Companies and Citizens. This is a first — In other words, "
            "he wants Tax Payers to pay for the Ballroom, instead of Donors and Patriots! "
            "The Ballroom is FREE to our Country, A GIFT, and vital for our National Security. "
            "This Judge, who works for another Judge who was just MANDAMUSED for the unfair and "
            "biased way he treats me, should be ashamed of himself! President DONALD J. TRUMP"
        ),
    },
    {
        "id": "no_tax_tips",
        "label": "No Tax on Tips",
        "headlines": [
            "Trump heads to Las Vegas to rally support for no-tax-on-tips legislation",
            "Senate Republicans advance bill to exempt tip income from federal taxes",
            "Service workers cheer as Trump pushes no-tax-on-tips ahead of midterms",
            "Las Vegas hospitality industry embraces Trump's tip tax elimination plan",
        ],
        "ground_truth": (
            "I'm on Air Force One heading to Las Vegas and Arizona for Greetings and Speeches "
            "on NO TAX ON TIPS, a WINDFALL for our Great American Citizens. I am watching one of "
            "the Least Attractive and Talented People on all of Television, Jessica Tarlov. "
            "Her voice is so grating and terrible, I had to 'turn her off!' Her Democrat soundbites "
            "are FAKE. She makes up 'Poll Numbers,' and nobody challenges her, because she is so boring. "
            "I have among the best Poll Numbers I have ever had, and why shouldn't I, "
            "ALL THE COUNTRY DOES IS WIN. GET HER OFF THE AIR, SHE IS BAD FOR OUR COUNTRY! "
            "President DJT"
        ),
    },
]


# ── Utility: clean generated text ─────────────────────────────────────────────
_URL_RE = re.compile(r'https?://\S+|pic\.twitter\.com/\S+|www\.\S+', re.I)

def clean(text):
    text = _URL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"!{4,}", "!!!", text)
    text = re.sub(r"\?{4,}", "???", text)
    if len(text) > 280:
        text = text[:280].rsplit(" ", 1)[0]
    return text


# ── Markov ────────────────────────────────────────────────────────────────────
def run_markov(headlines, n=5):
    from markov_predictor import MarkovTweetGenerator
    g = MarkovTweetGenerator(order=2)
    # seed with keywords from headlines
    words = re.findall(r"[A-Z][a-z]+|[A-Z]{3,}", " ".join(headlines))[:6]
    keywords = words if words else ["The", "People", "Country"]
    tweets = g.generate_with_topic(keywords, num_tweets=n)
    return [clean(t) for t in tweets]


# ── N-gram ────────────────────────────────────────────────────────────────────
def run_ngram(headlines, n=5):
    from ngram_predictor import NgramTweetGenerator
    g = NgramTweetGenerator(n=3)
    words = re.findall(r"[A-Z][a-z]+|[A-Z]{3,}", " ".join(headlines))[:6]
    keywords = words if words else ["America", "Great", "People"]
    tweets = g.generate_with_keywords(keywords, num_tweets=n)
    return [clean(t) for t in tweets]


# ── Fine-tuned Llama ──────────────────────────────────────────────────────────
_llama_model = None
_llama_tok   = None
_llama_dev   = None

SYSTEM_PROMPT = (
    "You are Donald Trump. Given today's news headlines, "
    "write exactly one tweet in Trump's authentic voice — punchy, opinionated, "
    "capitalized for emphasis, sometimes using exclamation marks. "
    "React directly to the news. Keep it under 280 characters."
)

def load_llama():
    global _llama_model, _llama_tok, _llama_dev
    if _llama_model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _llama_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("  Loading fine-tuned Llama v3...")
    _llama_tok   = AutoTokenizer.from_pretrained("trump_llama_v3")
    _llama_model = AutoModelForCausalLM.from_pretrained(
        "trump_llama_v3", dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(_llama_dev)
    _llama_model.eval()

def run_finetune(headlines, n=5):
    import torch
    load_llama()
    hl = "\n".join(f"- {h}" for h in headlines)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Today's news headlines:\n{hl}"},
    ]
    ids = _llama_tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to(_llama_dev)
    mask = torch.ones_like(ids)
    plen = ids.shape[1]
    terminators = [_llama_tok.eos_token_id,
                   _llama_tok.convert_tokens_to_ids("<|eot_id|>")]
    tweets = []
    with torch.no_grad():
        for _ in range(n):
            out = _llama_model.generate(
                ids, attention_mask=mask, max_new_tokens=150,
                temperature=0.85, top_p=0.92, top_k=50, do_sample=True,
                repetition_penalty=1.2, eos_token_id=terminators,
                pad_token_id=_llama_tok.eos_token_id,
            )
            t = _llama_tok.decode(out[0][plen:], skip_special_tokens=True).strip()
            tweets.append(clean(t))
    return tweets


# ── RAG ───────────────────────────────────────────────────────────────────────
_rag_model  = None
_rag_tok    = None
_rag_dev    = None
_rag_index  = None
_rag_data   = None
_rag_embed  = None

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def load_rag():
    global _rag_model, _rag_tok, _rag_dev, _rag_index, _rag_data, _rag_embed
    if _rag_model is not None:
        return
    import torch, faiss
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    _rag_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("  Loading RAG components...")
    _rag_embed = SentenceTransformer("all-MiniLM-L6-v2")
    _rag_index = faiss.read_index("rag_index.faiss")
    _rag_data  = [json.loads(l) for l in open("rag_data.jsonl")]
    model_id   = "meta-llama/Meta-Llama-3-8B-Instruct"
    _rag_tok   = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    _rag_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True, token=HF_TOKEN
    ).to(_rag_dev)
    _rag_model.eval()

def run_rag(headlines, n=5, k=5):
    import torch, numpy as np
    load_rag()
    query    = " ".join(headlines)
    qvec     = _rag_embed.encode([query], normalize_embeddings=True).astype("float32")
    _, idxs  = _rag_index.search(qvec, k)
    examples = [_rag_data[i] for i in idxs[0] if i < len(_rag_data)]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user",      "content": f"Today's news headlines:\n{ex['news']}"})
        messages.append({"role": "assistant", "content": ex["tweet"]})
    hl = "\n".join(f"- {h}" for h in headlines)
    messages.append({"role": "user", "content": f"Today's news headlines:\n{hl}"})

    ids = _rag_tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids  = ids.to(_rag_dev)
    mask = torch.ones_like(ids)
    plen = ids.shape[1]
    terminators = [_rag_tok.eos_token_id,
                   _rag_tok.convert_tokens_to_ids("<|eot_id|>")]
    tweets = []
    with torch.no_grad():
        for _ in range(n):
            out = _rag_model.generate(
                ids, attention_mask=mask, max_new_tokens=150,
                temperature=0.85, top_p=0.92, top_k=50, do_sample=True,
                repetition_penalty=1.2, eos_token_id=terminators,
                pad_token_id=_rag_tok.eos_token_id,
            )
            t = _rag_tok.decode(out[0][plen:], skip_special_tokens=True).strip()
            tweets.append(clean(t))
    return tweets


# ── Scoring ───────────────────────────────────────────────────────────────────
smooth = SmoothingFunction().method1
rscorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def score(reference, candidates):
    ref_tok = nltk.word_tokenize(reference.lower())
    b1s, b2s, r1s, r2s, rls = [], [], [], [], []
    for cand in candidates:
        hyp = nltk.word_tokenize(cand.lower())
        b1s.append(sentence_bleu([ref_tok], hyp, weights=(1,0,0,0), smoothing_function=smooth))
        b2s.append(sentence_bleu([ref_tok], hyp, weights=(.5,.5,0,0), smoothing_function=smooth))
        rs = rscorer.score(reference, cand)
        r1s.append(rs["rouge1"].fmeasure)
        r2s.append(rs["rouge2"].fmeasure)
        rls.append(rs["rougeL"].fmeasure)
    avg = lambda x: sum(x) / len(x)
    return dict(bleu1=avg(b1s), bleu2=avg(b2s), rouge1=avg(r1s), rouge2=avg(r2s), rougeL=avg(rls))


# ── Training curve parsing ────────────────────────────────────────────────────
def parse_training_log(path="finetune_v3.log"):
    steps, losses, accs = [], [], []
    eval_steps, eval_losses, eval_accs = [], [], []
    raw = open(path, errors="ignore").read()
    lines = raw.replace("\r", "\n").split("\n")
    for line in lines:
        # train log lines: {'loss': '3.45', ..., 'epoch': '0.32'}
        m = re.search(r"'loss':\s*'([0-9.]+)'.*?'mean_token_accuracy':\s*'([0-9.]+)'.*?'epoch':\s*'([0-9.]+)'", line)
        if m:
            steps.append(float(m.group(3)))
            losses.append(float(m.group(1)))
            accs.append(float(m.group(2)))
        # eval log lines
        m2 = re.search(r"'eval_loss':\s*'([0-9.]+)'.*?'eval_mean_token_accuracy':\s*'([0-9.]+)'.*?'epoch':\s*'([0-9.]+)'", line)
        if m2:
            eval_steps.append(float(m2.group(3)))
            eval_losses.append(float(m2.group(1)))
            eval_accs.append(float(m2.group(2)))
    return steps, losses, accs, eval_steps, eval_losses, eval_accs


# ── Figure generation ─────────────────────────────────────────────────────────
def make_figures(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    os.makedirs("figures", exist_ok=True)
    MODEL_NAMES  = ["Markov", "N-gram", "Fine-tuned\nLlama 3", "RAG"]
    COLORS       = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    EXAMPLE_LABELS = [ex["label"] for ex in EXAMPLES]
    METRICS      = ["bleu1", "bleu2", "rouge1", "rouge2", "rougeL"]
    METRIC_LABELS= ["BLEU-1", "BLEU-2", "ROUGE-1", "ROUGE-2", "ROUGE-L"]

    # ── Fig 1: Grouped bar chart per metric (averaged across examples) ──────
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric, mlabel in zip(axes, METRICS, METRIC_LABELS):
        scores_per_model = []
        for model in ["markov", "ngram", "finetune", "rag"]:
            vals = [all_results[ex["id"]][model][metric] for ex in EXAMPLES]
            scores_per_model.append(np.mean(vals))
        bars = ax.bar(MODEL_NAMES, scores_per_model, color=COLORS, edgecolor="white", width=0.6)
        ax.set_title(mlabel, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(scores_per_model) * 1.35 + 0.01)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylabel("Score" if ax == axes[0] else "", fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, scores_per_model):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Automatic Evaluation — Averaged across 5 Examples", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/eval_metrics_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/eval_metrics_bar.png")

    # ── Fig 2: Per-example ROUGE-1 heatmap ──────────────────────────────────
    model_keys  = ["markov", "ngram", "finetune", "rag"]
    data = np.array([
        [all_results[ex["id"]][m]["rouge1"] for m in model_keys]
        for ex in EXAMPLES
    ])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=0.35)
    ax.set_xticks(range(4)); ax.set_xticklabels(MODEL_NAMES, fontsize=10)
    ax.set_yticks(range(5)); ax.set_yticklabels(EXAMPLE_LABELS, fontsize=10)
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    color="black" if data[i,j] < 0.2 else "white", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, label="ROUGE-1 F1")
    ax.set_title("ROUGE-1 per Example × Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/rouge1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/rouge1_heatmap.png")

    # ── Fig 3: Training + validation loss curve ──────────────────────────────
    steps, losses, accs, e_steps, e_losses, e_accs = parse_training_log()
    if steps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(steps, losses, color="#1f77b4", lw=1.8, label="Train loss")
        if e_steps:
            ax1.plot(e_steps, e_losses, "o--", color="#d62728", lw=1.8, ms=7, label="Val loss")
        ax1.set_xlabel("Epoch", fontsize=11); ax1.set_ylabel("Loss (completion-only)", fontsize=11)
        ax1.set_title("Training & Validation Loss — Llama 3 v3", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10); ax1.spines[["top","right"]].set_visible(False)
        ax1.set_xlim(0, 1)

        ax2.plot(steps, accs, color="#1f77b4", lw=1.8, label="Train accuracy")
        if e_steps:
            ax2.plot(e_steps, e_accs, "o--", color="#d62728", lw=1.8, ms=7, label="Val accuracy")
        ax2.set_xlabel("Epoch", fontsize=11); ax2.set_ylabel("Token Accuracy (tweet tokens)", fontsize=11)
        ax2.set_title("Token Accuracy — Llama 3 v3", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10); ax2.spines[["top","right"]].set_visible(False)
        ax2.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig("figures/training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved figures/training_curves.png")
    else:
        print("  WARNING: No training log data found for curves")

    # ── Fig 4: Radar chart per model (avg across examples) ──────────────────
    cats  = METRIC_LABELS
    N     = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for model_key, mname, color in zip(model_keys, MODEL_NAMES, COLORS):
        vals = [np.mean([all_results[ex["id"]][model_key][m] for ex in EXAMPLES]) for m in METRICS]
        # normalize to 0-1 relative scale for radar
        max_vals = [max(np.mean([all_results[ex["id"]][mk][m] for ex in EXAMPLES])
                        for mk in model_keys) for m in METRICS]
        norm_vals = [v / mv if mv > 0 else 0 for v, mv in zip(vals, max_vals)]
        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals, lw=2, color=color, label=mname.replace("\n", " "))
        ax.fill(angles, norm_vals, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=11)
    ax.set_yticklabels([]); ax.set_title("Model Comparison (Normalized)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved figures/radar_chart.png")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Model':<18} {'BLEU-1':>7} {'BLEU-2':>7} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}")
    print("-"*70)
    for model_key, mname in zip(model_keys, ["Markov","N-gram","Fine-tuned Llama 3","RAG"]):
        avgs = {m: round(sum(all_results[ex["id"]][model_key][m] for ex in EXAMPLES)/len(EXAMPLES), 3)
                for m in METRICS}
        print(f"{mname:<18} {avgs['bleu1']:>7.3f} {avgs['bleu2']:>7.3f} "
              f"{avgs['rouge1']:>8.3f} {avgs['rouge2']:>8.3f} {avgs['rougeL']:>8.3f}")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = {}

    print("Loading statistical models...")
    # Pre-load both to share the CSV read
    from markov_predictor import MarkovTweetGenerator
    from ngram_predictor  import NgramTweetGenerator
    markov_g = MarkovTweetGenerator(order=2)
    ngram_g  = NgramTweetGenerator(n=3)

    for ex in EXAMPLES:
        print(f"\n{'='*60}")
        print(f"Example: {ex['label']}")
        print(f"Ground truth: {ex['ground_truth'][:80]}...")

        words = re.findall(r"[A-Z][a-z]+|[A-Z]{3,}", " ".join(ex["headlines"]))[:6]
        kw    = words if words else ["America", "People", "Country"]

        markov_out   = [clean(t) for t in markov_g.generate_with_topic(kw, num_tweets=5)]
        ngram_out    = [clean(t) for t in ngram_g.generate_with_keywords(kw, num_tweets=5)]
        finetune_out = run_finetune(ex["headlines"], n=5)
        rag_out      = run_rag(ex["headlines"], n=5)

        all_results[ex["id"]] = {
            "markov":   score(ex["ground_truth"], markov_out),
            "ngram":    score(ex["ground_truth"], ngram_out),
            "finetune": score(ex["ground_truth"], finetune_out),
            "rag":      score(ex["ground_truth"], rag_out),
            "outputs":  {"markov": markov_out, "ngram": ngram_out,
                         "finetune": finetune_out, "rag": rag_out},
        }

        for model in ["markov","ngram","finetune","rag"]:
            s = all_results[ex["id"]][model]
            print(f"  {model:<10} R1={s['rouge1']:.3f}  B1={s['bleu1']:.3f}")

    print("\nGenerating figures...")
    make_figures(all_results)

    with open("evaluation_results.json", "w") as f:
        json.dump({k: {m: v for m, v in v.items() if m != "outputs"}
                   for k, v in all_results.items()}, f, indent=2)
    print("\nDone. Results saved to evaluation_results.json")
