"""
finetune_llama.py
=================
Fine-tune Mistral 7B or Llama 3 8B with LoRA on Apple MPS (M-series chips).
No quantization needed — 32GB unified memory handles FP16 comfortably.

Models (pick one):
  mistralai/Mistral-7B-Instruct-v0.3   ← default, fully open, no token needed
  meta-llama/Meta-Llama-3-8B-Instruct  ← better, needs HF token + Meta license

Usage:
  python finetune_llama.py                            # Mistral 7B, default settings
  python finetune_llama.py --model llama3             # Llama 3 8B (needs HF_TOKEN)
  python finetune_llama.py --epochs 2 --max 20000     # faster run
  python finetune_llama.py --rank 32                  # higher LoRA rank = more capacity

Output: trump_llama_model/   (drop-in replacement for trump_gpt2_model)
"""

import argparse
import json
import re
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3":  "meta-llama/Meta-Llama-3-8B-Instruct",
}

LORA_TARGET_MODULES = {
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
    "llama3":  ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"],
}

SYSTEM_PROMPT = (
    "You are Donald Trump. Given today's news headlines, "
    "write exactly one tweet in Trump's authentic voice — punchy, opinionated, "
    "capitalized for emphasis, sometimes using exclamation marks. "
    "React directly to the news. Keep it under 280 characters."
)


# ── Data loading & formatting ─────────────────────────────────────────────────

_URL_RE = re.compile(r'https?://\S+|pic\.twitter\.com/\S+|www\.\S+', re.IGNORECASE)


def parse_raw_example(text: str):
    """Extract (headlines_str, tweet) from the <|news|>...<|tweet|>... format."""
    news_token  = "<|news|>"
    tweet_token = "<|tweet|>"
    eos_token   = "<|endoftext|>"

    if news_token not in text or tweet_token not in text:
        return None, None

    news_part  = text.split(news_token)[1].split(tweet_token)[0].strip()
    tweet_part = text.split(tweet_token)[1].replace(eos_token, "").strip()

    if not news_part or not tweet_part:
        return None, None

    # Filter: retweets (Twitter and TruthSocial styles)
    if re.match(r'^RT\s*[@:]', tweet_part, re.IGNORECASE):
        return None, None

    # Filter: reply tweets (start with @mention)
    if tweet_part.startswith("@"):
        return None, None

    # Strip all URLs from the tweet text, then validate what remains
    tweet_part = _URL_RE.sub("", tweet_part).strip()
    tweet_part = re.sub(r'\s+', ' ', tweet_part)  # collapse whitespace

    # Filter: too short or too long after stripping URLs
    if len(tweet_part) < 30 or len(tweet_part) > 400:
        return None, None

    return news_part, tweet_part


def format_for_chat(model_key: str, headlines_str: str, tweet: str) -> dict:
    """Return a prompt/completion example dict for completion-only loss.

    TRL tokenizes prompt and completion separately, then builds a completion_mask
    so loss is computed ONLY on the tweet (assistant turn), not on the system prompt
    or news headlines. This prevents the garbled/leaked-prompt outputs from v1.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Today's news headlines:\n{headlines_str}"},
        ],
        "completion": [
            {"role": "assistant", "content": tweet},
        ],
    }


def load_dataset_from_jsonl(path: str, model_key: str, max_examples: int):
    from datasets import Dataset

    examples = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(examples) >= max_examples:
                break
            try:
                obj = json.loads(line)
                raw = obj.get("text", "")
                headlines_str, tweet = parse_raw_example(raw)
                if headlines_str is None:
                    skipped += 1
                    continue
                examples.append(format_for_chat(model_key, headlines_str, tweet))
            except Exception:
                skipped += 1

    print(f"  Loaded {len(examples)} examples ({skipped} skipped)")
    return Dataset.from_list(examples)


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="mistral", choices=list(MODELS),
                        help="Model to fine-tune")
    parser.add_argument("--train",    default="finetune_train.jsonl")
    parser.add_argument("--val",      default="finetune_val.jsonl")
    parser.add_argument("--out",      default="trump_llama_model")
    parser.add_argument("--epochs",   type=int,   default=2)
    parser.add_argument("--max",      type=int,   default=50000,
                        help="Max training examples (use fewer to test quickly)")
    parser.add_argument("--rank",     type=int,   default=16,
                        help="LoRA rank — higher = more capacity, more memory")
    parser.add_argument("--lr",       type=float, default=2e-4)
    parser.add_argument("--batch",    type=int,   default=2)
    parser.add_argument("--grad-acc", type=int,   default=16)
    parser.add_argument("--max-len",  type=int,   default=256)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    model_id  = MODELS[args.model]
    device    = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype     = torch.bfloat16

    print("=" * 60)
    print(f"LoRA Fine-tune: {model_id}")
    print(f"Device: {device.upper()}  |  dtype: {dtype}")
    print("=" * 60)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = load_dataset_from_jsonl(args.train, args.model, args.max)
    val_ds   = load_dataset_from_jsonl(args.val,   args.model, max(500, args.max // 10))

    # ── Base model ────────────────────────────────────────────────────────────
    print(f"\nLoading base model ({model_id})...")
    print("  (First run downloads ~15GB — subsequent runs use cache)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.enable_input_require_grads()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params/1e9:.1f}B")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print(f"\nApplying LoRA (rank={args.rank})...")
    lora_cfg = LoraConfig(
        task_type        = TaskType.CAUSAL_LM,
        r                = args.rank,
        lora_alpha       = args.rank * 2,
        lora_dropout     = 0.05,
        target_modules   = LORA_TARGET_MODULES[args.model],
        bias             = "none",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M "
          f"({100*trainable/total:.2f}%)")

    # ── Training config ───────────────────────────────────────────────────────
    eff_batch = args.batch * args.grad_acc
    print(f"\nTraining config:")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Effective batch:{eff_batch}  ({args.batch} x {args.grad_acc} grad accum)")
    print(f"  LR:             {args.lr}")
    print(f"  Train examples: {len(train_ds)}")
    print(f"  Val examples:   {len(val_ds)}")

    training_args = SFTConfig(
        output_dir                  = args.out,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size  = args.batch,
        gradient_accumulation_steps = args.grad_acc,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 40,
        bf16                        = True,
        optim                       = "adamw_torch",
        logging_steps               = 50,
        eval_strategy               = "steps",
        eval_steps                  = 250,
        save_strategy               = "steps",
        save_steps                  = 250,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        report_to                   = "none",
        max_length                  = args.max_len,
        # Prompt/completion format — loss computed only on tweet tokens.
        # Prevents the garbled/leaked-prompt outputs from v1/v2.
        completion_only_loss        = True,
        dataloader_pin_memory       = False,   # required for MPS
        gradient_checkpointing      = True,    # trade compute for memory
    )

    trainer = SFTTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    trainer.train()

    # ── Save merged model ─────────────────────────────────────────────────────
    print(f"\nMerging LoRA weights and saving to {args.out}/...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Done! Model saved to {args.out}/")
    print(f"Next step: python generate_llama.py")


if __name__ == "__main__":
    main()
