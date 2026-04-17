"""
finetune_gpt2.py
================
Fine-tune GPT-2 on (news headlines → Trump tweet) pairs.

Uses HuggingFace Transformers + datasets.
Works on CPU, CUDA GPU, or AMD GPU via DirectML (Windows).

Models you can use (set MODEL_NAME below):
  "gpt2"          — 124M params, fastest, decent quality
  "gpt2-medium"   — 355M params, better quality, needs more RAM
  "gpt2-large"    — 774M params, best GPT-2 quality, needs GPU

Install dependencies first:
  pip install torch transformers datasets accelerate

Run:
  python finetune_gpt2.py
  python finetune_gpt2.py --quick      # fast CPU/GPU test with distilgpt2
"""

import os
import json
import math
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "gpt2"          # base model to fine-tune
                                  # alternatives: "distilgpt2" (faster), "gpt2-medium" (better quality)
TRAIN_FILE      = "finetune_train.jsonl"
VAL_FILE        = "finetune_val.jsonl"
OUTPUT_DIR      = "trump_gpt2_model"
CHECKPOINT_DIR  = "trump_gpt2_checkpoints"

MAX_LENGTH      = 512     # max tokens per example
BATCH_SIZE      = 4       # reduce to 2 if OOM
GRAD_ACCUM      = 8       # effective batch = BATCH_SIZE * GRAD_ACCUM = 32
EPOCHS          = 3
LR              = 5e-5
WARMUP_STEPS    = 200
SAVE_STEPS      = 500
EVAL_STEPS      = 500
LOGGING_STEPS   = 50

# Quick/test mode — set via --quick flag or QUICK_MODE=True
# Uses distilgpt2 + 3000 examples to validate the full pipeline
QUICK_MODE      = False
QUICK_EXAMPLES  = 3000

# Special tokens (must match prepare_finetune_data.py)
NEWS_TOKEN  = "<|news|>"
TWEET_TOKEN = "<|tweet|>"


def load_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with __import__("torch").no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1
    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: distilgpt2 + 3000 examples")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name (e.g. distilgpt2, gpt2-medium)")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    quick      = args.quick or QUICK_MODE
    model_name = args.model or ("distilgpt2" if quick else MODEL_NAME)
    epochs     = args.epochs or (1 if quick else EPOCHS)
    batch_size = 2 if quick else BATCH_SIZE
    max_length = 256 if quick else MAX_LENGTH

    print("=" * 60)
    print("FINE-TUNE GPT-2: News → Trump Tweet Predictor")
    if quick:
        print("  [QUICK MODE — small subset to validate pipeline]")
    print("=" * 60)

    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR

    # ── Pick device ───────────────────────────────────────────────────────────
    print("\nChecking device...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Apple MPS (Metal GPU)")
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"  DirectML (AMD GPU): {torch_directml.device_name(0)}")
        except ImportError:
            device = torch.device("cpu")
            print("  CPU (no GPU found)")

    print(f"  torch: {torch.__version__}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    for f in [TRAIN_FILE, VAL_FILE]:
        if not Path(f).exists():
            print(f"\nMissing {f} — run prepare_finetune_data.py first")
            return

    print(f"\nLoading training data...")
    train_data = load_jsonl(TRAIN_FILE)
    val_data   = load_jsonl(VAL_FILE)

    if quick:
        import random
        random.seed(42)
        train_data = random.sample(train_data, min(QUICK_EXAMPLES, len(train_data)))
        val_data   = random.sample(val_data, min(300, len(val_data)))
        print(f"  [Quick mode] {len(train_data)} train / {len(val_data)} val")
    print(f"  Train: {len(train_data):,} examples")
    print(f"  Val:   {len(val_data):,} examples")

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    print(f"\nLoading {model_name} tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [NEWS_TOKEN, TWEET_TOKEN]})
    print(f"  Vocab size: {len(tokenizer):,}")

    # ── 3. Lazy Dataset (tokenize on-the-fly to avoid OOM) ───────────────────
    from torch.utils.data import Dataset

    class TweetDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.examples[idx]["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels         = input_ids.clone()
            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    print("\nBuilding datasets (lazy tokenization)...")
    train_dataset = TweetDataset(train_data)
    val_dataset   = TweetDataset(val_data)
    print(f"  Ready.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print(f"\nLoading {model_name} model...")
    from transformers import GPT2Config
    config = GPT2Config.from_pretrained(model_name)
    # DirectML doesn't support dropout backward — disable it so backprop stays on GPU
    if str(device) == "privateuseone":  # DirectML only
        config.attn_pdrop   = 0.0
        config.resid_pdrop  = 0.0
        config.embd_pdrop   = 0.0
        print("  Dropout disabled (DirectML compatibility)")
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.0f}M")

    # ── 5. Optimizer + scheduler ──────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(train_dataset) / (batch_size * GRAD_ACCUM))
    total_steps     = steps_per_epoch * epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\nTraining configuration:")
    print(f"  Model:           {model_name}")
    print(f"  Device:          {device}")
    print(f"  Train examples:  {len(train_data):,}")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {batch_size} x {GRAD_ACCUM} grad accum = {batch_size * GRAD_ACCUM} effective")
    print(f"  Max length:      {max_length} tokens")
    print(f"  Total steps:     {total_steps:,}")
    print(f"\nStarting training...\n")

    # ── 6. Training loop ──────────────────────────────────────────────────────
    global_step   = 0
    best_val_loss = float("inf")
    model.train()

    for epoch in range(epochs):
        epoch_loss  = 0.0
        accum_loss  = 0.0
        n_accum     = 0
        t0          = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss / GRAD_ACCUM
            loss.backward()

            accum_loss  += loss.item()   # syncs GPU→CPU here (intentional checkpoint)
            n_accum     += 1

            if n_accum == GRAD_ACCUM or batch_idx == len(train_loader) - 1:
                # Manual grad clipping — avoids _foreach_norm which is unsupported on DirectML
                total_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm_sq += p.grad.detach().norm(2).item() ** 2
                clip_coef = 1.0 / max(total_norm_sq ** 0.5, 1.0)
                if clip_coef < 1.0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.detach().mul_(clip_coef)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss  += accum_loss

                if global_step % LOGGING_STEPS == 0:
                    elapsed = time.time() - t0
                    print(f"  Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                          f"loss: {accum_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} | "
                          f"{elapsed:.0f}s elapsed")

                if global_step % EVAL_STEPS == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  >>> Val loss: {val_loss:.4f} | perplexity: {math.exp(val_loss):.2f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ckpt = f"{CHECKPOINT_DIR}/best"
                        model.save_pretrained(ckpt)
                        tokenizer.save_pretrained(ckpt)
                        print(f"  >>> Saved best checkpoint (val_loss={val_loss:.4f})")

                accum_loss = 0.0
                n_accum    = 0

        avg_epoch_loss = epoch_loss / max(global_step, 1) * GRAD_ACCUM
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1} done | avg loss: {avg_epoch_loss:.4f} | {elapsed/60:.1f} min\n")

    # ── 7. Final eval + save ──────────────────────────────────────────────────
    val_loss   = evaluate(model, val_loader, device)
    perplexity = math.exp(val_loss)
    print(f"\nFinal evaluation:")
    print(f"  Loss:       {val_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    print(f"\nSaving model to {OUTPUT_DIR}/...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done! Model saved to {OUTPUT_DIR}/")
    print("Next step: python generate_tweet.py")


def _make_dict_loader(loader):
    """Wrap a TensorDataset loader to yield dicts for evaluate()."""
    for input_ids, attention_mask, labels in loader:
        yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    main()
