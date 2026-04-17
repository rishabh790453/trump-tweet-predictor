# ── Kaggle: Fine-tune Llama 3 8B with QLoRA on Trump tweets ──────────────────
# GPU: T4 x2 (or P100), 4-bit quantization, ~12h session
# Upload finetune_train.jsonl + finetune_val.jsonl as a Kaggle dataset first.

import os, json, re
import numpy as np
import pandas as pd

# Show input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ── Install dependencies ───────────────────────────────────────────────────────
os.system("pip install -q transformers peft trl bitsandbytes accelerate datasets")

# ── Config ────────────────────────────────────────────────────────────────────
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_ID    = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set via: export HF_TOKEN=hf_...
TRAIN_PATH  = "/kaggle/input/datasets/rishabhg28/trumptweets/finetune_train.jsonl"
VAL_PATH    = "/kaggle/input/datasets/rishabhg28/trumptweets/finetune_val.jsonl"
OUT_DIR     = "/kaggle/working/trump_llama_model"
MAX_EXAMPLES = 6000    # 187 steps × ~138s/step (observed) = ~7.2h — safe within 12h
MAX_LEN      = 256
EPOCHS       = 1
LORA_RANK    = 16
LR           = 2e-4
BATCH        = 4
GRAD_ACC     = 8       # effective batch = 32

SYSTEM_PROMPT = (
    "You are Donald Trump. Given today's news headlines, "
    "write exactly one tweet in Trump's authentic voice — punchy, opinionated, "
    "capitalized for emphasis, sometimes using exclamation marks. "
    "React directly to the news. Keep it under 280 characters."
)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]

# ── Data loading ──────────────────────────────────────────────────────────────
NEWS_TOKEN  = "<|news|>"
TWEET_TOKEN = "<|tweet|>"
EOS_TOKEN   = "<|endoftext|>"

def parse_example(text):
    if NEWS_TOKEN not in text or TWEET_TOKEN not in text:
        return None, None
    news_part  = text.split(NEWS_TOKEN)[1].split(TWEET_TOKEN)[0].strip()
    tweet_part = text.split(TWEET_TOKEN)[1].replace(EOS_TOKEN, "").strip()
    if not news_part or not tweet_part:
        return None, None
    if tweet_part.startswith("RT @") or re.match(r"^https?://", tweet_part):
        return None, None
    if len(tweet_part) < 20 or len(tweet_part) > 600:
        return None, None
    return news_part, tweet_part

def load_dataset(path, tokenizer, max_examples):
    texts, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(texts) >= max_examples:
                break
            try:
                obj = json.loads(line)
                news, tweet = parse_example(obj.get("text", ""))
                if news is None:
                    skipped += 1
                    continue
                messages = [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": f"Today's news headlines:\n{news}"},
                    {"role": "assistant", "content": tweet},
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append({"text": formatted})
            except Exception:
                skipped += 1
    print(f"  Loaded {len(texts):,} examples ({skipped:,} skipped)")
    return Dataset.from_list(texts)

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("QLoRA Fine-tune: Llama 3 8B Instruct → Trump Tweet Predictor")
print("=" * 60)
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

# Tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Datasets
print("\nLoading datasets...")
train_ds = load_dataset(TRAIN_PATH, tokenizer, MAX_EXAMPLES)
val_ds   = load_dataset(VAL_PATH,   tokenizer, max(2000, MAX_EXAMPLES // 10))

# 4-bit quantization config (QLoRA)
print("\nSetting up 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit               = True,
    bnb_4bit_quant_type        = "nf4",
    bnb_4bit_compute_dtype     = torch.bfloat16,
    bnb_4bit_use_double_quant  = True,
)

# Load model in 4-bit
print(f"Loading {MODEL_ID} in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = HF_TOKEN,
    trust_remote_code   = True,
)
model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()

# LoRA
print(f"\nApplying LoRA (rank={LORA_RANK})...")
lora_cfg = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = LORA_RANK,
    lora_alpha     = LORA_RANK * 2,
    lora_dropout   = 0.05,
    target_modules = LORA_TARGETS,
    bias           = "none",
)
model = get_peft_model(model, lora_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)")

# Training
print(f"\nTraining config:")
print(f"  Epochs:          {EPOCHS}")
print(f"  Effective batch: {BATCH * GRAD_ACC}  ({BATCH} x {GRAD_ACC} grad accum)")
print(f"  LR:              {LR}")
print(f"  Train examples:  {len(train_ds):,}")
print(f"  Val examples:    {len(val_ds):,}")

training_args = SFTConfig(
    output_dir                  = OUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH,
    per_device_eval_batch_size  = BATCH,
    gradient_accumulation_steps = GRAD_ACC,
    learning_rate               = LR,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    bf16                        = True,
    optim                       = "paged_adamw_8bit",   # memory-efficient optimizer
    logging_steps               = 50,
    eval_strategy               = "steps",
    eval_steps                  = 200,
    save_strategy               = "steps",
    save_steps                  = 200,
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    report_to                   = "none",
    max_length                  = MAX_LEN,
    dataset_text_field          = "text",
)

trainer = SFTTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = train_ds,
    eval_dataset     = val_ds,
    processing_class = tokenizer,
)

print("\nStarting training...\n")
trainer.train()

# Save merged model
print(f"\nMerging LoRA weights and saving to {OUT_DIR}...")
merged = model.merge_and_unload()
merged.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"Done! Model saved to {OUT_DIR}")
print("Download the output from Kaggle → use with generate_llama.py")
