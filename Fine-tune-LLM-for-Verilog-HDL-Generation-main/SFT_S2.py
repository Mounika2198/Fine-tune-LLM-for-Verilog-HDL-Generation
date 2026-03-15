import os
import json
import random

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel
from trl import SFTTrainer

# -----------------------------
# 1) BASIC CONFIG
# -----------------------------
# Use the SAME base model as Stage 1
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"

# Path to your Stage 1 adapter
STAGE1_ADAPTER = r"d:\ECE465Project\golden_qlora" 
#r"d:\ECE465Project\qlora_adapter_E3"

# Golden / unique dataset you built
GOLDEN_JSONL = r"d:\ECE465Project\S2_cleam_verilog.jsonl"
#r"d:\ECE465Project\golden_verilog_dataset.jsonl"

# Where to save the Stage 2 adapter
OUTPUT_DIR = r"d:\ECE465Project\S2_clean"

MAX_STAGE2_SAMPLES = 1000      # randomly select up to 1000
VAL_RATIO = 0.10               # 10% validation
MAX_SEQ_LENGTH = 512
SEED = 42

# Polishing training settings (smaller than Stage 1)
NUM_EPOCHS = 2
LR = 5e-5
BATCH_SIZE = 1
GRAD_ACCUM = 8


# -----------------------------
# 2) LOAD & SUBSAMPLE GOLDEN DATASET
# -----------------------------
def load_golden_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            task = obj.get("task", "").strip()
            code = obj.get("golden_verilog", "").strip()
            if not task or not code:
                continue

            text = f"Instruction:\n{task}\n\nResponse:\n{code}"
            records.append({"text": text})
    return records


print("Loading golden dataset from JSONL...")
all_records = load_golden_jsonl(GOLDEN_JSONL)
print("Total golden samples available:", len(all_records))

# Shuffle and take up to MAX_STAGE2_SAMPLES
random.seed(SEED)
random.shuffle(all_records)
subset = all_records[:MAX_STAGE2_SAMPLES]
print("Using Stage-2 subset size:", len(subset))

# Build HF Dataset
full_ds = Dataset.from_list(subset)

# Train/val split
split = full_ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
train_ds = split["train"]
val_ds   = split["test"]

print(f"Stage-2 Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")


# -----------------------------
# 3) LOAD TOKENIZER & 4-BIT MODEL (SAME AS STAGE 1)
# -----------------------------
print("Loading tokenizer and base model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",        # same as Stage 1
    trust_remote_code=True,
)
print(">>> Base model device:", next(base_model.parameters()).device)

# -----------------------------
# 4) ATTACH STAGE-1 LORA ADAPTER (MAKE IT TRAINABLE)
# -----------------------------
print("Loading Stage-1 LoRA adapter from:", STAGE1_ADAPTER)
model = PeftModel.from_pretrained(
    base_model,
    STAGE1_ADAPTER,
    is_trainable=True,    # <-- important
)

# Sanity-check trainable params (should NOT be 0)
trainable, total = 0, 0
for name, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(
    f"Trainable params (Stage 2): {trainable} || all params: {total} "
    f"|| trainable%: {100 * trainable / total:.4f}"
)


# -----------------------------
# 5) TRAINING ARGUMENTS (MATCH STYLE OF STAGE 1)
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LR,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    logging_steps=5,
    eval_strategy="steps",      # same key name you used in Stage 1
    eval_steps=20,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),

    dataloader_pin_memory=True,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


# -----------------------------
# 6) SFTTrainer WITH EXISTING PEFT MODEL
# -----------------------------
print("Setting up SFTTrainer for Stage-2...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    formatting_func=lambda example: example["text"],
)


# -----------------------------
# 7) TRAIN + PRINT LOSS HISTORY
# -----------------------------
print("Starting Stage-2 training...")
train_result = trainer.train()

print("\n=== LOSS HISTORY (train & eval) ===")
for log in trainer.state.log_history:
    if "loss" in log or "eval_loss" in log:
        print(log)

# -----------------------------
# 8) SAVE STAGE-2 ADAPTER
# -----------------------------
print("Saving Stage-2 QLoRA adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done! Stage-2 adapter saved at:", OUTPUT_DIR)



