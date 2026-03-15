
#%%

from huggingface_hub import login
login(token="Add the token",  # <- paste token here
    add_to_git_credential=False               # <- important so it avoids git
)


import os
import random

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# -----------------------------
# 1) BASIC CONFIG
# -----------------------------
# Base model (pick something you have access to)
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"  # or another instruct model
DATASET_ID = "balanced_50k_verilog.jsonl"
#"bnadimi/PyraNet-Verilog"              # your Verilog dataset
OUTPUT_DIR = r"d:\ECE465Project\golden_qlora"        # where to save QLoRA adapter

print("Loading dataset...")
raw_ds = load_dataset(
    "json",
    data_files= DATASET_ID,
    split="train",   # the JSONL is treated as a single split called 'train'
)
# raw_ds = load_dataset(DATASET_ID, split="train")

MAX_SAMPLES = 38617        # total samples (train + val)
#print("Total samples in dataset:", MAX_SAMPLES)

VAL_RATIO = 0.10          # 10% validation
MAX_SEQ_LENGTH = 512
SEED = 42

# -----------------------------
# 2) LOAD & SUBSAMPLE DATASET
# -----------------------------


# Convert to list and subsample
data_list = list(raw_ds)
random.seed(SEED)
random.shuffle(data_list)
subset = data_list[:MAX_SAMPLES]

val_size = int(MAX_SAMPLES * VAL_RATIO)
train_list = subset[:-val_size]
val_list   = subset[-val_size:]

print(f"Train samples: {len(train_list)}, Val samples: {len(val_list)}")

# -----------------------------
# 3) FORMAT DATA FOR INSTRUCTION TUNING
# -----------------------------
def format_example(example):
    # Extract natural-language description
    desc_block = example.get("task", {})
    if isinstance(desc_block, dict):
        instruction = desc_block.get("task", "").strip()
    else:
        instruction = str(desc_block).strip()

    if not instruction:
        return None  # skip bad examples

    # Extract raw Verilog code
    code = example.get("golden_verilog") or ""
    code = code.strip()

    # Skip junk (no module keyword → not useful!)
    if "module" not in code or "endmodule" not in code:
        return None

    # Clean the code (remove lines like 'endprogram', weird wrappers, chaff)
    cleaned = []
    for line in code.splitlines():
        if line.strip().startswith("endprogram"): continue
        if line.strip().startswith("program"): continue
        cleaned.append(line)
    code = "\n".join(cleaned)

    # Final template
    text = f"Instruction:\n{instruction}\n\nResponse:\n{code}"
    return {"text": text}


from datasets import Dataset

train_ds = Dataset.from_list(train_list).map(format_example)
train_ds = train_ds.filter(lambda x: x is not None)

val_ds = Dataset.from_list(val_list).map(format_example)
val_ds = val_ds.filter(lambda x: x is not None)

# -----------------------------
# 4) LOAD TOKENIZER & 4-BIT MODEL (QLoRA)
# -----------------------------
print("Loading tokenizer and model in 4-bit...")

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

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",        # spreads across available GPUs
    trust_remote_code=True,   # needed for some newer models
)
print(">>> Model device:", next(model.parameters()).device)
# -----------------------------
# 5) QLoRA CONFIG
# -----------------------------
# Target modules depend on model architecture; these are typical for Mistral/LLaMA
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# -----------------------------
# 6) TRAINING ARGUMENTS
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,     # <- was 2, start safe
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,     # keep overall effective batch size similar

    learning_rate=5e-4,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    logging_steps=5,
    eval_strategy="steps",
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
# 7) SFTTrainer WITH QLoRA
# -----------------------------

print("Setting up SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=lora_config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    formatting_func=lambda example: example["text"],  # <- per-sample formatter
)




# -----------------------------
# 8) TRAIN + PRINT LOSS HISTORY
# -----------------------------
print("Starting training...")
train_result = trainer.train()


print("\n=== LOSS HISTORY (train & eval) ===")
for log in trainer.state.log_history:
    if "loss" in log or "eval_loss" in log:
        print(log)

# -----------------------------
# 9) SAVE FINAL ADAPTER
# -----------------------------
print("Saving QLoRA adapter...")
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR))
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done!")




