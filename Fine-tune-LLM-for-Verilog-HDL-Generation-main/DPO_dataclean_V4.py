
import os
import json
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm

# ---------------- USER SETTINGS ----------------
HF_TOKEN = "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"     # <- your token
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"        # base model ID
DATASET_ID = "bnadimi/PyraNet-Verilog"                 # PyraNet dataset
SPLIT = "train"

# We want 600 RANDOM samples from the entire dataset
MAX_EXAMPLES = 600

OUTPUT_FILE = "dpo_dataset_clean.jsonl"
# v4 was with qlorastage2, golden dataset
# v5 with 25% AND/OR, 25% Adder/half adder, 25% mux, 25% decoder

# Path to your Stage-2 QLoRA adapter (SFT Stage 2 or whatever you call "stage2")
STAGE2_ADAPTER_DIR = r"d:\ECE465Project\golden_qlora"  # <-- put your local folder here

# ---------------- LOAD TOKENIZER & BASE MODEL -------------------
print("Loading tokenizer...")
base_tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN,
)

if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
base_model.eval()

# ---------------- LOAD STAGE-2 QLORA ADAPTER -------------------
print(f"Loading Stage-2 QLoRA adapter from: {STAGE2_ADAPTER_DIR}")
stage2_model = PeftModel.from_pretrained(
    base_model,
    STAGE2_ADAPTER_DIR,
)
stage2_model.eval()

# ---------------- LOAD DATASET (600 RANDOM EXAMPLES) -----------
print(f"Loading dataset: {DATASET_ID} ({SPLIT})")
dataset = load_dataset(DATASET_ID, split=SPLIT)

full_len = len(dataset)
print(f"Full dataset size: {full_len}")

if MAX_EXAMPLES is not None:
    if MAX_EXAMPLES > full_len:
        raise ValueError(f"Requested {MAX_EXAMPLES} examples, but dataset only has {full_len}.")
   
    # For reproducibility, you can fix the seed
    random.seed(42)

    # Sample 600 random indices from the whole dataset (no replacement)
    indices = random.sample(range(full_len), MAX_EXAMPLES)
    dataset = dataset.select(indices)
    print(f"Using {len(dataset)} RANDOM examples from PyraNet.")
else:
    print(f"Using ALL {full_len} examples (MAX_EXAMPLES=None).")

print("Columns:", dataset.column_names)  # should show ['code', 'description']

# --------- MAP YOUR COLUMNS → prompt/chosen --------------------
def get_prompt_and_chosen(example):
    # description → prompt
    # code        → chosen
    prompt = example["description"]
    chosen = example["code"]
    return prompt, chosen

# ------------- GENERATE REJECTED (USING STAGE-2 ADAPTER) -------
def generate_rejected(prompt, max_new_tokens=256):
    """
    Generate the 'rejected' response using the Stage-2 QLoRA adapter.
    """
    formatted_prompt = f"Instruction:\n{prompt}\n\nResponse:\n"

    model = stage2_model
    tokenizer = base_tokenizer

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Cut everything before "Response:" — same logic as your app
    if "Response:" in text:
        text = text.split("Response:", 1)[1].strip()

    return text


# ------------- BUILD DPO JSONL -----------------
print(f"Building DPO dataset → {OUTPUT_FILE}")

num_written = 0
num_skipped_identical = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for i, ex in enumerate(tqdm(dataset)):
        prompt, chosen = get_prompt_and_chosen(ex)
        rejected = generate_rejected(prompt)

        # Skip if chosen and rejected are identical (ignoring whitespace)
        if rejected.strip() == chosen.strip():
            num_skipped_identical += 1
            # Optional debug:
            # print(f"Skipping example {i} because chosen == rejected")
            continue

        dpo_example = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

        # Debug: print first 3 kept examples so you can see structure
        if num_written < 3:
            print("\nExample (kept)", num_written)
            print(json.dumps(dpo_example, indent=2)[:800])

        f_out.write(json.dumps(dpo_example, ensure_ascii=False) + "\n")
        num_written += 1

print("Done.")
print(f"Saved {num_written} DPO examples to: {OUTPUT_FILE}")
print(f"Skipped {num_skipped_identical} examples where chosen == rejected.")