
import json
import random
from typing import Dict, List

# ================================
# CONFIG — EDIT THESE
# ================================
INPUT_JSONL = "golden_verilog_dataset.jsonl"      # cleaned file from previous step
OUTPUT_JSONL = "S2_cleam_verilog.jsonl"       # new output file

TARGET_TOTAL = 50_000
NUM_BUCKETS = 4
TARGET_PER_BUCKET =500 
#TARGET_TOTAL // NUM_BUCKETS   # 12,500 each

DESC_FIELD = "task"  # adjust if your field name is different
CODE_FIELD = "golden_verilog"

# If your cleaned JSONL has a category field like "adder", "mux", etc,
# set this to the correct key and use it in `categorize()` if you want:
CATEGORY_FIELD = "category"    # change to the real name if needed, or leave unused

# ================================
# CATEGORY KEYWORDS (TIGHTENED)
# ================================
# IMPORTANT: we removed generic "and", "or" to avoid classifying
# everything as a gate just because the code uses logical operators.

#GATE_KEYWORDS = [
    #"and gate", "or gate", "nand gate", "nor gate",
    #"xor gate", "xnor gate", "inverter", "not gate"
#]

#ADDER_KEYWORDS = [
    #"full adder", "half adder", "ripple carry adder",
    #"carry lookahead", "carry-lookahead",
    #"carry select", "carry-select",
    #"carry save", "carry-save",
    #"k-bit adder", "adder"
#]

MUX_KEYWORDS = [
    "multiplexer", "2-to-1 mux", "2:1 mux",
    "4-to-1 mux", "4:1 mux",
    "8-to-1 mux", "8:1 mux",
    "16-to-1 mux", "16:1 mux",
    "mux", "select line", "sel"
]

COMP_DEC_ENC_KEYWORDS = [
    # comparators
    #"comparator", "magnitude comparator", "greater than", "less than",
    #">= ", "<= ", " == ", " != ",
    # decoders
    "decoder", "2-to-4 decoder", "3-to-8 decoder", "4-to-16 decoder",
    # encoders
    "encoder", "priority encoder"
]

# Labels (just for printing)
BUCKET_NAMES = {
    "gates": "Gates (AND/NOT/NOR/etc.)",
    "adders": "Adders / Half Adders",
    "muxes": "MUXes",
    "comp_dec_enc": "Comparators / Decoders / Encoders"
}

# ================================
# HELPERS
# ================================

def get_text(example: Dict) -> str:
    """Concatenate description + code (lowercased) for keyword matching."""
    desc = example.get(DESC_FIELD, "") or ""
    code = example.get(CODE_FIELD, "") or ""
    return (str(desc) + " " + str(code)).lower()

def matches_any(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)

# OPTIONAL: category-based mapping if your JSONL has something like:
#   "category": "adder", "mux", "comparator", etc.
def categorize_by_field(example: Dict) -> str | None:
    cat_val = example.get(CATEGORY_FIELD, None)
    if cat_val is None:
        return None
    s = str(cat_val).lower()

    # Adjust these to match whatever labels PyraNet uses
    if "adder" in s or "add" in s:
        return "adders"
    if "mux" in s or "multiplexer" in s:
        return "muxes"
    if "comp" in s or "comparator" in s or "decoder" in s or "encoder" in s:
        return "comp_dec_enc"
    if "gate" in s or "logic" in s:
        return "gates"
    return None

def categorize(example: Dict) -> str | None:
    """
    Return one of:
      - "gates"
      - "adders"
      - "muxes"
      - "comp_dec_enc"
      - None (if it doesn't clearly belong anywhere)

    Priority:
      1) Category field (if it matches)
      2) Adders
      3) MUXes
      4) Comparators / Decoders / Encoders
      5) Gates
    """

    # 1) Try category field first (if present / usable)
    cat_field_bucket = categorize_by_field(example)
    if cat_field_bucket is not None:
        return cat_field_bucket

    # 2) Fallback to text-based keywords
    t = get_text(example)

    # more specific classes FIRST
    #if matches_any(t, ADDER_KEYWORDS):
        #return "adders"
    if matches_any(t, MUX_KEYWORDS):
        return "muxes"
    if matches_any(t, COMP_DEC_ENC_KEYWORDS):
        return "comp_dec_enc"
    #if matches_any(t, GATE_KEYWORDS):
        #return "gates"

    return None

# ================================
# LOAD AND BUCKET DATA
# ================================

print(f"Reading cleaned file: {INPUT_JSONL}")
buckets = {
    "gates": [],
    "adders": [],
    "muxes": [],
    "comp_dec_enc": [],
}

total_read = 0
total_categorized = 0

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total_read += 1
        ex = json.loads(line)
        cat = categorize(ex)
        if cat is not None:
            buckets[cat].append(ex)
            total_categorized += 1

print(f"Total examples read: {total_read}")
print(f"Total examples categorized into one of the 4 buckets: {total_categorized}")
for key, name in BUCKET_NAMES.items():
    print(f"  {name}: {len(buckets[key])} examples")

# ================================
# SAMPLE FROM EACH BUCKET
# ================================

final_examples = []

for bucket_key, bucket_name in BUCKET_NAMES.items():
    examples = buckets[bucket_key]
    random.shuffle(examples)

    desired = TARGET_PER_BUCKET
    available = len(examples)

    if available == 0:
        print(f"[WARNING] No examples found for bucket: {bucket_name}")
        continue

    if available < desired:
        print(
            f"[INFO] Bucket '{bucket_name}' has only {available} examples "
            f"(desired {desired}). Using all {available}."
        )
        chosen = examples  # all of them
    else:
        chosen = examples[:desired]

    print(f"Using {len(chosen)} examples for bucket: {bucket_name}")
    final_examples.extend(chosen)

# ================================
# CHECK TOTAL SIZE
# ================================

unique_ids = set()
unique_final = []
for ex in final_examples:
    obj_id = id(ex)
    if obj_id not in unique_ids:
        unique_ids.add(obj_id)
        unique_final.append(ex)

final_count = len(unique_final)

if final_count < TARGET_TOTAL:
    print(
        f"[INFO] Could not reach {TARGET_TOTAL} unique examples. "
        f"Final size: {final_count}"
    )
else:
    unique_final = unique_final[:TARGET_TOTAL]
    final_count = TARGET_TOTAL

print(f"Writing {final_count} examples to {OUTPUT_JSONL} ...")

# ================================
# WRITE OUTPUT JSONL
# ================================

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for ex in unique_final:
        json.dump(ex, f, ensure_ascii=False)
        f.write("\n")

print("Done.")
print(f"Balanced dataset saved to: {OUTPUT_JSONL}")
