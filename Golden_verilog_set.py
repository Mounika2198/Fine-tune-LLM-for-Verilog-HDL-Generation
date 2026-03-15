
import json
import re
from datasets import load_dataset

# ================================
# CONFIG — EDIT THESE
# ================================

HF_DATASET_ID = "bnadimi/PyraNet-Verilog"   # <- put your HF dataset name here
SPLIT = "train"                                     # or "train[:1000]" for testing
OUTPUT_JSONL = "golden_verilog_dataset.jsonl"       # output file

# If your dataset has a compile_status column, keep only "No error!"
FILTER_ON_COMPILE_STATUS = True
COMPILE_STATUS_FIELD = "compile_status"
COMPILE_OK_VALUE = "No error!"


# ================================
# HELPER: normalize code for dedupe
# ================================

def normalize_code(code: str) -> str:
    """
    Normalize Verilog code for deduplication:
    - strip leading/trailing whitespace
    - strip trailing spaces on each line
    """
    lines = [ln.rstrip() for ln in code.strip().splitlines()]
    return "\n".join(lines)


# ================================
# HELPER: parse input/output names
# ================================

PORT_RE = re.compile(r'^\s*(input|output)\s+(.*?);', re.MULTILINE)

def extract_io_names(verilog_code: str):
    """
    Extract lists of input and output signal names from Verilog code.

    Handles lines like:
        input a;
        input wire [3:0] a, b;
        output reg y;

    Returns:
        (input_names, output_names) as sorted unique lists of strings.
    """
    inputs = []
    outputs = []

    for direction, rest in PORT_RE.findall(verilog_code):
        # Split by comma
        tokens = [t.strip() for t in rest.split(",") if t.strip()]
        names = []
        for tok in tokens:
            # Remove types and ranges: wire, reg, logic, signed/unsigned, [3:0], etc.
            tok = re.sub(r'\b(wire|reg|logic|signed|unsigned)\b', '', tok)
            tok = re.sub(r'\[[^]]+\]', '', tok)  # remove [N:M]
            tok = tok.strip()
            if tok:
                names.append(tok)

        if direction == "input":
            inputs.extend(names)
        elif direction == "output":
            outputs.extend(names)

    # dedupe + sort for consistency
    inputs = sorted(set(inputs))
    outputs = sorted(set(outputs))
    return inputs, outputs


# ================================
# MAIN SCRUB + EXPORT
# ================================

def main():
    print(f"Loading dataset: {HF_DATASET_ID} [{SPLIT}]...")
    ds = load_dataset(HF_DATASET_ID, split=SPLIT)

    seen_codes = set()
    num_total = 0
    num_kept = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for ex in ds:
            num_total += 1

            code = ex.get("code", "")
            desc = ex.get("description", "")

            if not code or not desc:
                # drop empty stuff
                continue

            # Optional: filter by compile_status if the field exists
            if FILTER_ON_COMPILE_STATUS and COMPILE_STATUS_FIELD in ex:
                if ex[COMPILE_STATUS_FIELD] != COMPILE_OK_VALUE:
                    continue

            norm = normalize_code(code)
            if norm in seen_codes:
                # duplicate code; skip
                continue
            seen_codes.add(norm)

            # Extract IO names
            input_names, output_names = extract_io_names(code)
            if not input_names or not output_names:
                # if we can't find any inputs/outputs, skip (probably not a clean module)
                continue

            record = {
                "task": desc.strip(),            # you can also shorten/clean description here
                "input_names": input_names,
                "output_names": output_names,
                "golden_verilog": code.strip(),
            }

            fout.write(json.dumps(record) + "\n")
            num_kept += 1

    print(f"Done.")
    print(f"Total raw examples: {num_total}")
    print(f"Unique, cleaned, IO-parsed examples written: {num_kept}")
    print(f"Output JSONL: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
