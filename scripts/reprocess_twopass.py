#!/usr/bin/env python3
"""Re-process two-pass response files using the fixed JSON parser.

Reads pass1_output from existing responses, re-parses with the updated
parse_phi_json (which now strips markdown fences), applies programmatic
redaction, and writes corrected files.

Also deduplicates any files with duplicate query_ids (e.g., from crash-resume).
"""

import json
import sys
from pathlib import Path
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import parse_phi_json, apply_redactions, load_dataset


def reprocess_file(filepath, dataset_lookup):
    """Re-process a single responses.jsonl file."""
    filepath = Path(filepath)

    with open(filepath) as f:
        lines = f.readlines()

    # Deduplicate: keep last occurrence of each query_id
    seen = OrderedDict()
    for line in lines:
        d = json.loads(line.strip())
        qid = d.get("query_id")
        seen[qid] = d

    records = list(seen.values())
    original_count = len(lines)
    deduped_count = len(records)

    fixed = 0
    still_broken = 0

    for rec in records:
        # Skip canaries and non-error records
        if rec.get("is_canary"):
            continue
        if rec.get("error") != "json_parse_failure":
            continue

        pass1 = rec.get("pass1_output")
        if not pass1:
            still_broken += 1
            continue

        # Re-parse with fixed parser
        phi_list = parse_phi_json(pass1)

        if phi_list is None:
            still_broken += 1
            continue

        if len(phi_list) == 0:
            # No PHI found; output should be the original query text
            qid = rec.get("query_id")
            if qid in dataset_lookup:
                rec["raw_output"] = dataset_lookup[qid]["query"]
            else:
                rec["raw_output"] = rec.get("input_text", "")
            rec["error"] = None
            rec["parse_status"] = "reprocessed_empty"
            fixed += 1
            continue

        # Apply programmatic redaction
        original_text = rec.get("input_text", "")
        if not original_text:
            qid = rec.get("query_id")
            if qid in dataset_lookup:
                original_text = dataset_lookup[qid]["query"]

        redacted = apply_redactions(original_text, phi_list)
        rec["raw_output"] = redacted
        rec["error"] = None
        rec["parse_status"] = "reprocessed_fixed"
        fixed += 1

    # Write back
    with open(filepath, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dupes_removed = original_count - deduped_count
    return fixed, still_broken, dupes_removed


def main():
    dataset = load_dataset()
    query_lookup = {q["query_id"]: q for q in dataset["queries"]}

    raw_dir = Path("raw")
    twopass_files = sorted(raw_dir.glob("*/two-pass*/run_*/responses.jsonl"))

    print(f"Found {len(twopass_files)} two-pass response files to reprocess\n")

    total_fixed = 0
    total_broken = 0
    total_deduped = 0

    for f in twopass_files:
        fixed, broken, deduped = reprocess_file(f, query_lookup)
        total_fixed += fixed
        total_broken += broken
        total_deduped += deduped
        rel = str(f)
        status = []
        if fixed: status.append(f"{fixed} fixed")
        if broken: status.append(f"{broken} still broken")
        if deduped: status.append(f"{deduped} dupes removed")
        print(f"  {rel}: {', '.join(status) if status else 'no changes'}")

    print(f"\nTotal: {total_fixed} fixed, {total_broken} still broken, {total_deduped} dupes removed")

    # Also deduplicate any non-two-pass files that have dupes
    print("\nChecking all other files for duplicates...")
    other_files = sorted(raw_dir.glob("*/*/run_*/responses.jsonl"))
    for f in other_files:
        if "two-pass" in str(f):
            continue
        with open(f) as fh:
            lines = fh.readlines()
        seen = OrderedDict()
        for line in lines:
            d = json.loads(line.strip())
            seen[d.get("query_id")] = d
        if len(seen) < len(lines):
            dupes = len(lines) - len(seen)
            with open(f, "w") as fh:
                for rec in seen.values():
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  {f}: {dupes} dupes removed")
            total_deduped += dupes

    print(f"\nDone. Total dupes removed across all files: {total_deduped}")


if __name__ == "__main__":
    main()
