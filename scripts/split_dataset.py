#!/usr/bin/env python3
"""
split_dataset.py
Split ASQ-PHI synthetic_clinical_queries.txt into structured JSON files
for the SLM de-identification evaluation study.

Produces:
  - positive_queries.json  (832 queries with PHI)
  - negative_queries.json  (219 queries without PHI)
  - all_queries.json       (1,051 queries combined with metadata)

Each record contains:
  - query_id: sequential identifier (Q-0001, Q-0002, ...)
  - query_text: the clinical query string
  - phi_tags: list of {identifier_type, value} dicts
  - phi_count: number of PHI elements in the query
  - is_negative: boolean flag (True if no PHI present)
"""

import json
import sys
from pathlib import Path


def parse_dataset(filepath: str) -> list[dict]:
    """Parse the ===QUERY=== / ===PHI_TAGS=== delimited text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on ===QUERY=== and discard the first empty segment
    raw_blocks = content.split("===QUERY===")
    raw_blocks = [b for b in raw_blocks if b.strip()]

    records = []
    for idx, block in enumerate(raw_blocks, start=1):
        if "===PHI_TAGS===" not in block:
            print(f"WARNING: Block {idx} missing ===PHI_TAGS=== delimiter, skipping.")
            continue

        query_part, tags_part = block.split("===PHI_TAGS===", 1)
        query_text = query_part.strip()

        # Parse PHI tags (one JSON object per line)
        phi_tags = []
        for line in tags_part.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                tag = json.loads(line)
                phi_tags.append({
                    "identifier_type": tag["identifier_type"],
                    "value": tag["value"]
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"WARNING: Failed to parse tag in block {idx}: {line} ({e})")

        query_id = f"Q-{idx:04d}"
        is_negative = len(phi_tags) == 0

        records.append({
            "query_id": query_id,
            "query_text": query_text,
            "phi_tags": phi_tags,
            "phi_count": len(phi_tags),
            "is_negative": is_negative
        })

    return records


def write_json(data, filepath: str) -> None:
    """Write data to a JSON file with pretty formatting."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if isinstance(data, dict) and "queries" in data:
        count = len(data["queries"])
    elif isinstance(data, list):
        count = len(data)
    else:
        count = len(data)
    print(f"  Written: {filepath} ({count} records)")


def main():
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    source_file = data_dir / "synthetic_clinical_queries.txt"

    if not source_file.exists():
        print(f"ERROR: Source file not found: {source_file}")
        sys.exit(1)

    print(f"Parsing: {source_file}")
    all_records = parse_dataset(str(source_file))
    print(f"  Total records parsed: {len(all_records)}")

    # Split into positive and negative
    positive = [r for r in all_records if not r["is_negative"]]
    negative = [r for r in all_records if r["is_negative"]]

    print(f"  Positive (with PHI): {len(positive)}")
    print(f"  Negative (no PHI):   {len(negative)}")

    # Compute total PHI elements
    total_phi = sum(r["phi_count"] for r in all_records)
    print(f"  Total PHI elements:  {total_phi}")

    # Validate against expected counts
    if len(positive) != 832:
        print(f"  WARNING: Expected 832 positive queries, got {len(positive)}")
    if len(negative) != 219:
        print(f"  WARNING: Expected 219 negative queries, got {len(negative)}")
    if total_phi != 2973:
        print(f"  WARNING: Expected 2,973 PHI elements, got {total_phi}")

    # Write output files
    print("\nWriting output files:")
    write_json(positive, str(data_dir / "positive_queries.json"))
    write_json(negative, str(data_dir / "negative_queries.json"))

    # All queries with metadata envelope
    all_output = {
        "metadata": {
            "dataset": "ASQ-PHI",
            "version": "1.0",
            "source_file": "synthetic_clinical_queries.txt",
            "total_queries": len(all_records),
            "positive_queries": len(positive),
            "negative_queries": len(negative),
            "total_phi_elements": total_phi,
            "phi_type_distribution": {},
            "query_complexity_distribution": {}
        },
        "queries": all_records
    }

    # Compute PHI type distribution
    type_counts = {}
    for r in all_records:
        for tag in r["phi_tags"]:
            t = tag["identifier_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
    # Sort descending
    all_output["metadata"]["phi_type_distribution"] = dict(
        sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # Compute query complexity distribution
    complexity = {}
    for r in all_records:
        c = r["phi_count"]
        complexity[c] = complexity.get(c, 0) + 1
    all_output["metadata"]["query_complexity_distribution"] = dict(
        sorted(complexity.items(), key=lambda x: x[0])
    )

    write_json(all_output, str(data_dir / "all_queries.json"))

    print("\nDone. All files written to:", data_dir)


if __name__ == "__main__":
    main()
