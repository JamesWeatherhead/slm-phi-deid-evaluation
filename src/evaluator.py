#!/usr/bin/env python3
"""
Three-Tier Evaluation Engine for SLM De-Identification Study

Tier 1: String absence (did PHI value disappear?)
Tier 2: Position-aware tag matching (was a correctly typed tag placed near PHI?)
Tier 3: Output fidelity (Levenshtein similarity >= 0.85, no hallucination)

Usage:
    python -m src.evaluator --model phi4-mini --prompt zero-shot-structured
    python -m src.evaluator --all
"""

import argparse
import json
import logging
import re
from pathlib import Path

from src.utils import (
    BASE_DIR, load_dataset, load_prompt_templates,
    PHI_TYPE_TO_TAG, TAG_VARIANT_MAP,
    find_all_tags, normalize_tag_type, count_tags_in_text,
    levenshtein_similarity, setup_logging, utc_timestamp,
    ParseResult,
)

logger = logging.getLogger("evaluator")


# ---------------------------------------------------------------------------
# Tier 1: String Absence
# ---------------------------------------------------------------------------

def evaluate_tier1_positive(query, raw_output):
    """Tier 1 evaluation for PHI-positive query.

    For each ground-truth PHI element, check if its value is absent from output.

    Returns:
        dict with per-element results.
    """
    if raw_output is None:
        return {
            "phi_detected": 0,
            "phi_missed": len(query["phi_tags"]),
            "recall": 0.0,
            "missed_values": [p["value"] for p in query["phi_tags"]],
            "detected_values": [],
        }

    output_lower = raw_output.strip().lower()
    detected = []
    missed = []

    for phi in query["phi_tags"]:
        value = phi["value"].strip()
        if value.lower() not in output_lower:
            detected.append({
                "value": value,
                "type": phi["identifier_type"],
                "tier1": True,
            })
        else:
            missed.append({
                "value": value,
                "type": phi["identifier_type"],
                "tier1": False,
            })

    total = len(query["phi_tags"])
    return {
        "phi_detected": len(detected),
        "phi_missed": len(missed),
        "recall": len(detected) / total if total > 0 else 1.0,
        "missed_values": [m["value"] for m in missed],
        "detected_values": [d["value"] for d in detected],
        "per_element": detected + missed,
    }


def evaluate_tier1_negative(query, raw_output):
    """Tier 1 evaluation for PHI-negative query.

    Check if any redaction tags were inserted (false positives).
    Uses comprehensive 14-pattern tag detector.

    Returns:
        dict with false positive info.
    """
    if raw_output is None:
        return {
            "tags_found": 0,
            "tag_list": [],
            "preserved": True,
        }

    tags = find_all_tags(raw_output)
    return {
        "tags_found": len(tags),
        "tag_list": [t["tag_text"] for t in tags],
        "tag_types": [t["tag_type"] for t in tags],
        "tag_formats": [t["format"] for t in tags],
        "preserved": len(tags) == 0,
    }


# ---------------------------------------------------------------------------
# Tier 2: Position-Aware Tag Matching
# ---------------------------------------------------------------------------

def evaluate_tier2_positive(query, raw_output):
    """Tier 2 evaluation for PHI-positive query.

    For each PHI element that passed Tier 1 (value absent from output),
    check if a correctly typed tag was placed near the PHI's original position.

    Returns:
        dict with per-element Tier 2 results.
    """
    if raw_output is None:
        return {
            "phi_detected": 0,
            "phi_missed": len(query["phi_tags"]),
            "recall": 0.0,
            "details": [],
        }

    query_text = query["query_text"]
    output_lower = raw_output.lower()
    all_tags = find_all_tags(raw_output)

    detected = 0
    missed = 0
    details = []

    for phi in query["phi_tags"]:
        value = phi["value"].strip()
        phi_type = phi["identifier_type"]
        expected_canonical = PHI_TYPE_TO_TAG.get(phi_type, phi_type)

        # Tier 1 check first
        if value.lower() in output_lower:
            missed += 1
            details.append({
                "value": value,
                "type": phi_type,
                "tier1_pass": False,
                "tier2_pass": False,
                "reason": "phi_still_present",
            })
            continue

        # Find PHI position in original input
        phi_pos = query_text.lower().find(value.lower())
        if phi_pos == -1:
            # PHI value not found in input (should not happen with clean data)
            detected += 1
            details.append({
                "value": value,
                "type": phi_type,
                "tier1_pass": True,
                "tier2_pass": True,
                "reason": "phi_not_in_input",
            })
            continue

        # Calculate approximate position ratio (where in the text is the PHI?)
        input_ratio = phi_pos / max(len(query_text), 1)

        # Look for matching tags near the expected position
        # Use a window based on position ratio in the output
        expected_output_pos = int(input_ratio * len(raw_output))
        window = 80  # Characters of tolerance

        tag_match_found = False
        for tag in all_tags:
            tag_canonical = tag["tag_type"]
            tag_pos = tag["position"]

            # Check if tag type matches (canonical comparison)
            type_matches = (tag_canonical == expected_canonical)

            # Check if tag is near expected position
            position_matches = abs(tag_pos - expected_output_pos) <= window

            # Also check: is the tag a generic redaction near the right spot?
            generic_match = (
                tag_canonical in ("UNKNOWN", "REDACTED", "REMOVED", "PHI")
                and position_matches
            )

            if (type_matches and position_matches) or generic_match:
                tag_match_found = True
                details.append({
                    "value": value,
                    "type": phi_type,
                    "expected_tag": f"[{expected_canonical}]",
                    "found_tag": tag["tag_text"],
                    "tag_type_match": type_matches,
                    "position_offset": abs(tag_pos - expected_output_pos),
                    "tier1_pass": True,
                    "tier2_pass": True,
                    "reason": "tag_matched",
                })
                break

        if tag_match_found:
            detected += 1
        else:
            # PHI was removed but no matching tag found: possible deletion
            missed += 1
            # Check if it was a deletion (surrounding context intact, PHI just gone)
            is_deletion = _check_deletion(
                query_text, raw_output, value, phi_pos
            )
            details.append({
                "value": value,
                "type": phi_type,
                "expected_tag": f"[{expected_canonical}]",
                "tier1_pass": True,
                "tier2_pass": False,
                "is_deletion": is_deletion,
                "reason": "deletion" if is_deletion else "no_matching_tag",
            })

    total = len(query["phi_tags"])
    return {
        "phi_detected": detected,
        "phi_missed": missed,
        "recall": detected / total if total > 0 else 1.0,
        "details": details,
    }


def _check_deletion(original, output, phi_value, phi_pos):
    """Check if PHI was deleted (not replaced) by examining surrounding context."""
    # Get context words around PHI in original
    window = 30
    before = original[max(0, phi_pos - window):phi_pos].strip().split()
    after_start = phi_pos + len(phi_value)
    after = original[after_start:after_start + window].strip().split()

    # Check if context words exist in output
    output_lower = output.lower()
    before_present = sum(1 for w in before[-3:] if w.lower() in output_lower)
    after_present = sum(1 for w in after[:3] if w.lower() in output_lower)

    # If surrounding context is present but PHI is not, it was likely deleted
    return (before_present >= 2 or after_present >= 2)


# ---------------------------------------------------------------------------
# Tier 3: Output Fidelity
# ---------------------------------------------------------------------------

def evaluate_tier3(query, raw_output, tier2_result=None):
    """Tier 3 evaluation: output fidelity check.

    Checks:
    1. Levenshtein similarity >= threshold (0.85 positive, 0.90 negative)
    2. Character ratio within bounds (0.80 to 1.20)
    3. No appended content (output not significantly longer than expected)
    4. No hallucinated new content

    Returns:
        dict with fidelity metrics.
    """
    if raw_output is None:
        return {
            "pass": False,
            "levenshtein_similarity": 0.0,
            "char_ratio": 0.0,
            "appended_content": False,
            "new_phi_detected": False,
            "failure_reason": "no_output",
        }

    query_text = query["query_text"]
    is_negative = query.get("is_negative", len(query.get("phi_tags", [])) == 0)

    # Build expected output by replacing PHI with tags
    if not is_negative:
        expected = query_text
        phi_tags = sorted(
            query.get("phi_tags", []),
            key=lambda p: len(p["value"]),
            reverse=True,
        )
        for phi in phi_tags:
            tag = f"[{PHI_TYPE_TO_TAG.get(phi['identifier_type'], 'ID')}]"
            expected = expected.replace(phi["value"], tag)
    else:
        expected = query_text

    # Fidelity checks
    lev_sim = levenshtein_similarity(expected, raw_output)
    char_ratio = len(raw_output) / max(len(expected), 1)

    # Appended content detection
    appended = False
    if len(raw_output) > len(expected) * 1.3:
        appended = True

    # New PHI detection (basic regex for obvious patterns)
    new_phi = _detect_new_phi(raw_output, query_text)

    # Thresholds
    sim_threshold = 0.90 if is_negative else 0.85
    ratio_lo, ratio_hi = 0.80, 1.20

    passed = (
        lev_sim >= sim_threshold
        and ratio_lo <= char_ratio <= ratio_hi
        and not appended
        and not new_phi
    )

    failure_reason = None
    if not passed:
        reasons = []
        if lev_sim < sim_threshold:
            reasons.append(f"low_similarity_{lev_sim:.2f}")
        if not (ratio_lo <= char_ratio <= ratio_hi):
            reasons.append(f"char_ratio_{char_ratio:.2f}")
        if appended:
            reasons.append("appended_content")
        if new_phi:
            reasons.append("new_phi_detected")
        failure_reason = "; ".join(reasons)

    return {
        "pass": passed,
        "levenshtein_similarity": round(lev_sim, 4),
        "char_ratio": round(char_ratio, 4),
        "appended_content": appended,
        "new_phi_detected": new_phi,
        "failure_reason": failure_reason,
    }


def _detect_new_phi(output, original):
    """Detect if the model hallucinated new PHI not present in the input.

    Uses simple regex patterns for obvious PHI patterns (names, dates, numbers).
    """
    # Extract potential PHI-like patterns from output
    # Pattern: Capitalized first + last name not in original
    name_pattern = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]*\.?\b')
    output_names = set(name_pattern.findall(output))
    original_names = set(name_pattern.findall(original))
    new_names = output_names - original_names

    # Filter out common clinical terms that look like names
    clinical_non_names = {
        "New York", "United States", "African American", "Native American",
        "Emergency Room", "Intensive Care", "Blood Pressure", "Heart Rate",
    }
    new_names -= clinical_non_names

    if new_names:
        return True

    # Pattern: Phone numbers, SSNs, dates not in original
    id_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    output_ids = set(id_pattern.findall(output))
    original_ids = set(id_pattern.findall(original))
    if output_ids - original_ids:
        return True

    return False


# ---------------------------------------------------------------------------
# False Positive Taxonomy (for negative queries)
# ---------------------------------------------------------------------------

def classify_false_positives(query, raw_output):
    """Classify false positive tags on negative queries into taxonomy types.

    Type A: Medical term incorrectly tagged
    Type B: Tags in hallucinated content
    Type C: Tags in explanatory text
    Type D: Ambiguous entity
    """
    if raw_output is None:
        return None

    tags = find_all_tags(raw_output)
    if not tags:
        return None

    query_text = query["query_text"]
    classification = {
        "type_a_count": 0, "type_b_count": 0,
        "type_c_count": 0, "type_d_count": 0,
        "tagged_terms": [],
    }

    # Check if output is significantly longer than input (hallucination indicator)
    is_expanded = len(raw_output) > len(query_text) * 1.2

    for tag in tags:
        tag_pos = tag["position"]
        tag_text = tag["tag_text"]
        tag_len = len(tag_text)

        # Extract surrounding context
        ctx_start = max(0, tag_pos - 30)
        ctx_end = min(len(raw_output), tag_pos + tag_len + 30)
        context = raw_output[ctx_start:ctx_end]

        if is_expanded:
            # Check if this tag is in the "excess" portion
            if tag_pos > len(query_text):
                # Tag is beyond original text length: likely hallucinated
                if any(kw in context.lower() for kw in
                       ["for example", "such as", "e.g.", "like"]):
                    classification["type_b_count"] += 1
                elif any(kw in context.lower() for kw in
                         ["replaced", "de-identified", "redacted", "removed"]):
                    classification["type_c_count"] += 1
                else:
                    classification["type_b_count"] += 1
                classification["tagged_terms"].append(tag_text)
                continue

        # Check if a medical term was tagged (Type A)
        # Look at what was in the original text at approximately this position
        input_ratio = tag_pos / max(len(raw_output), 1)
        approx_input_pos = int(input_ratio * len(query_text))
        input_window = query_text[
            max(0, approx_input_pos - 20):approx_input_pos + 20
        ]

        # Common clinical terms that should never be tagged
        clinical_terms = [
            "diabetes", "copd", "ms", "afib", "gerd", "hypertension",
            "cancer", "asthma", "nsclc", "chf", "cad", "ckd",
            "metformin", "lisinopril", "atorvastatin", "aspirin",
            "surgery", "bypass", "hysterectomy", "chemotherapy",
        ]
        if any(term in input_window.lower() for term in clinical_terms):
            classification["type_a_count"] += 1
        else:
            # Default: ambiguous entity
            classification["type_d_count"] += 1

        classification["tagged_terms"].append(tag_text)

    return classification


# ---------------------------------------------------------------------------
# Evaluate a single query (all tiers)
# ---------------------------------------------------------------------------

def evaluate_query(query, raw_output, parse_status=None):
    """Run all three evaluation tiers on a single query.

    Args:
        query: dict from all_queries.json
        raw_output: the model's output text (already parsed/extracted)
        parse_status: ParseResult value for P5 outputs

    Returns:
        dict with complete evaluation results.
    """
    is_negative = query.get("is_negative", len(query.get("phi_tags", [])) == 0)

    result = {
        "query_id": query["query_id"],
        "is_negative": is_negative,
        "phi_count": query.get("phi_count", len(query.get("phi_tags", []))),
        "parse_status": parse_status if parse_status else "clean",
    }

    if is_negative:
        # Negative query evaluation
        t1 = evaluate_tier1_negative(query, raw_output)
        result["tier1"] = t1
        result["tier2"] = {
            "tags_found": t1["tags_found"],
            "preserved": t1["preserved"],
        }
        result["tier3"] = evaluate_tier3(query, raw_output)
        result["fp_taxonomy"] = classify_false_positives(query, raw_output)
    else:
        # Positive query evaluation
        t1 = evaluate_tier1_positive(query, raw_output)
        t2 = evaluate_tier2_positive(query, raw_output)
        t3 = evaluate_tier3(query, raw_output, t2)
        result["tier1"] = {
            "phi_detected": t1["phi_detected"],
            "phi_missed": t1["phi_missed"],
            "recall": t1["recall"],
            "missed_values": t1["missed_values"],
        }
        result["tier2"] = {
            "phi_detected": t2["phi_detected"],
            "phi_missed": t2["phi_missed"],
            "recall": t2["recall"],
        }
        result["tier3"] = t3
        result["all_detected_tags"] = [
            t["tag_text"] for t in find_all_tags(raw_output or "")
        ]

    return result


# ---------------------------------------------------------------------------
# Process a full responses.jsonl file
# ---------------------------------------------------------------------------

def evaluate_responses_file(responses_path, dataset, output_dir=None):
    """Evaluate all responses in a JSONL file.

    Args:
        responses_path: Path to raw responses.jsonl
        dataset: loaded all_queries.json dict
        output_dir: directory for processed output (default: auto-detect)

    Returns:
        list of per-query evaluation dicts
    """
    responses_path = Path(responses_path)

    # Build query lookup
    query_lookup = {q["query_id"]: q for q in dataset["queries"]}

    # Read responses
    responses = []
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    results = []
    for resp in responses:
        query_id = resp.get("query_id")

        # Skip canary queries
        if resp.get("is_canary"):
            continue

        if query_id not in query_lookup:
            logger.warning("Query %s not found in dataset, skipping", query_id)
            continue

        query = query_lookup[query_id]

        # Determine the output text to evaluate
        # For P5: use extracted_output if available
        # For P4: use raw_output (which is pass2 output)
        # For others: use raw_output
        raw_output = resp.get("extracted_output") or resp.get("raw_output")
        parse_status = resp.get("parse_status")

        # Handle error responses
        if resp.get("error"):
            raw_output = None

        eval_result = evaluate_query(query, raw_output, parse_status)
        eval_result["run_number"] = resp.get("run_number")
        eval_result["latency_ms"] = resp.get("latency_ms", -1)
        eval_result["tokens_output"] = resp.get("tokens_output", 0)
        eval_result["error"] = resp.get("error")
        eval_result["output_truncated"] = resp.get("output_truncated", False)

        results.append(eval_result)

    return results


# ---------------------------------------------------------------------------
# Aggregate metrics across runs
# ---------------------------------------------------------------------------

def aggregate_evaluation(eval_results, model_slug, prompt_slug):
    """Aggregate per-query evaluation results into summary metrics.

    Args:
        eval_results: list of per-query evaluation dicts (across all runs)
        model_slug: model identifier
        prompt_slug: prompt identifier

    Returns:
        dict matching the evaluation.json schema
    """
    positive = [r for r in eval_results if not r["is_negative"] and not r.get("error")]
    negative = [r for r in eval_results if r["is_negative"] and not r.get("error")]
    errors = [r for r in eval_results if r.get("error")]

    # Group by run
    runs = sorted(set(r.get("run_number", 1) for r in eval_results))

    # Tier 1 recall per run (positive queries)
    t1_recall_per_run = []
    for run in runs:
        run_pos = [r for r in positive if r.get("run_number") == run]
        if run_pos:
            total_detected = sum(r["tier1"]["phi_detected"] for r in run_pos)
            total_phi = sum(r["phi_count"] for r in run_pos)
            t1_recall_per_run.append(total_detected / total_phi if total_phi > 0 else 0.0)

    # Tier 2 recall per run
    t2_recall_per_run = []
    for run in runs:
        run_pos = [r for r in positive if r.get("run_number") == run]
        if run_pos:
            total_detected = sum(r["tier2"]["phi_detected"] for r in run_pos)
            total_phi = sum(r["phi_count"] for r in run_pos)
            t2_recall_per_run.append(total_detected / total_phi if total_phi > 0 else 0.0)

    # Tier 3 fidelity pass rate per run
    t3_pass_per_run = []
    for run in runs:
        run_all = [r for r in eval_results if r.get("run_number") == run and not r.get("error")]
        if run_all:
            passed = sum(1 for r in run_all if r["tier3"]["pass"])
            t3_pass_per_run.append(passed / len(run_all))

    # Specificity per run (negative queries)
    spec_per_run = []
    for run in runs:
        run_neg = [r for r in negative if r.get("run_number") == run]
        if run_neg:
            preserved = sum(1 for r in run_neg if r["tier1"]["preserved"])
            spec_per_run.append(preserved / len(run_neg))

    # Per-category recall
    per_category = {}
    for r in positive:
        if "per_element" in r.get("tier1", {}):
            for elem in r["tier1"]["per_element"]:
                cat = elem["type"]
                if cat not in per_category:
                    per_category[cat] = {"detected": 0, "total": 0}
                per_category[cat]["total"] += 1
                if elem.get("tier1"):
                    per_category[cat]["detected"] += 1

    # FP taxonomy aggregation
    fp_types = {"type_a": 0, "type_b": 0, "type_c": 0, "type_d": 0}
    for r in negative:
        fp = r.get("fp_taxonomy")
        if fp:
            fp_types["type_a"] += fp.get("type_a_count", 0)
            fp_types["type_b"] += fp.get("type_b_count", 0)
            fp_types["type_c"] += fp.get("type_c_count", 0)
            fp_types["type_d"] += fp.get("type_d_count", 0)

    # Latency stats
    latencies = [r["latency_ms"] for r in eval_results if r.get("latency_ms", -1) > 0]

    import numpy as np
    lat_arr = np.array(latencies) if latencies else np.array([0])

    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def _safe_std(arr):
        return float(np.std(arr)) if len(arr) > 1 else 0.0

    return {
        "model_name": model_slug,
        "prompt_style": prompt_slug,
        "runs_completed": len(runs),
        "queries_per_run": len(eval_results) // max(len(runs), 1),
        "total_inferences": len(eval_results),
        "errors": {
            "total_failures": len(errors),
            "timeout": sum(1 for e in errors if "timeout" in str(e.get("error", ""))),
            "parse_error": sum(1 for e in errors if "parse" in str(e.get("error", ""))),
            "other": sum(1 for e in errors if "timeout" not in str(e.get("error", "")) and "parse" not in str(e.get("error", ""))),
        },
        "positive_queries": {
            "count": len(set(r["query_id"] for r in positive)),
            "tier1": {
                "recall_mean": _safe_mean(t1_recall_per_run),
                "recall_std": _safe_std(t1_recall_per_run),
                "recall_per_run": [round(r, 4) for r in t1_recall_per_run],
            },
            "tier2": {
                "recall_mean": _safe_mean(t2_recall_per_run),
                "recall_std": _safe_std(t2_recall_per_run),
                "recall_per_run": [round(r, 4) for r in t2_recall_per_run],
            },
            "tier3": {
                "fidelity_pass_rate_mean": _safe_mean(t3_pass_per_run),
            },
            "per_category": {
                cat: {
                    "recall": vals["detected"] / vals["total"] if vals["total"] > 0 else 0.0,
                    "count": vals["total"],
                    "detected": vals["detected"],
                }
                for cat, vals in per_category.items()
            },
        },
        "negative_queries": {
            "count": len(set(r["query_id"] for r in negative)),
            "tier1": {
                "specificity_mean": _safe_mean(spec_per_run),
                "specificity_std": _safe_std(spec_per_run),
                "specificity_per_run": [round(s, 4) for s in spec_per_run],
            },
            "fp_taxonomy": fp_types,
        },
        "latency": {
            "mean_ms": round(float(np.mean(lat_arr)), 1),
            "median_ms": round(float(np.median(lat_arr)), 1),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 1),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 1),
        },
        "timestamp": utc_timestamp(),
    }


# ---------------------------------------------------------------------------
# Process all results for a model/prompt combination
# ---------------------------------------------------------------------------

def process_model_prompt(model_slug, prompt_slug, output_dir=None):
    """Evaluate all runs for one model/prompt configuration.

    Reads from raw/{model}/{prompt}/run_*/responses.jsonl
    Writes to processed/{model}/{prompt}/per_query.jsonl and evaluation.json
    """
    if output_dir is None:
        output_dir = BASE_DIR
    output_dir = Path(output_dir)

    dataset = load_dataset()
    raw_dir = output_dir / "raw" / model_slug / prompt_slug

    if not raw_dir.exists():
        logger.warning("No raw data found at %s", raw_dir)
        return

    # Find all run directories
    run_dirs = sorted(raw_dir.glob("run_*/responses.jsonl"))
    if not run_dirs:
        logger.warning("No response files found in %s", raw_dir)
        return

    logger.info(
        "Evaluating %s/%s: %d run(s) found",
        model_slug, prompt_slug, len(run_dirs),
    )

    all_results = []
    for resp_file in run_dirs:
        results = evaluate_responses_file(resp_file, dataset, output_dir)
        all_results.extend(results)
        logger.info(
            "  %s: %d queries evaluated", resp_file.parent.name, len(results)
        )

    # Write per_query.jsonl
    proc_dir = output_dir / "processed" / model_slug / prompt_slug
    proc_dir.mkdir(parents=True, exist_ok=True)

    per_query_path = proc_dir / "per_query.jsonl"
    with open(per_query_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write evaluation.json
    agg = aggregate_evaluation(all_results, model_slug, prompt_slug)
    eval_path = proc_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    logger.info(
        "  Wrote %s (%d records) and %s",
        per_query_path, len(all_results), eval_path,
    )

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Three-Tier Evaluation Engine"
    )
    parser.add_argument("--model", type=str, help="Model slug to evaluate")
    parser.add_argument("--prompt", type=str, help="Prompt slug to evaluate")
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all model/prompt combinations found in raw/",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(BASE_DIR),
        help="Root output directory",
    )
    parser.add_argument("--log-file", type=str, default=None)

    args = parser.parse_args()
    setup_logging(log_file=args.log_file)

    if args.all:
        raw_dir = Path(args.output_dir) / "raw"
        if not raw_dir.exists():
            logger.error("No raw/ directory found at %s", args.output_dir)
            return

        for model_dir in sorted(raw_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for prompt_dir in sorted(model_dir.iterdir()):
                if not prompt_dir.is_dir():
                    continue
                process_model_prompt(
                    model_dir.name, prompt_dir.name, args.output_dir
                )
    elif args.model and args.prompt:
        process_model_prompt(args.model, args.prompt, args.output_dir)
    else:
        parser.error("Either --all or both --model and --prompt required.")


if __name__ == "__main__":
    main()
