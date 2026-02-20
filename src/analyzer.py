#!/usr/bin/env python3
"""
Cross-Model Analysis Pipeline for SLM De-Identification Study

Generates:
- comparison_table.csv (main results matrix)
- per_category_recall.csv
- false_positive_taxonomy.csv
- statistical_tests.json (bootstrap CIs, McNemar, Cochran's Q)

Usage:
    python -m src.analyzer --output-dir /path/to/slm-evaluation
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from src.utils import (
    BASE_DIR, setup_logging, utc_timestamp,
    stratified_bootstrap_metrics,
)

logger = logging.getLogger("analyzer")


# ---------------------------------------------------------------------------
# Load all evaluation.json files
# ---------------------------------------------------------------------------

def load_all_evaluations(output_dir):
    """Load all evaluation.json files from processed/."""
    processed_dir = Path(output_dir) / "processed"
    evaluations = []

    if not processed_dir.exists():
        logger.error("No processed/ directory found")
        return evaluations

    for model_dir in sorted(processed_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            eval_path = prompt_dir / "evaluation.json"
            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                evaluations.append(data)
                logger.info("Loaded: %s/%s", model_dir.name, prompt_dir.name)

    logger.info("Total configurations loaded: %d", len(evaluations))
    return evaluations


def load_per_query_results(output_dir, model_slug, prompt_slug):
    """Load per_query.jsonl for a specific model/prompt."""
    path = Path(output_dir) / "processed" / model_slug / prompt_slug / "per_query.jsonl"
    results = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def generate_comparison_table(evaluations, output_dir):
    """Generate the main comparison_table.csv."""
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "prompt_style",
        "t1_recall_mean", "t1_recall_std",
        "t2_recall_mean", "t2_recall_std",
        "t3_fidelity_mean",
        "t1_specificity_mean", "t1_specificity_std",
        "fp_type_a", "fp_type_b", "fp_type_c", "fp_type_d",
        "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
        "runs_completed", "errors_total",
    ]

    output_path = analysis_dir / "comparison_table.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ev in evaluations:
            pos = ev.get("positive_queries", {})
            neg = ev.get("negative_queries", {})
            lat = ev.get("latency", {})
            fp = neg.get("fp_taxonomy", {})

            row = {
                "model": ev.get("model_name", ""),
                "prompt_style": ev.get("prompt_style", ""),
                "t1_recall_mean": round(pos.get("tier1", {}).get("recall_mean", 0), 4),
                "t1_recall_std": round(pos.get("tier1", {}).get("recall_std", 0), 4),
                "t2_recall_mean": round(pos.get("tier2", {}).get("recall_mean", 0), 4),
                "t2_recall_std": round(pos.get("tier2", {}).get("recall_std", 0), 4),
                "t3_fidelity_mean": round(pos.get("tier3", {}).get("fidelity_pass_rate_mean", 0), 4),
                "t1_specificity_mean": round(neg.get("tier1", {}).get("specificity_mean", 0), 4),
                "t1_specificity_std": round(neg.get("tier1", {}).get("specificity_std", 0), 4),
                "fp_type_a": fp.get("type_a", 0),
                "fp_type_b": fp.get("type_b", 0),
                "fp_type_c": fp.get("type_c", 0),
                "fp_type_d": fp.get("type_d", 0),
                "latency_mean_ms": lat.get("mean_ms", 0),
                "latency_median_ms": lat.get("median_ms", 0),
                "latency_p95_ms": lat.get("p95_ms", 0),
                "runs_completed": ev.get("runs_completed", 0),
                "errors_total": ev.get("errors", {}).get("total_failures", 0),
            }
            writer.writerow(row)

    logger.info("Wrote comparison_table.csv (%d rows)", len(evaluations))
    return output_path


# ---------------------------------------------------------------------------
# Per-category recall
# ---------------------------------------------------------------------------

def generate_category_recall(evaluations, output_dir):
    """Generate per_category_recall.csv."""
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model", "prompt_style", "phi_type",
        "count", "detected", "t1_recall",
    ]

    output_path = analysis_dir / "per_category_recall.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ev in evaluations:
            per_cat = ev.get("positive_queries", {}).get("per_category", {})
            for cat, vals in sorted(per_cat.items()):
                writer.writerow({
                    "model": ev.get("model_name", ""),
                    "prompt_style": ev.get("prompt_style", ""),
                    "phi_type": cat,
                    "count": vals.get("count", 0),
                    "detected": vals.get("detected", 0),
                    "t1_recall": round(vals.get("recall", 0), 4),
                })

    logger.info("Wrote per_category_recall.csv")
    return output_path


# ---------------------------------------------------------------------------
# False positive taxonomy
# ---------------------------------------------------------------------------

def generate_fp_taxonomy(output_dir):
    """Generate false_positive_taxonomy.csv from per-query results."""
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_id", "model", "prompt_style", "run_number",
        "total_fp_tags", "type_a_count", "type_b_count",
        "type_c_count", "type_d_count", "tagged_terms",
    ]

    output_path = analysis_dir / "false_positive_taxonomy.csv"
    processed_dir = Path(output_dir) / "processed"

    rows = []
    if processed_dir.exists():
        for model_dir in sorted(processed_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for prompt_dir in sorted(model_dir.iterdir()):
                if not prompt_dir.is_dir():
                    continue
                results = load_per_query_results(
                    output_dir, model_dir.name, prompt_dir.name
                )
                for r in results:
                    if not r.get("is_negative"):
                        continue
                    fp = r.get("fp_taxonomy")
                    if not fp:
                        continue
                    total_fp = (
                        fp.get("type_a_count", 0) + fp.get("type_b_count", 0)
                        + fp.get("type_c_count", 0) + fp.get("type_d_count", 0)
                    )
                    if total_fp == 0:
                        continue
                    rows.append({
                        "query_id": r["query_id"],
                        "model": model_dir.name,
                        "prompt_style": prompt_dir.name,
                        "run_number": r.get("run_number", 1),
                        "total_fp_tags": total_fp,
                        "type_a_count": fp.get("type_a_count", 0),
                        "type_b_count": fp.get("type_b_count", 0),
                        "type_c_count": fp.get("type_c_count", 0),
                        "type_d_count": fp.get("type_d_count", 0),
                        "tagged_terms": "|".join(fp.get("tagged_terms", [])),
                    })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info("Wrote false_positive_taxonomy.csv (%d rows)", len(rows))
    return output_path


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def compute_bootstrap_cis(output_dir, n_iterations=10000):
    """Compute stratified bootstrap BCa CIs for all configurations."""
    processed_dir = Path(output_dir) / "processed"
    results = {}

    if not processed_dir.exists():
        return results

    for model_dir in sorted(processed_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue

            per_query = load_per_query_results(
                output_dir, model_dir.name, prompt_dir.name
            )
            if not per_query:
                continue

            # Use first run only for bootstrap (avoid inflating N with replicates)
            run1 = [r for r in per_query if r.get("run_number", 1) == 1]
            if not run1:
                run1 = per_query

            # Build bootstrap inputs
            positive_data = []
            negative_data = []

            for r in run1:
                if r.get("error"):
                    continue
                if r["is_negative"]:
                    tags_found = r.get("tier1", {}).get("tags_found", 0)
                    negative_data.append({"tags_found": tags_found})
                else:
                    t1 = r.get("tier1", {})
                    positive_data.append({
                        "phi_detected": t1.get("phi_detected", 0),
                        "phi_total": r.get("phi_count", 0),
                        "fp_count": max(0,
                            len(r.get("all_detected_tags", []))
                            - r.get("phi_count", 0)
                        ),
                    })

            if positive_data and negative_data:
                logger.info(
                    "Computing bootstrap CIs for %s/%s (%d pos, %d neg)...",
                    model_dir.name, prompt_dir.name,
                    len(positive_data), len(negative_data),
                )
                ci = stratified_bootstrap_metrics(
                    positive_data, negative_data,
                    n_iterations=n_iterations, random_state=42,
                )
                key = f"{model_dir.name}/{prompt_dir.name}"
                results[key] = ci

    return results


# ---------------------------------------------------------------------------
# McNemar's test for pre-specified contrasts
# ---------------------------------------------------------------------------

def mcnemar_test(results_a, results_b):
    """Run McNemar's test comparing two configurations at query level.

    Uses all-or-nothing recall: binary outcome per positive query
    (did the model get ALL PHI elements in this query correct?).

    Args:
        results_a, results_b: lists of per-query evaluation dicts

    Returns:
        dict with test statistic, p-value, and contingency table
    """
    # Build lookup by query_id (use run 1 only)
    a_lookup = {}
    b_lookup = {}

    for r in results_a:
        if r.get("is_negative") or r.get("error") or r.get("run_number", 1) != 1:
            continue
        a_lookup[r["query_id"]] = r["tier1"]["phi_missed"] == 0

    for r in results_b:
        if r.get("is_negative") or r.get("error") or r.get("run_number", 1) != 1:
            continue
        b_lookup[r["query_id"]] = r["tier1"]["phi_missed"] == 0

    # Match on shared query IDs
    shared = set(a_lookup.keys()) & set(b_lookup.keys())
    if len(shared) < 10:
        return {"error": "insufficient_shared_queries", "n": len(shared)}

    # Build 2x2 contingency table
    # b=both correct, c=A correct B wrong, b_=A wrong B correct, d=both wrong
    b = sum(1 for q in shared if a_lookup[q] and b_lookup[q])
    c = sum(1 for q in shared if a_lookup[q] and not b_lookup[q])
    b_ = sum(1 for q in shared if not a_lookup[q] and b_lookup[q])
    d = sum(1 for q in shared if not a_lookup[q] and not b_lookup[q])

    # McNemar's test (with continuity correction)
    n_discordant = c + b_
    if n_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n": len(shared),
            "contingency": {"both_correct": b, "a_only": c, "b_only": b_, "both_wrong": d},
            "cohens_g": 0.0,
        }

    chi2 = (abs(c - b_) - 1) ** 2 / (c + b_)
    p_value = 1.0 - _chi2_cdf(chi2, df=1)

    # Cohen's g (effect size for McNemar)
    cohens_g = (c - b_) / (c + b_) if (c + b_) > 0 else 0.0

    return {
        "statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "n": len(shared),
        "contingency": {"both_correct": b, "a_only": c, "b_only": b_, "both_wrong": d},
        "cohens_g": round(float(cohens_g), 4),
    }


def _chi2_cdf(x, df):
    """Chi-squared CDF using scipy."""
    return float(sp_stats.chi2.cdf(x, df))


# ---------------------------------------------------------------------------
# Cochran's Q test (omnibus)
# ---------------------------------------------------------------------------

def cochrans_q_test(all_results_by_config, held_constant_configs):
    """Run Cochran's Q test for omnibus comparison.

    Tests whether there is any difference among K configurations
    on binary outcomes (all-or-nothing recall per query).

    Args:
        all_results_by_config: dict mapping config_key -> per-query results
        held_constant_configs: list of config keys to compare

    Returns:
        dict with test statistic, df, p-value
    """
    if len(held_constant_configs) < 2:
        return {"error": "need_at_least_2_configs"}

    # Build binary outcome matrix: queries x configs
    # Use run 1 only
    config_outcomes = {}
    for key in held_constant_configs:
        results = all_results_by_config.get(key, [])
        run1 = {
            r["query_id"]: (r["tier1"]["phi_missed"] == 0)
            for r in results
            if not r.get("is_negative") and not r.get("error")
            and r.get("run_number", 1) == 1
        }
        config_outcomes[key] = run1

    # Find shared query IDs
    shared_ids = None
    for key, outcomes in config_outcomes.items():
        if shared_ids is None:
            shared_ids = set(outcomes.keys())
        else:
            shared_ids &= set(outcomes.keys())

    if shared_ids is None or len(shared_ids) < 10:
        return {"error": "insufficient_shared_queries"}

    shared_ids = sorted(shared_ids)
    K = len(held_constant_configs)
    N = len(shared_ids)

    # Build matrix
    matrix = np.zeros((N, K), dtype=int)
    for j, key in enumerate(held_constant_configs):
        for i, qid in enumerate(shared_ids):
            matrix[i, j] = int(config_outcomes[key].get(qid, False))

    # Cochran's Q statistic
    row_sums = matrix.sum(axis=1)  # T_i: successes per query
    col_sums = matrix.sum(axis=0)  # C_j: successes per config
    T = row_sums.sum()

    num = (K - 1) * (K * np.sum(col_sums ** 2) - T ** 2)
    denom = K * T - np.sum(row_sums ** 2)

    if denom == 0:
        return {"statistic": 0.0, "df": K - 1, "p_value": 1.0, "n": N}

    Q = float(num / denom)
    df = K - 1
    p_value = 1.0 - _chi2_cdf(Q, df)

    return {
        "statistic": round(Q, 4),
        "df": df,
        "p_value": round(p_value, 6),
        "n": N,
        "configs_compared": held_constant_configs,
    }


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(output_dir, n_bootstrap=10000):
    """Run the complete analysis pipeline."""
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluations
    evaluations = load_all_evaluations(output_dir)
    if not evaluations:
        logger.error("No evaluation files found. Run evaluator first.")
        return

    # 1. Comparison table
    generate_comparison_table(evaluations, output_dir)

    # 2. Per-category recall
    generate_category_recall(evaluations, output_dir)

    # 3. False positive taxonomy
    generate_fp_taxonomy(output_dir)

    # 4. Bootstrap CIs
    logger.info("Computing bootstrap CIs (n=%d)...", n_bootstrap)
    bootstrap_results = compute_bootstrap_cis(output_dir, n_iterations=n_bootstrap)

    # 5. Statistical tests
    logger.info("Running statistical tests...")
    stat_tests = {"timestamp": utc_timestamp()}

    # Load all per-query results for statistical tests
    all_per_query = {}
    processed_dir = output_dir / "processed"
    if processed_dir.exists():
        for model_dir in sorted(processed_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for prompt_dir in sorted(model_dir.iterdir()):
                if not prompt_dir.is_dir():
                    continue
                key = f"{model_dir.name}/{prompt_dir.name}"
                all_per_query[key] = load_per_query_results(
                    output_dir, model_dir.name, prompt_dir.name
                )

    # Cochran's Q: model comparison (holding prompt constant at zero-shot-structured)
    model_slugs = sorted(set(e["model_name"] for e in evaluations))
    zss_configs = [f"{m}/zero-shot-structured" for m in model_slugs
                   if f"{m}/zero-shot-structured" in all_per_query]

    if len(zss_configs) >= 2:
        cochrans_model = cochrans_q_test(all_per_query, zss_configs)
        stat_tests["cochrans_q_model_comparison"] = cochrans_model
        logger.info(
            "Cochran's Q (model comparison): Q=%.2f, p=%.4f",
            cochrans_model.get("statistic", 0),
            cochrans_model.get("p_value", 1),
        )

    # Cochran's Q: prompt comparison (holding model constant at phi4-mini)
    prompt_slugs = sorted(set(e["prompt_style"] for e in evaluations))
    phi4_configs = [f"phi4-mini/{p}" for p in prompt_slugs
                    if f"phi4-mini/{p}" in all_per_query]

    if len(phi4_configs) >= 2:
        cochrans_prompt = cochrans_q_test(all_per_query, phi4_configs)
        stat_tests["cochrans_q_prompt_comparison"] = cochrans_prompt
        logger.info(
            "Cochran's Q (prompt comparison): Q=%.2f, p=%.4f",
            cochrans_prompt.get("statistic", 0),
            cochrans_prompt.get("p_value", 1),
        )

    # Pre-specified McNemar contrasts
    contrasts = [
        ("phi4-mini/zero-shot-structured", "llama32-3b/zero-shot-structured",
         "phi4-mini vs llama32-3b (zero-shot-structured)"),
        ("phi4-mini/few-shot", "phi4-mini/zero-shot-minimal",
         "few-shot vs zero-shot-minimal (phi4-mini)"),
        ("phi4-mini/two-pass", "phi4-mini/zero-shot-minimal",
         "two-pass vs zero-shot-minimal (phi4-mini)"),
        ("phi4-mini/chain-of-thought", "phi4-mini/zero-shot-structured",
         "chain-of-thought vs zero-shot-structured (phi4-mini)"),
    ]

    mcnemar_results = []
    for config_a, config_b, label in contrasts:
        if config_a in all_per_query and config_b in all_per_query:
            result = mcnemar_test(all_per_query[config_a], all_per_query[config_b])
            result["comparison"] = label
            result["config_a"] = config_a
            result["config_b"] = config_b
            mcnemar_results.append(result)
            logger.info(
                "McNemar (%s): chi2=%.2f, p=%.4f",
                label,
                result.get("statistic", 0),
                result.get("p_value", 1),
            )

    # Apply Holm-Bonferroni correction to McNemar p-values
    if mcnemar_results:
        p_values = [r.get("p_value", 1.0) for r in mcnemar_results]
        corrected = holm_bonferroni(p_values)
        for i, r in enumerate(mcnemar_results):
            r["p_value_corrected"] = corrected[i]

    stat_tests["mcnemar_pairwise"] = mcnemar_results
    stat_tests["bootstrap"] = {
        "n_iterations": n_bootstrap,
        "method": "BCa (stratified by positive/negative queries)",
        "results_per_config": bootstrap_results,
    }

    # Write statistical_tests.json
    stat_path = analysis_dir / "statistical_tests.json"
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stat_tests, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Wrote statistical_tests.json")
    logger.info("Analysis pipeline complete.")


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns:
        list of corrected p-values in original order.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value (keep track of original indices)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    corrected = [0.0] * n
    prev_corrected = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        adjusted = max(adjusted, prev_corrected)  # Enforce monotonicity
        adjusted = min(adjusted, 1.0)
        corrected[orig_idx] = round(adjusted, 6)
        prev_corrected = adjusted

    return corrected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Model Analysis Pipeline"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(BASE_DIR),
        help="Root output directory (default: slm-evaluation/)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap iterations (default: 10000)",
    )
    parser.add_argument("--log-file", type=str, default=None)

    args = parser.parse_args()
    setup_logging(log_file=args.log_file)
    run_analysis(args.output_dir, args.n_bootstrap)


if __name__ == "__main__":
    main()
