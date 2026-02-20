#!/usr/bin/env python3
"""
Variance Pilot: Determinism Assessment

Runs 100 queries x 10 runs x 1 model x 1 prompt to measure:
- Byte-identical output rate across runs
- Variance in recall/precision across runs
- Recommendation for N (number of runs in the full experiment)

MUST run BEFORE the full experiment to determine the appropriate number of replicate runs.

Usage:
    python -m src.pilot --host http://<your-ollama-host>:11434
    python -m src.pilot --host http://localhost:11434 --model phi4-mini --prompt zero-shot-structured
"""

import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from src.utils import (
    BASE_DIR, load_dataset, load_prompt_templates, load_model_configs,
    load_inference_params, setup_logging, utc_timestamp,
    find_all_tags, PHI_TYPE_TO_TAG,
)
from src.runner import (
    query_with_retry, build_prompt, warm_up_model, flush_model,
    parse_direct_output, DEFAULT_HOST,
)

logger = logging.getLogger("pilot")

PILOT_N_QUERIES = 100
PILOT_N_RUNS = 10


def run_pilot(host, model_slug="phi4-mini", prompt_slug="zero-shot-structured",
              n_queries=PILOT_N_QUERIES, n_runs=PILOT_N_RUNS, output_dir=None):
    """Run the variance pilot.

    Args:
        host: Ollama host URL
        model_slug: model to test (default: phi4-mini)
        prompt_slug: prompt to test (default: zero-shot-structured)
        n_queries: number of queries to sample
        n_runs: number of repeat runs
        output_dir: where to write results

    Returns:
        dict with pilot results and recommendation
    """
    if output_dir is None:
        output_dir = BASE_DIR
    output_dir = Path(output_dir)

    dataset = load_dataset()
    templates = load_prompt_templates()
    model_configs = load_model_configs()
    inf_params = load_inference_params()

    # Resolve configs
    model_config = None
    for m in model_configs["models"]:
        if m["slug"] == model_slug:
            model_config = m
            break
    if model_config is None:
        raise ValueError(f"Model {model_slug} not found in configs")

    prompt_config = None
    for key, cfg in templates.items():
        if key.startswith("_"):
            continue
        if cfg.get("slug") == prompt_slug:
            prompt_config = cfg
            break
    if prompt_config is None:
        raise ValueError(f"Prompt {prompt_slug} not found in configs")

    model_tag = model_config["ollama_tag"]
    default_params = inf_params["default"]

    # Sample 100 queries (stratified: ~79 positive, ~21 negative)
    queries = dataset["queries"]
    positive = [q for q in queries if not q.get("is_negative", False)]
    negative = [q for q in queries if q.get("is_negative", False)]

    rng = random.Random(42)
    n_pos = min(int(n_queries * 0.79), len(positive))
    n_neg = n_queries - n_pos

    sampled_pos = rng.sample(positive, n_pos)
    sampled_neg = rng.sample(negative, min(n_neg, len(negative)))
    sampled = sampled_pos + sampled_neg
    rng.shuffle(sampled)

    logger.info("Pilot: %d queries x %d runs = %d inferences",
                len(sampled), n_runs, len(sampled) * n_runs)
    logger.info("Model: %s (%s)", model_slug, model_tag)
    logger.info("Prompt: %s", prompt_slug)

    # Warmup
    warm_up_model(host, model_tag, prompt_config, default_params, sampled, n=10)

    # Run all queries for each run
    all_outputs = {}  # query_id -> list of outputs (one per run)
    all_recalls = []  # per-run recall values

    for run_num in range(1, n_runs + 1):
        logger.info("--- Pilot Run %d/%d ---", run_num, n_runs)
        run_outputs = {}
        run_tp = 0
        run_total_phi = 0

        for i, query in enumerate(sampled):
            query_id = query["query_id"]
            query_text = query["query_text"]
            system_prompt, user_prompt = build_prompt(prompt_config, query_text)

            result = query_with_retry(
                host, model_tag, system_prompt, user_prompt,
                default_params, query_id, max_retries=2, timeout=120,
            )

            raw_output = ""
            if result["error"] is None and result["raw_output"]:
                raw_output = parse_direct_output(result["raw_output"])

            run_outputs[query_id] = raw_output

            # Track recall for positive queries
            if not query.get("is_negative", False):
                for phi in query.get("phi_tags", []):
                    run_total_phi += 1
                    if phi["value"].lower() not in raw_output.lower():
                        run_tp += 1

            if (i + 1) % 25 == 0:
                logger.info("  Run %d: %d/%d queries done", run_num, i + 1, len(sampled))

        # Store outputs
        for qid, output in run_outputs.items():
            if qid not in all_outputs:
                all_outputs[qid] = []
            all_outputs[qid].append(output)

        # Per-run recall
        recall = run_tp / run_total_phi if run_total_phi > 0 else 0.0
        all_recalls.append(recall)
        logger.info("  Run %d recall: %.4f (%d/%d)", run_num, recall, run_tp, run_total_phi)

    # Analysis
    logger.info("=" * 50)
    logger.info("PILOT ANALYSIS")
    logger.info("=" * 50)

    # Byte-identical rate
    total_queries = len(all_outputs)
    identical_count = 0
    for qid, outputs in all_outputs.items():
        hashes = [hashlib.sha256(o.encode("utf-8")).hexdigest() for o in outputs]
        if len(set(hashes)) == 1:
            identical_count += 1

    identical_rate = identical_count / total_queries if total_queries > 0 else 0.0
    logger.info("Byte-identical rate: %.1f%% (%d/%d queries)",
                identical_rate * 100, identical_count, total_queries)

    # Recall statistics
    recalls_arr = np.array(all_recalls)
    recall_mean = float(np.mean(recalls_arr))
    recall_std = float(np.std(recalls_arr))
    recall_range = float(np.max(recalls_arr) - np.min(recalls_arr))

    logger.info("Recall mean: %.4f", recall_mean)
    logger.info("Recall std: %.4f", recall_std)
    logger.info("Recall range: %.4f", recall_range)

    # Per-query Levenshtein variance (sample)
    from src.utils import levenshtein_similarity
    lev_diffs = []
    for qid, outputs in all_outputs.items():
        if len(outputs) >= 2:
            for j in range(1, len(outputs)):
                lev_diffs.append(levenshtein_similarity(outputs[0], outputs[j]))

    mean_lev = float(np.mean(lev_diffs)) if lev_diffs else 1.0
    logger.info("Mean pairwise Levenshtein similarity: %.4f", mean_lev)

    # Recommendation
    if identical_rate >= 0.99:
        recommendation = "N=3"
        rationale = (
            f"Byte-identical rate is {identical_rate*100:.1f}% (>= 99%). "
            "Model is effectively deterministic with temperature=0, seed=42. "
            "Run N=3 for confirmation only. Report single canonical run with "
            "2 confirmatory runs for reproducibility."
        )
    elif recall_std < 0.005:
        recommendation = "N=5"
        rationale = (
            f"Byte-identical rate is {identical_rate*100:.1f}% (< 99%) but "
            f"recall SD is {recall_std:.4f} (< 0.005). Minor non-determinism "
            "exists but does not meaningfully affect metrics. Run N=5 with "
            "different seeds {42, 123, 456, 789, 1024}."
        )
    else:
        recommendation = "N=5_different_seeds"
        rationale = (
            f"Byte-identical rate is {identical_rate*100:.1f}% and recall SD "
            f"is {recall_std:.4f} (>= 0.005). Meaningful non-determinism "
            "detected. Run N=5 with different seeds. Report mean and SD."
        )

    logger.info("RECOMMENDATION: %s", recommendation)
    logger.info("RATIONALE: %s", rationale)

    # Save results
    pilot_results = {
        "timestamp": utc_timestamp(),
        "model_slug": model_slug,
        "model_tag": model_tag,
        "prompt_slug": prompt_slug,
        "n_queries": len(sampled),
        "n_runs": n_runs,
        "n_positive": n_pos,
        "n_negative": len(sampled_neg),
        "byte_identical_rate": round(identical_rate, 4),
        "byte_identical_count": identical_count,
        "recall_mean": round(recall_mean, 4),
        "recall_std": round(recall_std, 4),
        "recall_range": round(recall_range, 4),
        "recall_per_run": [round(r, 4) for r in all_recalls],
        "mean_pairwise_levenshtein": round(mean_lev, 4),
        "recommendation": recommendation,
        "rationale": rationale,
        "inference_params": default_params,
    }

    pilot_dir = output_dir / "analysis"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = pilot_dir / "pilot_results.json"

    with open(pilot_path, "w", encoding="utf-8") as f:
        json.dump(pilot_results, f, indent=2, ensure_ascii=False)

    logger.info("Pilot results saved to %s", pilot_path)
    return pilot_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Variance Pilot: Determinism Assessment"
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST,
        help=f"Ollama host URL (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--model", type=str, default="phi4-mini",
        help="Model slug to test (default: phi4-mini)",
    )
    parser.add_argument(
        "--prompt", type=str, default="zero-shot-structured",
        help="Prompt slug to test (default: zero-shot-structured)",
    )
    parser.add_argument(
        "--n-queries", type=int, default=PILOT_N_QUERIES,
        help=f"Number of queries to sample (default: {PILOT_N_QUERIES})",
    )
    parser.add_argument(
        "--n-runs", type=int, default=PILOT_N_RUNS,
        help=f"Number of runs (default: {PILOT_N_RUNS})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(BASE_DIR),
        help="Root output directory",
    )
    parser.add_argument("--log-file", type=str, default=None)

    args = parser.parse_args()
    setup_logging(log_file=args.log_file)

    run_pilot(
        host=args.host,
        model_slug=args.model,
        prompt_slug=args.prompt,
        n_queries=args.n_queries,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
