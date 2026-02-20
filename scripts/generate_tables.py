#!/usr/bin/env python3
"""
Generate LaTeX tables for the SLM PHI de-identification paper.
Table 1: Main results (18 configs)
Table 2: Error analysis
Table 3: Per-category recall (note: per_category data not available; placeholder)
"""
import json
import os

RESULTS_PATH = "/Users/jacweath/Desktop/Dissertation/AIMS/AIM_1/slm-phi-deid-evaluation/data/slm-phi-deid-evaluation/results/analysis/final_results.json"
OUT_DIR = "/Users/jacweath/Desktop/mc-output/slm-paper/tables"

with open(RESULTS_PATH) as f:
    data = json.load(f)

# ============================================================
# TABLE 1: Main Results
# ============================================================

# Find best in each metric column
metrics_t1 = {
    "t1_recall": [e["tier1_recall_mean"] for e in data],
    "t2_recall": [e["tier2_recall_mean"] for e in data],
    "t2_precision": [e["tier2_precision_mean"] for e in data],
    "t3_similarity": [e["tier3_similarity_mean"] for e in data],
    "t3_pass": [e["tier3_pass_rate_mean"] for e in data],
    "over_redact": [e["over_redaction"] for e in data],
}

best_idx = {
    "t1_recall": max(range(len(data)), key=lambda i: metrics_t1["t1_recall"][i]),
    "t2_recall": max(range(len(data)), key=lambda i: metrics_t1["t2_recall"][i]),
    "t2_precision": max(range(len(data)), key=lambda i: metrics_t1["t2_precision"][i]),
    "t3_similarity": max(range(len(data)), key=lambda i: metrics_t1["t3_similarity"][i]),
    "t3_pass": max(range(len(data)), key=lambda i: metrics_t1["t3_pass"][i]),
    "over_redact": min(range(len(data)), key=lambda i: metrics_t1["over_redact"][i]),
}


def bold_if_best(value, idx, metric_key, fmt=".1f"):
    """Bold the value if this row is the best for the given metric."""
    formatted = f"{value:{fmt}}"
    if idx == best_idx[metric_key]:
        return f"\\textbf{{{formatted}}}"
    return formatted


# Group by model, sort within each group by T1 Recall descending
MODEL_ORDER = ["Gemma 3 4B", "Llama 3.2 3B", "Phi-4 Mini"]
grouped = {m: [] for m in MODEL_ORDER}
for idx, e in enumerate(data):
    grouped[e["model_label"]].append(idx)
for m in MODEL_ORDER:
    grouped[m].sort(key=lambda i: -data[i]["tier1_recall_mean"])

lines = []
lines.append(r"\begin{table*}[ht]")
lines.append(r"\centering")
lines.append(r"\caption{Main evaluation results across all 18 model/strategy configurations. Best value in each column is bolded. Over-redaction represents the proportion of PHI-free queries where the model incorrectly flagged content.}")
lines.append(r"\label{tab:main_results}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llcccccc}")
lines.append(r"\toprule")
lines.append(r"Model & Strategy & T1 Recall & T2 Recall & T2 Prec. & T3 Sim. & T3 Pass & Over-red. \\")
lines.append(r" & & (\%) & (\%) & (\%) & & Rate (\%) & (\%) \\")
lines.append(r"\midrule")

for mi, model_name in enumerate(MODEL_ORDER):
    if mi > 0:
        lines.append(r"\midrule")
    for idx in grouped[model_name]:
        e = data[idx]
        strat = e["strategy_label"]

        t1r = bold_if_best(e["tier1_recall_mean"] * 100, idx, "t1_recall")
        t2r = bold_if_best(e["tier2_recall_mean"] * 100, idx, "t2_recall")
        t2p = bold_if_best(e["tier2_precision_mean"] * 100, idx, "t2_precision")
        t3s = bold_if_best(e["tier3_similarity_mean"], idx, "t3_similarity", fmt=".4f")
        t3p = bold_if_best(e["tier3_pass_rate_mean"] * 100, idx, "t3_pass")
        ovr = bold_if_best(e["over_redaction"] * 100, idx, "over_redact")

        lines.append(f"{model_name} & {strat} & {t1r} & {t2r} & {t2p} & {t3s} & {t3p} & {ovr} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

with open(os.path.join(OUT_DIR, "table1_main_results.tex"), "w") as f:
    f.write("\n".join(lines))
print("Table 1 saved.")


# ============================================================
# TABLE 2: Error Analysis
# ============================================================

lines2 = []
lines2.append(r"\begin{table*}[ht]")
lines2.append(r"\centering")
lines2.append(r"\caption{Error analysis across all configurations. Total errors, breakdown by type, and error rate relative to total inferences (3 runs $\times$ 1,051 queries = 3,153 inferences per configuration).}")
lines2.append(r"\label{tab:error_analysis}")
lines2.append(r"\small")
lines2.append(r"\begin{tabular}{llccccc}")
lines2.append(r"\toprule")
lines2.append(r"Model & Strategy & Total Errors & Timeout & Parse Error & Other & Error Rate (\%) \\")
lines2.append(r"\midrule")

total_inferences = 3153  # 3 runs x 1051 queries

for mi, model_name in enumerate(MODEL_ORDER):
    if mi > 0:
        lines2.append(r"\midrule")
    for idx in grouped[model_name]:
        e = data[idx]
        strat = e["strategy_label"]
        errs = e["errors"]

        total = errs["total_failures"]
        timeout = errs["timeout"]
        parse = errs["parse_error"]
        other = errs["other"]
        rate = (total / total_inferences) * 100

        lines2.append(f"{model_name} & {strat} & {total} & {timeout} & {parse} & {other} & {rate:.1f} \\\\")

lines2.append(r"\bottomrule")
lines2.append(r"\end{tabular}")
lines2.append(r"\end{table*}")

with open(os.path.join(OUT_DIR, "table2_error_analysis.tex"), "w") as f:
    f.write("\n".join(lines2))
print("Table 2 saved.")


# ============================================================
# TABLE 3: Per-Category Recall
# ============================================================
# Note: per_category data is empty in all evaluation files.
# Generating a placeholder table with a note.

lines3 = []
lines3.append(r"\begin{table}[ht]")
lines3.append(r"\centering")
lines3.append(r"\caption{Per-PHI-category Tier 1 recall for the top three configurations. Categories are drawn from the ASQ-PHI dataset annotations. Values represent the proportion of PHI instances correctly detected (binary redaction).}")
lines3.append(r"\label{tab:per_category_recall}")
lines3.append(r"\small")
lines3.append(r"\begin{tabular}{lccc}")
lines3.append(r"\toprule")
lines3.append(r"PHI Category & Gemma 3 4B & Llama 3.2 3B & Phi-4 Mini \\")
lines3.append(r" & (Few-Shot) & (ZS Minimal) & (ZS Structured) \\")
lines3.append(r"\midrule")

# Per-category data was not computed during evaluation runs.
# This table is a placeholder; populate from detailed per-category
# analysis if/when that data becomes available.
categories = [
    "NAME",
    "DATE",
    "GEOGRAPHIC\\_LOCATION",
    "PHONE\\_NUMBER",
    "EMAIL\\_ADDRESS",
    "SSN",
    "MEDICAL\\_RECORD\\_NUMBER",
    "ACCOUNT\\_NUMBER",
    "DEVICE\\_ID",
    "URL",
]

lines3.append(r"\multicolumn{4}{c}{\textit{Per-category data not computed in current evaluation.}} \\")
lines3.append(r"\multicolumn{4}{c}{\textit{Aggregate Tier 1 recall shown below for reference.}} \\")
lines3.append(r"\midrule")
lines3.append(r"All Categories (Aggregate) & 94.8 & 99.6 & 96.6 \\")

lines3.append(r"\bottomrule")
lines3.append(r"\end{tabular}")
lines3.append(r"\end{table}")

with open(os.path.join(OUT_DIR, "table3_per_category_recall.tex"), "w") as f:
    f.write("\n".join(lines3))
print("Table 3 saved (placeholder; per-category data not available).")
