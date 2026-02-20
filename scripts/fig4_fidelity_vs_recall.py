#!/usr/bin/env python3
"""
Figure 4: Recall-fidelity tradeoff scatter plot.
X-axis: Tier 1 Recall (%). Y-axis: Tier 3 Similarity (output fidelity, 0-1).
Dashed lines at 95% recall and 0.85 similarity.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = "/Users/jacweath/Desktop/Dissertation/AIMS/AIM_1/slm-phi-deid-evaluation/data/slm-phi-deid-evaluation/results/analysis/final_results.json"
OUT_DIR = "/Users/jacweath/Desktop/mc-output/slm-paper/figures"

MODEL_COLORS = {
    "Gemma 3 4B": "#4A90D9",
    "Llama 3.2 3B": "#E74C3C",
    "Phi-4 Mini": "#27AE60",
}

STRATEGY_CATEGORIES = {
    "Zero-Shot Minimal": "zero-shot",
    "Zero-Shot Structured": "zero-shot",
    "Structured (Aggressive)": "zero-shot",
    "Few-Shot": "few-shot",
    "Two-Pass (Programmatic)": "two-pass",
    "Two-Pass (LLM)": "two-pass",
}

CATEGORY_MARKERS = {
    "zero-shot": "o",
    "few-shot": "^",
    "two-pass": "s",
}

CATEGORY_LABELS = {
    "zero-shot": "Zero-Shot Variants",
    "few-shot": "Few-Shot",
    "two-pass": "Two-Pass Variants",
}

with open(RESULTS_PATH) as f:
    data = json.load(f)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

# 180mm two-column width
fig, ax = plt.subplots(figsize=(7.09, 5.5))

# --- Add jitter for dense clusters ---
np.random.seed(42)

# --- Plot all points ---
for entry in data:
    model_label = entry["model_label"]
    strat_label = entry["strategy_label"]
    recall = entry["tier1_recall_mean"] * 100
    similarity = entry["tier3_similarity_mean"]
    cat = STRATEGY_CATEGORIES[strat_label]

    color = MODEL_COLORS[model_label]
    marker = CATEGORY_MARKERS[cat]

    # Small jitter for overlapping points
    jx = np.random.uniform(-0.4, 0.4)
    jy = np.random.uniform(-0.008, 0.008)

    ax.scatter(
        recall + jx, similarity + jy,
        c=color, marker=marker, s=75, edgecolors='black',
        linewidths=0.5, zorder=5, alpha=0.85,
    )

# --- Threshold lines ---
ax.axvline(x=95, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

# Shade ideal quadrant (high recall AND high fidelity)
ax.fill_between(
    [95, 100], 0.85, 1.0, alpha=0.08, color='green', zorder=1
)

# "Ideal Region" label at top of shaded zone, centered within 95-100 band
ax.text(
    97, 0.94, "Ideal Region",
    ha='center', va='center', fontsize=8, fontweight='bold',
    color='#1a6b1a', alpha=0.7, zorder=3,
)

# --- Annotate Gemma few-shot (best balanced config) ---
for entry in data:
    if entry["model_label"] == "Gemma 3 4B" and entry["strategy_label"] == "Few-Shot":
        r = entry["tier1_recall_mean"] * 100
        s = entry["tier3_similarity_mean"]
        ax.annotate(
            "Gemma 3 4B\n(Few-Shot)",
            xy=(r, s),
            xytext=(r - 25, s - 0.06),
            fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8,
                            connectionstyle='arc3,rad=0.15'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#999999', alpha=0.95),
            zorder=6,
        )

# --- Annotate Gemma two-pass outliers (high recall, destroyed fidelity) ---
gemma_twopass = [
    e for e in data
    if e["model_label"] == "Gemma 3 4B"
    and e["strategy_label"] in ("Two-Pass (Programmatic)", "Two-Pass (LLM)")
]
if gemma_twopass:
    # Average position of the two nearly-overlapping points
    avg_r = np.mean([e["tier1_recall_mean"] * 100 for e in gemma_twopass])
    avg_s = np.mean([e["tier3_similarity_mean"] for e in gemma_twopass])
    ax.annotate(
        "Gemma Two-Pass:\nfidelity collapsed",
        xy=(avg_r, avg_s),
        xytext=(70, 0.22),
        fontsize=8,
        arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8,
                        connectionstyle='arc3,rad=0.15'),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD',
                  edgecolor='#D4A017', alpha=0.95),
        zorder=6,
    )

# --- Legends in lower-left (clear area) ---
model_handles = []
for label, color in MODEL_COLORS.items():
    model_handles.append(
        mpatches.Patch(facecolor=color, edgecolor='black',
                       linewidth=0.5, label=label)
    )

strat_handles = []
for cat, marker in CATEGORY_MARKERS.items():
    strat_handles.append(
        plt.Line2D([0], [0], marker=marker, color='gray',
                   markerfacecolor='gray', markersize=7,
                   linestyle='None', label=CATEGORY_LABELS[cat])
    )

first_legend = ax.legend(
    handles=model_handles, title="Model", loc='lower left',
    framealpha=0.95, edgecolor='#cccccc', fontsize=8, title_fontsize=9,
)
ax.add_artist(first_legend)
ax.legend(
    handles=strat_handles, title="Strategy",
    loc='lower left', bbox_to_anchor=(0.20, 0.0),
    framealpha=0.95, edgecolor='#cccccc', fontsize=8, title_fontsize=9,
)

# X-axis is percentage (0-100), Y-axis is similarity score (0-1)
ax.set_xlabel("Tier 1 Recall (%)")
ax.set_ylabel("Tier 3 Similarity (Output Fidelity)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.0)
ax.set_title("Recall vs. Output Fidelity Tradeoff")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_fidelity_vs_recall.pdf", bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig4_fidelity_vs_recall.png", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")
