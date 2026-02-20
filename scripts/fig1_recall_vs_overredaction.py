#!/usr/bin/env python3
"""
Figure 1: Tier 1 Recall vs. Over-redaction scatter plot.
Each point is one model/strategy combination (18 total).
Color by model, shape by strategy category.
Dashed lines at 95% recall and 10% over-redaction define the target quadrant.
Y-axis 0-100, X-axis 0-100.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Configuration ---
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

# --- Load data ---
with open(RESULTS_PATH) as f:
    data = json.load(f)

# --- Style ---
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

# 180mm two-column width = 7.09 inches
fig, ax = plt.subplots(figsize=(7.09, 5.5))

# --- Reproducible jitter for overlapping points ---
np.random.seed(42)

best_dist = float('inf')
best_point = None
best_entry = None

for entry in data:
    model_label = entry["model_label"]
    strat_label = entry["strategy_label"]
    recall = entry["tier1_recall_mean"] * 100
    over_redact = entry["over_redaction"] * 100
    cat = STRATEGY_CATEGORIES[strat_label]

    color = MODEL_COLORS[model_label]
    marker = CATEGORY_MARKERS[cat]

    # Small jitter to separate overlapping points in dense clusters
    jx = np.random.uniform(-1.2, 1.2)
    jy = np.random.uniform(-0.4, 0.4)

    ax.scatter(
        over_redact + jx, recall + jy,
        c=color, marker=marker, s=70, edgecolors='black',
        linewidths=0.5, zorder=5, alpha=0.85,
    )

    # Track point closest to the target corner (over_redact=10, recall=95)
    dist = np.sqrt((over_redact - 10)**2 + (recall - 95)**2)
    if dist < best_dist:
        best_dist = dist
        best_point = (over_redact + jx, recall + jy)
        best_entry = (model_label, strat_label)

# --- Target threshold lines ---
ax.axhline(y=95, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
ax.axvline(x=10, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

# --- Shade target quadrant ---
ax.fill_between(
    [0, 10], 95, 100, alpha=0.08, color='green', zorder=1
)
ax.text(
    5, 97.5, "Target\nRegion",
    ha='center', va='center', fontsize=9, fontweight='bold',
    color='#1a6b1a', alpha=0.8, zorder=3,
)

# --- Annotate closest point with clean arrow ---
if best_point and best_entry:
    bx, by = best_point
    bmodel, bstrat = best_entry
    ax.annotate(
        f"{bmodel}\n({bstrat})",
        xy=(bx, by),
        xytext=(bx + 12, by - 8),
        fontsize=8,
        arrowprops=dict(
            arrowstyle='->', color='#333333', lw=0.8,
            connectionstyle='arc3,rad=0.2',
        ),
        bbox=dict(
            boxstyle='round,pad=0.3', facecolor='white',
            edgecolor='#999999', alpha=0.95,
        ),
        zorder=6,
    )

# --- Legends: lower-left area (well clear of data with Y 0-100) ---
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

# Model legend at lower-left
first_legend = ax.legend(
    handles=model_handles, title="Model", loc='lower left',
    framealpha=0.95, edgecolor='#cccccc', fontsize=8, title_fontsize=9,
)
ax.add_artist(first_legend)

# Strategy legend shifted right of model legend
ax.legend(
    handles=strat_handles, title="Strategy",
    loc='lower left', bbox_to_anchor=(0.20, 0.0),
    framealpha=0.95, edgecolor='#cccccc', fontsize=8, title_fontsize=9,
)

# --- Axis formatting: both axes 0 to 100 ---
ax.set_xlabel("Over-redaction (%)")
ax.set_ylabel("Tier 1 Recall (%)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_title("Recall vs. Over-redaction Across Model/Strategy Configurations")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_recall_vs_overredaction.pdf", bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig1_recall_vs_overredaction.png", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")
