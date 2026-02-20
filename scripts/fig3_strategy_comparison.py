#!/usr/bin/env python3
"""
Figure 3: Grouped bar chart comparing Tier 1 Recall across strategies.
X-axis: 6 strategies. For each, 3 bars (one per model).
Horizontal dashed line at 95%. Error bars from per-run std.
Y-axis: 0 to 100. Value labels inside bars as white bold text.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "/Users/jacweath/Desktop/Dissertation/AIMS/AIM_1/slm-phi-deid-evaluation/data/slm-phi-deid-evaluation/results/analysis/final_results.json"
OUT_DIR = "/Users/jacweath/Desktop/mc-output/slm-paper/figures"

MODEL_COLORS = {
    "Gemma 3 4B": "#4A90D9",
    "Llama 3.2 3B": "#E74C3C",
    "Phi-4 Mini": "#27AE60",
}

STRATEGY_ORDER = [
    "Zero-Shot Minimal",
    "Zero-Shot Structured",
    "Structured (Aggressive)",
    "Few-Shot",
    "Two-Pass (Programmatic)",
    "Two-Pass (LLM)",
]

STRATEGY_SHORT = [
    "ZS\nMinimal",
    "ZS\nStructured",
    "Structured\n(Aggressive)",
    "Few-Shot",
    "Two-Pass\n(Prog.)",
    "Two-Pass\n(LLM)",
]

MODEL_ORDER = ["Gemma 3 4B", "Llama 3.2 3B", "Phi-4 Mini"]

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

# Build lookup: (model_label, strategy_label) -> entry
lookup = {}
for entry in data:
    lookup[(entry["model_label"], entry["strategy_label"])] = entry

n_strategies = len(STRATEGY_ORDER)
n_models = len(MODEL_ORDER)
x = np.arange(n_strategies)
bar_width = 0.25

# 180mm width, taller aspect ratio
fig, ax = plt.subplots(figsize=(7.09, 6))

all_bars = []
all_recalls = []

for i, model in enumerate(MODEL_ORDER):
    recalls = []
    stds = []
    for strat in STRATEGY_ORDER:
        entry = lookup.get((model, strat))
        if entry:
            recalls.append(entry["tier1_recall_mean"] * 100)
            stds.append(entry["tier1_recall_std"] * 100)
        else:
            recalls.append(0)
            stds.append(0)

    offset = (i - 1) * bar_width
    bars = ax.bar(
        x + offset, recalls, bar_width,
        label=model, color=MODEL_COLORS[model],
        yerr=stds, capsize=2, edgecolor='black', linewidth=0.4,
        error_kw={'linewidth': 0.8, 'capthick': 0.6},
        zorder=3,
    )
    all_bars.append(bars)
    all_recalls.append(recalls)

# --- Value labels INSIDE bars as white bold text ---
for bars, recalls in zip(all_bars, all_recalls):
    for bar, val in zip(bars, recalls):
        if val > 0:
            # Place label inside bar near the top
            label_y = bar.get_height() - 1.5
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f'{val:.1f}',
                ha='center', va='top', fontsize=8, rotation=90,
                fontweight='bold', color='white',
                zorder=4,
            )

# 95% threshold
ax.axhline(
    y=95, color='#555555', linestyle='--', linewidth=1.3,
    alpha=0.7, zorder=2, label='95% Threshold',
)

ax.set_xlabel("Prompting Strategy")
ax.set_ylabel("Tier 1 Recall (%)")
ax.set_title("Tier 1 Recall by Strategy and Model")
ax.set_xticks(x)
ax.set_xticklabels(STRATEGY_SHORT, fontsize=8)
ax.set_ylim(0, 100)
ax.legend(loc='lower left', framealpha=0.95, edgecolor='#cccccc', fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_strategy_comparison.pdf", bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig3_strategy_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")
