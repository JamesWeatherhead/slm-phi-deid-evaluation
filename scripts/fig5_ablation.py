#!/usr/bin/env python3
"""
Figure 5: Ablation study comparing Structured (conservative) vs. Structured (Aggressive).
Two subplots: left = T1 Recall, right = Over-redaction.
Paired bars for each model.
Y-axes: 0 to 100 on both subplots.
Value labels INSIDE bars as white bold text, rotated 90 degrees to avoid overlap.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "/Users/jacweath/Desktop/Dissertation/AIMS/AIM_1/slm-phi-deid-evaluation/data/slm-phi-deid-evaluation/results/analysis/final_results.json"
OUT_DIR = "/Users/jacweath/Desktop/mc-output/slm-paper/figures"

MODEL_ORDER = ["Gemma 3 4B", "Llama 3.2 3B", "Phi-4 Mini"]

with open(RESULTS_PATH) as f:
    data = json.load(f)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

# Build lookup
lookup = {}
for entry in data:
    lookup[(entry["model_label"], entry["strategy_label"])] = entry

conservative_label = "Zero-Shot Structured"
aggressive_label = "Structured (Aggressive)"

# 180mm width, taller for breathing room
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 5.5))

x = np.arange(len(MODEL_ORDER))
bar_width = 0.32

# --- Left subplot: T1 Recall ---
cons_recall = []
aggr_recall = []
cons_recall_std = []
aggr_recall_std = []

for model in MODEL_ORDER:
    c = lookup.get((model, conservative_label))
    a = lookup.get((model, aggressive_label))
    cons_recall.append(c["tier1_recall_mean"] * 100 if c else 0)
    aggr_recall.append(a["tier1_recall_mean"] * 100 if a else 0)
    cons_recall_std.append(c["tier1_recall_std"] * 100 if c else 0)
    aggr_recall_std.append(a["tier1_recall_std"] * 100 if a else 0)

bars1 = ax1.bar(
    x - bar_width/2, cons_recall, bar_width,
    label='Structured (Conservative)',
    color='#5DADE2', edgecolor='black', linewidth=0.4,
    yerr=cons_recall_std, capsize=4,
    error_kw={'linewidth': 1.0, 'capthick': 0.8},
    zorder=3,
)
bars2 = ax1.bar(
    x + bar_width/2, aggr_recall, bar_width,
    label='Structured (Aggressive)',
    color='#F39C12', edgecolor='black', linewidth=0.4,
    yerr=aggr_recall_std, capsize=4,
    error_kw={'linewidth': 1.0, 'capthick': 0.8},
    zorder=3,
)

# 95% threshold: clearly visible with label on the left
ax1.axhline(y=95, color='#555555', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
ax1.text(
    -0.45, 95.8, '95%',
    ha='left', va='bottom', fontsize=8, color='#555555', fontweight='bold',
)

ax1.set_ylabel("Tier 1 Recall (%)")
ax1.set_title("(a) Tier 1 Recall")
ax1.set_xticks(x)
ax1.set_xticklabels(MODEL_ORDER, fontsize=8)
ax1.set_ylim(0, 100)
# Legend in center area (well below bars which are 96-99%)
ax1.legend(loc='center left', bbox_to_anchor=(0.0, 0.35),
           framealpha=0.95, edgecolor='#cccccc', fontsize=8)

# Value labels INSIDE recall bars (white bold text, rotated 90 degrees)
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h - 2,
             f'{h:.1f}%', ha='center', va='top', fontsize=8,
             fontweight='bold', color='white', rotation=90, zorder=4)
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h - 2,
             f'{h:.1f}%', ha='center', va='top', fontsize=8,
             fontweight='bold', color='white', rotation=90, zorder=4)

# --- Right subplot: Over-redaction ---
cons_over = []
aggr_over = []

for model in MODEL_ORDER:
    c = lookup.get((model, conservative_label))
    a = lookup.get((model, aggressive_label))
    cons_over.append(c["over_redaction"] * 100 if c else 0)
    aggr_over.append(a["over_redaction"] * 100 if a else 0)

bars3 = ax2.bar(
    x - bar_width/2, cons_over, bar_width,
    label='Structured (Conservative)',
    color='#5DADE2', edgecolor='black', linewidth=0.4,
    zorder=3,
)
bars4 = ax2.bar(
    x + bar_width/2, aggr_over, bar_width,
    label='Structured (Aggressive)',
    color='#F39C12', edgecolor='black', linewidth=0.4,
    zorder=3,
)

# 10% threshold: prominent line with label on the left
ax2.axhline(y=10, color='#C0392B', linestyle='--', linewidth=2.0, alpha=0.9, zorder=4)
ax2.text(
    -0.45, 11.5, '10%',
    ha='left', va='bottom', fontsize=8, color='#C0392B', fontweight='bold',
)

ax2.set_ylabel("Over-redaction (%)")
ax2.set_title("(b) Over-redaction")
ax2.set_xticks(x)
ax2.set_xticklabels(MODEL_ORDER, fontsize=8)
ax2.set_ylim(0, 100)
# Legend in center area (well below bars which are 89-100%)
ax2.legend(loc='center left', bbox_to_anchor=(0.0, 0.35),
           framealpha=0.95, edgecolor='#cccccc', fontsize=8)

# Value labels INSIDE over-redaction bars (white bold text, rotated 90 degrees)
for bar in bars3:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h - 2,
             f'{h:.1f}%', ha='center', va='top', fontsize=8,
             fontweight='bold', color='white', rotation=90, zorder=4)
for bar in bars4:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h - 2,
             f'{h:.1f}%', ha='center', va='top', fontsize=8,
             fontweight='bold', color='white', rotation=90, zorder=4)

fig.suptitle(
    "Ablation: Conservative vs. Aggressive Uncertainty Instruction",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_ablation.pdf", bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig5_ablation.png", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved.")
