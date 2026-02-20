#!/usr/bin/env python3
"""
Figure 2: Three-tier evaluation heatmap.
Rows: model x strategy (18 configs), sorted by T1 Recall descending.
Columns: T1 Recall, T2 Recall, T2 Precision, T3 Similarity.
Color scale: red (0) -> yellow (0.5) -> green (1.0).

Fixes applied:
  - Column headers now all include (%) for consistency
  - Cell annotation fontsize bumped to 9pt for print readability
  - Figure width set to 180mm (7.09 in)
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

RESULTS_PATH = "/Users/jacweath/Desktop/Dissertation/AIMS/AIM_1/slm-phi-deid-evaluation/data/slm-phi-deid-evaluation/results/analysis/final_results.json"
OUT_DIR = "/Users/jacweath/Desktop/mc-output/slm-paper/figures"

with open(RESULTS_PATH) as f:
    data = json.load(f)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'figure.dpi': 300,
})

# --- Build matrix ---
# All column headers include (%) since values are displayed as percentages
columns = ["T1 Recall\n(%)", "T2 Recall\n(%)", "T2 Precision\n(%)", "T3 Similarity\n(%)"]
rows = []
matrix = []

for entry in data:
    label = f"{entry['model_label']} / {entry['strategy_label']}"
    vals = [
        entry["tier1_recall_mean"],
        entry["tier2_recall_mean"],
        entry["tier2_precision_mean"],
        entry["tier3_similarity_mean"],
    ]
    rows.append(label)
    matrix.append(vals)

matrix = np.array(matrix)

# Sort by T1 Recall descending
sort_idx = np.argsort(-matrix[:, 0])
matrix = matrix[sort_idx]
rows = [rows[i] for i in sort_idx]

# --- Custom colormap: red -> yellow -> green ---
cmap = mcolors.LinearSegmentedColormap.from_list(
    "ryg",
    [(0.0, "#d73027"), (0.5, "#fee08b"), (1.0, "#1a9850")],
)

# 180mm width = 7.09 in; tall enough for 18 rows
fig, ax = plt.subplots(figsize=(7.09, 9))

im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

# --- Annotate cells (9pt for print readability, minimum 8pt met) ---
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        text_color = "white" if val < 0.35 or val > 0.85 else "black"
        ax.text(j, i, f"{val*100:.1f}%", ha='center', va='center',
                fontsize=9, fontweight='bold', color=text_color)

# --- Axis labels ---
ax.set_xticks(np.arange(len(columns)))
ax.set_xticklabels(columns, fontsize=10, fontweight='bold')
ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows, fontsize=8.5)

ax.set_xlabel("")
ax.set_title("Three-Tier Evaluation Across All Configurations", fontsize=13, pad=12)

# Move x-axis labels to top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# --- Gridlines ---
ax.set_xticks(np.arange(len(columns)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
ax.grid(which='minor', color='white', linewidth=1.5)
ax.grid(which='major', visible=False)
ax.tick_params(which='minor', length=0)

# --- Colorbar ---
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
cbar.set_label("Score", fontsize=11)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_three_tier_heatmap.pdf", bbox_inches='tight')
plt.savefig(f"{OUT_DIR}/fig2_three_tier_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")
