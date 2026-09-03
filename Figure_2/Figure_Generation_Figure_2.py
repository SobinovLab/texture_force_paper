"""
Figure_Generation_Figure_2.py

Renders manuscript Figure 2: a 2x2 grid of per-subject strip plots
(A: normal force, B: tangential force, C: CV normal, D: CV tangential),
each showing all three tasks (Hardness, Slipperiness, Roughness) with a
black median bar per task, and three pairwise significance brackets
(H-S, H-R, S-R) annotated with Wilcoxon signed-rank p and rank-biserial r,
matching the layout in the manuscript (e.g. Fig 2A: "p<0.001 r=0.82" spanning
H-R, "p=0.001 r=0.76" spanning H-S, "p=0.27" spanning S-R).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from regression_utils import wilcoxon_p_and_rank_biserial_r, format_p  # noqa: E402


BLOCK_ORDER = ['H', 'S', 'R']
BLOCK_LABELS = {'H': 'Hardness', 'S': 'Slipperiness', 'R': 'Roughness'}
# Pairs to annotate, and their vertical stacking order (innermost pair drawn lowest)
PAIRS = [('H', 'S'), ('S', 'R'), ('H', 'R')]


def _draw_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color='black')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=9)


def _panel(ax, df, metric, label, ylabel):
    pivot = df.pivot(index='Subject', columns='Block', values=metric)
    pivot = pivot[[b for b in BLOCK_ORDER if b in pivot.columns]]

    plot_df = pivot.reset_index().melt(id_vars='Subject', var_name='Block', value_name='Value')

    sns.stripplot(
        data=plot_df, x='Block', y='Value', order=BLOCK_ORDER,
        color='black', size=5, jitter=0.15, ax=ax
    )

    for i, block in enumerate(BLOCK_ORDER):
        med = pivot[block].median()
        ax.plot([i - 0.2, i + 0.2], [med, med], lw=3, color='black')

    ymax = plot_df['Value'].max()
    ymin = plot_df['Value'].min()
    span = ymax - ymin if ymax > ymin else 1.0
    step = span * 0.12

    for level, (b1, b2) in enumerate(PAIRS):
        if b1 not in pivot.columns or b2 not in pivot.columns:
            continue
        p, r = wilcoxon_p_and_rank_biserial_r(pivot[b1], pivot[b2])
        x1, x2 = BLOCK_ORDER.index(b1), BLOCK_ORDER.index(b2)
        y = ymax + step * (level + 1)
        r_text = "r=NA" if np.isnan(r) else f"r={r:.2f}"
        _draw_bracket(ax, x1, x2, y, step * 0.3, f"{format_p(p)}, {r_text}")

    ax.set_ylim(ymin - step * 0.5, ymax + step * (len(PAIRS) + 1.5))
    ax.set_xticks(range(len(BLOCK_ORDER)))
    ax.set_xticklabels([BLOCK_LABELS[b] for b in BLOCK_ORDER])
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_title(label, loc='left', weight='bold')
    ax.spines[['top', 'right']].set_visible(False)


def plot_figure2(base_path):
    data_path = os.path.join(base_path, "Figure_2", "Data_Figure_2.csv")
    df = pd.read_csv(data_path)

    fig, axs = plt.subplots(2, 2, figsize=(11, 10))

    _panel(axs[0, 0], df, 'Normalized Median Fz', 'A', 'Adjusted Median Force (z)')
    _panel(axs[0, 1], df, 'Normalized Median Tangential', 'B', 'Adjusted Median Force (z)')
    _panel(axs[1, 0], df, 'CV Fz', 'C', 'Coefficient of Variation')
    _panel(axs[1, 1], df, 'CV Tangential', 'D', 'Coefficient of Variation')

    plt.tight_layout()

    out_dir = os.path.join(base_path, "Figure_2")
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f"Figure_2.{ext}"), dpi=300, bbox_inches='tight')
    print(f"Saved Figure 2 to: {out_dir}")

    return fig


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import OUTPUT_DIR

    plot_figure2(OUTPUT_DIR)
    plt.show()
