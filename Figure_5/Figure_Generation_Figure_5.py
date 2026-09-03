"""
Figure_Generation_Figure_5.py

Renders manuscript Figure 5:
    A) Normalized slipperiness rating (y) vs. friction coefficient (x)
    B) Normalized roughness rating (y) vs. friction coefficient (x)
One point per trial, plain black dots (no per-texture coloring, to match
the manuscript's plain scatter style), with a Theil-Sen regression line
and Spearman rho/p annotated -- matching the robust-regression choice
already used elsewhere in your friction pipeline (friction coefficients
are ratios and can be heavy-tailed, so a rank-based/robust approach was
kept rather than switching to OLS/Pearson).

NOTE on axes: earlier drafts plotted Rating (x) vs. Friction (y). The
manuscript figure has Friction on the x-axis and Rating on the y-axis --
flipped here to match.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import TheilSenRegressor

PANELS = [('S', 'Slipperiness', 'A'), ('R', 'Roughness', 'B')]


def plot_figure5(base_path):
    data_path = os.path.join(base_path, "Figure_5", "Data_Figure_5.csv")
    df = pd.read_csv(data_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, (block, label, panel) in zip(axes, PANELS):
        sub = df[df['Block'] == block].dropna(subset=['Dynamic Friction Coefficient', 'Rating'])
        if sub.empty:
            ax.axis('off')
            continue

        x = sub['Dynamic Friction Coefficient'].values
        y = sub['Rating'].values

        ax.scatter(x, y, s=30, color='black', alpha=0.7)

        model = TheilSenRegressor(random_state=0)
        model.fit(x.reshape(-1, 1), y)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = model.predict(x_fit.reshape(-1, 1))
        ax.plot(x_fit, y_fit, color='black', linewidth=1.5)

        rho, p = stats.spearmanr(x, y)
        p_text = "p < 0.0001" if p < 0.0001 else f"p = {p:.2g}"
        ax.text(
            0.95, 0.95, f"r = {rho:.2f}\n{p_text}\nN = {len(sub)} trials",
            transform=ax.transAxes, ha='right', va='top', fontsize=11, color='red'
        )

        ax.set_title(f"{panel}) {label}", loc='left', weight='bold', fontsize=13)
        ax.set_xlabel("Friction Coefficient", fontsize=11)
        ax.set_ylabel(f"Normalized {label} Rating", fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    out_dir = os.path.join(base_path, "Figure_5")
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f"Figure_5.{ext}"), dpi=300, bbox_inches='tight')
    print(f"Saved Figure 5 to: {out_dir}")

    return fig


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import OUTPUT_DIR

    plot_figure5(OUTPUT_DIR)
    plt.show()
