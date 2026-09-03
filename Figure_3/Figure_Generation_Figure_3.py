"""
Figure_Generation_Figure_3.py

Renders manuscript Figure 3: a 3x3 grid.
    Columns = Hardness, Slipperiness, Roughness (A-C, D-F, G-I)
    Rows    = Mean fingertip speed | Max normal force | Max tangential force

Each panel: texture-level mean +/- SEM (both axes), an OLS trend line fit
on the texture-level means (for display only), and an annotation with the
in-sample r, cross-validated pseudo-r2, and permutation p-value pulled from
Data_Figure_3_stats.csv (computed in Data_Generation_Figure_3.py).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

BLOCKS = ['H', 'S', 'R']
BLOCK_LABELS = {'H': 'Hardness', 'S': 'Slipperiness', 'R': 'Roughness'}
PREDICTORS = ['Mean_Speed_Keypoint_8', 'Max_Force_Fz', 'Max_Force_2D_Tangential_Vector']
YLABELS = {
    'Mean_Speed_Keypoint_8': 'Normalized Mean Fingertip Speed',
    'Max_Force_Fz': 'Normalized Max Normal Force',
    'Max_Force_2D_Tangential_Vector': 'Normalized Max Tangential Force',
}
# Column-major lettering so Hardness (A-C) is the first column, Slipperiness
# (D-F) the second, Roughness (G-I) the third.
PANEL_LABELS = [
    ['A', 'D', 'G'],
    ['B', 'E', 'H'],
    ['C', 'F', 'I'],
]


def _format_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = NA"
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def plot_figure3(base_path, suffix=''):
    """
    suffix : '' for the primary figure, '_no_outliers' to render the reliability
        sensitivity run written by generate_figure3_data(skip_pairs=...).
    """
    fig_dir = os.path.join(base_path, "Figure_3" + suffix)
    data_path = os.path.join(fig_dir, "Data_Figure_3.csv")
    stats_path = os.path.join(fig_dir, "Data_Figure_3_stats.csv")
    df = pd.read_csv(data_path)
    stats_df = pd.read_csv(stats_path)

    fig, axs = plt.subplots(3, 3, figsize=(13, 13))

    for col, block in enumerate(BLOCKS):
        for row, predictor in enumerate(PREDICTORS):
            ax = axs[row, col]
            sub = df[(df['Block'] == block) & (df['Predictor'] == predictor)]
            stat_row = stats_df[(stats_df['Block'] == block) & (stats_df['Predictor'] == predictor)]

            if sub.empty:
                ax.axis('off')
                continue

            ax.errorbar(
                sub['Mean Rating'], sub['Mean Predictor'],
                xerr=sub['SEM Rating'], yerr=sub['SEM Predictor'],
                fmt='o', ecolor='gray', color='black',
                capsize=3, markersize=6, linestyle='none'
            )

            valid = sub.dropna(subset=['Mean Rating', 'Mean Predictor'])
            if len(valid) >= 2:
                fit = linregress(valid['Mean Rating'], valid['Mean Predictor'])
                x_vals = np.linspace(valid['Mean Rating'].min(), valid['Mean Rating'].max(), 100)
                y_vals = fit.slope * x_vals + fit.intercept
                ax.plot(x_vals, y_vals, color='black')

            if not stat_row.empty:
                r = stat_row['r'].iloc[0]
                cv_r2 = stat_row['cv_r2'].iloc[0]
                p_value = stat_row['p_value'].iloc[0]
                r_text = "r=NA" if pd.isna(r) else f"r={r:.2f}"
                r2_text = "r2=NA" if pd.isna(cv_r2) else f"r2={cv_r2:.2f}"
                ax.text(
                    0.95, 0.95, f"{_format_p(p_value)}\n{r2_text}\n{r_text}",
                    color='red', ha='right', va='top',
                    transform=ax.transAxes, fontsize=10
                )

            label = PANEL_LABELS[row][col]
            ax.set_title(f"{label}) {BLOCK_LABELS[block]}", loc='left', weight='bold', fontsize=12)
            ax.set_xlabel("Normalized Rating", fontsize=10)
            ax.set_ylabel(YLABELS[predictor], fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=9)

    plt.tight_layout()

    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(fig_dir, f"Figure_3.{ext}"), dpi=300, bbox_inches='tight')
    print(f"Saved Figure 3 to: {fig_dir}")

    return fig


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import OUTPUT_DIR

    # For the reliability sensitivity run, set suffix='_no_outliers' to match
    # generate_figure3_data(skip_pairs=...).
    plot_figure3(OUTPUT_DIR, suffix='_no_outliers')
    plt.show()
