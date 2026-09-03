"""
Figure_Generation_Figure_4.py

Thin wrapper around Anton's band_analysis_cd.py (stage2_pooled): renders
the 3x1 pooled-violin figure (one row per condition -- Hardness,
Slipperiness, Roughness -- since WINDOWS = ['last'] only). Each panel shows
one violin per frequency band, subject-level Spearman rho distribution,
group median bar colored red if the pooled Wilcoxon signed-rank test
(vs. rho=0) is significant at alpha, with optional Benjamini-Hochberg
correction across the 5 bands within each condition.

Reads band_stats.csv produced by Data_Generation_Figure_4.py.
"""

import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
import band_analysis_cd as bac  # noqa: E402


def plot_figure4(output_path, alpha=0.05, adjust=True, suffix=''):
    """
    Parameters
    ----------
    output_path : str
        Directory containing band_stats.csv (same as Data_Generation_Figure_4's
        output_path). Figures are saved alongside it as
        rating_vs_power_pooled_<Condition>_last.{png,svg} (per band_analysis_cd's
        internal naming) -- actual filenames are printed to stdout.
    alpha : float
        Significance threshold for the pooled Wilcoxon test (default 0.05).
    adjust : bool
        Whether to apply Benjamini-Hochberg FDR correction across the 5
        frequency bands within each condition (recommended; matches
        "corrected using Benjamini-Hochberg for 5 frequency bands" in the
        Figure 4 caption).
    suffix : str
        '' for the primary figure, '_no_outliers' to render the reliability
        sensitivity run written by generate_figure4_data(skip_pairs=...).
    """
    if suffix:
        output_path = output_path.rstrip('/\\') + suffix
    bac.stage2_pooled(output_path, alpha=alpha, adjust=adjust)
    print(f"Saved Figure 4 (pooled violin) to: {output_path}")


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import OUTPUT_DIR

    output_path = os.path.join(OUTPUT_DIR, "Figure_4")
    # For the reliability sensitivity run, set suffix='_no_outliers' to match
    # generate_figure4_data(skip_pairs=...).
    plot_figure4(output_path, suffix='_no_outliers')
    plt.show()
