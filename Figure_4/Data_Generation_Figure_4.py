"""
Data_Generation_Figure_4.py

Thin wrapper around Anton's band_analysis_cd.py (Stage 1): scans raw
per-trial force data, extracts RMS vibration amplitude in each of the 5
frequency bands over the final WINDOW_SEC (=2.0 s, matching manuscript
Methods) of contact, and computes per-subject Spearman correlations
between band amplitude and perceptual rating. Writes band_stats.csv,
which Figure_Generation_Figure_4.py (Stage 2, pooled) reads.

This is kept as a separate wrapper (rather than editing band_analysis_cd.py's
CLI directly) so the Data-Generation / Figure-Generation split matches the
convention used for every other figure in this package. band_analysis_cd.py
itself is left runnable standalone too (it has its own __main__/argparse).
"""

import os
import re
import sys

sys.path.append(os.path.dirname(__file__))
import band_analysis_cd as bac  # noqa: E402


def generate_figure4_data(session_path, output_path, cv=None, skip_pairs=None):
    """
    Parameters
    ----------
    session_path : str
        Path to the TextureForce Data directory containing
        SubjectN_SessionM folders (each with a 'Force' subfolder and
        'reports.csv').
    output_path : str
        Directory to write band_stats.csv into.
    cv : None, 'loo', or int
        Cross-validation mode for the per-subject Spearman stats
        (passed through to band_analysis_cd.stage1). Default: None
        (standard Spearman, no CV) -- matches the manuscript, which
        does not describe cross-validating the Figure 4 correlations
        (unlike Figure 3 / Table 2, which explicitly do).
    skip_pairs : iterable of (Subject, Block) tuples, optional
        (participant, block) outliers to exclude, e.g. [('Subject7', 'H')].
        Block is the one-letter code 'H'/'S'/'R'. Default None keeps every
        participant (the published analysis). Used for the reliability
        sensitivity re-run.

    Returns
    -------
    None. Writes {output_path}/band_stats.csv
    """
    all_items = os.listdir(session_path)
    session_folders = sorted([
        f for f in all_items
        if (os.path.isdir(os.path.join(session_path, f))
            and re.match(r'^Subject\d+_Session\d+$', f)
            and os.path.exists(os.path.join(session_path, f, 'Force'))
            and os.path.exists(os.path.join(session_path, f, 'reports.csv')))
    ])
    print(f"Found {len(session_folders)} sessions in {session_path}")

    # A sensitivity run (skip_pairs set) writes to a parallel _no_outliers
    # directory so it never overwrites the primary Figure 4 outputs.
    if skip_pairs:
        output_path = output_path.rstrip('/\\') + '_no_outliers'
        os.makedirs(output_path, exist_ok=True)

    bac.stage1(session_path, output_path, session_folders, cv=cv,
               skip_pairs=skip_pairs)
    print(f"Saved band_stats.csv to: {output_path}")


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import SESSION_DIR, OUTPUT_DIR

    output_path = os.path.join(OUTPUT_DIR, "Figure_4")
    os.makedirs(output_path, exist_ok=True)

    session_path = SESSION_DIR

    # (participant, block) reliability outliers to exclude for a sensitivity
    # re-run; block is the one-letter code H/S/R. Leave empty for the published
    # analysis. Populate from Data_Generation_Reliability.py's outlier printout,
    # e.g. [('Subject7', 'H'), ('Subject6', 'S')].
    outlier_pairs = [('Subject7', 'H'), ('Subject6', 'S')]
    generate_figure4_data(session_path, output_path, skip_pairs=outlier_pairs)
