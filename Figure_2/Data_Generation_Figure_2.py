"""
Data_Generation_Figure_2.py

Builds the per-subject, per-task summary data behind manuscript Figure 2:
    A) Adjusted median normal force per subject, Hardness / Slipperiness / Roughness
    B) Adjusted median tangential force per subject, same 3 tasks
    C) Coefficient of variation of normal force per subject, same 3 tasks
    D) Coefficient of variation of tangential force per subject, same 3 tasks

Per Methods: "Median force is normalized by z-scoring each subject's median
force with the subject's average force across all tasks." I.e. the z-score
baseline (mean/std) for a subject is computed across ALL of that subject's
trials (all 3 blocks combined), then each block's median is expressed in
those z-score units. This differs from normalizing separately within each
block.

Output: one row per (Subject, Block) with 4 metrics, for all three blocks
(H, S, R) -- unlike earlier drafts that only carried H and S through to the
plot.
"""

import os
import numpy as np
import pandas as pd


def generate_figure2_data(force_df, base_path):
    """
    Parameters
    ----------
    force_df : pandas.DataFrame
        Binned force dataset. Required columns:
        'Session', 'Block Type', 'Bin Median Force Fz',
        'Bin Median Force 2D_Tangential_Vector',
        'Bin_Force_CV_0.2s_Fz', 'Bin_Force_CV_0.2s_2D_Tangential_Vector'.
    base_path : str
        Path to the Manuscript_Figures-equivalent output directory.
        Output written to {base_path}/Figure_2/Data_Figure_2.csv

    Returns
    -------
    df_final : pandas.DataFrame
        One row per (Subject, Block) with:
        ['Subject', 'Block', 'Normalized Median Fz', 'Normalized Median Tangential',
         'CV Fz', 'CV Tangential']
    """
    required_cols = [
        'Session', 'Block Type', 'Bin Median Force Fz',
        'Bin Median Force 2D_Tangential_Vector',
        'Bin_Force_CV_0.2s_Fz', 'Bin_Force_CV_0.2s_2D_Tangential_Vector'
    ]
    missing = [c for c in required_cols if c not in force_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in force_df: {missing}")

    force_df = force_df.copy()
    force_df['Session'] = force_df['Session'].astype(str).str.replace('_preprocessed$', '', regex=True)

    subject_list = sorted(
        force_df['Session'].str.extract(r'Subject(\d+)')[0].dropna().unique(),
        key=lambda x: int(x)
    )
    subject_list = [f'Subject{x}' for x in subject_list]

    blocks = ['H', 'S', 'R']
    results = []

    for subject in subject_list:
        sub_df = force_df[force_df['Session'].str.contains(subject + '_')]
        if sub_df.empty:
            continue

        # z-score baseline: across ALL blocks for this subject
        fz_all = sub_df['Bin Median Force Fz'].dropna()
        tan_all = sub_df['Bin Median Force 2D_Tangential_Vector'].dropna()
        fz_mean, fz_std = fz_all.mean(), fz_all.std()
        tan_mean, tan_std = tan_all.mean(), tan_all.std()

        for block in blocks:
            block_df = sub_df[sub_df['Block Type'] == block]
            if block_df.empty:
                results.append({
                    'Subject': subject, 'Block': block,
                    'Normalized Median Fz': np.nan,
                    'Normalized Median Tangential': np.nan,
                    'CV Fz': np.nan,
                    'CV Tangential': np.nan
                })
                continue

            fz_med = block_df['Bin Median Force Fz'].median()
            tan_med = block_df['Bin Median Force 2D_Tangential_Vector'].median()
            fz_cv = block_df['Bin_Force_CV_0.2s_Fz'].mean()
            tan_cv = block_df['Bin_Force_CV_0.2s_2D_Tangential_Vector'].mean()

            results.append({
                'Subject': subject,
                'Block': block,
                'Normalized Median Fz': ((fz_med - fz_mean) / fz_std) if fz_std > 0 else np.nan,
                'Normalized Median Tangential': ((tan_med - tan_mean) / tan_std) if tan_std > 0 else np.nan,
                'CV Fz': fz_cv,
                'CV Tangential': tan_cv
            })

    df_final = pd.DataFrame(results)

    output_path = os.path.join(base_path, "Figure_2", "Data_Figure_2.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Saved Figure 2 data to: {output_path}")

    return df_final


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import BINNED_FORCE_CSV, OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    force_df = pd.read_csv(BINNED_FORCE_CSV)
    generate_figure2_data(force_df, OUTPUT_DIR)
