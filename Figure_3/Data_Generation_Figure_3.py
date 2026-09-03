"""
Data_Generation_Figure_3.py

Builds the texture-level summary data behind manuscript Figure 3:
a 3 (predictor) x 3 (task) grid --
    rows:    A/D/G = Mean fingertip speed
             B/E/H = Max normal force
             C/F/I = Max tangential force
    columns: A-C = Hardness, D-F = Slipperiness, G-I = Roughness

For every (block, predictor) pair, this script:
    1. z-scores the predictor and the rating within each subject (subject's
       own mean/std across all of that subject's trials in that block).
    2. Averages to one value per (subject, texture), then averages across
       subjects to get one point per texture (N=14), with SEM computed
       across subjects (matches Fig 3 caption: "Whiskers indicate standard
       error of the mean computed on inter-subject variability").
    3. Computes the in-sample Pearson r between the N=14 texture means,
       the leave-one-out cross-validated pseudo-R2, and a permutation
       p-value (1000 shuffles) -- see shared/regression_utils.py.

Note: unlike an earlier draft that picked a different "top 3" predictor set
per block, the manuscript figure uses the SAME three predictors for every
block (fingertip speed, max normal force, max tangential force); only the
task-specific pattern of significance differs.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from regression_utils import (  # noqa: E402
    zscore_per_subject, texture_level_mean_sem,
    cv_r2_and_permutation_p, pearson_r
)

PREDICTORS = ['Mean_Speed_Keypoint_8', 'Max_Force_Fz', 'Max_Force_2D_Tangential_Vector']
BLOCKS = ['H', 'S', 'R']
RESPONSE = 'Normalized Rating'


def generate_figure3_data(force_df, image_df, base_path, n_perm=1000, seed=0,
                          skip_pairs=None):
    """
    Parameters
    ----------
    force_df : pandas.DataFrame
        Must contain 'Session', 'Block Type', 'Trial', 'Texture',
        'Max_Force_Fz', 'Max_Force_2D_Tangential_Vector', 'Normalized Rating'
        (one row per trial, or per-bin rows that share the same per-trial
        value -- either works since we average within subject x texture).
    image_df : pandas.DataFrame
        Must contain 'Session', 'Block Type', 'Trial', 'Texture',
        'Mean_Speed_Keypoint_8'.
    base_path : str
        Output directory. Writes {base_path}/Figure_3/Data_Figure_3.csv
        and {base_path}/Figure_3/Data_Figure_3_stats.csv
    skip_pairs : iterable of (Subject, Block) tuples, optional
        (participant, block) outliers to drop before the analysis, e.g.
        [('Subject7', 'H')]. Block is the one-letter code 'H'/'S'/'R'.
        Default None keeps every participant (the published analysis). Used
        for the reliability sensitivity re-run.

    Returns
    -------
    plot_df : pandas.DataFrame
        One row per (Block, Predictor, Texture) with Mean/SEM Predictor,
        Mean/SEM Rating.
    stats_df : pandas.DataFrame
        One row per (Block, Predictor) with r, cv_r2, p_value, n_textures.
    """
    force_df = force_df.copy()
    image_df = image_df.copy()
    force_df['Session'] = force_df['Session'].astype(str).str.replace('_preprocessed$', '', regex=True)
    image_df['Session'] = image_df['Session'].astype(str).str.replace('_preprocessed$', '', regex=True)

    force_df['Subject'] = force_df['Session'].str.extract(r'(Subject\d+)')
    image_df['Subject'] = image_df['Session'].str.extract(r'(Subject\d+)')

    plot_rows = []
    stats_rows = []

    # Collapse each dataframe to one row per (Subject, Trial, Texture) first.
    # Source CSVs are bin-level (many rows per trial); averaging the relevant
    # columns down to one value per trial BEFORE z-scoring per subject avoids
    # implicitly over-weighting subjects/trials with more bins.
    force_trial_cols = ['Subject', 'Trial', 'Texture', 'Block Type', RESPONSE] + \
        [c for c in ['Max_Force_Fz', 'Max_Force_2D_Tangential_Vector'] if c in force_df.columns]
    force_trial = (
        force_df[force_trial_cols]
        .groupby(['Subject', 'Trial', 'Texture', 'Block Type'], as_index=False)
        .mean(numeric_only=True)
    )

    image_trial_cols = ['Subject', 'Trial', 'Texture', 'Block Type'] + \
        [c for c in ['Mean_Speed_Keypoint_8'] if c in image_df.columns]
    image_trial = (
        image_df[image_trial_cols]
        .groupby(['Subject', 'Trial', 'Texture', 'Block Type'], as_index=False)
        .mean(numeric_only=True)
    )

    # Drop manually-entered (participant, block) outlier pairs before any
    # averaging, so an excluded participant contributes nothing to that block.
    if skip_pairs:
        skip = {(str(s), str(b)) for s, b in skip_pairs}

        def _drop_pairs(df):
            keep = np.array([(s, b) not in skip
                             for s, b in zip(df['Subject'], df['Block Type'])])
            return df[keep]

        n0f, n0i = len(force_trial), len(image_trial)
        force_trial = _drop_pairs(force_trial)
        image_trial = _drop_pairs(image_trial)
        print(f"Skipping outlier pairs {sorted(skip)}: "
              f"force trials {n0f}->{len(force_trial)}, "
              f"image trials {n0i}->{len(image_trial)}")

    for block in BLOCKS:
        f_block = force_trial[force_trial['Block Type'] == block].copy()
        i_block = image_trial[image_trial['Block Type'] == block].copy()

        # z-score rating and force predictors within subject (force_df carries the rating too)
        f_block = zscore_per_subject(f_block, 'Subject', RESPONSE, out_col='_rating_z')

        for predictor in PREDICTORS:
            if predictor in f_block.columns:
                src = zscore_per_subject(f_block, 'Subject', predictor, out_col='_pred_z')
                rating_col = '_rating_z'
            elif predictor in i_block.columns:
                src = zscore_per_subject(i_block, 'Subject', predictor, out_col='_pred_z')
                # image_df may not carry rating directly; merge it in from force_df at trial level
                if '_rating_z' not in src.columns:
                    rating_lookup = f_block[['Subject', 'Texture', 'Trial', '_rating_z']]
                    src = src.merge(rating_lookup, on=['Subject', 'Texture', 'Trial'], how='left')
                rating_col = '_rating_z'
            else:
                print(f"Warning: predictor '{predictor}' not found in either dataframe, skipping.")
                continue

            pred_tex = texture_level_mean_sem(src, 'Texture', 'Subject', '_pred_z')
            rating_tex = texture_level_mean_sem(src, 'Texture', 'Subject', rating_col)

            merged = pred_tex.join(rating_tex, lsuffix='_pred', rsuffix='_rating', how='inner').reset_index()
            merged = merged.rename(columns={'Texture': 'Texture'})

            for _, row in merged.iterrows():
                plot_rows.append({
                    'Block': block,
                    'Predictor': predictor,
                    'Texture': row['Texture'],
                    'Mean Predictor': row['mean_pred'],
                    'SEM Predictor': row['sem_pred'],
                    'Mean Rating': row['mean_rating'],
                    'SEM Rating': row['sem_rating'],
                })

            valid = merged.dropna(subset=['mean_pred', 'mean_rating'])
            r = pearson_r(valid['mean_rating'], valid['mean_pred'])
            cv_r2, p_value = cv_r2_and_permutation_p(
                valid['mean_rating'].values, valid['mean_pred'].values,
                n_perm=n_perm, seed=seed
            )

            stats_rows.append({
                'Block': block,
                'Predictor': predictor,
                'r': r,
                'cv_r2': cv_r2,
                'p_value': p_value,
                'n_textures': len(valid)
            })

    plot_df = pd.DataFrame(plot_rows)
    stats_df = pd.DataFrame(stats_rows)

    # A sensitivity run (skip_pairs set) writes to a parallel _no_outliers
    # folder so it never overwrites the primary Figure 3 outputs.
    suffix = '_no_outliers' if skip_pairs else ''
    out_dir = os.path.join(base_path, "Figure_3" + suffix)
    os.makedirs(out_dir, exist_ok=True)
    plot_df.to_csv(os.path.join(out_dir, "Data_Figure_3.csv"), index=False)
    stats_df.to_csv(os.path.join(out_dir, "Data_Figure_3_stats.csv"), index=False)
    print(f"Saved Figure 3 data + stats to: {out_dir}")

    return plot_df, stats_df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import BINNED_FORCE_CSV, BINNED_IMAGE_CSV, OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    force_df = pd.read_csv(BINNED_FORCE_CSV)
    image_df = pd.read_csv(BINNED_IMAGE_CSV)

    # (participant, block) reliability outliers to exclude for a sensitivity
    # re-run; block is the one-letter code H/S/R. Leave empty for the published
    # analysis. Populate from Data_Generation_Reliability.py's outlier printout,
    # e.g. [('Subject7', 'H'), ('Subject6', 'S')].
    outlier_pairs = [('Subject7', 'H'), ('Subject6', 'S')]
    generate_figure3_data(force_df, image_df, OUTPUT_DIR,
                          skip_pairs=outlier_pairs)
