"""
Data_Generation_Table_2.py

Builds manuscript Table 2: for each of 8 predictors, independently
(NOT stepwise / not best-subset), regressed against Normalized Rating,
separately for each of the 3 tasks. Reports the leave-one-out
cross-validated pseudo-R2 and a one-sided permutation-test p-value
(1000 shuffles) -- see shared/regression_utils.py, which is the exact
same implementation used for Figure 3, so the two are guaranteed
consistent.

This replaces an earlier draft (Table_3_correct / Cross_validated_data_
generation_table_3.py) that used forward stepwise / best-subset selection
across multiple predictors -- that doesn't match the manuscript, which
reports each of the 8 predictors as its own independent single-predictor
model (Table 2: "the table lists the predictors, the cross-validated
pseudo-R2 of the model, and one-sided permutation test p-value").

The 8 predictors (manuscript naming -> column naming):
    Index Fingertip Speed, Mean   -> Mean_Speed_Keypoint_8      (image_df)
    Index Fingertip Speed, Max    -> Max_Speed_Keypoint_8       (image_df)
    Normal Force, Median          -> Median_Force_Fz            (force_df)
    Normal Force, Max             -> Max_Force_Fz               (force_df)
    Normal Force Rate, Median     -> Median_Force_Rate_Fz       (force_df)
    Tan. Force, Median            -> Median_Force_2D_Tangential_Vector
    Tan. Force, Max               -> Max_Force_2D_Tangential_Vector
    Tan. Force Rate, Median       -> Median_Force_Rate_2D_Tangential_Vector
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from regression_utils import (  # noqa: E402
    zscore_per_subject, texture_level_mean_sem, cv_r2_and_permutation_p
)

BLOCKS = ['H', 'S', 'R']
RESPONSE = 'Normalized Rating'

PREDICTOR_TYPE = {
    'Mean_Speed_Keypoint_8': 'kinematics',
    'Max_Speed_Keypoint_8': 'kinematics',
    'Median_Force_Fz': 'normal',
    'Max_Force_Fz': 'normal',
    'Median_Force_Rate_Fz': 'normal',
    'Median_Force_2D_Tangential_Vector': 'tangential',
    'Max_Force_2D_Tangential_Vector': 'tangential',
    'Median_Force_Rate_2D_Tangential_Vector': 'tangential',
}
PREDICTORS = list(PREDICTOR_TYPE.keys())


def generate_table2_data(force_df, image_df, base_path, n_perm=1000, seed=0,
                         skip_pairs=None):
    """
    Returns
    -------
    stats_df : pandas.DataFrame
        One row per (Block, Regressor) with columns:
        ['Block', 'Regressor', 'Type', 'cv_r2', 'p_value', 'n_textures']
        Saved to {base_path}/Table_2/Data_Table_2_long.csv
    """
    force_df = force_df.copy()
    image_df = image_df.copy()
    force_df['Session'] = force_df['Session'].astype(str).str.replace('_preprocessed$', '', regex=True)
    image_df['Session'] = image_df['Session'].astype(str).str.replace('_preprocessed$', '', regex=True)
    force_df['Subject'] = force_df['Session'].str.extract(r'(Subject\d+)')
    image_df['Subject'] = image_df['Session'].str.extract(r'(Subject\d+)')

    # Collapse to one row per (Subject, Trial, Texture) before z-scoring,
    # for the same reason as in Figure 3: source CSVs are bin-level.
    force_predictor_cols = [c for c in PREDICTORS if c in force_df.columns]
    force_trial = (
        force_df[['Subject', 'Trial', 'Texture', 'Block Type', RESPONSE] + force_predictor_cols]
        .groupby(['Subject', 'Trial', 'Texture', 'Block Type'], as_index=False)
        .mean(numeric_only=True)
    )

    image_predictor_cols = [c for c in PREDICTORS if c in image_df.columns]
    image_trial = (
        image_df[['Subject', 'Trial', 'Texture', 'Block Type'] + image_predictor_cols]
        .groupby(['Subject', 'Trial', 'Texture', 'Block Type'], as_index=False)
        .mean(numeric_only=True)
    )

    # Drop (participant, block) reliability outliers before averaging, so an
    # excluded participant contributes nothing to that block's texture means.
    if skip_pairs:
        skip = {(str(s), str(b)) for s, b in skip_pairs}

        def _drop_pairs(df):
            keep = [(str(s), str(b)) not in skip
                    for s, b in zip(df['Subject'], df['Block Type'])]
            return df[keep]

        force_trial = _drop_pairs(force_trial)
        image_trial = _drop_pairs(image_trial)

    stats_rows = []

    for block in BLOCKS:
        f_block = force_trial[force_trial['Block Type'] == block].copy()
        i_block = image_trial[image_trial['Block Type'] == block].copy()

        f_block = zscore_per_subject(f_block, 'Subject', RESPONSE, out_col='_rating_z')

        for predictor in PREDICTORS:
            if predictor in f_block.columns:
                src = zscore_per_subject(f_block, 'Subject', predictor, out_col='_pred_z')
                rating_col = '_rating_z'
            elif predictor in i_block.columns:
                src = zscore_per_subject(i_block, 'Subject', predictor, out_col='_pred_z')
                rating_lookup = f_block[['Subject', 'Texture', 'Trial', '_rating_z']]
                src = src.merge(rating_lookup, on=['Subject', 'Texture', 'Trial'], how='left')
                rating_col = '_rating_z'
            else:
                print(f"Warning: predictor '{predictor}' not found for block {block}, skipping.")
                continue

            pred_tex = texture_level_mean_sem(src, 'Texture', 'Subject', '_pred_z')
            rating_tex = texture_level_mean_sem(src, 'Texture', 'Subject', rating_col)
            merged = pred_tex.join(rating_tex, lsuffix='_pred', rsuffix='_rating', how='inner')
            valid = merged.dropna(subset=['mean_pred', 'mean_rating'])

            cv_r2, p_value = cv_r2_and_permutation_p(
                valid['mean_rating'].values, valid['mean_pred'].values,
                n_perm=n_perm, seed=seed
            )

            stats_rows.append({
                'Block': block,
                'Regressor': predictor,
                'Type': PREDICTOR_TYPE[predictor],
                'cv_r2': cv_r2,
                'p_value': p_value,
                'n_textures': len(valid)
            })

    stats_df = pd.DataFrame(stats_rows)

    out_dir = os.path.join(base_path, "Table_2")
    os.makedirs(out_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(out_dir, "Data_Table_2_long.csv"), index=False)
    print(f"Saved Table 2 raw stats to: {out_dir}")

    return stats_df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import BINNED_FORCE_CSV, BINNED_IMAGE_CSV, OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    force_df = pd.read_csv(BINNED_FORCE_CSV)
    image_df = pd.read_csv(BINNED_IMAGE_CSV)

    # (participant, block) reliability outliers excluded by default (from
    # Data_Generation_Reliability.py). Set to [] to keep every participant.
    outlier_pairs = [('Subject7', 'H'), ('Subject6', 'S')]
    generate_table2_data(force_df, image_df, OUTPUT_DIR, skip_pairs=outlier_pairs)
