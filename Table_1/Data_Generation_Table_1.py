"""
Data_Generation_Table_1.py

Computes the raw summary statistics behind manuscript Table 1: for every
(Texture, Task, Force Type) combination, the median, IQR, and 99th
percentile of force. This mirrors the previously-working pipeline (found
under the "Table_2_Correct" folder in earlier drafts -- the content was
correct, just filed under the wrong table number) with no logic changes,
split here into a data-generation step (this file, raw long-form stats)
and a formatting step (Table_Generation_Table_1.py, the wide display table).
"""

import os
import numpy as np
import pandas as pd


def generate_table1_stats(force_df, base_path):
    """
    Parameters
    ----------
    force_df : pandas.DataFrame
        Must contain 'Texture', 'Block Type', 'Bin Median Force Fz',
        'Bin Median Force 2D_Tangential_Vector'.
    base_path : str
        Output directory. Writes {base_path}/Table_1/Data_Table_1_long.csv

    Returns
    -------
    agg : pandas.DataFrame
        Long-form: one row per (Texture, Task, Force Type) with
        ['median', 'iqr', 'p99'].
    """
    required_cols = [
        'Texture', 'Block Type', 'Bin Median Force Fz',
        'Bin Median Force 2D_Tangential_Vector'
    ]
    missing = [c for c in required_cols if c not in force_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    block_map = {'H': 'Hardness', 'R': 'Roughness', 'S': 'Slipperiness',
                 'Hardness': 'Hardness', 'Roughness': 'Roughness', 'Slipperiness': 'Slipperiness'}
    df = force_df.copy()
    df['Task'] = df['Block Type'].map(block_map).fillna(df['Block Type'])

    force_map = {'Normal': 'Bin Median Force Fz', 'Tangential': 'Bin Median Force 2D_Tangential_Vector'}
    long_force = []
    for f_label, f_col in force_map.items():
        tmp = df[['Texture', 'Task', f_col]].rename(columns={f_col: 'Value'}).copy()
        tmp['Force Type'] = f_label
        long_force.append(tmp)
    long_force = pd.concat(long_force, ignore_index=True).dropna(subset=['Value'])

    def _iqr(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 2:
            return np.nan
        q75, q25 = np.percentile(x, [75, 25])
        return q75 - q25

    def _p99(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        return np.percentile(x, 99) if x.size else np.nan

    agg = (
        long_force
        .groupby(['Texture', 'Task', 'Force Type'])
        .agg(median=('Value', 'median'), iqr=('Value', _iqr), p99=('Value', _p99))
        .reset_index()
    )

    out_dir = os.path.join(base_path, "Table_1")
    os.makedirs(out_dir, exist_ok=True)
    agg.to_csv(os.path.join(out_dir, "Data_Table_1_long.csv"), index=False)
    print(f"Saved Table 1 raw stats to: {out_dir}")

    return agg


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import BINNED_FORCE_CSV, OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    force_df = pd.read_csv(BINNED_FORCE_CSV)
    generate_table1_stats(force_df, OUTPUT_DIR)
