"""
Data_Generation_Figure_5.py

Builds the trial-level dataset behind manuscript Figure 5:
    A) Normalized slipperiness rating vs. friction coefficient (per trial)
    B) Normalized roughness rating vs. friction coefficient (per trial)

Pipeline:
    1. friction_utils.compute_all_trial_events() detects swipe events per
       trial and computes an event-level dynamic friction coefficient
       (Ft/Fn at each event's peak-velocity index, averaged over 3 samples).
    2. THIS script collapses events to one value per (Session, Trial,
       Texture): the MEDIAN dynamic friction coefficient across that
       trial's events -- matching the Methods wording, "we computed the
       median ratio of tangential to normal force as an estimate of the
       coefficient of friction," and matching the paper's reported N
       (190 slipperiness trials, 204 roughness trials -- i.e. trial-level,
       not event-level or texture-averaged).
    3. Merges with each trial's Normalized Rating from the binned force
       data.

Requires raw per-session data (SubjectX_SessionY/Force/trial_N.csv,
Images/trial_N/..., reports.csv) in addition to the two binned CSVs --
this cannot run on the binned CSVs alone.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from friction_utils import compute_all_trial_events  # noqa: E402


def generate_figure5_data(binned_image_data, binned_force_data, data_folder, base_path,
                           subject_min=1, subject_max=17):
    """
    Returns
    -------
    trial_df : pandas.DataFrame
        One row per (Session, Trial, Texture, Block) with:
        'Dynamic Friction Coefficient' (median across that trial's events),
        'N Events' (how many events contributed), 'Rating'.
        Saved to {base_path}/Figure_5/Data_Figure_5.csv
    """
    event_df = compute_all_trial_events(
        binned_image_data, binned_force_data, data_folder,
        subject_min=subject_min, subject_max=subject_max, blocks=('R', 'S')
    )

    if event_df.empty:
        raise RuntimeError("No friction events detected -- check data_folder path and raw file layout.")

    # --- Collapse events -> one row per trial (median across events) ---
    trial_friction = (
        event_df.groupby(['Session', 'Trial', 'Texture', 'Block'], as_index=False)
        .agg(**{
            'Dynamic Friction Coefficient': ('Dynamic Friction Coefficient', 'median'),
            'N Events': ('Dynamic Friction Coefficient', 'size'),
        })
    )

    # --- Merge in ratings from the binned force data ---
    rating_df = (
        binned_force_data
        .assign(Session=lambda d: d['Session'].str.replace('_preprocessed', '', regex=True))
        .groupby(['Session', 'Trial', 'Texture', 'Block Type'])['Normalized Rating']
        .apply(lambda x: x.iloc[0])
        .reset_index()
        .rename(columns={'Normalized Rating': 'Rating', 'Block Type': 'Block'})
    )

    trial_df = pd.merge(
        trial_friction, rating_df,
        on=['Session', 'Trial', 'Texture', 'Block'], how='inner'
    )

    out_dir = os.path.join(base_path, "Figure_5")
    os.makedirs(out_dir, exist_ok=True)
    trial_df.to_csv(os.path.join(out_dir, "Data_Figure_5.csv"), index=False)

    n_s = (trial_df['Block'] == 'S').sum()
    n_r = (trial_df['Block'] == 'R').sum()
    print(f"Saved Figure 5 data to: {out_dir}")
    print(f"  N trials -- Slipperiness: {n_s}, Roughness: {n_r} "
          f"(paper reports N=190 and N=204 respectively)")

    return trial_df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import BINNED_FORCE_CSV, BINNED_IMAGE_CSV, SESSION_DIR, OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    binned_force_data = pd.read_csv(BINNED_FORCE_CSV)
    binned_image_data = pd.read_csv(BINNED_IMAGE_CSV)

    generate_figure5_data(binned_image_data, binned_force_data, SESSION_DIR,
                          OUTPUT_DIR)
