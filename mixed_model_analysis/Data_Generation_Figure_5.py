#!python3.11
"""
Data_Generation_Figure_5.py -- control analysis for Figure 5 (friction).

The manuscript concludes that friction is strongly related to perceived
slipperiness but not to perceived roughness. Rather than reporting that one
correlation was significant and the other was not, this script tests the
stronger, direct claim -- that the two relationships DIFFER -- with a single
mixed-effects model carrying a Friction x Task interaction:

    rating ~ friction_z * Task + (1 + friction_z || Subject) + (1 | Texture)

Task is Slipperiness vs Roughness (the two blocks in which friction is
estimated), coded as the dummy `Slippery` (1 = slipperiness, 0 = roughness), so
the reference slope is roughness and the interaction `friction_z:Slippery` is
exactly "how much steeper the friction-rating slope is in slipperiness than in
roughness". A significant interaction is the direct statistical support for the
claim -- substantially stronger than "one correlation was significant".

Friction measure -- matches how Figure 5 was actually made
(Manuscript_Figures/Correct_Folder_Names/Table_1_correct/Multiprocessing.py).
That pipeline estimates the dynamic friction coefficient mu = Ft/Fn at every
in-contact sample, bins it by exploration speed (50 bins over 0-100 cm/s,
`Speed mu` in dynamic_friction_long_table.csv), and the friction-vs-rating
relationship (`generate_friction_relationships` -> `mu_rating_at_speed_range`,
final_range = [0, 30] cm/s) uses the mean mu over the 0-30 cm/s bins. This
speed-controlled estimate is used here as the friction predictor, so the
friction is already matched for exploration speed -- which is the cleanest
control, because dynamic friction is measured during movement and exploration
velocity covaries with perceived slipperiness. `Multiprocessing.py` also imports
`reporting_pool`/`prehension`, which are not in this repository, so its numeric
constants (speed bins, 0-30 cm/s window) are reproduced here rather than
imported; the Tukey-IQR outlier option below mirrors its `outliers_iqr`.

Models (all with crossed Subject/Texture random effects and an uncorrelated
Subject random slope for friction, the `||`):

    1. Friction x Task                       -- the primary interaction test.
    2. Friction x Task + Speed               -- sensitivity: exploration speed
       (mean fingertip speed) added as a covariate, on top of the already
       speed-controlled friction.
    3. (friction_between + friction_within) x Task -- robustness. Friction is
       texture-intrinsic (mostly between-texture); a plain (1|Texture) model can
       absorb a between-texture predictor (as it did for the Figure 4 roughness
       bands), so friction is split into its texture mean (friction_between) and
       trial deviation (friction_within), with the key term
       friction_between:Slippery.

Each row reports the friction slope in roughness (reference), the slope in
slipperiness (linear combination), the interaction with SE/CI/p, the
Subject/Texture/slope variance components, their ICCs, and marginal/conditional
R^2. Pass --drop_outliers to remove Tukey-IQR friction outliers within block,
matching the paper's main-text figure (no_outliers=True).

Inputs:
    dynamic_friction_long_table.csv   from Multiprocessing.py (Session, Trial,
                                      Texture, Block, Dynamic Friction
                                      Coefficient, Speed mu, ...)
    binned force CSV                  for the per-trial Normalized Rating
    binned image CSV                 for exploration speed (sensitivity model)

Outputs (into <base_path>/Figure_5_control/):
    Data_Figure_5_interaction.csv   one row per model
    Data_Figure_5_variance.csv      full variance components for each model
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from revision_utils import (  # noqa: E402
    collapse_bins_to_trials, ensure_dir, OUTLIER_PAIRS, drop_outlier_pairs,
)
from Data_Generation_MixedModels import (  # noqa: E402
    fit_crossed, variance_summary, nakagawa_r2, _pick_variance, _term_row,
    zscore,
)

SPEED_COL = 'Mean_Speed_Keypoint_8'
TASKS = {'S': 'Slipperiness', 'R': 'Roughness'}   # blocks with friction

# Reproduced from Multiprocessing.py (DN_SPEED_MIN_BORDER=0, MAX=100, NUMBINS=50;
# final_range = [0, 30] cm/s in generate_friction_relationships).
SPEED_BINS = np.linspace(0, 100, 50)
FRICTION_SPEED_RANGE = (0.0, 30.0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _speed_controlled_friction(speed_mu_str):
    """Mean mu over the 0-30 cm/s speed bins, as Figure 5 uses it."""
    vals = np.array([float(v) for v in str(speed_mu_str).split()
                     if v != ''], dtype=float)
    if vals.size != SPEED_BINS.size:
        return np.nan
    sel = ((SPEED_BINS > FRICTION_SPEED_RANGE[0]) &
           (SPEED_BINS <= FRICTION_SPEED_RANGE[1]))
    band = vals[sel]
    return float(np.nanmean(band)) if np.any(np.isfinite(band)) else np.nan


def load_friction_trials(long_table_csv, force_csv, image_csv):
    """
    Trial-level table: Subject, Texture, Block Type (S/R), Friction
    (speed-controlled, 0-30 cm/s), rating, and Mean_Speed_Keypoint_8.
    """
    fr = pd.read_csv(long_table_csv)
    fr['Session'] = (fr['Session'].astype(str)
                     .str.replace('_preprocessed$', '', regex=True))
    fr['Subject'] = fr['Session'].str.extract(r'(Subject\d+)')
    fr['Friction'] = fr['Speed mu'].apply(_speed_controlled_friction)
    fr = fr.rename(columns={'Block': 'Block Type'})
    fr = fr[fr['Block Type'].isin(TASKS)].copy()
    fr['Trial'] = fr['Trial'].astype(int)
    fr['Texture'] = fr['Texture'].astype(float).astype(int)

    # Block-specific per-trial ratings, exactly as Multiprocessing.py merges them.
    force = pd.read_csv(force_csv)
    force['Session'] = (force['Session'].astype(str)
                        .str.replace('_preprocessed$', '', regex=True))
    rating = (force.groupby(['Session', 'Trial', 'Texture', 'Block Type'])
                   ['Normalized Rating'].first().reset_index()
                   .rename(columns={'Normalized Rating': 'rating'}))
    rating['Trial'] = rating['Trial'].astype(int)
    rating['Texture'] = rating['Texture'].astype(float).astype(int)

    d = fr.merge(rating, on=['Session', 'Trial', 'Texture', 'Block Type'],
                 how='inner')
    if d.empty:
        print('WARNING: friction<->rating merge produced 0 rows -- check that '
              'Session naming and the Trial/Texture key dtypes match between the '
              'long table and the binned force CSV')

    speed = collapse_bins_to_trials(pd.read_csv(image_csv), [SPEED_COL])
    speed['Trial'] = speed['Trial'].astype(int)
    speed['Texture'] = speed['Texture'].astype(float).astype(int)
    d = d.merge(speed, on=['Subject', 'Trial', 'Texture', 'Block Type'],
                how='left')
    return d


def _drop_tukey_outliers(d, col='Friction', k=1.5):
    """Drop Tukey-IQR outliers of `col` within each block (cf. outliers_iqr)."""
    keep = pd.Series(True, index=d.index)
    for _, idx in d.groupby('Block Type').groups.items():
        x = pd.to_numeric(d.loc[idx, col], errors='coerce')
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        keep.loc[idx] = (x >= q1 - k * iqr) & (x <= q3 + k * iqr)
    return d[keep]


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def _combo(res, spec):
    """Stats for a linear combination of fixed-effect coefficients."""
    if res is None:
        return {'beta': np.nan, 'se': np.nan, 'z': np.nan, 'p': np.nan}
    fe = list(res.fe_params.index)
    if not all(t in fe for t in spec):
        return {'beta': np.nan, 'se': np.nan, 'z': np.nan, 'p': np.nan}
    c = np.zeros((1, len(fe)))
    for t, w in spec.items():
        c[0, fe.index(t)] = w
    test = res.t_test(c)
    return {'beta': float(np.squeeze(test.effect)),
            'se': float(np.squeeze(test.sd)),
            'z': float(np.squeeze(test.statistic)),
            'p': float(np.squeeze(test.pvalue))}


def summarize_interaction(res, vc_names, slope_term, label):
    """
    One summary row for a `rating ~ slope_term * Slippery [...]` model:
    roughness slope (reference), slipperiness slope (combo), the interaction,
    variance components, ICCs and R^2.
    """
    inter = f'{slope_term}:Slippery'
    row = {'Model': label,
           'N trials': int(res.nobs) if res is not None else np.nan}

    r = _term_row(res, slope_term, {})            # slope in roughness (ref)
    row.update({'beta_Roughness': r['beta'], 'se_Roughness': r['se'],
                'p_Roughness': r['p'], 'CI_R_low': r['CI low'],
                'CI_R_high': r['CI high']})

    s = _combo(res, {slope_term: 1.0, inter: 1.0})  # slope in slipperiness
    row.update({'beta_Slipperiness': s['beta'], 'se_Slipperiness': s['se'],
                'p_Slipperiness': s['p']})

    i = _term_row(res, inter, {})                 # the interaction (S - R)
    row.update({'beta_interaction': i['beta'], 'se_interaction': i['se'],
                'z_interaction': i['z'], 'p_interaction': i['p'],
                'CI_int_low': i['CI low'], 'CI_int_high': i['CI high']})

    vs = variance_summary(res, vc_names, label)
    row.update(_pick_variance(vs))
    row.update(nakagawa_r2(res))
    return row, vs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_figure5_control(trial_df, base_path, random_slope=True,
                             drop_outliers=False, outlier_pairs=None):
    d = trial_df.dropna(subset=['Friction', 'rating']).copy()
    d = d[d['Friction'] > 0]
    pairs = OUTLIER_PAIRS if outlier_pairs is None else outlier_pairs
    if pairs:
        n0 = len(d)
        d = drop_outlier_pairs(d, pairs)
        print(f'Excluded reliability outliers {pairs}: {n0}->{len(d)} trials '
              '(only Subject6/S applies to the slipperiness/roughness blocks)')
    if drop_outliers:
        n0 = len(d)
        d = _drop_tukey_outliers(d, 'Friction')
        print(f'Dropped {n0 - len(d)} Tukey-IQR friction outliers within block')
    d['friction_z'] = zscore(d['Friction'])
    d['Slippery'] = (d['Block Type'] == 'S').astype(int)
    d = d.dropna(subset=['friction_z'])

    n_s = int((d['Block Type'] == 'S').sum())
    n_r = int((d['Block Type'] == 'R').sum())
    print(f'Friction trials -- Slipperiness: {n_s}, Roughness: {n_r}, '
          f'subjects: {d["Subject"].nunique()}, textures: {d["Texture"].nunique()}')

    rows, var_rows = [], []
    slope = 'friction_z' if random_slope else None

    # 1. Primary interaction.
    print('\n[1] Friction x Task')
    res, vc = fit_crossed(d, 'rating ~ friction_z * Slippery',
                          random_slope_for=slope)
    row, vs = summarize_interaction(res, vc, 'friction_z', 'Friction x Task')
    rows.append(row); var_rows.append(vs)

    # 2. Sensitivity: additionally control for exploration speed.
    ds = d.dropna(subset=[SPEED_COL]).copy()
    if len(ds) >= 30 and ds[SPEED_COL].std() > 0:
        ds['speed_z'] = zscore(ds[SPEED_COL])
        print('\n[2] Friction x Task + Speed')
        res, vc = fit_crossed(ds, 'rating ~ friction_z * Slippery + speed_z',
                              random_slope_for=slope)
        row, vs = summarize_interaction(res, vc, 'friction_z',
                                        'Friction x Task + Speed')
        rows.append(row); var_rows.append(vs)
    else:
        print('\n[2] Speed unavailable; skipping the speed sensitivity model')

    # 3. Robustness: between/within-texture friction.
    d['friction_between'] = d.groupby('Texture')['friction_z'].transform('mean')
    d['friction_within'] = d['friction_z'] - d['friction_between']
    print('\n[3] (friction_between + friction_within) x Task')
    res, vc = fit_crossed(
        d, 'rating ~ (friction_between + friction_within) * Slippery',
        random_slope_for='friction_within' if random_slope else None)
    row, vs = summarize_interaction(res, vc, 'friction_between',
                                    'Between/within friction x Task')
    rows.append(row); var_rows.append(vs)

    out = pd.DataFrame(rows)
    out_dir = ensure_dir(os.path.join(base_path, 'Figure_5_control'))
    out.to_csv(os.path.join(out_dir, 'Data_Figure_5_interaction.csv'),
               index=False)
    pd.DataFrame(var_rows).to_csv(
        os.path.join(out_dir, 'Data_Figure_5_variance.csv'), index=False)
    print(f'\nSaved Figure 5 control analysis to: {out_dir}')

    show = ['Model', 'beta_Roughness', 'p_Roughness', 'beta_Slipperiness',
            'p_Slipperiness', 'beta_interaction', 'p_interaction']
    print(out[[c for c in show if c in out.columns]].to_string(index=False))
    print('\nKey statistic: p_interaction -- a significant Friction x Task '
          'interaction shows friction is related to slipperiness more than to '
          'roughness (stronger than "one correlation was significant").')
    return out


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import FRICTION_LONG_TABLE, BINNED_FORCE_CSV, BINNED_IMAGE_CSV, OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument('--long_table_csv', default=FRICTION_LONG_TABLE,
                    help='dynamic_friction_long_table.csv from Multiprocessing.py')
    ap.add_argument('--force_csv', default=BINNED_FORCE_CSV)
    ap.add_argument('--image_csv', default=BINNED_IMAGE_CSV)
    ap.add_argument('--base_path', default=OUTPUT_DIR)
    ap.add_argument('--drop_outliers', action='store_true',
                    help='remove Tukey-IQR friction outliers within block '
                         '(matches the paper main-text figure)')
    ap.add_argument('--no_random_slope', action='store_true')
    ap.add_argument('--keep_outliers', action='store_true',
                    help='keep the reliability outliers (default: exclude '
                         'Subject6/S, the only one in the S/R blocks)')
    args = ap.parse_args()

    trial_df = load_friction_trials(args.long_table_csv, args.force_csv,
                                    args.image_csv)
    generate_figure5_control(trial_df, args.base_path,
                             random_slope=not args.no_random_slope,
                             drop_outliers=args.drop_outliers,
                             outlier_pairs=[] if args.keep_outliers else None)
