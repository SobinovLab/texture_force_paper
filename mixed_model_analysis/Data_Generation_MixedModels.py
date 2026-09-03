#!python3.11
"""
Data_Generation_MixedModels.py -- editor point 4, participant- and texture-level
mixed-effects models.

Every model here is fit on TRIAL-level data with participant and texture as
crossed random intercepts, so nothing is averaged away before inference:

    Figure 3 / Table 2   rating ~ x_between + x_within + (1|Subject) + (1|Texture)
                         optional random slope: + (0 + x_within|Subject)
    Figure 4             same decomposition per band; the between-texture slope is
                         the Figure 4 question. BH over the 5 bands within block.
    Figure 2             log(force) ~ C(Block) + (1|Subject) + (1|Texture),
                         BH-corrected pairwise block contrasts
    Supp Figure 2        log(force) ~ 1 + (1|Subject) + (1|Texture); the subject
                         variance component and its likelihood-ratio test
                         replace the pseudoreplicated top/bottom-quartile ANOVA

Each per-model row is a compact summary: the fixed-effect slope(s) with SE, 95%
CI, Wald p and BH-adjusted p; the Subject / Texture / slope variance components
and their ICCs (fraction of the unexplained variance from each); and marginal
R^2 (fixed effects) and conditional R^2 (fixed + random). Full variance detail
for every model is also written to Data_Mixed_Variance.csv.

Conventions and their reasons:

* Each continuous predictor is z-scored across all trials of a block (grand z)
  and then SPLIT into a between-texture component (its texture mean, x_between)
  and a within-texture component (deviation from it, x_within). x_between answers
  the texture-level question ("textures eliciting more of the behaviour are rated
  differently") and is estimated alongside the texture random intercept rather
  than being absorbed by it; x_within answers the within-texture question ("for
  the same texture, more of the behaviour -> higher/lower rating"). Both slopes
  are in rating units per grand-SD of the predictor, so they are comparable. This
  split matters most for cues that live between textures (e.g. vibration and
  roughness): a plain single-slope model with a texture random intercept would
  null them out. Predictors are NOT z-scored within subject -- the subject random
  intercept does that job.
* The response for Figures 3/4 and Table 2 is `Normalized Rating` as stored
  (per-subject, per-block max-normalized). Subject intercepts absorb the residual
  scale differences.
* Force responses are log-transformed. Individual differences in applied force are
  multiplicative (the top four participants used ~5x the force of the bottom
  four), so a random intercept on the log scale absorbs them; on the raw scale it
  cannot.
* Crossed random effects are fit by giving statsmodels a single group containing
  all rows and declaring Subject and Texture as variance components. p-values on
  fixed effects are Wald z tests; they are anticonservative relative to
  Satterthwaite/Kenward-Roger at these sample sizes. Pass --engine pymer4 to get
  lme4/lmerTest Satterthwaite p-values instead, if pymer4 and R are installed.

Outputs (into <base_path>/MixedModels/):
    Data_Mixed_Figure_3.csv      one row per (Block, Predictor): between/within
                                 slopes (+BH), variance components, ICCs, R^2
    Data_Mixed_Table_2.csv       as Figure 3, all 8 predictors, BH within block
    Data_Mixed_Figure_4.csv      one row per (Block, Band): between/within slopes
                                 (+BH over 5 bands), variance components, ICCs, R^2
    Data_Mixed_Figure_2.csv      block contrasts per force measure, with the
                                 model's variance components, ICCs and R^2
    Data_Mixed_Variance.csv      full variance components and ICCs for every model
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from revision_utils import (  # noqa: E402
    bh_adjust, collapse_bins_to_trials, ensure_dir,
    OUTLIER_PAIRS, drop_outlier_pairs,
)

BLOCKS = ['H', 'S', 'R']
BLOCK_NAMES = {'H': 'Hardness', 'S': 'Slipperiness', 'R': 'Roughness'}
RESPONSE = 'Normalized Rating'

FIGURE_3_PREDICTORS = [
    'Mean_Speed_Keypoint_8',
    'Max_Force_Fz',
    'Max_Force_2D_Tangential_Vector',
]

# The eight trial-level features described in Methods / Table 2
TABLE_2_PREDICTORS = [
    'Median_Force_2D_Tangential_Vector',
    'Max_Force_2D_Tangential_Vector',
    'Absolute_Force_Rate_2D_Tangential_Vector',
    'Median_Force_Fz',
    'Max_Force_Fz',
    'Absolute_Force_Rate_Fz',
    'Mean_Speed_Keypoint_8',
    'Max_Speed_Keypoint_8',
]

FIGURE_2_RESPONSES = [
    'Median_Force_Fz',
    'Median_Force_2D_Tangential_Vector',
    'Force_CV_Fz',
    'Force_CV_2D_Tangential_Vector',
]


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_crossed(df, formula, random_slope_for=None, reml=True):
    """
    Fit `formula` with crossed Subject and Texture random intercepts.

    Returns (result, vc_names) or (None, None) if the fit fails. The single
    constant group plus vc_formula is the documented statsmodels idiom for
    crossed random effects.
    """
    d = df.copy()
    d['_all'] = 1
    d['Subject'] = d['Subject'].astype(str)
    d['Texture'] = d['Texture'].astype(str)

    vc = {'Subject': '0 + C(Subject)', 'Texture': '0 + C(Texture)'}
    if random_slope_for is not None:
        vc['Subject_slope'] = f'0 + C(Subject):{random_slope_for}'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            md = smf.mixedlm(formula, data=d, groups=d['_all'],
                             re_formula='0', vc_formula=vc)
            res = md.fit(reml=reml, method=['lbfgs', 'bfgs', 'powell'])
            return res, vc_names_of(md, vc)
        except Exception as exc:                                # noqa: BLE001
            print(f'    crossed fit failed ({exc})')
            if random_slope_for is not None:
                print('    retrying without the random slope')
                return fit_crossed(df, formula, random_slope_for=None,
                                   reml=reml)
            # Fallback: some statsmodels builds reject re_formula='0'. Use
            # Subject as the grouping factor with a random intercept and keep
            # Texture as a variance component. Note this nests Texture within
            # Subject instead of crossing them, so the texture variance is no
            # longer directly comparable; the printed warning flags it.
            print('    falling back to groups=Subject with Texture nested '
                  '(NOT crossed) -- report this model as approximate')
            try:
                md = smf.mixedlm(formula, data=d, groups=d['Subject'],
                                 re_formula='1',
                                 vc_formula={'Texture': '0 + C(Texture)'})
                res = md.fit(reml=reml, method=['lbfgs', 'bfgs', 'powell'])
                return res, vc_names_of(md, {'Texture': ''})
            except Exception as exc2:                           # noqa: BLE001
                print(f'    fallback also failed ({exc2})')
                return None, None


def vc_names_of(model, vc_formula):
    """
    Variance-component names in the order statsmodels stores them.

    Older statsmodels versions sort vc_formula keys while newer ones preserve
    insertion order, so read the names off the fitted model rather than trusting
    the dict order.
    """
    names = getattr(getattr(model, 'exog_vc', None), 'names', None)
    return list(names) if names else list(vc_formula.keys())


def variance_summary(res, vc_names, label):
    """Extract variance components, residual variance and ICCs from a fit."""
    if res is None:
        return {'Model': label}
    vcomp = np.atleast_1d(np.asarray(res.vcomp, float))
    out = {'Model': label, 'Residual var': float(res.scale),
           'logLik': float(res.llf), 'N obs': int(res.nobs)}
    total = float(res.scale)

    named = list(zip(vc_names or [], vcomp))
    # A random intercept fitted through `groups` (the fallback path) shows up in
    # cov_re rather than vcomp; fold it in under the grouping factor's name.
    if res.model.k_re > 0:
        named.insert(0, ('Subject', float(np.asarray(res.cov_re)[0, 0])))

    for name, v in named:
        out[f'Var {name}'] = float(v)
        total += float(v)
    for name, v in named:
        out[f'ICC {name}'] = float(v) / total if total > 0 else np.nan
    out['Total var'] = total
    return out


def lrt_drop_component(df, formula, drop='Subject'):
    """
    Likelihood-ratio test for one variance component, fixed effects held fixed.

    REML likelihoods are comparable here because the fixed-effect structure is
    identical between the two models. The chi2(1) p-value is halved because the
    null sits on the boundary of the parameter space (a 50:50 mixture of chi2(0)
    and chi2(1)).
    """
    d = df.copy()
    d['_all'] = 1
    d['Subject'] = d['Subject'].astype(str)
    d['Texture'] = d['Texture'].astype(str)

    full_vc = {'Subject': '0 + C(Subject)', 'Texture': '0 + C(Texture)'}
    reduced_vc = {k: v for k, v in full_vc.items() if k != drop}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            full = smf.mixedlm(formula, data=d, groups=d['_all'],
                               re_formula='0', vc_formula=full_vc
                               ).fit(reml=True, method=['lbfgs', 'bfgs'])
            red = smf.mixedlm(formula, data=d, groups=d['_all'],
                              re_formula='0', vc_formula=reduced_vc
                              ).fit(reml=True, method=['lbfgs', 'bfgs'])
        except Exception as exc:                                # noqa: BLE001
            print(f'    LRT failed: {exc}')
            return {'LRT chi2': np.nan, 'LRT p': np.nan}

    chi2 = 2.0 * (full.llf - red.llf)
    chi2 = max(chi2, 0.0)
    p = 0.5 * stats.chi2.sf(chi2, 1)
    return {'LRT chi2': float(chi2), 'LRT p': float(p), 'LRT df': 1,
            'LRT component': drop}


def _term_row(res, term, extra):
    row = dict(extra)
    if res is None or term not in res.params.index:
        row.update({'beta': np.nan, 'se': np.nan, 'z': np.nan, 'p': np.nan,
                    'CI low': np.nan, 'CI high': np.nan, 'N trials': np.nan})
        return row
    ci = res.conf_int()
    row.update({
        'beta': float(res.params[term]),
        'se': float(res.bse[term]),
        'z': float(res.tvalues[term]),
        'p': float(res.pvalues[term]),
        'CI low': float(ci.loc[term, 0]),
        'CI high': float(ci.loc[term, 1]),
        'N trials': int(res.nobs),
    })
    return row


def zscore(series):
    v = pd.to_numeric(series, errors='coerce')
    sd = v.std()
    return (v - v.mean()) / sd if sd and sd > 0 else v * np.nan


def nakagawa_r2(res):
    """
    Marginal R^2 (fixed effects only) and conditional R^2 (fixed + random),
    following Nakagawa & Schielzeth. var_fixed is the variance of the
    fixed-effect linear predictor (X @ fe_params) across observations.
    """
    if res is None:
        return {'Marginal R2': np.nan, 'Conditional R2': np.nan}
    try:
        fe_pred = (np.asarray(res.model.exog, float)
                   @ np.asarray(res.fe_params, float))
        var_f = float(np.var(fe_pred))
    except Exception:                                           # noqa: BLE001
        return {'Marginal R2': np.nan, 'Conditional R2': np.nan}
    var_re = float(np.nansum(np.atleast_1d(np.asarray(res.vcomp, float))))
    if getattr(res.model, 'k_re', 0) > 0:      # fallback path random intercept
        var_re += float(np.asarray(res.cov_re)[0, 0])
    total = var_f + var_re + float(res.scale)
    if total <= 0:
        return {'Marginal R2': np.nan, 'Conditional R2': np.nan}
    return {'Marginal R2': var_f / total,
            'Conditional R2': (var_f + var_re) / total}


def _decomp_terms(res):
    """beta/se/z/p/CI for the between-texture and within-texture components."""
    out = {}
    for term, tag in [('x_between', 'between'), ('x_within', 'within')]:
        r = _term_row(res, term, {})
        out.update({f'beta_{tag}': r['beta'], f'se_{tag}': r['se'],
                    f'z_{tag}': r['z'], f'p_{tag}': r['p'],
                    f'CI_{tag}_low': r['CI low'], f'CI_{tag}_high': r['CI high']})
    return out


def _pick_variance(vs):
    """Compact variance/ICC subset for the per-model summary row."""
    keys = ['Var Subject', 'Var Texture', 'Var Subject_slope', 'Residual var',
            'ICC Subject', 'ICC Texture']
    return {k: vs.get(k, np.nan) for k in keys}


def _bh_within_block(bdf, pcols):
    """Add a BH-adjusted column next to each raw p-column, over the rows given."""
    for pcol in pcols:
        ok = bdf[pcol].notna()
        bdf.loc[ok, pcol + '_BH'] = bh_adjust(bdf.loc[ok, pcol].to_numpy())
    return bdf


def _add_between_within(d, raw_col):
    """
    Split a predictor into a between-texture part (its texture mean) and a
    within-texture part (deviation from it), both on the grand-z scale so the two
    slopes are in rating units per grand-SD and are directly comparable.
    """
    d['z'] = zscore(d[raw_col])
    d = d.dropna(subset=['z'])
    d['x_between'] = d.groupby('Texture')['z'].transform('mean')
    d['x_within'] = d['z'] - d['x_between']
    return d


# ---------------------------------------------------------------------------
# Figure 3 / Table 2
# ---------------------------------------------------------------------------

def run_rating_models(trial_df, predictors, label, random_slope=True,
                      outlier_pairs=None):
    """
    One model per (block, predictor):

        rating ~ x_between + x_within + (1|Subject) + (1|Texture)
                 [+ (0 + x_within|Subject)]

    where the grand-z predictor is split into its texture mean (x_between,
    the between-texture question -- "textures eliciting more of the behaviour
    are rated differently") and the deviation from it (x_within, the
    within-texture question -- "for the same texture, more of the behaviour ->
    higher/lower rating"). Estimating x_between alongside the texture random
    intercept keeps the between-texture signal instead of letting the intercept
    absorb it.

    Each output row is a compact summary: both slopes with SE/CI/p (BH-adjusted
    across predictors within block, separately for the between and within
    slopes), the Subject/Texture/slope variance components and their ICCs, and
    marginal/conditional R^2.
    """
    trial_df = drop_outlier_pairs(trial_df, outlier_pairs)
    rows, var_rows = [], []
    for block in BLOCKS:
        b = trial_df[trial_df['Block Type'] == block]
        if b.empty:
            continue
        block_rows = []
        for pred in predictors:
            if pred not in b.columns:
                print(f'  {label} {block}: predictor {pred} absent, skipping')
                continue
            d = b[['Subject', 'Texture', RESPONSE, pred]].dropna().copy()
            if d['Subject'].nunique() < 3 or len(d) < 30:
                print(f'  {label} {block}/{pred}: too few data, skipping')
                continue
            d = _add_between_within(d, pred)
            d = d.rename(columns={RESPONSE: 'rating'})

            print(f'  {label} {block}/{pred}: n={len(d)}, '
                  f'subjects={d["Subject"].nunique()}, '
                  f'textures={d["Texture"].nunique()}')
            res, vc_names = fit_crossed(
                d, 'rating ~ x_between + x_within',
                random_slope_for='x_within' if random_slope else None)

            row = {'Block': block, 'Block name': BLOCK_NAMES[block],
                   'Predictor': pred,
                   'N subjects': int(d['Subject'].nunique()),
                   'N textures': int(d['Texture'].nunique()),
                   'N trials': int(len(d))}
            row.update(_decomp_terms(res))
            vs = variance_summary(res, vc_names, f'{label} {block} {pred}')
            row.update(_pick_variance(vs))
            row.update(nakagawa_r2(res))
            block_rows.append(row)
            var_rows.append(vs)

        if block_rows:
            rows.append(_bh_within_block(pd.DataFrame(block_rows),
                                         ['p_between', 'p_within']))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, pd.DataFrame(var_rows)


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------

def run_band_models(band_dir, random_slope=True, outlier_pairs=None):
    """
    One model per (block, band) on the trial-level band table written by
    Data_Generation_Band_Trials.py, using the same between-texture /
    within-texture decomposition as the rating models. For Figure 4 the
    between-texture slope is the relevant one: it asks whether textures that
    elicit more vibration in a band are rated differently -- the between-texture
    relationship the published per-subject Spearman measured, which a plain
    (1|Texture) model would otherwise absorb. p-values are BH-corrected across
    the 5 bands within block, separately for the between and within slopes.
    """
    pairs = OUTLIER_PAIRS if outlier_pairs is None else outlier_pairs
    rows, var_rows = [], []
    for condition in ['Hardness', 'Slipperiness', 'Roughness']:
        path = os.path.join(band_dir, f'Data_Band_trials_{condition}.csv')
        if not os.path.exists(path):
            print(f'  {path} not found; run Data_Generation_Band_Trials.py first')
            continue
        df = pd.read_csv(path)
        drop_subj = {str(s) for s, b in pairs if str(b) == condition[0]}
        if drop_subj:
            n0 = len(df)
            df = df[~df['Subject'].astype(str).isin(drop_subj)]
            print(f'  {condition}: excluded outlier {sorted(drop_subj)} '
                  f'({n0}->{len(df)} trials)')
        # Band columns are labelled like "5-25 Hz"; the '-' distinguishes them
        # from the "Sample rate Hz" metadata column, which also ends with "Hz"
        # and would otherwise be fit as a spurious sixth band.
        bands = [c for c in df.columns if c.endswith('Hz') and '-' in c]
        block = condition[0]

        block_rows = []
        for band in bands:
            d = df[['Subject', 'Texture', 'Normalized Rating', band]].dropna().copy()
            if len(d) < 30:
                continue
            d = _add_between_within(d, band)
            d = d.rename(columns={'Normalized Rating': 'rating'})

            print(f'  Figure 4 {condition}/{band}: n={len(d)}, '
                  f'subjects={d["Subject"].nunique()}')
            res, vc_names = fit_crossed(
                d, 'rating ~ x_between + x_within',
                random_slope_for='x_within' if random_slope else None)

            row = {'Block': block, 'Condition': condition, 'Band': band,
                   'N subjects': int(d['Subject'].nunique()),
                   'N textures': int(d['Texture'].nunique()),
                   'N trials': int(len(d))}
            row.update(_decomp_terms(res))
            vs = variance_summary(res, vc_names, f'Figure 4 {condition} {band}')
            row.update(_pick_variance(vs))
            row.update(nakagawa_r2(res))
            block_rows.append(row)
            var_rows.append(vs)

        if block_rows:
            bdf = _bh_within_block(pd.DataFrame(block_rows),
                                   ['p_between', 'p_within'])
            bdf['Correction'] = (f'BH over {int(bdf["p_between"].notna().sum())} '
                                 f'bands within block')
            rows.append(bdf)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, pd.DataFrame(var_rows)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def run_block_contrast_models(trial_df, responses=None, log_transform=True):
    """
    log(force) ~ C(Block Type) + (1|Subject) + (1|Texture), with the three
    pairwise block contrasts extracted and BH-corrected across all contrasts of
    all four measures -- the correction the published Figure 2 omits.
    """
    responses = responses or FIGURE_2_RESPONSES
    rows, var_rows = [], []

    for resp in responses:
        if resp not in trial_df.columns:
            print(f'  Figure 2: response {resp} absent, skipping')
            continue
        d = trial_df[['Subject', 'Texture', 'Block Type', resp]].dropna().copy()
        d = d[np.isfinite(pd.to_numeric(d[resp], errors='coerce'))]
        if log_transform:
            d = d[d[resp] > 0]
            d['y'] = np.log(d[resp])
            scale_note = 'log'
        else:
            d['y'] = d[resp]
            scale_note = 'raw'
        if d['Block Type'].nunique() < 2:
            continue

        d['block'] = pd.Categorical(d['Block Type'], categories=BLOCKS)
        print(f'  Figure 2 {resp}: n={len(d)}, '
              f'subjects={d["Subject"].nunique()}')
        res, vc_names = fit_crossed(d, 'y ~ C(block)')
        vs = variance_summary(res, vc_names, f'Figure 2 {resp}')
        var_rows.append(vs)
        if res is None:
            continue
        # Same variance components / ICCs / R^2 attached to each contrast row of
        # this response (the three contrasts share one model).
        model_summary = {**_pick_variance(vs), **nakagawa_r2(res)}

        # H is the reference level; S and R come out as direct contrasts, S-R is
        # obtained from the linear combination of the two coefficients.
        terms = {b: f'C(block)[T.{b}]' for b in ['S', 'R']}
        # t_test operates on the fixed effects only, so the contrast must be a
        # (1, k_fe) row over res.fe_params. res.params also carries the
        # variance-component parameters, which are not part of the fixed-effect
        # design -- a 1-D vector over all params is both the wrong length and the
        # wrong rank, which is what raised the IndexError on r_matrix.shape[1].
        fe_names = list(res.fe_params.index)
        for pair, spec in [(('H', 'S'), {terms['S']: 1.0}),
                           (('H', 'R'), {terms['R']: 1.0}),
                           (('S', 'R'), {terms['R']: 1.0, terms['S']: -1.0})]:
            if not all(t in fe_names for t in spec):
                continue
            contrast = np.zeros((1, len(fe_names)))
            for t, w in spec.items():
                contrast[0, fe_names.index(t)] = w
            test = res.t_test(contrast)
            est = float(np.squeeze(test.effect))
            rows.append({
                'Response': resp, 'Scale': scale_note,
                'Contrast': f'{pair[1]} - {pair[0]}',
                'beta': est,
                'se': float(np.squeeze(test.sd)),
                'z': float(np.squeeze(test.statistic)),
                'p': float(np.squeeze(test.pvalue)),
                'Fold change': float(np.exp(est)) if log_transform else np.nan,
                'N trials': int(res.nobs),
                'N subjects': int(d['Subject'].nunique()),
                **model_summary,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        ok = out['p'].notna()
        out.loc[ok, 'p BH'] = bh_adjust(out.loc[ok, 'p'].to_numpy())
        out['Correction'] = f'BH over {int(ok.sum())} contrasts'
    return out, pd.DataFrame(var_rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_mixed_model_data(force_df, image_df, base_path, band_dir=None,
                              random_slope=True, outlier_pairs=None):
    pairs = OUTLIER_PAIRS if outlier_pairs is None else outlier_pairs
    print('Rating-model outlier exclusion: '
          + (f'{pairs}' if pairs else 'none (all participants kept)'))
    force_trial = collapse_bins_to_trials(
        force_df, [RESPONSE] + list(dict.fromkeys(
            TABLE_2_PREDICTORS + FIGURE_2_RESPONSES)))
    image_trial = collapse_bins_to_trials(
        image_df, ['Mean_Speed_Keypoint_8', 'Max_Speed_Keypoint_8'])

    trial_df = force_trial.merge(
        image_trial, on=['Subject', 'Trial', 'Texture', 'Block Type'],
        how='left')
    print(f'Trial-level table: {len(trial_df)} rows, '
          f'{trial_df["Subject"].nunique()} subjects, '
          f'{trial_df["Texture"].nunique()} textures')

    out_dir = ensure_dir(os.path.join(base_path, 'MixedModels'))
    var_frames = []

    print('\nFigure 3 models')
    f3, v = run_rating_models(trial_df, FIGURE_3_PREDICTORS, 'Figure 3',
                              random_slope=random_slope, outlier_pairs=pairs)
    f3.to_csv(os.path.join(out_dir, 'Data_Mixed_Figure_3.csv'), index=False)
    var_frames.append(v)

    print('\nTable 2 models')
    t2, v = run_rating_models(trial_df, TABLE_2_PREDICTORS, 'Table 2',
                              random_slope=random_slope, outlier_pairs=pairs)
    t2.to_csv(os.path.join(out_dir, 'Data_Mixed_Table_2.csv'), index=False)
    var_frames.append(v)

    print('\nFigure 2 models')
    f2, v = run_block_contrast_models(trial_df)
    f2.to_csv(os.path.join(out_dir, 'Data_Mixed_Figure_2.csv'), index=False)
    var_frames.append(v)

    if band_dir:
        print('\nFigure 4 models')
        f4, v = run_band_models(band_dir, random_slope=random_slope,
                                outlier_pairs=pairs)
        f4.to_csv(os.path.join(out_dir, 'Data_Mixed_Figure_4.csv'), index=False)
        var_frames.append(v)

    var_df = pd.concat([f for f in var_frames if not f.empty],
                       ignore_index=True) if var_frames else pd.DataFrame()
    var_df.to_csv(os.path.join(out_dir, 'Data_Mixed_Variance.csv'), index=False)

    print(f'\nSaved mixed-model output to: {out_dir}')
    return trial_df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import BINNED_FORCE_CSV, BINNED_IMAGE_CSV, OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--force_csv',
        default=BINNED_FORCE_CSV)
    ap.add_argument(
        '--image_csv',
        default=BINNED_IMAGE_CSV)
    ap.add_argument(
        '--base_path',
        default=OUTPUT_DIR)
    ap.add_argument(
        '--band_dir', default=None,
        help='directory holding Data_Band_trials_*.csv (usually '
             '<base_path>/Figure_4_mixed). Omit to skip the Figure 4 models.')
    ap.add_argument('--no_random_slope', action='store_true')
    ap.add_argument('--keep_outliers', action='store_true',
                    help='keep the reliability outliers (default: exclude '
                         'Subject7/H and Subject6/S from the rating models; '
                         'Figure 2 keeps all participants either way)')
    args = ap.parse_args()

    band_dir = args.band_dir or os.path.join(args.base_path, 'Figure_4_mixed')
    if not os.path.isdir(band_dir):
        band_dir = None

    generate_mixed_model_data(
        pd.read_csv(args.force_csv),
        pd.read_csv(args.image_csv),
        args.base_path,
        band_dir=band_dir,
        random_slope=not args.no_random_slope,
        outlier_pairs=[] if args.keep_outliers else None)
