"""
band_analysis.py  –  FFT frequency-band regression pipeline

Stage 1 (--stage 1):
    Load trial force data, detect contact window, extract RMS amplitude in
    five frequency bands, run per-subject OLS regression (band amplitude →
    perceptual rating), save adjusted R² and p-values to a single CSV.

Stage 2 (--stage 2):
    Load saved statistics and render a 3 × 2 swarm-plot figure
        rows : Hardness | Slipperiness | Roughness
        cols : First ? s | Last ? s
    with one swarm per frequency band; dots coloured by significance.

Demo (--stage demo):
    Write a synthetic band_stats.csv and call Stage 2 to demonstrate
    the plotting without real data.

This module is imported by the Figure 4 scripts (Data_Generation_Figure_4.py
calls stage1; Figure_Generation_Figure_4.py calls stage2_pooled) rather than run
directly.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.signal import welch
from scipy.stats import t as t_dist, spearmanr, wilcoxon, theilslopes
from tqdm import tqdm

# np.trapezoid was added in NumPy 2.0; fall back to np.trapz on older installs
_trapz = getattr(np, 'trapezoid', np.trapz)

# ── Configuration ─────────────────────────────────────────────────────────────

BAND_EDGES  = [5, 25, 50, 100, 400, 1000]          # Hz
BAND_LABELS = [f"{BAND_EDGES[i]}-{BAND_EDGES[i+1]} Hz"
               for i in range(len(BAND_EDGES) - 1)]

CONDITIONS  = ['Hardness', 'Slipperiness', 'Roughness']
WINDOWS     = ['last']
SIGNAL_TYPE = 'Fn'          # normal force magnitude (Fz)

MIN_CONTACT_SEC = 2.0       # trials shorter than this are skipped
WINDOW_SEC      = 1.0       # duration of the analysis window; set to match
                            # Methods: "the final 2 s of the exploration
                            # period were extracted". Since this equals
                            # MIN_CONTACT_SEC, every included trial
                            # contributes its full final 2 s.

STATS_FILE = 'band_stats.csv'   # written by Stage 1, read by Stage 2

# ── Contact-detection helpers ─────────────────────────────────────────────────

def _find_first(x):
    idx = x.view(bool).argmax() // x.itemsize
    return int(idx) if x[idx] else -1


def _find_last(x):
    ff = _find_first(np.flip(x))
    return -1 if ff == -1 else len(x) - ff - 1

# ── Signal processing ─────────────────────────────────────────────────────────

def band_rms(signal, sample_rate):
    """
    Compute RMS amplitude in each frequency band via Welch PSD.
    Returns dict {band_label: float}.
    """
    signal = signal - np.nanmean(signal)
    freqs, psd = welch(signal, fs=sample_rate, scaling='density')
    out = {}
    for i, label in enumerate(BAND_LABELS):
        mask = (freqs >= BAND_EDGES[i]) & (freqs < BAND_EDGES[i + 1])
        out[label] = (
            float(np.sqrt(_trapz(psd[mask], freqs[mask])))
            if np.any(mask) else np.nan
        )
    return out

# ── Regression ────────────────────────────────────────────────────────────────

def spearman_stats(x, y):
    """
    Spearman rank correlation between x and y.
    Returns (rho, p_value), or (nan, nan) if degenerate.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(y) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    result = spearmanr(x, y)
    # .statistic was added in scipy 1.9; fall back to .correlation on older builds
    rho = float(result.statistic if hasattr(result, 'statistic') else result.correlation)
    return rho, float(result.pvalue)


def cv_spearman_stats(x, y, cv='loo'):
    """
    Cross-validated Spearman correlation of y on x.

    On each iteration a Theil-Sen line is fitted on the held-IN textures and
    used to predict ratings for the held-OUT texture(s).  Spearman ρ and its
    p-value are then computed on the pooled out-of-fold (predicted, actual)
    pairs.  This prevents inflated ρ from the same textures appearing in both
    training and test.

    Parameters
    ----------
    x  : band RMS amplitudes, shape (n_textures,)
    y  : perceptual ratings,   shape (n_textures,)
    cv : 'loo' – leave-one-out (default); int k – k-fold (capped at n)

    Returns
    -------
    (rho, p_value)  or  (nan, nan) if degenerate.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok   = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n    = len(x)

    if n < 4:
        return np.nan, np.nan

    idx = np.arange(n)
    if cv == 'loo':
        splits = [(np.delete(idx, i), np.array([i])) for i in range(n)]
    else:
        k     = max(2, min(int(cv), n))
        folds = np.array_split(idx, k)
        splits = [
            (np.concatenate([folds[j] for j in range(k) if j != fi]), folds[fi])
            for fi in range(k)
        ]

    y_pred = np.full(n, np.nan)

    for train_idx, test_idx in splits:
        if len(train_idx) < 3 or x[train_idx].std() == 0:
            continue
        # res[0] = slope, res[1] = intercept (safe across scipy versions)
        res = theilslopes(y[train_idx], x[train_idx])
        y_pred[test_idx] = res[1] + res[0] * x[test_idx]

    valid = ~np.isnan(y_pred)
    if valid.sum() < 3:
        return np.nan, np.nan

    return spearman_stats(y_pred[valid], y[valid])

# ── Stage 1 – data extraction ─────────────────────────────────────────────────

def load_trials(session_folders, session_path, condition, window):
    """
    Iterate over sessions for *condition*, detect contact windows, extract
    band amplitudes for the chosen *window* ('first' or 'last' WINDOW_SEC).

    Returns a DataFrame with one row per valid trial:
        Subject | Texture | Normalized Rating | <band columns>
    """
    rows = []

    for sf in tqdm(session_folders, desc=f'{condition}/{window}'):
        reports_path = os.path.join(session_path, sf, 'reports.csv')
        if not os.path.exists(reports_path):
            continue

        reports = pd.read_csv(reports_path)
        if reports['block type'].iloc[0] != condition[0]:
            continue

        norm_path    = os.path.join(session_path, sf, 'normalized_ratings.csv')
        has_norm_file = os.path.exists(norm_path)
        has_norm_col  = 'normalized block rating' in reports.columns
        if not has_norm_file and not has_norm_col:
            print(f'  No rating source for {sf}, skipping.')
            continue

        norm_ratings = pd.read_csv(norm_path) if has_norm_file else None
        subject      = sf.split('_Session')[0]

        for _, trial in reports.iterrows():
            trial_num = int(trial['trial number'])
            texture   = trial['texture used']
            rating    = (norm_ratings['normalized rating'][trial_num - 1]
                         if has_norm_file
                         else trial['normalized block rating'])

            force_path = os.path.join(session_path, sf, 'Force',
                                      f'trial_{trial_num}.csv')
            if not os.path.exists(force_path):
                continue

            fd   = pd.read_csv(force_path)
            time = fd['Exact Times'].values
            Fn   = fd['Fz'].values.copy()
            Ft   = np.sqrt(fd['Fx'].values ** 2 + fd['Fy'].values ** 2)
            sr   = len(time) / (time[-1] - time[0])

            # ── Contact detection ──────────────────────────────────────────
            Fn -= np.median(Fn[time < 0.33])
            thr = 10 * np.std(np.abs(Fn[time < 0.33]))
            st  = _find_first(np.abs(Fn) > thr)
            en  = _find_last(np.abs(Fn) > thr)

            if st < 0 or en < 0 or (en - st) / sr < MIN_CONTACT_SEC:
                continue

            sig = (Ft if SIGNAL_TYPE == 'Ft' else Fn)[st:en]
            n   = int(WINDOW_SEC * sr)
            sig = sig[-n:] if window == 'last' else sig[:n]

            row = {'Subject': subject, 'Texture': texture,
                   'Normalized Rating': rating}
            row.update(band_rms(sig, sr))
            rows.append(row)

    return pd.DataFrame(rows)


def per_subject_regressions(trial_df, cv=None):
    """
    For each subject, average trials per texture, then compute Spearman ρ
    between each band amplitude and Normalized Rating.

    Parameters
    ----------
    trial_df : DataFrame from load_trials()
    cv       : None  – standard Spearman (default)
               'loo' – leave-one-out cross-validation
               int   – k-fold cross-validation (k ≥ 2)

    Returns DataFrame: Subject | Band | rho | p_value
    """
    stat_fn = spearman_stats if cv is None else (
        lambda x, y: cv_spearman_stats(x, y, cv=cv))
    records = []
    for subj, sdf in trial_df.groupby('Subject'):
        tex = (sdf.groupby('Texture')[BAND_LABELS + ['Normalized Rating']]
               .mean()
               .reset_index())
        y = tex['Normalized Rating'].values
        for band in BAND_LABELS:
            rho, pval = stat_fn(tex[band].values, y)
            records.append({'Subject': subj, 'Band': band,
                            'rho': rho, 'p_value': pval})
    return pd.DataFrame(records)


def stage1(session_path, output_path, session_folders, cv=None,
           skip_pairs=None):
    """
    Compute and save band statistics for all conditions × windows.

    Parameters
    ----------
    cv : None | 'loo' | int  – passed to per_subject_regressions.
         When not None, ρ and p are computed via cross-validation.
    skip_pairs : iterable of (Subject, Block) tuples, optional
         (participant, block) outliers to exclude, e.g. [('Subject7', 'H')].
         Block is the one-letter code 'H'/'S'/'R' (= condition[0]). An excluded
         participant contributes no ρ for that block. Default None keeps every
         participant (the published analysis). Used for the reliability
         sensitivity re-run.
    """
    skip_by_block = {}
    for s, b in (skip_pairs or []):
        skip_by_block.setdefault(str(b), set()).add(str(s))
    os.makedirs(output_path, exist_ok=True)
    all_stats = []
    cv_tag = '' if cv is None else f' (CV={cv})'
    print(f'Computing per-subject Spearman ρ{cv_tag}')

    for condition in CONDITIONS:
        for window in WINDOWS:
            trial_df = load_trials(session_folders, session_path,
                                   condition, window)
            if trial_df.empty:
                print(f'  No usable trials: {condition}/{window}')
                continue

            # Drop manually-entered (participant, block) outliers for this block.
            drop = skip_by_block.get(condition[0], set())
            if drop:
                n0 = len(trial_df)
                trial_df = trial_df[~trial_df['Subject'].isin(drop)]
                print(f'  {condition}: skipped {sorted(drop)} '
                      f'({n0}->{len(trial_df)} trials)')
                if trial_df.empty:
                    print(f'  No usable trials after skip: {condition}/{window}')
                    continue

            # Save trial-level amplitudes for reference / debugging
            tag = f'{condition}_{window}'
            trial_df.to_csv(
                os.path.join(output_path, f'trials_{tag}.csv'), index=False)

            stats = per_subject_regressions(trial_df, cv=cv)
            stats['Condition'] = condition
            stats['Window']    = window
            all_stats.append(stats)

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        out = os.path.join(output_path, STATS_FILE)
        combined.to_csv(out, index=False)
        print(f'\nSaved {out}  ({len(combined)} rows)')
    else:
        print('No data was processed.')

# ── Stage 2 – plotting ────────────────────────────────────────────────────────

def bh_adjust(pvals):
    """
    Benjamini-Hochberg FDR adjustment (step-up procedure).
    Returns adjusted p-values (q-values) in the same order as input.
    Values that are NaN are left as NaN.
    """
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    finite = np.isfinite(pvals)
    if finite.sum() == 0:
        return pvals.copy()

    # Work only on finite values
    idx_finite = np.where(finite)[0]
    p_fin = pvals[idx_finite]
    order = np.argsort(p_fin)
    ranks = np.arange(1, len(p_fin) + 1)

    # BH adjusted value for each sorted position: p * n / rank
    q = p_fin[order] * len(p_fin) / ranks

    # Enforce monotone non-increasing from the right (step-up)
    for i in range(len(q) - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])

    # Map back
    out = pvals.copy()
    out[idx_finite[order]] = np.clip(q, 0.0, 1.0)
    return out


def _pooled_wilcoxon_p(vals):
    """
    Two-tailed Wilcoxon signed-rank test of *vals* against 0.
    Non-parametric analogue of the one-sample t-test.
    Returns p-value, or nan if degenerate (n < 3 or all values identical).
    """
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return np.nan
    try:
        _, p = wilcoxon(vals, alternative='two-sided')
        return float(p)
    except ValueError:
        return np.nan


def stage2(output_path, alpha=0.05, adjust=False):
    """
    Load band_stats.csv and render the 3 × 2 swarm-plot figure.

    Visual encoding
    ---------------
    Dot colour : red  = subject significant (Spearman p < alpha)  |  black = n.s.
    Black bar  : median Spearman ρ across subjects
    p-value    : pooled Wilcoxon p printed above bands significant at *alpha*

    Multiple comparison correction (when adjust=True)
    --------------------------------------------------
    Subject-level : Benjamini-Hochberg FDR applied across the 5 frequency
                    bands, separately for each subject × condition × window.
    Pooled        : BH FDR applied across the 5 band-level pooled p-values,
                    separately for each condition × window (i.e. first and
                    last ? s are corrected independently).
    """
    stats_path = os.path.join(output_path, STATS_FILE)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f'band_stats.csv not found at {stats_path}. Run --stage 1 first.')

    df = pd.read_csv(stats_path)
    df['Band'] = pd.Categorical(df['Band'], categories=BAND_LABELS, ordered=True)

    # ── Optional BH correction on subject-level p-values ─────────────────────
    # Adjustment is across the 5 bands, within each subject × condition × window.
    if adjust:
        def _bh_group(g):
            g = g.copy()
            fin = np.isfinite(g['p_value'].values)
            if fin.sum() > 1:
                g.loc[g.index[fin], 'p_value'] = bh_adjust(
                    g.loc[g.index[fin], 'p_value'].values)
            return g
        df = (df.groupby(['Condition', 'Window', 'Subject'], group_keys=False)
                .apply(_bh_group))

    df['sig'] = (df['p_value'] < alpha).map({True: 'sig', False: 'ns'})
    palette   = {'sig': '#d62728', 'ns': 'black'}

    corr_label = ' (BH-FDR corrected)' if adjust else ''
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(11, 9), sharey='all')
    fig.suptitle(f'Frequency-band Spearman $\\rho$  (per subject){corr_label}',
                 fontsize=13)

    col_titles = {'first': 'First ? s', 'last': 'Last ? s'}

    for r, cond in enumerate(CONDITIONS):
        for c, win in enumerate(WINDOWS):
            ax  = axes[r, c]
            sub = df[(df['Condition'] == cond) & (df['Window'] == win)].copy()

            # ── Pre-compute pooled p-values for all 5 bands ───────────────
            pooled_pvals = {
                band: _pooled_wilcoxon_p(
                    sub.loc[sub['Band'] == band, 'rho'].dropna().values)
                for band in BAND_LABELS
            }

            # Optional BH correction on pooled p-values across bands
            # (within this condition × window; first and last treated separately)
            if adjust:
                finite_bands = [b for b in BAND_LABELS
                                if np.isfinite(pooled_pvals[b])]
                if len(finite_bands) > 1:
                    raw_ps = np.array([pooled_pvals[b] for b in finite_bands])
                    adj_ps = bh_adjust(raw_ps)
                    for b, p in zip(finite_bands, adj_ps):
                        pooled_pvals[b] = float(p)

            sns.swarmplot(
                data=sub,
                x='Band', y='rho',
                hue='sig', palette=palette,
                order=BAND_LABELS,
                hue_order=['sig', 'ns'],
                size=5, dodge=False, ax=ax, legend=False,
            )

            # Median bar + pooled significance star per band
            for xi, band in enumerate(BAND_LABELS):
                vals = sub.loc[sub['Band'] == band, 'rho'].dropna().values
                if len(vals) == 0:
                    continue

                ax.plot([xi - 0.28, xi + 0.28],
                        [float(np.median(vals))] * 2,
                        color='k', linewidth=2.0, zorder=5,
                        solid_capstyle='round')

                pp = pooled_pvals[band]
                if np.isfinite(pp) and pp < alpha:
                    top   = float(np.max(vals))
                    p_lbl = 'p<0.001' if pp < 0.001 else f'p={pp:.3f}'
                    ax.text(xi, top, p_lbl, ha='center', va='bottom',
                            fontsize=7, color='#d62728', zorder=6)

            ax.axhline(0, color='0.55', linewidth=0.8, linestyle='--')

            # ── x-axis: remove spine and ticks for all rows ───────────────
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', length=0)
            ax.set_xticks(range(len(BAND_LABELS)))
            if r == 2:
                ax.set_xticklabels(BAND_LABELS, fontsize=8)
                ax.set_xlabel('Frequency band', fontsize=9)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel('')

            # ── y-axis ────────────────────────────────────────────────────
            if c == 0:
                ax.set_ylabel(f'{cond} Spearman $\\rho$', fontsize=9)
            else:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

            if r == 0:
                ax.set_title(col_titles[win], fontsize=10)

            ax.spines[['top', 'right']].set_visible(False)

    # Shared legend
    p_label = f'q < {alpha}' if adjust else f'p < {alpha}'
    legend_handles = [
        Patch(facecolor='#d62728', edgecolor='none',
              label=f'{p_label} (subject, Spearman)'),
        Patch(facecolor='black',   edgecolor='none',
              label=f'n.s. (subject)'),
        plt.Line2D([0], [0], color='#d62728', linewidth=0, marker='',
                   label=f'pooled Wilcoxon {"q" if adjust else "p"} printed above'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.045, 1, 1])

    suffix = '_BH' if adjust else ''
    for ext in ('png', 'svg'):
        out = os.path.join(output_path, f'swarm_rho{suffix}.{ext}')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'Saved {out}')

    # plt.show()
    return fig

# ── Stage 2b – pooled-only plot ───────────────────────────────────────────────

def stage2_pooled(output_path, alpha=0.05, adjust=False):
    """
    Load band_stats.csv and render a 3 × 2 figure showing only the
    pooled (across-subject) performance.

    Visual layout
    -------------
    Each panel contains one violin per frequency band (rows = conditions,
    columns = time windows).  The violin body shows the kernel-density
    estimate of the per-subject Spearman ρ distribution.  Inside each
    violin, a thick vertical bar spans the interquartile range (Q1–Q3) and
    a white dot marks the median.  Individual subject values are overlaid as
    jittered dots.  A horizontal bar at the group median is colour-coded by
    pooled significance (red = significant, grey = n.s.).  A dashed
    horizontal line at ρ = 0 serves as a reference.

    Pooled significance test
    ------------------------
    For each band the n per-subject ρ values are treated as a single sample
    and tested against the null hypothesis that the population median ρ = 0
    using the two-tailed Wilcoxon signed-rank test
    (scipy.stats.wilcoxon, alternative='two-sided').

    The signed-rank test ranks the absolute deviations |ρᵢ| and weights
    positive and negative deviations by their rank, making it more
    sensitive than the sign test while remaining distribution-free.  It is
    the non-parametric analogue of the one-sample t-test and is appropriate
    here because per-subject ρ values are bounded (−1 to 1) and may not be
    normally distributed, especially with small subject counts.  Values
    exactly equal to 0 are excluded from ranking (zero_method='wilcox',
    scipy default).

    The test is applied independently to each of the
    len(BAND_LABELS) × len(CONDITIONS) × len(WINDOWS) band–condition–window
    combinations.

    Multiple-comparison correction (when adjust=True)
    -------------------------------------------------
    Benjamini-Hochberg FDR correction is applied to the pooled p-values
    across the len(BAND_LABELS) bands within each condition × window
    combination independently (i.e. first and last windows are corrected
    separately).  The resulting q-values replace the raw p-values for
    thresholding at *alpha*.
    """
    stats_path = os.path.join(output_path, STATS_FILE)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f'band_stats.csv not found at {stats_path}. Run --stage 1 first.')

    df = pd.read_csv(stats_path)
    df['Band'] = pd.Categorical(df['Band'], categories=BAND_LABELS, ordered=True)

    corr_label = ' (BH-FDR corrected)' if adjust else ''
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(11, 9), sharey='all')
    fig.suptitle(
        f'Frequency-band Spearman $\\rho$ — pooled across subjects{corr_label}',
        fontsize=13)

    col_titles = {'first': 'First ? s', 'last': 'Last ? s'}
    rng = np.random.default_rng(0)

    for r, cond in enumerate(CONDITIONS):
        for c, win in enumerate(WINDOWS):
            ax  = axes[r, c]
            sub = df[(df['Condition'] == cond) & (df['Window'] == win)].copy()

            pooled_pvals = {
                band: _pooled_wilcoxon_p(
                    sub.loc[sub['Band'] == band, 'rho'].dropna().values)
                for band in BAND_LABELS
            }

            if adjust:
                finite_bands = [b for b in BAND_LABELS
                                if np.isfinite(pooled_pvals[b])]
                if len(finite_bands) > 1:
                    raw_ps = np.array([pooled_pvals[b] for b in finite_bands])
                    adj_ps = bh_adjust(raw_ps)
                    for b, p in zip(finite_bands, adj_ps):
                        pooled_pvals[b] = float(p)

            for xi, band in enumerate(BAND_LABELS):
                vals = sub.loc[sub['Band'] == band, 'rho'].dropna().values
                if len(vals) < 2:
                    continue

                pp    = pooled_pvals[band]
                sig   = np.isfinite(pp) and pp < alpha
                color = '#d62728' if sig else '#888888'

                # Violin body (needs ≥ 4 points for KDE to be sensible)
                if len(vals) >= 4:
                    parts = ax.violinplot(
                        vals, positions=[xi], widths=0.7,
                        showmeans=False, showmedians=False, showextrema=False)
                    for pc in parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.40)
                        pc.set_edgecolor('none')

                    # IQR bar + median dot inside the violin
                    q1, med, q3 = np.percentile(vals, [25, 50, 75])
                    ax.plot([xi, xi], [q1, q3], color='0.25',
                            linewidth=3.5, solid_capstyle='round', zorder=4)
                    ax.plot(xi, med, 'o', color='white', markersize=4.5,
                            markeredgecolor='0.25', markeredgewidth=0.8, zorder=5)

                # Individual subject dots (jittered)
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(xi + jitter, vals, color='k', s=18,
                           alpha=0.75, zorder=6, linewidths=0)

                # Group median bar (coloured by significance)
                med_val = float(np.median(vals))
                ax.plot([xi - 0.28, xi + 0.28], [med_val, med_val],
                        color=color, linewidth=2.5, zorder=7,
                        solid_capstyle='round')

                if sig:
                    top   = float(np.max(vals))
                    p_lbl = 'p<0.001' if pp < 0.001 else f'p={pp:.3f}'
                    ax.text(xi, top, p_lbl, ha='center', va='bottom',
                            fontsize=7, color='#d62728', zorder=8)

            ax.axhline(0, color='0.55', linewidth=0.8, linestyle='--')
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', length=0)
            ax.set_xticks(range(len(BAND_LABELS)))
            if r == 2:
                ax.set_xticklabels(BAND_LABELS, fontsize=8)
                ax.set_xlabel('Frequency band', fontsize=9)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel('')

            if c == 0:
                ax.set_ylabel(f'{cond} Spearman $\\rho$', fontsize=9)
            else:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)

            if r == 0:
                ax.set_title(col_titles[win], fontsize=10)

            ax.spines[['top', 'right']].set_visible(False)

    p_label = f'q < {alpha}' if adjust else f'p < {alpha}'
    legend_handles = [
        Patch(facecolor='#d62728', alpha=0.40, edgecolor='none',
              label=f'Pooled Wilcoxon {p_label}'),
        Patch(facecolor='#888888', alpha=0.40, edgecolor='none',
              label='Pooled n.s.'),
        plt.Line2D([0], [0], color='#d62728', linewidth=2.5,
                   label='Median (significant)'),
        plt.Line2D([0], [0], color='#d62728', linewidth=0, marker='',
                   label=f'pooled Wilcoxon {"q" if adjust else "p"} printed above'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.055, 1, 1])

    suffix = '_BH' if adjust else ''
    for ext in ('png', 'svg'):
        out = os.path.join(output_path, f'pooled_rho{suffix}.{ext}')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f'Saved {out}')

    # plt.show()
    return fig

# ── Stage 3 – rating vs. band power scatter ───────────────────────────────────

def plot_rating_vs_power(output_path, condition='Roughness', window='last',
                         alpha=0.05):
    """
    Scatter Normalized Rating (x) against RMS band amplitude (y) for each
    frequency band on a single figure with one subplot per band.

    Data are averaged per (Subject × Texture) before plotting so that each dot
    represents one texture for one subject.  Subjects are colour-coded.  A
    Theil-Sen regression line (non-parametric) is overlaid together with
    Spearman ρ and its p-value as a panel annotation.

    Requires the trial-level CSV written by Stage 1:
        trials_{condition}_{window}.csv
    """
    trial_csv = os.path.join(output_path, f'trials_{condition}_{window}.csv')
    if not os.path.exists(trial_csv):
        raise FileNotFoundError(
            f'{trial_csv} not found – run --stage 1 first.')

    df  = pd.read_csv(trial_csv)

    # One point per (Subject, Texture): average over repeated trials
    avg = (df.groupby(['Subject', 'Texture'])[BAND_LABELS + ['Normalized Rating']]
             .mean()
             .reset_index())

    subjects = sorted(avg['Subject'].unique())
    palette  = dict(zip(subjects, sns.color_palette('tab20', len(subjects))))

    n_bands = len(BAND_LABELS)
    fig, axes = plt.subplots(1, n_bands,
                             figsize=(3.2 * n_bands, 4.4),
                             sharey=False)
    if n_bands == 1:
        axes = [axes]

    fig.suptitle(
        f'{condition} — band RMS amplitude vs. rating  [{window} ? s]',
        fontsize=12)

    for ax, band in zip(axes, BAND_LABELS):
        sub = avg[['Subject', 'Normalized Rating', band]].dropna()
        x   = sub['Normalized Rating'].values
        y   = sub[band].values

        # Per-subject scatter
        for s in subjects:
            m = sub['Subject'] == s
            ax.scatter(sub.loc[m, 'Normalized Rating'],
                       sub.loc[m, band],
                       color=palette[s], s=24, alpha=0.85,
                       linewidths=0, zorder=3)

        # Theil-Sen regression line (non-parametric fit)
        if len(x) >= 4:
            res  = theilslopes(y, x)
            # res[0] = slope, res[1] = intercept (safe across scipy versions)
            x_lo, x_hi = x.min(), x.max()
            ax.plot([x_lo, x_hi],
                    [res[1] + res[0] * x_lo, res[1] + res[0] * x_hi],
                    color='k', linewidth=1.5, zorder=4)

        # Spearman ρ + p-value + N annotation
        rho, pval = spearman_stats(x, y)
        n_pts = len(x)
        if np.isfinite(rho):
            p_str = 'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
            ax.set_title(
                f'{band}  (N={n_pts})\nρ = {rho:.2f},  {p_str}',
                fontsize=8)
        else:
            ax.set_title(f'{band}  (N={n_pts})', fontsize=8)

        ax.set_xlabel('Normalized rating', fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('RMS amplitude', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    # Subject colour legend below the figure
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=palette[s], markersize=7, label=s)
        for s in subjects
    ]
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(subjects), 7), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    for ext in ('png', 'svg'):
        out_path = os.path.join(
            output_path,
            f'rating_vs_power_{condition}_{window}.{ext}')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f'Saved {out_path}')

    # plt.show()
    return fig


def stage_scatter(output_path, alpha=0.05):
    """Call plot_rating_vs_power for every available trial-level CSV."""
    found = False
    for cond in CONDITIONS:
        for win in WINDOWS:
            csv = os.path.join(output_path, f'trials_{cond}_{win}.csv')
            if os.path.exists(csv):
                plot_rating_vs_power(output_path, condition=cond,
                                     window=win, alpha=alpha)
                found = True
    if not found:
        print('No trial-level CSV files found in output_path. '
              'Run --stage 1 first.')

# ── Demo data generation ──────────────────────────────────────────────────────

def generate_demo(output_path, n_subjects=14, seed=42):
    """
    Write a synthetic band_stats.csv that mimics Stage 1 output.
    Higher bands tend to predict roughness better; first-window values
    are somewhat lower, reflecting the typical pattern in the real data.
    """
    rng = np.random.default_rng(seed)

    # Mean Spearman ρ per band (index 0 = lowest band)
    band_mu = np.array([0.04, 0.12, 0.30, 0.44, 0.18])

    cond_scale = {'Hardness': 0.55, 'Slipperiness': 0.50, 'Roughness': 1.00}
    win_scale  = {'first': 0.72,    'last': 1.00}
    noise_sd   = 0.18
    n_textures = 8   # used for Spearman t-approximation p-value

    subjects = [f'Subject{i + 1:02d}' for i in range(n_subjects)]
    rows = []

    for cond in CONDITIONS:
        for win in WINDOWS:
            for subj in subjects:
                for bi, band in enumerate(BAND_LABELS):
                    mu  = band_mu[bi] * cond_scale[cond] * win_scale[win]
                    rho = float(np.clip(rng.normal(mu, noise_sd), -0.97, 0.97))

                    # p-value via Spearman t-approximation: t = ρ√((n-2)/(1-ρ²))
                    t_val = rho * np.sqrt((n_textures - 2) /
                                         max(1 - rho ** 2, 1e-9))
                    pval  = float(2 * t_dist.sf(abs(t_val), df=n_textures - 2))

                    rows.append({'Subject': subj, 'Band': band,
                                 'rho': rho, 'p_value': pval,
                                 'Condition': cond, 'Window': win})

    os.makedirs(output_path, exist_ok=True)
    out = os.path.join(output_path, STATS_FILE)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Demo stats written to {out}')

    # ── Also write trial-level CSVs for plot_rating_vs_power ─────────────────
    # One row per (subject, texture): amplitude linearly correlated with rating.
    textures       = [f'Texture{i + 1:02d}' for i in range(n_textures)]
    texture_rating = np.linspace(0.1, 0.9, n_textures)

    for cond in CONDITIONS:
        for win in WINDOWS:
            trial_rows = []
            for subj in subjects:
                subj_ratings = np.clip(
                    texture_rating + rng.normal(0, 0.05, n_textures), 0.0, 1.0)
                for ti, tex in enumerate(textures):
                    row = {'Subject': subj, 'Texture': tex,
                           'Normalized Rating': float(subj_ratings[ti])}
                    for bi, band in enumerate(BAND_LABELS):
                        mu    = band_mu[bi] * cond_scale[cond] * win_scale[win]
                        amp   = 0.40 + 0.60 * mu * subj_ratings[ti] + \
                                rng.normal(0, 0.10)
                        row[band] = float(max(0.0, amp))
                    trial_rows.append(row)

            trial_out = os.path.join(output_path, f'trials_{cond}_{win}.csv')
            pd.DataFrame(trial_rows).to_csv(trial_out, index=False)
    print(f'Demo trial data written to {output_path}/'
          f'trials_{{Condition}}_{{Window}}.csv')

