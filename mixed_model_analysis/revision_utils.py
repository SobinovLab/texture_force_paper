#!python3.11
"""
revision_utils.py -- shared helpers for the R1 revision analyses.

Deliberately mirrors the conventions of
`Manuscript_Figures/Correct_Folder_Names/Table_1_correct/Multiprocessing.py`:
a jobs table is exported to CSV, a process pool maps a single-trial worker over
it, results are collected into a long table, and summaries/figures are generated
from that long table. The csv helpers below are copied from that file verbatim so
the on-disk formats match.

Two substitutions were necessary: `reporting_pool.ReportingPool` and
`prehension.tools.stats` are not in this repository, so `simple_pool()` below
provides the same start()/failed-job behaviour on top of the standard library
`multiprocessing`, and `format_nonparam_mwu_rbc()` reimplements the one
`prehension` stats formatter that was used.
"""

import csv
import multiprocessing as mp
import os
import traceback

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# CSV helpers (copied from Table_1_correct/Multiprocessing.py)
# ---------------------------------------------------------------------------

def import_csv(filename, cast=float):
    """Imports csv into a simple structure (column-major)."""
    with open(filename, 'r') as f:
        rdr = csv.reader(f)
        line = next(rdr)
        column_names = [i.strip() for i in line]
        values = [[] for _ in column_names]
        for li in rdr:
            for idof, vdof in enumerate(li):
                try:
                    v = cast(vdof)
                except ValueError:
                    v = vdof
                values[idof].append(v)
    for idof in reversed(range(len(column_names))):
        if (len(column_names[idof]) == 0 and
                all(isinstance(v, str) and len(v) == 0 for v in values[idof])):
            del column_names[idof]
            del values[idof]
    return column_names, values


def export_csv(filename, column_names, values):
    """Exports a column-major structure into a csv file."""
    with open(filename, 'w', newline='') as f:
        wrr = csv.writer(f)
        wrr.writerow(column_names)
        for itrial in range(len(values[0])):
            wrr.writerow([values[k][itrial] for k in range(len(column_names))])


def export_csv_vertical(filename, column_names, values):
    """Exports a row-major structure into a csv file."""
    with open(filename, 'w', newline='') as f:
        wrr = csv.writer(f)
        wrr.writerow(column_names)
        for vals in values:
            wrr.writerow(vals)


def import_csv_vertical(filename, cast=float):
    """Imports csv into a row-major structure."""
    with open(filename, 'r') as f:
        rdr = csv.reader(f)
        line = next(rdr)
        column_names = [i.strip() for i in line]
        values = []
        for li in rdr:
            values.append([])
            for vdof in li:
                try:
                    v = cast(vdof)
                except ValueError:
                    v = vdof
                values[-1].append(v)
    return column_names, values


def import_csv_as_dic(filename, cast=float):
    return {cn: v for cn, v in zip(*import_csv(filename, cast=cast))}


def export_dic_to_csv(filename, dic):
    export_csv(filename, list(dic.keys()), list(dic.values()))


def summary_to_dictionary(summary):
    summary_dic = {}
    for el in summary:
        for k, v in el.items():
            summary_dic.setdefault(k, []).append(v)
    return summary_dic


def get_iqr(x):
    return np.nanpercentile(x, [25, 75])


def xy_numsubplots(numsubplots):
    yn = int(np.ceil(np.sqrt(numsubplots)))
    xn = int(np.ceil(numsubplots / yn))
    return xn, yn


# ---------------------------------------------------------------------------
# Process pool (stand-in for reporting_pool.ReportingPool)
# ---------------------------------------------------------------------------

def _guarded(args):
    fn, i, job = args
    try:
        return i, fn(*job), None
    except Exception:
        return i, None, traceback.format_exc()


def simple_pool(fn, jobs, processes=13, desc='jobs'):
    """
    Map `fn` over `jobs` with a process pool, tolerating per-job failures.

    Returns (results, failed_i_jobs). `results` preserves job order and holds
    None for failed jobs. Mirrors ReportingPool's behaviour closely enough that
    call sites read the same.
    """
    payload = [(fn, i, job) for i, job in enumerate(jobs)]
    results = [None] * len(jobs)
    failed = []
    with mp.Pool(processes=processes) as pool:
        for n, (i, res, err) in enumerate(pool.imap_unordered(_guarded, payload), 1):
            if err is None:
                results[i] = res
            else:
                failed.append(i)
                print(f'\n[{desc}] job {i} failed:\n{err}')
            print(f'\r[{desc}] {n}/{len(jobs)}', end='', flush=True)
    print()
    return results, failed


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def bh_adjust(pvals):
    """
    Benjamini-Hochberg step-up adjusted p-values.

    Same algorithm as `bh_adjust` in Final_Manuscript_Code/Figure_4/
    band_analysis_cd.py, reproduced here so the revision analyses correct
    exactly as the published Figure 4 did.
    """
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def format_nonparam_mwu_rbc(a, b, alternative='two-sided'):
    """
    Mann-Whitney U with rank-biserial correlation, formatted as a string.
    Reimplementation of prehension.tools.stats.format_nonparam_mwu_rbc.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 'U=nan, p=nan, r=nan'
    u, p = stats.mannwhitneyu(a, b, alternative=alternative)
    rbc = 1.0 - 2.0 * u / (a.size * b.size)
    return (f'U={u:.1f}, p={p:.3g}, r={rbc:.2f}, '
            f'N={a.size}/{b.size}, median={np.median(a):.3g}/{np.median(b):.3g}')


def icc_3_1(matrix):
    """
    ICC(3,1): two-way mixed effects, consistency, single measurement.

    Parameters
    ----------
    matrix : array (n_targets, k_raters)
        Rows are the rated objects (here: textures), columns are the repeated
        measurements (here: the 1st/2nd/3rd presentation of each texture).
        Rows containing any NaN are dropped.

    Returns
    -------
    dict with icc, n_targets, k_raters, ms_rows, ms_error.
        icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error)
    """
    m = np.asarray(matrix, float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return {'icc': np.nan, 'n_targets': n, 'k_raters': k,
                'ms_rows': np.nan, 'ms_error': np.nan}

    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)

    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((m - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error
    icc = (ms_rows - ms_error) / denom if denom != 0 else np.nan
    return {'icc': icc, 'n_targets': n, 'k_raters': k,
            'ms_rows': ms_rows, 'ms_error': ms_error}


def mean_pairwise_spearman(matrix):
    """Mean Spearman rho over all pairs of columns of `matrix` (targets x repeats)."""
    m = np.asarray(matrix, float)
    k = m.shape[1]
    rhos = []
    for i in range(k):
        for j in range(i + 1, k):
            ok = np.isfinite(m[:, i]) & np.isfinite(m[:, j])
            if ok.sum() < 3:
                continue
            if np.std(m[ok, i]) == 0 or np.std(m[ok, j]) == 0:
                continue
            rhos.append(stats.spearmanr(m[ok, i], m[ok, j]).statistic)
    return float(np.mean(rhos)) if rhos else np.nan


def collapse_bins_to_trials(df, value_cols, session_col='Session'):
    """
    Collapse the bin-level master CSVs to one row per (Subject, Trial, Texture,
    Block Type), exactly as Final_Manuscript_Code/Figure_3/
    Data_Generation_Figure_3.py lines 83-97 does.

    Trial-level features are stored only on the first row of each
    (Session, Trial) group with NaN elsewhere (see
    binned_data_generator_analysis_forces.calculate_trial_stat_first_row), so the
    mean over the group recovers the single non-NaN value.
    """
    df = df.copy()
    df[session_col] = (df[session_col].astype(str)
                       .str.replace('_preprocessed$', '', regex=True))
    df['Subject'] = df[session_col].str.extract(r'(Subject\d+)')
    keys = ['Subject', 'Trial', 'Texture', 'Block Type']
    cols = keys + [c for c in value_cols if c in df.columns]
    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        print(f'  collapse_bins_to_trials: columns absent from input: {missing}')
    return (df[cols]
            .groupby(keys, as_index=False)
            .mean(numeric_only=True))


def session_order_table(df, session_col='Session'):
    """
    Recover each participant's block order empirically from session numbering.

    Each session contains exactly one block (see the `reports['block type']
    .iloc[0]` checks throughout the pipeline), so sorting a subject's sessions by
    number gives the order in which that subject performed Hardness /
    Slipperiness / Roughness. Used to substantiate the Methods claim that block
    order was randomized across participants.
    """
    d = df.copy()
    d[session_col] = (d[session_col].astype(str)
                      .str.replace('_preprocessed$', '', regex=True))
    d['Subject'] = d[session_col].str.extract(r'(Subject\d+)')
    d['SessionNum'] = d[session_col].str.extract(r'Session(\d+)').astype(float)
    per = (d.dropna(subset=['Subject', 'SessionNum', 'Block Type'])
             .groupby(['Subject', 'SessionNum'])['Block Type']
             .agg(lambda s: s.iloc[0])
             .reset_index()
             .sort_values(['Subject', 'SessionNum']))
    rows = []
    for subj, sdf in per.groupby('Subject'):
        order = ''.join(sdf['Block Type'].tolist())
        rows.append({'Subject': subj, 'Block order': order,
                     'N blocks': len(sdf)})
    return rows


# Participant-block reliability outliers, detected by
# Data_Generation_Reliability.py (below the per-block Tukey lower fence AND below
# an ICC of 0.5). Excluded by default from rating-based analyses (Figures 3, 4,
# 5, Table 2); force-only analyses (Figure 2, Supplementary Figure 2) keep every
# participant and should pass pairs=[].
OUTLIER_PAIRS = [('Subject7', 'H'), ('Subject6', 'S')]


def drop_outlier_pairs(df, pairs=None, subject_col='Subject',
                       block_col='Block Type'):
    """
    Drop rows whose (subject, block) is in `pairs`. `pairs=None` uses
    OUTLIER_PAIRS (the default exclusion); `pairs=[]` keeps every row.
    """
    pairs = OUTLIER_PAIRS if pairs is None else pairs
    if not pairs:
        return df
    skip = {(str(s), str(b)) for s, b in pairs}
    keep = np.array([(str(s), str(b)) not in skip
                     for s, b in zip(df[subject_col], df[block_col])])
    return df[keep]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
