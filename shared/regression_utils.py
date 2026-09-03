"""
regression_utils.py

Shared statistics helpers used by Figure 3 and Table 2, so both use the exact
same definitions of "z-scored per subject", "texture-level aggregation",
"cross-validated pseudo-R2", and "permutation p-value". Written once here so
the two outputs can't silently drift apart.

Methods this implements (see manuscript Methods + Table 2 caption + Fig 3 caption):
    - Explanatory variables and ratings are z-scored *within subject* before
      any aggregation (subject's own mean/std across all trials of that subject
      for that block).
    - For each texture, trial-level (subject x texture) values are averaged
      within subject, then averaged across subjects -> one point per texture
      (N=14 textures). SEM for error bars is computed across subjects (i.e.
      "whiskers indicate SEM computed on inter-subject variability", per Fig 3
      caption).
    - "r" reported in Figure 3 is the in-sample Pearson correlation between the
      N=14 texture-level means.
    - "r2" reported in Figure 3 / Table 2 is a leave-one-out cross-validated
      pseudo-R2 computed on those N=14 texture-level means (fit on 13, predict
      the 14th, repeat for each texture).
    - The p-value is a one-sided permutation test: shuffle the texture-level
      rating labels, recompute the CV R2 on the shuffled data, repeat
      n_perm (default 1000) times, and take the fraction of shuffles whose
      CV R2 is >= the observed CV R2 (with a +1/+1 correction so p is never
      exactly zero).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon, rankdata


# ---------------------------------------------------------------------------
# Per-subject normalization
# ---------------------------------------------------------------------------

def zscore_per_subject(df, subject_col, value_col, out_col=None):
    """
    Z-scores `value_col` within each subject (group defined by subject_col).
    Returns a copy of df with an added/overwritten column `out_col`
    (defaults to value_col, i.e. in place).

    A subject with zero variance (std == 0 or NaN) gets NaN in the output
    for that subject's rows, rather than raising or silently returning 0
    for everyone (which would be a subtle bug for constant subjects).
    """
    out_col = out_col or value_col
    df = df.copy()

    def _z(g):
        mean = g[value_col].mean()
        std = g[value_col].std()
        if not std or np.isnan(std) or std == 0:
            return pd.Series(np.nan, index=g.index)
        return (g[value_col] - mean) / std

    df[out_col] = df.groupby(subject_col, group_keys=False).apply(_z)
    return df


# ---------------------------------------------------------------------------
# Texture-level aggregation (subject-then-texture averaging)
# ---------------------------------------------------------------------------

def texture_level_mean_sem(df, texture_col, subject_col, value_col):
    """
    Two-stage aggregation:
      1. mean of value_col within each (texture, subject)
      2. mean + SEM of those per-subject means, across subjects, within texture

    Returns a DataFrame indexed by texture with columns ['mean', 'sem', 'n_subjects'].
    This is what produces the "inter-subject variability" SEM whiskers described
    in the Figure 3 caption.
    """
    per_subject = (
        df.groupby([texture_col, subject_col])[value_col]
        .mean()
        .reset_index()
    )

    def _agg(g):
        vals = g[value_col].dropna()
        n = len(vals)
        return pd.Series({
            'mean': vals.mean() if n > 0 else np.nan,
            'sem': (vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            'n_subjects': n
        })

    return per_subject.groupby(texture_col).apply(_agg)


# ---------------------------------------------------------------------------
# Cross-validated pseudo-R2 + permutation test (single predictor, OLS)
# ---------------------------------------------------------------------------

def _loo_cv_r2(x, y):
    """
    Leave-one-out cross-validated R^2 for a single-predictor OLS model.
    TSS is computed against the mean of the *full* y (standard convention
    for reporting a single overall CV-R2 comparable across folds).
    Returns np.nan if fewer than 4 valid (x, y) pairs are available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(y)
    if n < 4:
        return np.nan

    preds = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = sm.add_constant(x[mask], has_constant='add')
        try:
            model = sm.OLS(y[mask], X_train).fit()
            X_test = np.array([[1.0, x[i]]])
            preds[i] = model.predict(X_test)[0]
        except Exception:
            preds[i] = np.nan

    valid = np.isfinite(preds)
    if valid.sum() < 2:
        return np.nan

    ss_res = np.sum((y[valid] - preds[valid]) ** 2)
    ss_tot = np.sum((y[valid] - y.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def cv_r2_and_permutation_p(x, y, n_perm=1000, seed=0):
    """
    Computes the observed LOO CV R^2 for (x, y), then a one-sided permutation
    p-value: shuffle y n_perm times, recompute LOO CV R^2 each time, and
    report the fraction of shuffles with CV R^2 >= observed (Laplace-smoothed
    so p is never exactly 0).

    Returns (cv_r2, p_value). Both np.nan if degenerate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(y)

    cv_r2 = _loo_cv_r2(x, y)
    if not np.isfinite(cv_r2) or n < 4:
        return cv_r2, np.nan

    rng = np.random.default_rng(seed)
    null_r2 = np.full(n_perm, np.nan)
    for k in range(n_perm):
        y_perm = rng.permutation(y)
        null_r2[k] = _loo_cv_r2(x, y_perm)

    valid_null = np.isfinite(null_r2)
    if valid_null.sum() == 0:
        return cv_r2, np.nan

    p_value = (np.sum(null_r2[valid_null] >= cv_r2) + 1) / (valid_null.sum() + 1)
    return cv_r2, p_value


def pearson_r(x, y):
    """In-sample Pearson correlation coefficient. NaN-safe."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test + rank-biserial effect size (paired, e.g. Figure 2)
# ---------------------------------------------------------------------------

def wilcoxon_p_and_rank_biserial_r(a, b, continuity_correction=True):
    """
    Paired Wilcoxon signed-rank test between a and b, plus rank-biserial
    correlation r as the effect size (matches Figure 2 caption: "P-values
    represent Wilcoxon signed-rank test significance, and r-values -
    rank-biserial correlation").
    Returns (p_value, r). NaN-safe; drops zero differences per scipy default
    ('wilcox' zero_method).
    """
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]

    d = a - b
    nonzero = d != 0
    d = d[nonzero]
    n = len(d)
    if n == 0:
        return np.nan, np.nan

    try:
        _, p = wilcoxon(a[nonzero], b[nonzero], zero_method='wilcox', correction=False)
    except Exception:
        return np.nan, np.nan

    ranks = rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w == 0:
        return p, np.nan

    if continuity_correction:
        diff = w_plus - mean_w
        z = np.sign(diff) * (np.abs(diff) - 0.5) / np.sqrt(var_w)
    else:
        z = (w_plus - mean_w) / np.sqrt(var_w)

    r = z / np.sqrt(n)
    return p, r


def format_p(p):
    """MATLAB/paper-style p-value formatting: 'p < 0.001', 'p = 0.003', 'p = 0.27'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = NA"
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"
