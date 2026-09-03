#!python3.11
"""
Data_Generation_SuppFig2.py -- editor point 3, Supplementary Figure 2.

Replaces the published test. The panels keep the published intent (per-subject
distributions of normal and tangential force, subjects sorted by increasing
median normal force) and extend it from the hardness and slipperiness blocks to
all three blocks (hardness, slipperiness, roughness); the statistics are rebuilt
on the correct unit.

What was wrong. The only implementation in the repository matching the caption is
`trial_violin_per_subject()` in binned_data_generator_analysis_forces.py
(lines 805-897). It sorts subjects by median force, takes the top and bottom
quartiles (17 // 4 = 4 subjects each), then at lines 838-846 flattens *every row*
of those subjects into two lists and calls `f_oneway` on them. The default column
is 'Bin Median Force Fz', one row per 0.02 s bin, so each of the eight
participants contributes tens of thousands of observations. The test therefore
asks whether two pools of 20 ms bins came from one distribution, not whether two
groups of four people differ; p < 0.001 follows from the row count. The one-tailed
p is also derived by halving a two-sided F-test p (lines 875-878), and since
f_oneway's F is non-negative that branch always fires. The function is never
called from any committed driver script, and Final_Manuscript_Code/README.md
lines 77-79 records that this figure's code was never written, so the published
p-value cannot be traced.

What replaces it.
    1. Descriptive. Fold difference between the median force of the four
       heaviest-touch and four lightest-touch participants, computed on one
       summary value per participant. No test attached -- this is the effect the
       figure is actually reporting, and with four participants per group an
       exact two-sided rank test cannot go below 2/C(8,4) = 0.0286, so a 4-vs-4
       comparison could never have supported p < 0.001.
    2. Inferential, on the right unit. log(force) ~ 1 + (1|Subject) +
       (1|Texture) fit on trial-level data; the subject variance component, its
       ICC, and a likelihood-ratio test against the model without it. This tests
       "participants differ in the force they apply" using trials as replicates
       nested in participants, which is what the claim means, and it doubles as
       part of the point-4 mixed-effects analysis.
    3. Nonparametric companion. Kruskal-Wallis across all participants on
       trial-level force. Participant is the grouping factor and the trial is the
       replicate, so this is not the pseudoreplication above; it is reported
       alongside the LRT because it makes no distributional assumption.

Outputs (into <base_path>/Supp_Figure_2/):
    Data_SuppFig2_per_subject.csv     one row per (Block, Measure, Subject)
    Data_SuppFig2_stats.csv           one row per (Block, Measure)
    Supp_Figure_2.png / .pdf          the four panels
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from revision_utils import (  # noqa: E402
    collapse_bins_to_trials, get_iqr, ensure_dir,
)
from Data_Generation_MixedModels import (  # noqa: E402
    fit_crossed, variance_summary, lrt_drop_component,
)

# Columns and rows of the figure. The published legend showed hardness and
# slipperiness; roughness is added here as the third column.
BLOCKS = ['H', 'S', 'R']
BLOCK_NAMES = {'H': 'Hardness', 'S': 'Slipperiness', 'R': 'Roughness'}
MEASURES = {
    'Median_Force_Fz': 'Normal force (N)',
    'Median_Force_2D_Tangential_Vector': 'Tangential force (N)',
}
SORT_MEASURE = 'Median_Force_Fz'
N_EXTREME = 4          # the four highest- and four lowest-force participants


def per_subject_table(trial_df, block, measure):
    d = trial_df[(trial_df['Block Type'] == block)][['Subject', measure]].dropna()
    if d.empty:
        return pd.DataFrame(), {}
    per = (d.groupby('Subject')[measure]
             .agg(['median', 'mean', 'count'])
             .rename(columns={'median': 'Subject median',
                              'mean': 'Subject mean',
                              'count': 'N trials'})
             .reset_index())
    per = per.sort_values('Subject median').reset_index(drop=True)
    values = {s: g[measure].to_numpy(float)
              for s, g in d.groupby('Subject')}
    return per, values


def extreme_fold_difference(per):
    """Fold difference between the top-N and bottom-N participants' medians."""
    if len(per) < 2 * N_EXTREME:
        return {}
    low = per['Subject median'].to_numpy()[:N_EXTREME]
    high = per['Subject median'].to_numpy()[-N_EXTREME:]
    lo_med, hi_med = np.median(low), np.median(high)
    return {
        'N extreme per group': N_EXTREME,
        'Low group subjects': ', '.join(per['Subject'].to_numpy()[:N_EXTREME]),
        'High group subjects': ', '.join(per['Subject'].to_numpy()[-N_EXTREME:]),
        'Low group median force': float(lo_med),
        'High group median force': float(hi_med),
        'Fold difference': float(hi_med / lo_med) if lo_med > 0 else np.nan,
        'Min attainable two-sided exact p at 4v4': 2.0 / 70.0,
    }


def variance_component_stats(trial_df, block, measure):
    """Subject variance component, ICC and LRT, on log force at trial level."""
    d = trial_df[(trial_df['Block Type'] == block)][
        ['Subject', 'Texture', measure]].dropna().copy()
    d = d[d[measure] > 0]
    if d['Subject'].nunique() < 3 or len(d) < 30:
        return {}
    d['y'] = np.log(d[measure])

    res, vc_names = fit_crossed(d, 'y ~ 1')
    out = variance_summary(res, vc_names, f'Supp Fig 2 {block} {measure}')
    out.pop('Model', None)
    if res is not None:
        out['Subject SD (log units)'] = float(np.sqrt(out.get('Var Subject', np.nan)))
        out['Texture SD (log units)'] = float(np.sqrt(out.get('Var Texture', np.nan)))
    out.update(lrt_drop_component(d, 'y ~ 1', drop='Subject'))
    return out


def kruskal_across_subjects(values):
    """Kruskal-Wallis across participants, trials as replicates."""
    groups = [v for v in values.values() if len(v) >= 2]
    if len(groups) < 2:
        return {}
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return {'Kruskal H': float(h), 'Kruskal p': float(p),
            'Kruskal k groups': len(groups), 'Kruskal N trials': int(n),
            'Kruskal eta2': float((h - len(groups) + 1) / (n - len(groups)))
            if n > len(groups) else np.nan}


def generate_suppfig2_data(force_df, base_path):
    trial_df = collapse_bins_to_trials(force_df, list(MEASURES))

    per_rows, stat_rows, value_store = [], [], {}

    for block in BLOCKS:
        for measure in MEASURES:
            per, values = per_subject_table(trial_df, block, measure)
            if per.empty:
                print(f'  {block}/{measure}: no data')
                continue
            value_store[(block, measure)] = (per, values)

            for _, r in per.iterrows():
                per_rows.append({
                    'Block': block, 'Block name': BLOCK_NAMES[block],
                    'Measure': measure, 'Measure label': MEASURES[measure],
                    **r.to_dict()})

            entry = {'Block': block, 'Block name': BLOCK_NAMES[block],
                     'Measure': measure, 'Measure label': MEASURES[measure],
                     'N subjects': int(len(per))}
            lo, hi = get_iqr(np.concatenate(list(values.values())))
            entry['Trial-level median'] = float(
                np.median(np.concatenate(list(values.values()))))
            entry['Trial-level IQR low'] = float(lo)
            entry['Trial-level IQR high'] = float(hi)
            entry.update(extreme_fold_difference(per))
            print(f'  {block}/{measure}: fitting variance components')
            entry.update(variance_component_stats(trial_df, block, measure))
            entry.update(kruskal_across_subjects(values))
            stat_rows.append(entry)

    per_df = pd.DataFrame(per_rows)
    stats_df = pd.DataFrame(stat_rows)

    out_dir = ensure_dir(os.path.join(base_path, 'Supp_Figure_2'))
    per_df.to_csv(os.path.join(out_dir, 'Data_SuppFig2_per_subject.csv'),
                  index=False)
    stats_df.to_csv(os.path.join(out_dir, 'Data_SuppFig2_stats.csv'),
                    index=False)
    print(f'Saved Supplementary Figure 2 data to: {out_dir}')

    cols = [c for c in ['Block name', 'Measure label', 'Fold difference',
                        'ICC Subject', 'LRT chi2', 'LRT p', 'Kruskal p']
            if c in stats_df.columns]
    if cols:
        print(stats_df[cols].to_string(index=False))

    # Task-averaged force gap between the lightest- and heaviest-touch four
    # participants, on the sort measure (normal force). This is the single
    # descriptive number the main text can quote.
    fz = stats_df[stats_df['Measure'] == SORT_MEASURE]
    if 'Fold difference' in fz.columns:
        fz = fz[fz['Fold difference'].notna()]
    if not fz.empty:
        mean_fold = fz['Fold difference'].mean()
        mean_abs = (fz['High group median force']
                    - fz['Low group median force']).mean()
        print(f'\nNormal force, heaviest vs lightest {N_EXTREME} participants, '
              f'averaged across {len(fz)} tasks '
              f'({", ".join(fz["Block name"])}):')
        print(f'  mean fold difference     = {mean_fold:.2f}x')
        print(f'  mean absolute difference = {mean_abs:.3f} N')
        if 'ICC Subject' in fz.columns:
            mean_icc = np.nanmean(fz['ICC Subject'].to_numpy(float))
            # The participant variance component averaged across tasks -- the
            # number for the main text ("the participant term accounted for
            # X% of the variance in log normal force").
            print(f'  mean ICC Subject         = {mean_icc:.3f} '
                  f'({100 * mean_icc:.1f}% of log normal-force variance)')

    return per_df, stats_df, value_store


def generate_suppfig2_figure(value_store, base_path):
    out_dir = ensure_dir(os.path.join(base_path, 'Supp_Figure_2'))

    fig, axes = plt.subplots(len(MEASURES), len(BLOCKS),
                             figsize=(4.2 * len(BLOCKS), 3.4 * len(MEASURES)),
                             squeeze=False)

    # Subject order is fixed by increasing median NORMAL force, per the legend,
    # and reused for the tangential row so rows are comparable.
    order_by_block = {}
    for block in BLOCKS:
        key = (block, SORT_MEASURE)
        if key in value_store:
            order_by_block[block] = value_store[key][0]['Subject'].tolist()

    for irow, measure in enumerate(MEASURES):
        for icol, block in enumerate(BLOCKS):
            ax = axes[irow][icol]
            key = (block, measure)
            if key not in value_store:
                ax.set_visible(False)
                continue
            _, values = value_store[key]
            order = order_by_block.get(block, sorted(values))
            order = [s for s in order if s in values]
            data = [values[s] for s in order]

            parts = ax.violinplot(data, showmeans=False, showextrema=False,
                                  showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('0.6')
                pc.set_alpha(0.8)

            if len(order) >= 2 * N_EXTREME:
                for i in list(range(N_EXTREME)):
                    parts['bodies'][i].set_facecolor('#4472C4')
                for i in list(range(len(order) - N_EXTREME, len(order))):
                    parts['bodies'][i].set_facecolor('#C00000')

            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels([s.replace('Subject', '') for s in order],
                               fontsize=7)
            ax.set_ylabel(MEASURES[measure])
            ax.set_ylim(bottom=0)
            if irow == 0:
                ax.set_title(BLOCK_NAMES[block])
            if irow == len(MEASURES) - 1:
                ax.set_xlabel('Participant, sorted by median normal force')

    fig.suptitle('Supplementary Figure 2. Per-participant force distributions\n'
                 '(each violin is one participant, trial-level values; blue = '
                 'four lightest, red = four heaviest)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'Supp_Figure_2.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, 'Supp_Figure_2.pdf'))
    plt.close(fig)
    print(f'Saved Supplementary Figure 2 to: {out_dir}')


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import BINNED_FORCE_CSV, OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--force_csv',
        default=BINNED_FORCE_CSV)
    ap.add_argument(
        '--base_path',
        default=OUTPUT_DIR)
    args = ap.parse_args()

    force_df = pd.read_csv(args.force_csv)
    _, _, store = generate_suppfig2_data(force_df, args.base_path)
    generate_suppfig2_figure(store, args.base_path)
