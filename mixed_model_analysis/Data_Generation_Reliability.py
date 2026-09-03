#!python3.11
"""
Data_Generation_Reliability.py -- editor point 2, participant-level reliability.

Each participant rated all 14 textures three times per block, so reliability is
estimable without collecting anything new. For every (subject, block) this builds
the 14 x 3 texture-by-repetition rating matrix and reports:

    ICC(3,1)                two-way mixed, consistency, single measurement
    mean pairwise Spearman  rank-based companion, robust to the max-normalization
    Cronbach's alpha        reliability of the 3-presentation mean

The "repetition" index is the 1st / 2nd / 3rd occurrence of a texture in that
session's trial order (textures were presented in a permuted-block design, so
each texture appears exactly once per cycle of 14 -- see
random_texture_list_generator.rand_texture_generator).

Ratings come from the `Normalized Rating` column of the binned force CSV,
collapsed to one value per trial. That column is the per-session max-normalized
rating (rating_normalization.normalized_ratings): a strictly monotone,
within-subject transform, so the rank-based measures are unaffected by it and
ICC is computed within subject where the scaling is constant.

Also writes the empirical block-order table, which substantiates the Methods
statement that the order of the three perceptual blocks was randomized across
participants.

Outputs (into <base_path>/Reliability/):
    Data_Reliability_per_subject.csv   one row per (Subject, Block)
    Data_Reliability_summary.csv       median [IQR] per block
    Data_Block_order.csv               one row per Subject
    Reliability.png / .pdf             swarm of per-subject ICC by block
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from revision_utils import (  # noqa: E402
    icc_3_1, mean_pairwise_spearman, collapse_bins_to_trials,
    session_order_table, get_iqr, ensure_dir,
)

BLOCKS = ['H', 'S', 'R']
BLOCK_NAMES = {'H': 'Hardness', 'S': 'Slipperiness', 'R': 'Roughness'}
RESPONSE = 'Normalized Rating'
N_TEXTURES = 14
N_REPEATS = 3
ICC_OUTLIER_FLOOR = 0.5   # a participant/block is flagged only if its ICC is
                          # below both the Tukey lower fence and this floor


def cronbach_alpha(matrix):
    """Cronbach's alpha over the k columns (repetitions) of a targets x k matrix."""
    m = np.asarray(matrix, float)
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 2 or k < 2:
        return np.nan
    item_var = m.var(axis=0, ddof=1).sum()
    total_var = m.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_var / total_var)


def build_repeat_matrix(trial_df, subject, block):
    """
    Return a (n_textures x N_REPEATS) matrix of ratings for one subject/block.

    Repetition index = rank of Trial within (Texture), so column 0 is each
    texture's first presentation, column 1 the second, column 2 the third.
    """
    sel = trial_df[(trial_df['Subject'] == subject) &
                   (trial_df['Block Type'] == block)].copy()
    if sel.empty:
        return None
    sel['Repeat'] = (sel.sort_values('Trial')
                        .groupby('Texture')['Trial']
                        .rank(method='first')
                        .astype(int) - 1)
    sel = sel[sel['Repeat'] < N_REPEATS]
    wide = sel.pivot_table(index='Texture', columns='Repeat',
                           values=RESPONSE, aggfunc='mean')
    wide = wide.reindex(columns=range(N_REPEATS))
    return wide


def icc_outlier_pairs(per_subject):
    """
    (Subject, Block) pairs whose ICC(3,1) is below the per-block Tukey lower
    fence (Q1 - 1.5*IQR) AND below ICC_OUTLIER_FLOOR -- both statistically
    unusual within the task and practically poor.

    Returns (pairs, fences): a set of (Subject, block-letter) tuples and a
    dict {block-letter: fence value}.
    """
    pairs, fences = set(), {}
    for block in BLOCKS:
        sub = per_subject[per_subject['Block'] == block]
        icc = sub['ICC(3,1)']
        if icc.notna().sum() < 4:
            continue
        q1, q3 = get_iqr(icc.to_numpy(float))
        fence = q1 - 1.5 * (q3 - q1)
        fences[block] = fence
        for _, r in sub[(icc < fence) & (icc < ICC_OUTLIER_FLOOR)].iterrows():
            pairs.add((r['Subject'], block))
    return pairs, fences


def generate_reliability_data(force_df, base_path):
    trial_df = collapse_bins_to_trials(force_df, [RESPONSE])

    rows = []
    subjects = sorted(trial_df['Subject'].dropna().unique(),
                      key=lambda s: int(s.replace('Subject', '')))

    for subject in subjects:
        for block in BLOCKS:
            wide = build_repeat_matrix(trial_df, subject, block)
            if wide is None or wide.dropna(how='all').empty:
                continue
            mat = wide.to_numpy(dtype=float)
            res = icc_3_1(mat)
            rows.append({
                'Subject': subject,
                'Block': block,
                'Block name': BLOCK_NAMES[block],
                'N textures used': res['n_targets'],
                'N repeats': res['k_raters'],
                'ICC(3,1)': res['icc'],
                'Mean pairwise Spearman rho': mean_pairwise_spearman(mat),
                'Cronbach alpha': cronbach_alpha(mat),
                'N trials': int(np.isfinite(mat).sum()),
            })

    per_subject = pd.DataFrame(rows)

    # A participant whose block matrix has no complete texture triplet yields a
    # NaN ICC(3,1). Count such blocks toward N participants (they were attempted
    # and have rank/alpha where estimable) but flag them, rather than dropping
    # them silently from the N as the earlier `notna().sum()` count did.
    for _, r in per_subject[per_subject['ICC(3,1)'].isna()].iterrows():
        print(f'  WARNING {r["Subject"]} {r["Block"]}: ICC(3,1) not computable '
              f'({int(r["N trials"])} rating cells, no complete 3-presentation '
              f'triplet); block still counted in N participants')

    summary = []
    for block in BLOCKS:
        sub = per_subject[per_subject['Block'] == block]
        if sub.empty:
            continue
        entry = {'Block': block, 'Block name': BLOCK_NAMES[block],
                 'N participants': int(len(sub)),
                 'N with computable ICC': int(sub['ICC(3,1)'].notna().sum())}
        for col in ['ICC(3,1)', 'Mean pairwise Spearman rho', 'Cronbach alpha']:
            v = sub[col].to_numpy(float)
            lo, hi = get_iqr(v)
            entry[f'{col} median'] = np.nanmedian(v)
            entry[f'{col} IQR low'] = lo
            entry[f'{col} IQR high'] = hi
            entry[f'{col} min'] = np.nanmin(v)
        summary.append(entry)
    summary_df = pd.DataFrame(summary)

    # Overall reliability pooled across all participants and all blocks.
    overall = {'Block': 'Overall', 'Block name': 'All blocks',
               'N participants': int(len(per_subject)),
               'N with computable ICC':
                   int(per_subject['ICC(3,1)'].notna().sum())}
    for col in ['ICC(3,1)', 'Mean pairwise Spearman rho', 'Cronbach alpha']:
        v = per_subject[col].to_numpy(float)
        lo, hi = get_iqr(v)
        overall[f'{col} median'] = np.nanmedian(v)
        overall[f'{col} IQR low'] = lo
        overall[f'{col} IQR high'] = hi
        overall[f'{col} min'] = np.nanmin(v)
    summary_df = pd.concat([summary_df, pd.DataFrame([overall])],
                           ignore_index=True)

    order_df = pd.DataFrame(session_order_table(force_df))
    if not order_df.empty:
        counts = order_df['Block order'].value_counts().rename('N participants')
        order_counts = counts.reset_index().rename(columns={'index': 'Block order'})
    else:
        order_counts = pd.DataFrame()

    out_dir = ensure_dir(os.path.join(base_path, 'Reliability'))
    per_subject.to_csv(os.path.join(out_dir, 'Data_Reliability_per_subject.csv'),
                       index=False)
    summary_df.to_csv(os.path.join(out_dir, 'Data_Reliability_summary.csv'),
                      index=False)
    order_df.to_csv(os.path.join(out_dir, 'Data_Block_order.csv'), index=False)
    if not order_counts.empty:
        order_counts.to_csv(os.path.join(out_dir, 'Data_Block_order_counts.csv'),
                            index=False)

    icc_cols = ['Block', 'Block name', 'N participants', 'N with computable ICC',
                'ICC(3,1) median', 'ICC(3,1) IQR low', 'ICC(3,1) IQR high',
                'ICC(3,1) min']
    rho_cols = ['Block', 'Block name',
                'Mean pairwise Spearman rho median',
                'Mean pairwise Spearman rho IQR low',
                'Mean pairwise Spearman rho IQR high',
                'Mean pairwise Spearman rho min']

    print(f'Saved reliability data to: {out_dir}')
    print('\nParticipant counts and ICC(3,1) by block:')
    print(summary_df[[c for c in icc_cols if c in summary_df.columns]]
          .to_string(index=False))
    print('\nMean pairwise Spearman rho by block (H/S/R):')
    print(summary_df[[c for c in rho_cols if c in summary_df.columns]]
          .to_string(index=False))

    # ICC(3,1) outliers: below the per-block Tukey lower fence (Q1 - 1.5*IQR)
    # AND below the absolute floor -- both statistically unusual within the task
    # and practically poor. Candidates for a leave-one-out sensitivity re-run,
    # and marked red on the ICC panel of the figure.
    outliers, fences = icc_outlier_pairs(per_subject)
    print('\nICC(3,1) reliability outliers (below Tukey lower fence Q1-1.5*IQR '
          f'and below {ICC_OUTLIER_FLOOR}):')
    if outliers:
        for subj, block in sorted(outliers):
            icc_val = per_subject.loc[(per_subject['Subject'] == subj) &
                                      (per_subject['Block'] == block),
                                      'ICC(3,1)'].iloc[0]
            print(f'  {subj} {BLOCK_NAMES[block]}: ICC={icc_val:.3f} '
                  f'(fence={fences[block]:.3f})')
    else:
        print('  none')

    print()
    print('Block orders observed (from session numbering):')
    print(order_counts.to_string(index=False) if not order_counts.empty else '  n/a')

    return per_subject, summary_df, order_df


def generate_reliability_figure(per_subject, base_path):
    out_dir = ensure_dir(os.path.join(base_path, 'Reliability'))
    outliers, _ = icc_outlier_pairs(per_subject)

    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)
    for ax, col in zip(axes, ['ICC(3,1)', 'Mean pairwise Spearman rho',
                              'Cronbach alpha']):
        blocks_present = [b for b in BLOCKS if (per_subject['Block'] == b).any()]
        order = [BLOCK_NAMES[b] for b in blocks_present]

        if col == 'ICC(3,1)':
            # Long-form so the flagged outlier participants can be coloured red
            # individually (each dot = one participant, red = ICC outlier).
            d = per_subject[per_subject['Block'].isin(blocks_present)].dropna(
                subset=[col]).copy()
            d['Block name'] = d['Block'].map(BLOCK_NAMES)
            d['Outlier'] = [(s, b) in outliers
                            for s, b in zip(d['Subject'], d['Block'])]
            sns.swarmplot(data=d, x='Block name', y=col, order=order,
                          hue='Outlier', hue_order=[False, True],
                          palette={False: 'k', True: 'r'}, dodge=False,
                          size=4, ax=ax, legend=False)
            medians = [np.nanmedian(d.loc[d['Block name'] == bn, col])
                       for bn in order]
        else:
            data = {BLOCK_NAMES[b]: per_subject.loc[per_subject['Block'] == b, col]
                                              .to_numpy(float)
                    for b in blocks_present}
            sns.swarmplot(data=data, orient='v', ax=ax, color='k', size=4)
            medians = [np.nanmedian(v) for v in data.values()]

        for i, m in enumerate(medians):
            ax.hlines(m, i - 0.3, i + 0.3, color='b', linewidth=2)
        ax.axhline(0, color='0.7', linewidth=0.8)
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylim(-0.2, 1.05)
        ax.tick_params(axis='x', rotation=30)
    axes[0].set_ylabel('Reliability')
    fig.suptitle('Participant-level rating reliability (each dot = one '
                 'participant; red = ICC outlier, blue bar = median)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'Reliability.png'), dpi=300)
    fig.savefig(os.path.join(out_dir, 'Reliability.pdf'))
    plt.close(fig)
    print(f'Saved reliability figure to: {out_dir}')


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
    per_subject, _, _ = generate_reliability_data(force_df, args.base_path)
    generate_reliability_figure(per_subject, args.base_path)
