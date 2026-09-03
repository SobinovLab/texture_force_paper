#!python3.11
"""
Data_Generation_Band_Trials.py -- trial-level vibration band table for the
Figure 4 mixed-effects re-analysis (editor point 4).

The published Figure 4 pipeline
(Final_Manuscript_Code/Figure_4/band_analysis_cd.py) writes
`trials_<Condition>_last.csv` with only Subject / Texture / Normalized Rating /
5 band columns -- it drops the Session and Trial identifiers (see the `row` dict
at band_analysis_cd.py lines 224-226). A trial-level mixed model needs those, so
this script re-runs the *same* extraction with the identifiers retained.

Everything statistically relevant is imported from band_analysis_cd rather than
reimplemented, so band edges (5/25/50/100/400/1000 Hz), the Welch PSD band-RMS
metric, the 10-sigma contact detector and MIN_CONTACT_SEC cannot drift from the
published figure. WINDOW_SEC is the one exception: band_analysis_cd now sets it
to 2.0 (matching the Methods text), but every committed Figure 4 output was
generated at 1.0 s (Final_Manuscript_Code/README.md), so it is overridden to
WINDOW_SEC = 1.0 here to reproduce the published run, and the value used is
reported below.

Structured like Table_1_correct/Multiprocessing.py: build a jobs table, export it
to CSV, map a worker over it with a process pool, collect a long table.

Outputs (into <base_path>/Figure_4_mixed/):
    jobs_list_band_trials.csv                    the job table
    Data_Band_trials_<Condition>.csv             one row per surviving trial
    Data_Band_trial_counts.csv                   trials per subject per condition
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from revision_utils import (  # noqa: E402
    export_csv_vertical, import_csv_vertical, simple_pool,
    summary_to_dictionary, export_dic_to_csv, ensure_dir,
)

# band_analysis_cd sets WINDOW_SEC = 2.0 (Methods text), but the committed
# Figure 4 outputs were produced at 1.0 s. Override here so the trial-level
# re-extraction reproduces the published run. Everything else (bands, contact
# detector, MIN_CONTACT_SEC) is still imported from band_analysis_cd.
WINDOW_SEC = 1.0


def _load_band_analysis(figure4_dir):
    """Import band_analysis_cd from the Final_Manuscript_Code tree."""
    sys.path.insert(0, figure4_dir)
    import band_analysis_cd as bac  # noqa: E402
    return bac


# ---------------------------------------------------------------------------
# Worker -- one session, all its trials
# ---------------------------------------------------------------------------

def process_single_session(figure4_dir, session_path, session_folder, condition,
                           window, window_sec):
    """
    Extract band RMS amplitudes for every usable trial of one session.

    Returns a list of row dicts. This is a transcription of
    band_analysis_cd.load_trials' inner loop with 'Session' and 'Trial' added to
    the emitted row; the numeric operations are the imported ones.
    """
    bac = _load_band_analysis(figure4_dir)

    reports_path = os.path.join(session_path, session_folder, 'reports.csv')
    if not os.path.exists(reports_path):
        return []

    reports = pd.read_csv(reports_path)
    if reports['block type'].iloc[0] != condition[0]:
        return []

    norm_path = os.path.join(session_path, session_folder,
                             'normalized_ratings.csv')
    has_norm_file = os.path.exists(norm_path)
    has_norm_col = 'normalized block rating' in reports.columns
    if not has_norm_file and not has_norm_col:
        return []

    norm_ratings = pd.read_csv(norm_path) if has_norm_file else None
    subject = session_folder.split('_Session')[0]

    rows = []
    for _, trial in reports.iterrows():
        trial_num = int(trial['trial number'])
        texture = trial['texture used']
        rating = (norm_ratings['normalized rating'][trial_num - 1]
                  if has_norm_file else trial['normalized block rating'])

        force_path = os.path.join(session_path, session_folder, 'Force',
                                  f'trial_{trial_num}.csv')
        if not os.path.exists(force_path):
            continue

        fd = pd.read_csv(force_path)
        time = fd['Exact Times'].values
        Fn = fd['Fz'].values.copy()
        Ft = np.sqrt(fd['Fx'].values ** 2 + fd['Fy'].values ** 2)
        sr = len(time) / (time[-1] - time[0])

        # contact detection -- identical to band_analysis_cd.load_trials
        Fn -= np.median(Fn[time < 0.33])
        thr = 10 * np.std(np.abs(Fn[time < 0.33]))
        st = bac._find_first(np.abs(Fn) > thr)
        en = bac._find_last(np.abs(Fn) > thr)
        if st < 0 or en < 0 or (en - st) / sr < bac.MIN_CONTACT_SEC:
            continue

        sig = (Ft if bac.SIGNAL_TYPE == 'Ft' else Fn)[st:en]
        n = int(float(window_sec) * sr)
        sig = sig[-n:] if window == 'last' else sig[:n]

        row = {'Subject': subject,
               'Session': session_folder,
               'Condition': condition,
               'Trial': trial_num,
               'Texture': texture,
               'Normalized Rating': rating,
               'Contact duration s': (en - st) / sr,
               'Sample rate Hz': sr}
        row.update(bac.band_rms(sig, sr))
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Job table
# ---------------------------------------------------------------------------

def generate_jobs(figure4_dir, session_path, jobs_csv, conditions, window,
                  window_sec):
    session_folders = sorted([
        f for f in os.listdir(session_path)
        if (os.path.isdir(os.path.join(session_path, f))
            and re.match(r'^Subject\d+_Session\d+$', f)
            and os.path.exists(os.path.join(session_path, f, 'Force'))
            and os.path.exists(os.path.join(session_path, f, 'reports.csv')))
    ])
    print(f'Found {len(session_folders)} sessions in {session_path}')

    jobs = [(figure4_dir, session_path, sf, condition, window, str(window_sec))
            for condition in conditions
            for sf in session_folders]

    export_csv_vertical(
        jobs_csv,
        ['figure4_dir', 'session_path', 'session_folder', 'condition', 'window',
         'window_sec'],
        jobs)
    print(f'Exported job list ({len(jobs)} jobs) to {jobs_csv}.')
    return jobs


def generate_band_trials(jobs_csv, base_path, processes=13):
    _, jobs = import_csv_vertical(jobs_csv, cast=str)

    results, failed = simple_pool(process_single_session, jobs,
                                  processes=processes, desc='sessions')
    if failed:
        print(f'Failed jobs: {failed}')

    rows = [r for res in results if res for r in res]
    if not rows:
        print('No usable trials found.')
        return None

    long_df = pd.DataFrame(rows)

    # Each worker only emits rows when the session's block matches the requested
    # condition, and it stamps that condition onto every row, so the split is a
    # straight groupby.
    out_dir = ensure_dir(os.path.join(base_path, 'Figure_4_mixed'))

    count_rows = []
    for condition in sorted(long_df['Condition'].unique()):
        sub = long_df[long_df['Condition'] == condition]
        if sub.empty:
            continue
        path = os.path.join(out_dir, f'Data_Band_trials_{condition}.csv')
        sub.to_csv(path, index=False)
        print(f'  {condition}: {len(sub)} trials, '
              f'{sub["Subject"].nunique()} subjects -> {path}')
        for subj, sdf in sub.groupby('Subject'):
            count_rows.append({'Condition': condition, 'Subject': subj,
                               'N trials': len(sdf),
                               'N textures': sdf['Texture'].nunique()})

    counts = pd.DataFrame(count_rows)
    counts.to_csv(os.path.join(out_dir, 'Data_Band_trial_counts.csv'),
                  index=False)
    print('\nPer-subject trial counts written -- these are the ragged Ns that '
          'the per-subject Wilcoxon weights equally and the mixed model does '
          'not.')
    if not counts.empty:
        print(counts.groupby('Condition')['N trials']
              .agg(['min', 'median', 'max', 'sum']).to_string())

    return long_df


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import SESSION_DIR, OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--session_path',
        default=SESSION_DIR)
    ap.add_argument(
        '--figure4_dir',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Figure_4'),
        help='directory containing band_analysis_cd.py')
    ap.add_argument(
        '--base_path',
        default=OUTPUT_DIR)
    ap.add_argument('--processes', type=int, default=13)
    args = ap.parse_args()

    bac = _load_band_analysis(args.figure4_dir)
    print(f'band_analysis_cd config: bands={bac.BAND_LABELS}, '
          f'signal={bac.SIGNAL_TYPE}, MIN_CONTACT_SEC={bac.MIN_CONTACT_SEC}, '
          f'band_analysis_cd.WINDOW_SEC={bac.WINDOW_SEC} '
          f'(overridden here to WINDOW_SEC={WINDOW_SEC} to match the published '
          f'Figure 4 outputs)')

    out_dir = ensure_dir(os.path.join(args.base_path, 'Figure_4_mixed'))
    jobs_csv = os.path.join(out_dir, 'jobs_list_band_trials.csv')

    generate_jobs(args.figure4_dir, args.session_path, jobs_csv,
                  bac.CONDITIONS, 'last', WINDOW_SEC)
    generate_band_trials(jobs_csv, args.base_path, processes=args.processes)
