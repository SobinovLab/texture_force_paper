#!python3.11
import re
import os
import sys
import csv
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import TheilSenRegressor
from tqdm import tqdm
import seaborn

try:
    from reporting_pool import ReportingPool
except ImportError:
    ReportingPool = None  # only needed for the friction-extraction step
try:
    from prehension.tools import stats as pstats
except ImportError:
    pstats = None  # only needed for rating_variability_shuffle


# DEFINES
DN_SPEED_THRESHOLD = 0
# DN_SPEED_THRESHOLD = 5
# DN_SPEED_THRESHOLD = 50
DN_SPEED_MIN_BORDER = DN_SPEED_THRESHOLD
DN_SPEED_MAX_BORDER = 100
DN_SPEED_NUMBINS = 50
DN_SPEED_PHYSICS_THRESHOLD = 50


DN_MIN_EVENT_DURATION_S = 0.1
DN_MIN_EVENT_FRAMES = 5
# DN_MIN_SPEED_PEAK = 5  # cm/sec
DN_PROP_FN_THRESHOLD = .4
DN_PROP_FT_THRESHOLD = .2

# Reliability outliers (participant, block) excluded from the friction-rating
# relationship (Figure 5), matching Data_Generation_Reliability.py. Only
# (Subject6, S) falls in the S/R blocks used here; (Subject7, H) is listed for
# completeness but never matches (Figure 5 has no hardness block).
EXCLUDE_PAIRS = (('Subject6', 'S'), ('Subject7', 'H'))


# UTILS from prehension
def import_csv(filename, cast=float):
    '''Imports csv into a simple structure.

    Arguments:
        filename {str} -- filename to import.
    Keyword Arguments:
        cast {callable} -- class of variables to return. Defaults to float.
    Returns a tuple of:
        column_names {list of M str} -- list of all column names.
        values {list [M][N] of cast type if possible, str otherwise} -- list of all column values.
            First index corresponds to column number.
    '''
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

    # clear empty
    for idof in reversed(range(len(column_names))):
        if (len(column_names[idof]) == 0 and
                all(isinstance(v, str) and len(v) == 0 for v in values[idof])):
            del column_names[idof]
            del values[idof]
    return column_names, values


def export_csv(filename, column_names, values):
    '''Exports from a structure into a csv file.

    Arguments:
        filename {str} -- filename to write.
        column_names {list of M str} -- list of all column names.
        values {list [M][N]} -- list of all column values. First index corresponds to
            column number.
    '''
    with open(filename, 'w', newline='') as f:
        wrr = csv.writer(f)

        wrr.writerow(column_names)

        for itrial in range(len(values[0])):
            lo = [values[k][itrial] for k in range(len(column_names))]
            wrr.writerow(lo)


def export_csv_vertical(filename, column_names, values):
    '''Exports from a structure into a csv file. Orthogonal values structure to standard

    Arguments:
        filename {str} -- filename to write.
        column_names {list of M str} -- list of all column names.
        values {list [N][M]} -- list of all column values. First index corresponds to
            row number.
    '''
    with open(filename, 'w', newline='') as f:
        wrr = csv.writer(f)

        wrr.writerow(column_names)

        for vals in values:
            wrr.writerow(vals)


def import_csv_vertical(filename, cast=float):
    '''Imports csv into a simple structure.

    Arguments:
        filename {str} -- filename to import.
    Keyword Arguments:
        cast {callable} -- class of variables to return. Defaults to float.
    Returns a tuple of:
        column_names {list of M str} -- list of all column names.
        values {list [N][M] of cast type if possible, str otherwise} -- list of all column values.
            First index corresponds to row number.
    '''
    with open(filename, 'r') as f:
        rdr = csv.reader(f)

        line = next(rdr)
        column_names = [i.strip() for i in line]

        values = []

        for li in rdr:
            values.append([])
            for idof, vdof in enumerate(li):
                try:
                    v = cast(vdof)
                except ValueError:
                    v = vdof
                values[-1].append(v)

    return column_names, values


def actual_vline(ax, x, **kwargs):
    '''Draws a vline covering the whole y-range, preserving the previous ylim.'''
    ymin, ymax = ax.get_ylim()
    ax.vlines(x, ymin, ymax, **kwargs)
    ax.set_ylim(ymin, ymax)


def actual_hline(ax, x, **kwargs):
    '''Draws a hline covering the whole x-range, preserving the previous xlim.'''
    xmin, xmax = ax.get_xlim()
    ax.hlines(x, xmin, xmax, **kwargs)
    ax.set_xlim(xmin, xmax)


def annotated_vbar(ax, x, title, ha='center', color='#F44336', linestyle='--', continue_axs=None):
    '''Makes and annotates a vline, continues it onto other axes.'''
    _, ymax = ax.get_ylim()
    actual_vline(ax, x, color=color, linestyle=linestyle)
    ax.annotate(
        title, (x, ymax),
        xycoords='data', ha=ha, va='bottom')
    if continue_axs is not None:
        for ax_ in continue_axs:
            actual_vline(ax_, x, color=color, linestyle=linestyle)


def downsample_at_timeseries(times, data, times_new):
    '''Downsamples data (for example, pressure sensors) to a different set of times.

    times_new has to have a much lower frequency. Neither have to be uniform.
    times_new cannot be wider than times'''
    times = np.array(times)
    data = np.array(data)
    times_new = np.array(times_new)

    data_new = []
    diff_times_new_hvd = np.diff(times_new) / 2
    t_froms = np.insert(times_new[1:] - diff_times_new_hvd, 0, times_new[0])
    t_tos = np.insert(times_new[:-1] + diff_times_new_hvd, len(diff_times_new_hvd), times_new[-1])
    for t_from, t_to in zip(t_froms, t_tos):
        slc = np.logical_and(times >= t_from, times < t_to)
        if sum(slc) == 0:
            # TODO check where and why it happens
            # warnings.warn('downsample_at_timeseries: No corresponding interval found.')
            data_new.append(np.zeros(np.shape(data)[1:]))
        else:
            data_new.append(np.median(data[slc], axis=0))

    # # testing
    # plt.figure()
    # plt.plot(times, reduce_force_matrices(data), 'k')
    # plt.plot(times_new, reduce_force_matrices(data_new), 'r')
    # plt.show()

    return np.array(data_new)


def import_csv_as_dic(filename, cast=float):
    '''Imports csv into a simple structure. The column names are the dictionary keys.

    Arguments:
        filename {str} -- filename to import.
    Keyword Arguments:
        cast {callable} -- class of variables to return. Defaults to float.
    Returns a dictionary of:
        column_names: values {str: N of cast type if possible, str otherwise}
    '''
    return {cn: v for cn, v in zip(*import_csv(filename, cast=cast))}


def export_dic_to_csv(filename, dic):
    export_csv(filename, list(dic.keys()), list(dic.values()))


# UTILS local
def summary_to_dictionary(summary):
    summary_dic = {}
    for el in summary:
        for k, v in el.items():
            if k not in summary_dic.keys():
                summary_dic[k] = []
            summary_dic[k].append(v)
    return summary_dic


def get_iqr(x):
    return np.nanpercentile(x, [25, 75])


def xy_numsubplots(numsubplots):
    '''Calculates the number of columns/rows to fit numsubplots approximately in a square.'''
    yn_subplots = int(np.ceil(np.sqrt(numsubplots)))
    xn_subplots = int(np.ceil(numsubplots / yn_subplots))
    return xn_subplots, yn_subplots


# / UTILS


def process_single_trial(
    session,
    trial,
    image_csv,
    time_csv,
    force_csv,
    image_bin_start,
    image_bin_end,
    force_bin_start,
    force_bin_end,
    texture,
    block,
    diagnostic_dir,
):

    if not all(os.path.exists(p) for p in [image_csv, time_csv, force_csv]):
        return []

    image_df = pd.read_csv(image_csv)
    time_df = pd.read_csv(time_csv)
    force_df = pd.read_csv(force_csv)

    (event_vals, fn_percents, ft_percents, speed_mus, speed_mu_errs
     ) = compute_dynamic_friction_for_trial_all_points(
        image_df,
        time_df,
        force_df,
        image_bin_start,
        image_bin_end,
        force_bin_start,
        force_bin_end,
        session=session,
        trial=trial,
        texture=texture,
        diagnostic_plot=True,
        diagnostic_dir=diagnostic_dir
    )

    rows = []

    for event_idx, (mu, fn_pct, ft_pct, speed_mu, speed_mu_err) in enumerate(
        zip(event_vals, fn_percents, ft_percents, speed_mus, speed_mu_errs)
    ):
        if np.isnan(mu):
            continue

        rows.append({
            "Session": session,
            "Trial": trial,
            "Event": event_idx,
            "Texture": texture,
            "Block": block,
            "Dynamic Friction Coefficient": mu,
            "Fn Stability Percentage": fn_pct,
            "Ft Stability Percentage": ft_pct,
            "Speed mu": ' '.join(f'{v:.4f}' for v in speed_mu),
            "Speed mu err b": ' '.join(f'{v:.4f}' for v in speed_mu_err[0]),
            "Speed mu err a": ' '.join(f'{v:.4f}' for v in speed_mu_err[1]),
        })

    return rows


def compute_dynamic_friction_for_trial_all_points(
    image_df,
    time_df,
    force_df,
    image_bin_start,
    image_bin_end,
    force_bin_start,
    force_bin_end,
    session=None,
    trial=None,
    texture=None,
    diagnostic_plot=False,
    diagnostic_dir=None,
    debug=True
):
    # calculates it at all timepoints and returns median


    # --------------------------------------------------------
    # 1. Keep all rows, reset index
    # --------------------------------------------------------
    image_df = image_df.reset_index(drop=True)
    if len(image_df) < 3:
        return [], [], []

    # check: image frames must index into time_df
    if image_df["frame"].max() >= len(time_df):
        raise ValueError(
            f"Frame/time mismatch: max frame={image_df['frame'].max()}, "
            f"time_df length={len(time_df)} "
            f"(session={session}, trial={trial})"
        )

    # --------------------------------------------------------
    # 2. Speed
    # --------------------------------------------------------
    # to remember:
    # image_df has only a subset of frames with Index, frame as the first two columns
    # time_df has time stamps of all images. Exact times (when captured) and system times
    #       (when saved)
    # Thus:
    #   speed indices are only a subset of frames, matching the image_df organization

    dx = np.diff(image_df["keypoint_8_x"].to_numpy())
    dy = np.diff(image_df["keypoint_8_y"].to_numpy())

    image_frames = image_df["frame"].values
    image_times = time_df["Exact Times"].values
    xy_times = image_times[image_frames]

    dt = np.diff(xy_times)

    speed = np.sqrt(dx**2 + dy**2) / dt
    speed[speed > DN_SPEED_PHYSICS_THRESHOLD] = np.nan

    # --------------------------------------------------------
    # 6. Force times
    # --------------------------------------------------------
    force_times = force_df["Exact Times"].values
    time_offset = force_bin_start - image_bin_start

    # --------------------------------------------------------
    # 7. Forces
    # --------------------------------------------------------
    Fx = force_df["Fx"].values
    Fy = force_df["Fy"].values
    Fz = force_df["Fz"].values

    # There should not be negative normal force values
    # remove the first 500 ms median to reset to 0
    Fn = Fz
    Fn -= np.median(Fn[force_times < 0.5])
    Fn[Fn < 0] = 0

    Ft = np.sqrt(Fx**2 + Fy**2)

    # resample everything to speed
    speed_times = xy_times[:-1]
    Fn_rs = downsample_at_timeseries(force_times - time_offset, Fn, speed_times)
    Ft_rs = downsample_at_timeseries(force_times - time_offset, Ft, speed_times)

    Fn_mean = np.mean(Fn_rs)
    Ft_mean = np.mean(Ft_rs)

    # over the whole trial
    fn_threshold = DN_PROP_FN_THRESHOLD * Fn_mean
    ft_threshold = DN_PROP_FT_THRESHOLD * Ft_mean

    def get_sel_filter(speed_bottom, speed_top):
        sel_filter = np.logical_and(speed > speed_bottom, speed <= speed_top)
        sel_filter = np.logical_and(sel_filter, Fn_rs > fn_threshold)
        sel_filter = np.logical_and(sel_filter, Ft_rs > ft_threshold)
        return sel_filter

    def get_mus(speed_bottom, speed_top):
        sel_filter = get_sel_filter(speed_bottom, speed_top)
        mus = Ft_rs[sel_filter] / Fn_rs[sel_filter]
        return mus

    def get_flagged_estimate(speed_bottom, speed_top):
        mus = get_mus(speed_bottom, speed_top)
        return np.nanmedian(mus), get_iqr(mus)

    # All-time friction
    sel_filter = get_sel_filter(DN_SPEED_THRESHOLD, np.inf)
    mus = get_mus(DN_SPEED_THRESHOLD, np.inf)
    event_friction_values = [get_flagged_estimate(DN_SPEED_THRESHOLD, np.inf)[0]]
    above_threshold_values = [0.8]  # filler
    above_ft_threshold_values = [0.8]

    speed_bins = np.linspace(DN_SPEED_MIN_BORDER, DN_SPEED_MAX_BORDER, DN_SPEED_NUMBINS)
    speed_mus, speed_mu_iqrs = zip(*[
        get_flagged_estimate(sb, st)
        for sb, st in zip(speed_bins, np.append(speed_bins[1:], np.inf))])
    speed_mu_errs = [
        [m-a[0], a[1]-m] if not np.any(np.isnan(a)) else [np.nan, np.nan]
        for a, m in zip(speed_mu_iqrs, speed_mus)]
    speed_mu_errs = np.array(speed_mu_errs).T

    # ------------------------------------------------
    # 9. Diagnostic plotting
    # ------------------------------------------------

    if (diagnostic_plot and diagnostic_dir is not None):
        os.makedirs(diagnostic_dir, exist_ok=True)
        # traces
        # Figure
        fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

        # X and Y Position (image space)
        axes[0].plot(xy_times, image_df["keypoint_8_x"].values, color="k")
        axes[0].set_ylabel("X Position (cm)")

        axes[1].plot(xy_times, image_df["keypoint_8_y"].values, color="k")
        axes[1].set_ylabel("Y Position (cm)")

        # Speed (image space)
        axes[2].plot(speed_times, speed, color="k")
        axes[2].set_ylabel("Speed (cm/s)")
        axes[2].set_ylim(bottom=0)

        # Tangential force
        axes[3].plot(force_times - time_offset, Ft, color="k")
        axes[3].set_ylabel("Ft (N)")
        axes[3].set_ylim(bottom=0)

        # Normal force
        axes[4].plot(force_times - time_offset, Fn, color="k")
        axes[4].set_ylabel("Fn (N)")
        axes[4].set_ylim(bottom=0)
        axes[4].set_xlabel('Time (images), s')

        outname = f"{session}_trial{trial}_tex{texture}.png"
        fig.savefig(os.path.join(diagnostic_dir, outname), dpi=300)
        plt.close(fig)

        # mu vs speed
        fig, ax = plt.subplots(1, 1)
        plt.scatter(speed[sel_filter], mus, c='k', marker='.')
        plt.errorbar(speed_bins, speed_mus, yerr=speed_mu_errs, c='g')
        actual_hline(ax, event_friction_values[0], color='r')
        plt.xlabel('Speed (cm/s)')
        plt.ylabel('Friction coefficient')

        outname = f"{session}_trial{trial}_tex{texture}_scatter.png"
        fig.savefig(os.path.join(diagnostic_dir, outname), dpi=300)
        plt.close(fig)

        # mu vs Fn
        fig, ax = plt.subplots(1, 1)
        plt.scatter(Fn_rs[sel_filter], mus, c='k', marker='.')
        actual_hline(ax, event_friction_values[0], color='r')
        plt.xlabel('Normal force, N')
        plt.ylabel('Friction coefficient')
        outname = f"{session}_trial{trial}_tex{texture}_scatter_fn.png"
        fig.savefig(os.path.join(diagnostic_dir, outname), dpi=300)
        plt.close(fig)

        # mu vs Ft
        fig, ax = plt.subplots(1, 1)
        plt.scatter(Ft_rs[sel_filter], mus, c='k', marker='.')
        actual_hline(ax, event_friction_values[0], color='r')
        plt.xlabel('Tangential force, N')
        plt.ylabel('Friction coefficient')
        outname = f"{session}_trial{trial}_tex{texture}_scatter_ft.png"
        fig.savefig(os.path.join(diagnostic_dir, outname), dpi=300)
        plt.close(fig)

        # Fn vs speed
        fig, ax = plt.subplots(1, 1)
        plt.scatter(speed[sel_filter], Fn_rs[sel_filter], c='k', marker='.')
        plt.xlabel('Speed (cm/s)')
        plt.ylabel('Normal force, N')
        outname = f"{session}_trial{trial}_tex{texture}_scatter_speed_vs_fn.png"
        fig.savefig(os.path.join(diagnostic_dir, outname), dpi=300)
        plt.close(fig)

    return (event_friction_values, above_threshold_values, above_ft_threshold_values,
            [speed_mus], [speed_mu_errs])


def generate_jobs_long_table(
    binned_image_data,
    binned_force_df,
    data_folder,
    diagnostic_dir,
    jobs_csv,
    subject_min=9,
    subject_max=17
):
    """
    Computes EVENT-LEVEL dynamic friction coefficients.
    One row per (Session, Trial, Event).

    Only includes subjects in [subject_min, subject_max].
    """

    # --------------------------------------------------
    # Determine and FILTER sessions by subject range
    # AND block type (R/S only)
    # --------------------------------------------------
    all_sessions = (
        binned_force_df["Session"]
        .dropna()
        .str.replace("_preprocessed", "", regex=True)
        .unique()
    )

    sessions = []

    for session in all_sessions:
        match = re.search(r"Subject(\d+)", session)
        if match is None:
            continue

        subject_id = int(match.group(1))
        if not (subject_min <= subject_id <= subject_max):
            continue

        # Since each session has ONE block type, we can check it directly
        session_block = (
            binned_force_df
            .loc[
                binned_force_df["Session"].str.replace("_preprocessed", "", regex=True) == session,
                "Block Type"
            ]
            .dropna()
            .iloc[0]
        )

        if session_block not in {"R", "S"}:
            continue

        sessions.append(session)

    sessions = sorted(set(sessions))


    print(f"Including subjects {subject_min}–{subject_max}: {len(sessions)} sessions")

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    jobs = []

    image_sessions = set(binned_image_data["Session"].unique())

    for session in tqdm(sessions, desc="Sessions"):

        if session not in image_sessions:
            continue

        session_path = os.path.join(data_folder, session)
        preproc_session = session + "_preprocessed"
        preproc_path = os.path.join(data_folder, preproc_session)

        reports_path = os.path.join(session_path, "reports.csv")
        if not os.path.exists(reports_path):
            continue

        reports_df = pd.read_csv(reports_path)

        for trial in reports_df["trial number"].unique():
            image_csv = os.path.join(
                session_path, "Images", f"trial_{trial}",
                "prediction", "processed_keypoint_data_real_coords.csv"
            )
            time_csv = os.path.join(
                session_path, "Images", f"trial_{trial}",
                f"trial_{trial}_times.csv"
            )
            force_csv = os.path.join(
                preproc_path, "Force", f"trial_{trial}.csv"
            )

            image_bin_start = binned_image_data.query(
                "Session == @session and Trial == @trial"
            )["Bin Start Times"].min()

            image_bin_end = binned_image_data.query(
                "Session == @session and Trial == @trial"
            )["Bin Start Times"].max()

            force_bin_start = binned_force_df.query(
                "Session.str.contains(@preproc_session) and Trial == @trial",
                engine="python"
            )["Bin Start Times"].min()

            force_bin_end = binned_force_df.query(
                "Session.str.contains(@preproc_session) and Trial == @trial",
                engine="python"
            )["Bin Start Times"].max()

            texture = reports_df.loc[
                reports_df["trial number"] == trial,
                "texture used"
            ].values[0]

            block = binned_force_df.query(
                "Session.str.contains(@preproc_session) and Trial == @trial",
                engine="python"
            )["Block Type"].iloc[0]

            jobs.append((
                session,
                trial,
                image_csv,
                time_csv,
                force_csv,
                image_bin_start,
                image_bin_end,
                force_bin_start,
                force_bin_end,
                texture,
                block,
                diagnostic_dir
            ))

    # export to a file
    column_names = ['session',
                    'trial',
                    'image_csv',
                    'time_csv',
                    'force_csv',
                    'image_bin_start',
                    'image_bin_end',
                    'force_bin_start',
                    'force_bin_end',
                    'texture',
                    'block',
                    'diagnostic_dir']
    export_csv_vertical(jobs_csv, column_names, jobs)

    print(f'Exported job list to {jobs_csv}.')


def generate_dynamic_friction_long_table(
    jobs_csv,
    dynamic_friction_long_table_csv,
):
    column_names, jobs = import_csv_vertical(jobs_csv)
    processes = 13

    # # HACK work in progress
    # process_single_trial(*(jobs[0]))

    # sys.exit()

    # jobs = jobs[:5]

    # ################################

    pool = ReportingPool(process_single_trial, jobs, processes=processes,
                         report_on_change=True, track_failures=True)
    results = pool.start()

    if len(pool.failed_i_jobs) > 0:
        print()
        print(f'Failed to transform rows: {pool.failed_i_jobs}')

    # Save long table
    results = sum(results, [])
    long_dic = summary_to_dictionary(results)

    export_dic_to_csv(dynamic_friction_long_table_csv, long_dic)

    print(f"Long-form friction table saved with {len(results)} rows to"
          f" {dynamic_friction_long_table_csv}.")

    return long_dic


def generate_dynamic_friction_summary_table(
        dynamic_friction_long_table_csv, summary_table_csv,
        long_dic=None
):
    """
    Produces texture-level summary statistics from trial-level data.
    """
    if long_dic is None:
        long_dic = import_csv_as_dic(dynamic_friction_long_table_csv)

    summary = []
    textures = set(long_dic['Texture'])

    for texture in textures:
        texture_rows = [i for i, v in enumerate(long_dic['Texture']) if v == texture]
        values = [long_dic['Dynamic Friction Coefficient'][i] for i in texture_rows]

        n_events = len(texture_rows)
        n_subjects = len(set([long_dic['Session'][i] for i in texture_rows]))

        mean = np.nanmean(values)
        std = np.nanstd(values, ddof=1)
        ci = 1.96 * std / np.sqrt(n_events)  # TODO ARS not sure

        summary.append({
            "Texture": texture,
            "Number of subjects": n_subjects,
            "Number of events": n_events,
            "Mean dynamic friction coefficient": round(mean, 3),
            "95% CI": round(ci, 3)
        })

    summary_dic = summary_to_dictionary(summary)

    export_dic_to_csv(summary_table_csv, summary_dic)

    print(f"Summary table saved with {len(summary)} textures into {summary_table_csv}.")

    return summary_dic


def generate_fn_stability_vs_friction_figure(long_df, output_dir):
    """
    Event-level Fn stability percentage vs dynamic friction coefficient.
    Diagnostic / exploratory figure only.
    """

    required_cols = {
        "Dynamic Friction Coefficient",
        "Fn Stability Percentage",
        "Block"
    }
    if not required_cols.issubset(long_df.columns):
        raise ValueError(
            f"Missing required columns: {required_cols - set(long_df.columns)}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Save source data
    long_df.to_csv(
        os.path.join(
            output_dir,
            "Figure_FnStability_vs_Friction_SourceData.csv"
        ),
        index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, block in zip(axes, ["R", "S"]):

        sub = long_df[long_df["Block"] == block]
        if sub.empty:
            continue

        x = sub["Fn Stability Percentage"].values
        y = sub["Dynamic Friction Coefficient"].values

        ax.scatter(
            x,
            y,
            s=40,
            alpha=0.75,
            color="black"
        )

        # Robust regression
        model = TheilSenRegressor(random_state=0)
        model.fit(x.reshape(-1, 1), y)

        xfit = np.linspace(x.min(), x.max(), 200)
        yfit = model.predict(xfit.reshape(-1, 1))

        rho, p = stats.spearmanr(x, y)

        ax.plot(xfit, yfit, "--", color="black", linewidth=1.5)
        ax.text(
            0.05,
            0.95,
            f"Spearman $\\rho$={rho:.2f}\n$p$={p:.3g}",
            transform=ax.transAxes,
            va="top"
        )

        ax.set_title(f"{block} Block")
        ax.set_xlabel("Fn Stability Percentage (%)")
        ax.set_ylabel("Dynamic Friction Coefficient")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "FnStability_vs_Friction.png"),
        dpi=300
    )
    fig.savefig(
        os.path.join(output_dir, "FnStability_vs_Friction.pdf")
    )
    plt.close(fig)


def generate_friction_vs_rating_figure(long_df, binned_force_df, output_dir):
    """
    Plots ALL trial-level friction vs rating points, colored by texture.
    """

    # Trial-level ratings
    rating_df = (
        binned_force_df
        .assign(Session=lambda d: d["Session"].str.replace("_preprocessed", "", regex=True))
        .groupby(["Session", "Trial", "Texture", "Block Type"])["Normalized Rating"]
        .apply(lambda x: x.iloc[0])
        .reset_index()
        .rename(columns={"Normalized Rating": "Rating"})
    )

    merged = pd.merge(
        long_df,
        rating_df,
        on=["Session", "Trial", "Texture"],
        how="inner"
    )

    os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(
        os.path.join(output_dir, "Figure_Friction_vs_Rating_SourceData.csv"),
        index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    textures = sorted(merged["Texture"].unique())
    cmap = plt.cm.get_cmap("tab20", len(textures))
    tex_color = dict(zip(textures, cmap.colors))

    for ax, block in zip(axes, ["R", "S"]):

        sub = merged[merged["Block"] == block]
        if sub.empty:
            continue

        for tex in textures:
            df_tex = sub[sub["Texture"] == tex]
            if df_tex.empty:
                continue

            ax.scatter(
                df_tex["Rating"],
                df_tex["Dynamic Friction Coefficient"],
                color=tex_color[tex],
                alpha=0.75,
                s=40
            )

        x = sub["Rating"].values
        y = sub["Dynamic Friction Coefficient"].values

        # Robust Theil–Sen regression
        model = TheilSenRegressor(random_state=0)
        model.fit(x.reshape(-1, 1), y)

        xfit = np.linspace(x.min(), x.max(), 200)
        yfit = model.predict(xfit.reshape(-1, 1))

        # Spearman correlation for robust association strength
        rho, p = stats.spearmanr(x, y)

        ax.plot(xfit, yfit, "--", color="black", linewidth=1.5)
        ax.text(
            0.05, 0.95,
            f"Spearman $\\rho$={rho:.2f}\n$p$={p:.3g}",
            transform=ax.transAxes,
            va="top"
        )


        ax.set_title(f"{block} Block")
        ax.set_xlabel("Normalized Rating")
        ax.set_ylabel("Dynamic Friction Coefficient")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "Friction_vs_Rating.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "Friction_vs_Rating.pdf"))

    return merged


def generate_friction_vs_rating_texture_averaged_figure(
    dynamic_friction_long_table_csv,
    binned_force_df,
    output_dir,
    exclude_high_friction=False,
    friction_threshold=1.5645,
    per_subject=False
):
    """
    Texture-level friction (block-agnostic) vs
    texture-level ratings (block-specific).

    If per_subject=True, also generates one figure per subject
    (using the same logic, restricted to that subject).
    """
    # load the csv
    long_dic = import_csv_as_dic(dynamic_friction_long_table_csv)
    df = pd.DataFrame(long_dic)

    # if exclude_high_friction:
    #     df = df[df["Dynamic Friction Coefficient"] <= friction_threshold]

    # --------------------------------------------------
    # TEXTURE-LEVEL FRICTION (ignore block)
    # --------------------------------------------------
    friction_tex = (
        df
        .groupby("Texture")["Dynamic Friction Coefficient"]
        .mean()
        .reset_index()
        .rename(columns={"Dynamic Friction Coefficient": "Friction"})
    )

    # --------------------------------------------------
    # TEXTURE-LEVEL RATINGS PER BLOCK
    # --------------------------------------------------
    rating_df = (
        binned_force_df
        .assign(Session=lambda d: d["Session"].str.replace("_preprocessed", "", regex=True))
        .groupby(["Session", "Trial", "Texture", "Block Type"])["Normalized Rating"]
        .apply(lambda x: x.iloc[0])
        .reset_index()
        .rename(columns={"Normalized Rating": "Rating"})
    )

    rating_df = rating_df[rating_df["Block Type"].isin(["R", "S"])]

    rating_tex = (
        rating_df
        .groupby(["Texture", "Block Type"])["Rating"]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------
    # Merge: friction ⟷ ratings (GROUP LEVEL)
    # --------------------------------------------------
    merged = pd.merge(
        rating_tex,
        friction_tex,
        on="Texture",
        how="inner"
    )

    os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(
        os.path.join(
            output_dir,
            "Figure_Friction_vs_Rating_TextureAveraged_SourceData.csv"
        ),
        index=False
    )

    # --------------------------------------------------
    # GROUP-LEVEL PLOT
    # --------------------------------------------------
    def plot_panel(data, title_suffix, save_path):

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for ax, block in zip(axes, ["R", "S"]):

            sub = data[data["Block Type"] == block]
            if sub.empty:
                continue

            x = sub["Rating"].values
            y = sub["Friction"].values

            ax.scatter(x, y, s=90, color="black", alpha=0.85)

            model = TheilSenRegressor(random_state=0)
            model.fit(x.reshape(-1, 1), y)

            xfit = np.linspace(x.min(), x.max(), 200)
            yfit = model.predict(xfit.reshape(-1, 1))

            rho, p = stats.spearmanr(x, y)

            ax.plot(xfit, yfit, "--", color="black", linewidth=1.5)
            ax.text(
                0.05, 0.95,
                f"Spearman $\\rho$={rho:.2f}\n$p$={p:.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=11
            )

            ax.set_title(f"{block} Block {title_suffix}")
            ax.set_xlabel("Mean Normalized Rating")
            ax.set_ylabel("Dynamic Friction Coefficient")
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path + ".png", dpi=300)
        fig.savefig(save_path + ".pdf")
        # plt.show()
        plt.close(fig)

    # Group figure
    plot_panel(
        merged,
        title_suffix="(Texture-level)",
        save_path=os.path.join(
            output_dir,
            "Friction_vs_Rating_TextureAveraged"
        )
    )

    # --------------------------------------------------
    # PER-SUBJECT FIGURES
    # --------------------------------------------------
    if per_subject:

        # Attach subject to ratings
        rating_df["Subject"] = rating_df["Session"].str.extract(r"(Subject\d+)")

        df["Subject"] = df["Session"].str.extract(r"(Subject\d+)")

        for subject in sorted(df["Subject"].dropna().unique()):

            df_sub = df[df["Subject"] == subject]
            rating_sub = rating_df[rating_df["Subject"] == subject]

            if df_sub.empty or rating_sub.empty:
                continue

            friction_tex_sub = (
                df_sub
                .groupby("Texture")["Dynamic Friction Coefficient"]
                .mean()
                .reset_index()
                .rename(columns={"Dynamic Friction Coefficient": "Friction"})
            )

            rating_tex_sub = (
                rating_sub
                .groupby(["Texture", "Block Type"])["Rating"]
                .mean()
                .reset_index()
            )

            merged_sub = pd.merge(
                rating_tex_sub,
                friction_tex_sub,
                on="Texture",
                how="inner"
            )

            if merged_sub.empty:
                continue

            plot_panel(
                merged_sub,
                title_suffix=f"({subject})",
                save_path=os.path.join(
                    output_dir,
                    f"Friction_vs_Rating_TextureAveraged_{subject}"
                )
            )

    print("saved")
    return merged


def _prep_1d(xs):
    """Convert to float array and track finite values."""
    x = np.asarray(xs, dtype=float).ravel()
    finite = np.isfinite(x)
    xf = x[finite]
    return x, finite, xf


def outliers_iqr(xs, k=1.5):
    """
    Tukey fences using IQR (k=1.5 for mild outliers, k=3.0 for extreme).
    Returns: mask, indices, (lower, upper)
    """
    x, finite, xf = _prep_1d(xs)
    if xf.size == 0:
        return np.zeros_like(x, dtype=bool), np.array([], dtype=int), (np.nan, np.nan)

    q1, q3 = np.percentile(xf, [25, 75])
    iqr = stats.iqr(xf, rng=(25, 75), nan_policy="omit")
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = np.zeros_like(x, dtype=bool)
    mask[finite] = (xf < lower) | (xf > upper)
    return mask, np.flatnonzero(mask), (lower, upper)


def scatter_regressor(ax, xs, ys, no_outliers=False, show_outliers=True):
    nanflag = np.logical_and(~np.isnan(xs), ~np.isnan(ys))

    xs = np.array(xs)[nanflag]
    ys = np.array(ys)[nanflag]

    mask, fnzmask, (lower_bnd, upper_bnd) = outliers_iqr(xs, k=1.5)
    y_mask, y_fnzmask, (y_lower_bnd, y_upper_bnd) = outliers_iqr(ys, k=1.5)

    if show_outliers:
        ax.scatter(xs, ys, c='k', marker='.')
        ax.scatter(xs[mask], ys[mask], c='r', marker='.')
        ax.scatter(xs[y_mask], ys[y_mask], c='g', marker='o')

    if no_outliers:
        xs = xs[np.logical_and(~mask, ~y_mask)]
        ys = ys[np.logical_and(~mask, ~y_mask)]

    if not show_outliers:
        ax.scatter(xs, ys, c='k', marker='.')

    # regressor
    if len(xs) == 0:
        return
    result = scipy.stats.linregress(xs, ys)
    xfit = [min(xs), max(xs)]
    yfit = [result.slope * x + result.intercept for x in xfit]
    ax.plot(xfit, yfit, "--", color="black", linewidth=1.5)
    ax.text(
        0.05, 0.95,
        f"$p$={result.pvalue:.3g}$r$={result.rvalue:.2f}$N$={len(xs)}\n",
        transform=ax.transAxes,
        va="top",
        fontsize=11
    )


def average_mu_per_speed(subjects, textures, subject_sessions, long_dic, speed_mus, speed_bins):
    # average mu at each speed within subject
    xn_subplots, yn_subplots = xy_numsubplots(len(subjects))
    fig, axs = plt.subplots(xn_subplots, yn_subplots)
    axs = axs.flatten()

    for subject, ax in zip(subjects, axs):
        for texture in textures:
            s_texture_rows = [irow for irow in subject_sessions[subject]
                              if long_dic['Texture'][irow] == texture]
            s_texture_mus = np.nanmean(speed_mus[s_texture_rows], axis=0)
            ax.plot(speed_bins, s_texture_mus, label=texture)
        ax.set_xlabel('Speed, cm/s')
        ax.set_ylabel('Friction mu')
    plt.suptitle('Average friction at speed per subject per texture')


def iqr_mu_per_speed(subjects, textures, subject_sessions, long_dic, speed_mu_iqr, speed_bins):
    # iqr of mu at each speed within subject
    xn_subplots, yn_subplots = xy_numsubplots(len(subjects))
    fig, axs = plt.subplots(xn_subplots, yn_subplots)
    axs = axs.flatten()

    for subject, ax in zip(subjects, axs):
        for texture in textures:
            s_texture_rows = [irow for irow in subject_sessions[subject]
                              if long_dic['Texture'][irow] == texture]
            s_texture_mu_iqrs = np.nanmean(speed_mu_iqr[s_texture_rows], axis=0)
            ax.plot(speed_bins, s_texture_mu_iqrs, label=texture)
        ax.set_xlabel('Speed, cm/s')
        ax.set_ylabel('Friction mu IQR')
    plt.suptitle('IQR of friction at speed per subject per texture')


def mu_rating_at_speed_range(
        xn_subplots, yn_subplots, sp_fr, sp_to, speed_bins, subjects,
        textures, subject_sessions, long_dic, speed_mus, no_outliers=False
):
    fig, axs = plt.subplots(xn_subplots, yn_subplots)
    # axs = axs.flatten()

    speed_idcs = np.logical_and(speed_bins > sp_fr, speed_bins <= sp_to)
    xs_r_all = []
    ys_r_all = []
    xs_r_all_n = []
    xs_s_all = []
    ys_s_all = []
    xs_s_all_n = []

    for isubject, subject in enumerate(subjects):
        def tex_sub(ax, block_type):
            xs = []
            # xs_expand = []
            ys = []
            # ys_expand = []
            for texture in textures:
                texture_int = int(texture)
                s_texture_rows = [irow for irow in subject_sessions[subject]
                                  if long_dic['Texture'][irow] == texture]
                s_texture_mus = speed_mus[s_texture_rows]
                s_texture_mus = s_texture_mus[:, speed_idcs]
                # s_texture_mus_pt = np.nanmean(s_texture_mus, axis=1).tolist()
                s_texture_mus = np.nanmean(s_texture_mus)

                # rating
                mean_norm_rating = (
                    binned_force_df.loc[
                        (binned_force_df["Subject"] == subject) &
                        (binned_force_df["Block Type"] == block_type) &
                        (binned_force_df["Texture"] == texture_int),
                        "Normalized Rating"
                    ]
                    .mean()
                )

                # norm_ratings = []
                # for irow in s_texture_rows:
                #     rating = binned_force_df.loc[
                #         (binned_force_df["Subject"] == subject) &
                #         (binned_force_df["Block Type"] == block_type) &
                #         (binned_force_df["Texture"] == texture_int) &
                #         (binned_force_df["Session"] == sessions_l[irow]) &
                #         (binned_force_df["Trial"] == trials[irow]),
                #         "Normalized Rating"
                #     ].values
                #     if len(rating) == 0:
                #         norm_ratings.append(np.nan)
                #     else:
                #         norm_ratings.append(rating.tolist()[0])
                # if len(norm_ratings) != len(s_texture_mus_pt):
                #     print('lengths dom not match')
                #     print(s_texture_rows)
                #     print(len(norm_ratings))
                #     print(len(s_texture_mus_pt))
                #     raise ValueError()

                # s_texture_mus_pt = [
                #     s_texture_mus if np.isnan(v) else v for v in s_texture_mus_pt]
                # norm_ratings = [mean_norm_rating if np.isnan(v) else v for v in norm_ratings]

                xs.append(s_texture_mus)
                ys.append(mean_norm_rating)
                # xs_expand += s_texture_mus_pt
                # ys_expand += norm_ratings

            scatter_regressor(ax, xs, ys, no_outliers=no_outliers)
            return xs, ys

        # R
        ax = axs[0, isubject]
        xs, ys = tex_sub(ax, 'R')
        if (subject, 'R') not in EXCLUDE_PAIRS:
            xs_r_all += xs
            ys_r_all += ys
            mi_xs = min(xs)
            ma_xs = max(xs)
            xs_r_all_n += [(v - mi_xs) / (ma_xs - mi_xs) for v in xs]
        ax.set_title(subject)

        # S
        ax = axs[1, isubject]
        xs, ys = tex_sub(ax, 'S')
        if (subject, 'S') not in EXCLUDE_PAIRS:
            xs_s_all += xs
            ys_s_all += ys
            mi_xs = min(xs)
            ma_xs = max(xs)
            xs_s_all_n += [(v - mi_xs) / (ma_xs - mi_xs) for v in xs]
        ax.set_xlabel('Friction coefficient')


    fig.suptitle(f'Speeds from {sp_fr:.1f} to {sp_to:.1f}{" NO OUTLIERS" if no_outliers else ""}')
    axs[0, 0].set_ylabel('Roughness\nNormalized rating')
    axs[1, 0].set_ylabel('Slipperiness\nNormalized rating')

    # summaries - 1
    ax = axs[0, len(subjects)]
    scatter_regressor(ax, xs_r_all, ys_r_all, no_outliers=no_outliers)
    ax.set_title('All subjects')

    ax = axs[1, len(subjects)]
    scatter_regressor(ax, xs_s_all, ys_s_all, no_outliers=no_outliers)
    ax.set_xlabel('Friction coefficient')

    # summaries - 2
    ax = axs[0, len(subjects)+1]
    scatter_regressor(ax, xs_r_all_n, ys_r_all, no_outliers=no_outliers)
    ax.set_title('All subjects normed')

    ax = axs[1, len(subjects)+1]
    scatter_regressor(ax, xs_s_all_n, ys_s_all, no_outliers=no_outliers)
    ax.set_xlabel('Friction coefficient')

    # main text figure
    if no_outliers:
        fig, axs = plt.subplots(1, 2)
        ax = axs[0]
        scatter_regressor(ax, xs_r_all, ys_r_all, no_outliers=no_outliers, show_outliers=False)
        ax.set_title('Roughness')
        ax.set_xlabel('Friction coefficient')
        ax.set_ylabel('Normalized rating')

        ax = axs[1]
        scatter_regressor(ax, xs_s_all, ys_s_all, no_outliers=no_outliers, show_outliers=False)
        ax.set_title('Slipperiness')
        ax.set_xlabel('Friction coefficient')
        ax.set_ylabel('Normalized rating')

        fig.suptitle(f'Speeds from {sp_fr:.1f} to {sp_to:.1f} NO OUTLIERS')


def generate_friction_relationships(dynamic_friction_long_table_csv, binned_force_df):
    long_dic = import_csv_as_dic(dynamic_friction_long_table_csv)

    subjects = sorted(
        list(set([v.split('_')[0] for v in long_dic['Session']])),
        key=lambda x: int(x[7:]))
    textures = sorted(list(set(long_dic['Texture'])))
    trials = [int(v) for v in long_dic['Trial']]

    speed_mus = [[float(vv) for vv in v.split(' ')] for v in long_dic['Speed mu']]
    speed_mus = np.array(speed_mus)
    speed_mu_err_b = [[float(vv) for vv in v.split(' ')] for v in long_dic['Speed mu err b']]
    speed_mu_err_b = np.array(speed_mu_err_b)
    speed_mu_err_a = [[float(vv) for vv in v.split(' ')] for v in long_dic['Speed mu err a']]
    speed_mu_err_a = np.array(speed_mu_err_a)
    speed_mu_iqr = speed_mu_err_b + speed_mu_err_a

    # read the dataframe for ratings
    # block_types = ['R', 'S']
    # Attach subject to ratings
    binned_force_df["Subject"] = binned_force_df["Session"].str.extract(r"(Subject\d+)")
    binned_force_df["Session"] = binned_force_df["Session"].str.extract(r"(Session\d+)")

    subject_sessions = {
        s: [] for s in subjects
    }
    subjects_l = []
    sessions_l = []
    for ises, sess in enumerate(long_dic['Session']):
        for s in subjects:
            if sess.startswith(s+'_'):
                subject_sessions[s].append(ises)
                subjects_l.append(s)
                sessions_l.append(sess.split('_')[1])
                break
    # CONST
    speed_bins = np.linspace(DN_SPEED_MIN_BORDER, DN_SPEED_MAX_BORDER, DN_SPEED_NUMBINS)

    # average mu at each speed within subject
    # average_mu_per_speed(subjects, textures, subject_sessions, long_dic, speed_mus, speed_bins)


    # iqr of mu at each speed within subject
    # iqr_mu_per_speed(subjects, textures, subject_sessions, long_dic, speed_mu_iqr, speed_bins)

    # friction-rating relationships at each speed range
    speed_ranges = [0, 10, 20, 30, 40, 50, 75, 100]
    xn_subplots = 2
    yn_subplots = len(subjects) + 2

    # for sp_fr, sp_to in tqdm(zip(speed_ranges, speed_ranges[1:]), desc='Speeds', ncols=100):
    #     mu_rating_at_speed_range(
    #         xn_subplots, yn_subplots, sp_fr, sp_to, speed_bins, subjects,
    #         textures, subject_sessions, long_dic, speed_mus)


    final_range = [0, 30]  # cm/c
    mu_rating_at_speed_range(
        xn_subplots, yn_subplots, final_range[0], final_range[1], speed_bins, subjects,
        textures, subject_sessions, long_dic, speed_mus)


    mu_rating_at_speed_range(
        xn_subplots, yn_subplots, final_range[0], final_range[1], speed_bins, subjects,
        textures, subject_sessions, long_dic, speed_mus, no_outliers=True)

    plt.show()


def rating_variability_shuffle(binned_force_df):
    subjects = binned_force_df['Subject'].to_numpy()
    block_types = binned_force_df['Block Type'].to_numpy()
    textures = binned_force_df['Texture'].to_numpy()
    ratings = binned_force_df['Normalized Rating'].to_numpy()

    print(set(subjects))

    isnan = np.isnan(ratings)
    subjects = subjects[~isnan]
    block_types = block_types[~isnan]
    textures = textures[~isnan]
    ratings = ratings[~isnan]

    u_textures = sorted(list(set(textures)))
    u_subjects = sorted(list(set(subjects)))
    print(u_subjects)
    sys.exit()

    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    fig2, axs2 = plt.subplots(1, 3, figsize=(8, 4), sharey=True)
    # split ratings by subjects
    for block_type, ax, ax2 in zip(['H', 'R', 'S'], axs, axs2):
        s_bt = subjects[block_types == block_type]
        t_bt = textures[block_types == block_type]
        r_bt = ratings[block_types == block_type]

        bs_ratings = []
        bs_ratings_shuffled = []
        bs_ratings_avg = []
        bs_ratings_avg_shuffled = []
        for subject in u_subjects:
            bs_ratings_avg.append([])
            bs_ratings_avg_shuffled.append([])
            t_bt_s = t_bt[s_bt == subject]
            r_bt_s = r_bt[s_bt == subject]

            t_bt_s_shuffled = np.copy(t_bt_s)
            np.random.shuffle(t_bt_s_shuffled)

            for texture in u_textures:
                bs_ratings.append(scipy.stats.sem(r_bt_s[t_bt_s == texture]))
                bs_ratings_shuffled.append(scipy.stats.sem(r_bt_s[t_bt_s_shuffled == texture]))
                bs_ratings_avg[-1].append(np.nanmean(r_bt_s[t_bt_s == texture]))
                bs_ratings_avg_shuffled[-1].append(np.nanmean(r_bt_s[t_bt_s_shuffled == texture]))

        # sem between subjects
        bs_ratings_bws = []
        bs_ratings_bws_shuffled = []
        for it, t in enumerate(u_textures):
            bs_ratings_bws.append(scipy.stats.sem([r[it] for r in bs_ratings_avg],
                                                  nan_policy='omit'))
            bs_ratings_bws_shuffled.append(scipy.stats.sem([r[it] for r in bs_ratings_avg_shuffled],
                                                           nan_policy='omit'))

        # no subject split for sem
        bs_ratings_nosubj = []
        bs_ratings_nosubj_shuffled = []

        t_bt_shuffled = np.copy(t_bt)
        np.random.shuffle(t_bt_shuffled)

        for texture in u_textures:
            bs_ratings_nosubj.append(scipy.stats.sem(r_bt[t_bt == texture]))
            bs_ratings_nosubj_shuffled.append(scipy.stats.sem(r_bt[t_bt_shuffled == texture]))

        bs_ratings = np.array(bs_ratings)
        bs_ratings_shuffled = np.array(bs_ratings_shuffled)
        bs_ratings_nosubj = np.array(bs_ratings_nosubj)
        bs_ratings_nosubj_shuffled = np.array(bs_ratings_nosubj_shuffled)

        bs_ratings = bs_ratings[~np.isnan(bs_ratings)]
        bs_ratings_shuffled = bs_ratings_shuffled[~np.isnan(bs_ratings_shuffled)]
        bs_ratings_nosubj = bs_ratings_nosubj[~np.isnan(bs_ratings_nosubj)]
        bs_ratings_nosubj_shuffled = bs_ratings_nosubj_shuffled[~np.isnan(bs_ratings_nosubj_shuffled)]

        ax.violinplot(
            [bs_ratings, bs_ratings_shuffled,
             bs_ratings_nosubj, bs_ratings_nosubj_shuffled,
             bs_ratings_bws, bs_ratings_bws_shuffled],
            showmeans=False,
            showextrema=False,
            showmedians=True,
            quantiles=[[0.25, 0.75]]*6)
        ax.set_xticks([1, 2, 3, 4, 5, 6],
                      labels=['Real', 'Shuffled',
                              'Real nosubj', 'Shuffled nosubj',
                              'Real bwsub', 'Shuffled bwsub'])
        ax.set_xlim(0.25, 8 + 0.75)
        ax.set_ylim(bottom=0)

        ax.set_xlabel('Distribution')
        ax.set_ylabel('SEM of texture rating')
        ax.set_title(f"Distributions for Block Type {block_type} ")

        # conclusion - in the 0-1 space, the shuffled mean distances converge to 0.5,
        # which makes the metric useless
        # the SEM pulling between all subjects is the correct comparison

        # paper figure
        plt.sca(ax2)
        # ax2.violinplot(
        #     [bs_ratings_nosubj, bs_ratings_nosubj_shuffled],
        #     showmeans=False,
        #     showextrema=False,
        #     showmedians=True,
        #     quantiles=[[0.25, 0.75]]*2)
        # ax2.set_xticks([1, 2],
        #                labels=['Real', 'Shuffled'])
        # ax2.set_xlim(0.25, 2 + 0.75)

        seaborn.swarmplot(
            {'real': bs_ratings_nosubj, 'shuffled': bs_ratings_nosubj_shuffled},
            orient='v',
            palette={'shuffled': 'r', 'real': 'k'})

        ax2.set_ylim(bottom=0)

        ax2.set_ylabel('SEM of texture rating')
        ax2.set_title(f"Distributions for Block Type {block_type} ")

        # stat test
        s = pstats.format_nonparam_mwu_rbc(
            bs_ratings_nosubj, bs_ratings_nosubj_shuffled,
            alternative="less")
        print(f'\tComparing sem for ratings of {block_type}: {s}')
        print(u_textures)
        print(bs_ratings_nosubj)



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from config import (DATA_DIR, OUTPUT_DIR, SESSION_DIR, BINNED_FORCE_CSV,
                        BINNED_IMAGE_CSV, FRICTION_LONG_TABLE)

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    data_folder = SESSION_DIR

    output_dir = OUTPUT_DIR
    diagnostic_dir = os.path.join(output_dir, "Diagnostic_plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(diagnostic_dir, exist_ok=True)

    binned_force_path = BINNED_FORCE_CSV

    binned_image_path = BINNED_IMAGE_CSV

    jobs_csv = os.path.join(DATA_DIR, 'jobs_list.csv')
    dynamic_friction_long_table_csv = FRICTION_LONG_TABLE
    summary_table_csv = os.path.join(OUTPUT_DIR, "summary_table.csv")

    # --------------------------------------------------
    # Load binned metadata
    # --------------------------------------------------
    binned_force_df = pd.read_csv(binned_force_path)
    binned_force_df["Subject"] = binned_force_df["Session"].str.extract(r"(Subject\d+)")
    # binned_image_data = pd.read_csv(binned_image_path)

    # --------------------------------------------------
    # 1. Generate canonical LONG-FORM friction table
    # --------------------------------------------------
    # generate_jobs_long_table(  # DONE
    #     binned_image_data,
    #     binned_force_df,
    #     data_folder,
    #     diagnostic_dir,
    #     jobs_csv,
    #     subject_min=1,
    #     subject_max=17)

    # # Step - DONE
    # # After this - controlled by criteria
    # long_dic = generate_dynamic_friction_long_table(
    #     jobs_csv,
    #     dynamic_friction_long_table_csv,
    # )

    # # not needed in the new approach
    # generate_fn_stability_vs_friction_figure(
    #     long_df=long_df,
    #     output_dir=output_dir,
    # )

    # # --------------------------------------------------
    # # 2. Generate TEXTURE-LEVEL SUMMARY table
    # # --------------------------------------------------
    # generate_dynamic_friction_summary_table(
    #     dynamic_friction_long_table_csv,
    #     summary_table_csv
    # )  # DONE


    # --------------------------------------------------
    # 3. Generate FIGURE from the SAME long-form table   -- SKIPPED
    # --------------------------------------------------
    # fig_df = generate_friction_vs_rating_figure(
    #     long_df=long_df,
    #     binned_force_df=binned_force_df,
    #     output_dir=output_dir,
    #     exclude_low_contact=True
    # )

    # print(f"Figure source data saved with {len(fig_df)} rows")

    generate_friction_relationships(dynamic_friction_long_table_csv, binned_force_df)

    # --------------------------------------------------
    # FIGURE 2: Texture-averaged friction, block-specific ratings
    # --------------------------------------------------
    # fig_texture_df = generate_friction_vs_rating_texture_averaged_figure(
    #     dynamic_friction_long_table_csv,
    #     binned_force_df,
    #     output_dir,
    #     per_subject=True
    # )

    # print(f"Texture-averaged figure source data: {len(fig_texture_df)} rows")

    # Figure 2, ratings within texture/between textures
    # rating_variability_shuffle(binned_force_df)

    plt.show()
