"""
friction_utils.py

Dynamic-friction estimation, adapted from friction_best.py. Detects
swipe "events" from fingertip velocity (periods of near-zero velocity
bounding a fast sweep), matching the manuscript's description: "we
identified periods of fingertip motion in line with the distribution
of swiping speeds reported previously (Callier et al., 2015)."

For each detected event, the friction coefficient is estimated as the
average of Ft/Fn at the moment of peak velocity and the samples
immediately before/after it.

IMPORTANT: this module intentionally stops at event-level output
(one row per Session/Trial/Event). Collapsing events to one
median-per-trial value (to match the manuscript's stated N=190/204
trials, and its wording "the median ratio of tangential to normal
force") is done in Data_Generation_Figure_5.py, not here -- keeping
that aggregation step visible and separate from detection makes it
easy to inspect/adjust independently.

Tunable constants below (velocity_threshold, min_event_duration_s,
the 120 cm/s peak-velocity floor, and the 0.4*mean(Fn) contact
threshold) are carried over unchanged from the working version you
had -- per your note, we're running this as-is first and will revisit
specific constants only if the output numbers don't match the paper.
"""

import os
import numpy as np
import pandas as pd


def compute_dynamic_friction_for_trial(
    image_df,
    time_df,
    force_df,
    image_bin_start,
    image_bin_end,
    force_bin_start,
    force_bin_end,
    velocity_threshold=2.5,
    min_event_duration_s=0.1,
    peak_velocity_floor=120.0,
    fn_threshold_frac=0.4,
):
    """
    Detects swipe events in a single trial and returns per-event dynamic
    friction coefficients.

    Returns
    -------
    event_friction_values : list of float
        mu = Ft/Fn averaged over the sample before, at, and after each
        event's peak-velocity index.
    above_threshold_pct : list of float
        Percentage of a +/-10-sample window around the event's peak that
        has Fn above fn_threshold_frac * mean(Fn) for the trial (contact
        stability diagnostic; carried through but not required for Fig 5).
    """
    image_df = image_df.reset_index(drop=True)
    if len(image_df) < 3:
        return [], []

    dx = image_df["keypoint_8_x"].diff().values
    dy = image_df["keypoint_8_y"].diff().values
    image_frames = image_df["frame"].values
    image_time = time_df["Exact Times"].copy().values[image_frames]
    dt = np.diff(image_time, prepend=np.nan)
    velocity = np.sqrt(dx ** 2 + dy ** 2) / dt

    zero_vel_indices = np.where(np.abs(velocity) < velocity_threshold)[0]
    if len(zero_vel_indices) < 2:
        return [], []

    image_times = time_df["Exact Times"].values
    start_img_idx, end_img_idx, event_img_idx = [], [], []
    start_img_time, end_img_time, event_img_time = [], [], []

    for idx0, idx1 in zip(zero_vel_indices[:-1], zero_vel_indices[1:]):
        if idx1 <= idx0:
            continue

        frame0 = image_df.loc[idx0, "frame"]
        frame1 = image_df.loc[idx1, "frame"]
        t0, t1 = image_times[frame0], image_times[frame1]
        if (t1 - t0) < min_event_duration_s:
            continue

        v_slice = velocity[idx0:idx1 + 1]
        valid_mask = v_slice >= peak_velocity_floor
        if not np.any(valid_mask):
            continue

        local_max = np.argmax(v_slice)
        global_idx = idx0 + local_max

        frame0 = image_df.loc[idx0, "frame"]
        frame1 = image_df.loc[idx1, "frame"]
        framee = image_df.loc[global_idx, "frame"]
        if not (0 <= frame0 < len(image_times) and
                0 <= frame1 < len(image_times) and
                0 <= framee < len(image_times)):
            continue

        start_img_idx.append(idx0)
        end_img_idx.append(idx1)
        event_img_idx.append(global_idx)
        start_img_time.append(image_times[frame0])
        end_img_time.append(image_times[frame1])
        event_img_time.append(image_times[framee])

    if len(event_img_idx) == 0:
        return [], []

    force_times = force_df["Exact Times"].values
    time_offset = force_bin_start - image_bin_start
    start_force_time = [t + time_offset for t in start_img_time]
    end_force_time = [t + time_offset for t in end_img_time]
    event_force_time = [t + time_offset for t in event_img_time]

    valid_event_mask = []
    for i in range(len(event_img_idx)):
        in_image_bounds = (
            image_bin_start <= start_img_time[i] <= image_bin_end and
            image_bin_start <= end_img_time[i] <= image_bin_end
        )
        in_force_bounds = (
            force_bin_start <= start_force_time[i] <= force_bin_end and
            force_bin_start <= end_force_time[i] <= force_bin_end
        )
        valid_event_mask.append(in_image_bounds and in_force_bounds)

    event_img_idx = [v for v, m in zip(event_img_idx, valid_event_mask) if m]
    start_force_time = [v for v, m in zip(start_force_time, valid_event_mask) if m]
    end_force_time = [v for v, m in zip(end_force_time, valid_event_mask) if m]
    event_force_time = [v for v, m in zip(event_force_time, valid_event_mask) if m]

    if len(event_img_idx) == 0:
        return [], []

    Fx = force_df["Fx"].values.copy()
    Fy = force_df["Fy"].values.copy()
    Fz = force_df["Fz"].values.copy()
    Fx[Fx < 0] = 0
    Fy[Fy < 0] = 0
    Fz[Fz < 0] = 0

    Ft = np.sqrt(Fx ** 2 + Fy ** 2)
    Fn = np.abs(Fz)
    fn_threshold = fn_threshold_frac * np.mean(Fn)

    event_friction_values = []
    above_threshold_values = []

    for i in range(len(event_img_idx)):
        idxe_force = int(np.argmin(np.abs(force_times - event_force_time[i])))

        pre_idx = max(idxe_force - 10, 0)
        post_idx = min(idxe_force + 11, len(Fn))
        fn_event_window = Fn[pre_idx:post_idx]
        percent_above_threshold = (
            np.sum(fn_event_window >= fn_threshold) / len(fn_event_window)
        ) * 100 if len(fn_event_window) > 0 else np.nan

        idx_before = max(idxe_force - 1, 0)
        idx_after = min(idxe_force + 1, len(force_times) - 1)

        if Fn[idx_before] == 0 or Fn[idxe_force] == 0 or Fn[idx_after] == 0:
            continue

        mu_dynamic = (
            (Ft[idx_before] / Fn[idx_before]) +
            (Ft[idxe_force] / Fn[idxe_force]) +
            (Ft[idx_after] / Fn[idx_after])
        ) / 3

        event_friction_values.append(mu_dynamic)
        above_threshold_values.append(percent_above_threshold)

    return event_friction_values, above_threshold_values


def compute_all_trial_events(binned_image_data, binned_force_data, data_folder,
                              subject_min=1, subject_max=17, blocks=('R', 'S')):
    """
    Runs compute_dynamic_friction_for_trial across every session/trial in
    the given subject range and block set, returning one row per detected
    EVENT (Session, Trial, Event). Collapsing to one row per trial happens
    in Data_Generation_Figure_5.generate_figure5_data.
    """
    import re
    from tqdm import tqdm

    results = []

    all_sessions = (
        binned_force_data["Session"].dropna()
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
        session_block = (
            binned_force_data
            .loc[binned_force_data["Session"].str.replace("_preprocessed", "", regex=True) == session,
                 "Block Type"]
            .dropna()
        )
        if session_block.empty or session_block.iloc[0] not in blocks:
            continue
        sessions.append(session)
    sessions = sorted(set(sessions))
    print(f"Including subjects {subject_min}-{subject_max}, blocks {blocks}: {len(sessions)} sessions")

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
            image_csv = os.path.join(session_path, "Images", f"trial_{trial}",
                                      "prediction", "processed_keypoint_data_real_coords.csv")
            time_csv = os.path.join(session_path, "Images", f"trial_{trial}", f"trial_{trial}_times.csv")
            force_csv = os.path.join(preproc_path, "Force", f"trial_{trial}.csv")

            if not all(os.path.exists(p) for p in [image_csv, time_csv, force_csv]):
                continue

            image_df = pd.read_csv(image_csv)
            time_df = pd.read_csv(time_csv)
            force_df = pd.read_csv(force_csv)

            image_bin_start = binned_image_data.query(
                "Session == @session and Trial == @trial")["Bin Start Times"].min()
            image_bin_end = binned_image_data.query(
                "Session == @session and Trial == @trial")["Bin Start Times"].max()
            force_bin_start = binned_force_data.query(
                "Session.str.contains(@preproc_session) and Trial == @trial",
                engine="python")["Bin Start Times"].min()
            force_bin_end = binned_force_data.query(
                "Session.str.contains(@preproc_session) and Trial == @trial",
                engine="python")["Bin Start Times"].max()

            texture = reports_df.loc[reports_df["trial number"] == trial, "texture used"].values[0]

            event_vals, pct_above = compute_dynamic_friction_for_trial(
                image_df=image_df, time_df=time_df, force_df=force_df,
                image_bin_start=image_bin_start, image_bin_end=image_bin_end,
                force_bin_start=force_bin_start, force_bin_end=force_bin_end,
            )
            if len(event_vals) == 0:
                continue

            block = binned_force_data.query(
                "Session.str.contains(@preproc_session) and Trial == @trial",
                engine="python")["Block Type"].iloc[0]

            for event_idx, (mu, pct) in enumerate(zip(event_vals, pct_above)):
                if np.isnan(mu):
                    continue
                results.append({
                    "Session": session, "Trial": trial, "Event": event_idx,
                    "Texture": texture, "Block": block,
                    "Dynamic Friction Coefficient": mu,
                    "Fn Stability Percentage": pct
                })

    return pd.DataFrame(results)
