#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect.py  -  find anomalies in your own OpenSky file and list the sections that should be cleaned.

Input:    Parquet or CSV file in the format of the daily files (one row per second and flight):
          timestamp, latitude, longitude, altitude, geoaltitude, vertical_rate, groundspeed, track,
          flight_id (recommended), onground, icao24, callsign
Output (in --out_dir):
  cleaning_intervals.csv   merged sections per flight and feature group: from ... to, to be cleaned
  flagged_sequences.csv    every flagged 60 s sequence individually (with error/threshold and rule violations)
  flights_summary.csv      one row per flight: row counts, segments, skipped (reason), flagged sequences

Steps per flight (as in clean_flights.py): remove on-ground rows and repeated positions, traffic outlier filter
("aggressive"), features (compute_all_features), 60 s sequences with stride 15, scaling, three LSTM autoencoders,
thresholds from the validation data (mean_thr, per_ts_thr, times r; variant "mp").
Time gaps (dt > 1 s) split a flight into segments; every segment is checked on its own.
The pipeline's rule violations (dist_diff, gs_jump, vertical_rate, gs_diff, vs_diff, ...) are only saved as a hint;
they do not influence the model result.

Usage:
  python 3_detection/detect.py --input my_file.parquet --out_dir result [--r 1.5] [--seeds 42 | --seeds all]
"""
import os as _os, sys as _sys  # REPO_ROOT_SHIM
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_LEN, STRIDE = 60, 15
ALL_SEEDS = [42, 7, 123, 155, 17, 98]
MODEL_NAMES = {"position": "E_pure_mse", "speed": "I_latent_smoothing",
               "vertical": "I_altitude_latent_smoothing"}
VIEW_COLUMNS = {"position": "latitude, longitude", "speed": "groundspeed",
                "vertical": "altitude, vertical_rate"}
REQUIRED_COLS = ["timestamp", "latitude", "longitude", "altitude", "geoaltitude",
                 "vertical_rate", "groundspeed", "track"]


# ---------------------------------------------------------------------------
# Helper functions (without tensorflow/traffic, testable on their own)
# ---------------------------------------------------------------------------
def make_windows(n_rows, seq_len=SEQ_LEN, stride=STRIDE):
    """Start indices of the sequences. As in training (stride 15), plus one window at the end of the flight
    so that the remainder behind the last regular window is not left unchecked."""
    if n_rows < seq_len:
        return []
    starts = list(range(0, n_rows - seq_len + 1, stride))
    if starts[-1] != n_rows - seq_len:
        starts.append(n_rows - seq_len)
    return starts


def split_segments(ts, max_dt=1.0):
    """Segment number per row; a new segment starts when the time gap is > max_dt seconds."""
    dt = ts.diff().dt.total_seconds()
    return (dt > max_dt + 1e-6).cumsum().to_numpy()


def rule_hints(d, starts, rule_masks, gs_diff_max, vs_diff_max):
    """Per sequence: the pipeline rules violated within the window (a hint, not a model result)."""
    masks = {k: v[0] for k, v in rule_masks(d).items() if k != "gap"}
    if gs_diff_max is not None:
        masks["gs_diff"] = d["gs_diff"].to_numpy(dtype=float) >= gs_diff_max
    if vs_diff_max is not None:
        masks["vs_diff"] = d["vs_diff"].to_numpy(dtype=float) >= vs_diff_max
    cum = {k: np.r_[0, np.cumsum(np.asarray(m, dtype=int))] for k, m in masks.items()}
    out = []
    for s in starts:
        out.append(";".join(k for k, c in cum.items() if c[s + SEQ_LEN] - c[s] > 0))
    return out


def merge_intervals(df, gap_s=1.0):
    """df: flagged sequences of ONE flight, segment and group, columns start, end, features,
    max_ratio, rules. Overlapping or directly adjacent sequences are merged into one section."""
    df = df.sort_values("start")
    out, cur = [], None
    for r in df.itertuples():
        if cur is not None and r.start <= cur["end"] + pd.Timedelta(seconds=gap_s):
            cur["end"] = max(cur["end"], r.end)
            cur["n_sequences"] += 1
            cur["features"] |= set(filter(None, r.features.split(";")))
            cur["rules"] |= set(filter(None, r.rules.split(";")))
            cur["max_ratio"] = max(cur["max_ratio"], r.max_ratio)
        else:
            if cur is not None:
                out.append(cur)
            cur = dict(start=r.start, end=r.end, n_sequences=1,
                       features=set(filter(None, r.features.split(";"))),
                       rules=set(filter(None, r.rules.split(";"))), max_ratio=r.max_ratio)
    if cur is not None:
        out.append(cur)
    return out


def scale_thresholds(thr, r):
    """Variant 'mp' only: mean_thr and per_ts_thr are multiplied by r (extreme_ts_thr is not used)."""
    return {k: {f: float(v) * r for f, v in thr[k].items()} for k in ("per_ts_thr", "mean_thr")}


# ---------------------------------------------------------------------------
# Preprocessing of one flight (same steps as clean_flights.process_flight)
# ---------------------------------------------------------------------------
def prepare_flight(fid, df, args, mods):
    """Returns (summary row, list of segments).
    Segment: dict(seg, X (n_windows, 60, 9), start, end, rules, in_scope)."""
    Flight = mods["Flight"]
    CFG = mods["CFG"]
    info = dict(flight_id=fid, n_rows_raw=len(df), n_onground_removed=0, n_static_rows=0, n_rows_clean=0,
                n_segments=0, n_segments_checked=0, n_rows_checked=0, n_sequences=0,
                in_training_scope="", skipped_reason="")
    segs = []

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if "onground" not in df.columns:
        df["onground"] = False
    for col, default in (("icao24", str(fid)), ("callsign", "")):
        if col not in df.columns:
            df[col] = default
    df["flight_id"] = fid

    df, info["n_onground_removed"] = mods["remove_onground_rows"](df)
    df, info["n_static_rows"] = mods["drop_static_rows"](df, CFG["max_static_points"])
    info["n_rows_clean"] = len(df)
    if len(df) < CFG["min_rows"]:
        info["skipped_reason"] = "too_short"
        return info, segs

    clean = df
    if not args.no_outlier_filter:
        try:
            ff = mods["filtering"](Flight(df.copy()))
        except Exception as e:  # noqa: BLE001
            info["skipped_reason"] = f"filter_error: {repr(e)[:80]}"
            return info, segs
        if ff is None or len(ff.data) == 0:
            info["skipped_reason"] = "filter_failed"
            return info, segs
        clean = ff.data.copy().reset_index(drop=True)
        clean["flight_id"] = fid

    seg_no = split_segments(clean["timestamp"], CFG["max_dt"])
    info["n_segments"] = int(seg_no.max()) + 1 if len(seg_no) else 0
    for k in np.unique(seg_no):
        seg = clean.loc[seg_no == k].reset_index(drop=True)
        if len(seg) < CFG["min_rows"]:
            continue
        seg["flight_id"] = f"{fid}#{k}"
        try:
            fe = mods["compute_all_features"](Flight(seg))
        except Exception as e:  # noqa: BLE001
            info["skipped_reason"] = (info["skipped_reason"] + f" seg{k}:feature_error").strip()
            continue
        d = fe.data
        if len(d) < SEQ_LEN:
            continue
        starts = make_windows(len(d))
        try:
            hints = rule_hints(d, starts, mods["rule_masks"], mods["GS_DIFF_MAX"], mods["VS_DIFF_MAX"])
        except Exception:  # noqa: BLE001
            hints = [""] * len(starts)
        arr = d[mods["ALL_FEATURES"]].to_numpy(dtype=float)
        X = np.stack([arr[s:s + SEQ_LEN] for s in starts])
        ts = pd.to_datetime(d["timestamp"], utc=True).reset_index(drop=True)
        scope = bool(np.nanmax(d["altitude"].to_numpy(float)) >= CFG["min_altitude_ft"]
                     and np.nanmax(d["groundspeed"].to_numpy(float)) >= mods["GS_MIN"])
        segs.append(dict(seg=int(k), X=X,
                         start=ts.iloc[starts].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(),
                         end=ts.iloc[[s + SEQ_LEN - 1 for s in starts]].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(),
                         rules=hints, in_scope=scope))
        info["n_segments_checked"] += 1
        info["n_rows_checked"] += len(d)
        info["n_sequences"] += len(starts)
    if not segs and not info["skipped_reason"]:
        info["skipped_reason"] = "no_continuous_segment_of_60_rows"
    info["in_training_scope"] = bool(all(s["in_scope"] for s in segs)) if segs else ""
    return info, segs


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def find_scaler(root, view):
    """Path of the scaler of a view under root; raises FileNotFoundError if there is none."""
    for p in (f"{root}/sequences/{view}/scaler.joblib", f"{root}/{view}/scaler.joblib"):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Scaler for '{view}' not found under {root}/sequences/{view}/scaler.joblib")


def evaluate_view(view, X_all, args, mods):
    """Applies the models of one view to all sequences.

    Returns (flagged (n, n_feat) bool [majority vote over seeds], ratio (n, n_feat) float [median over seeds of
    mean error / threshold], n_seeds_flagged (n,))."""
    import joblib
    import tensorflow as tf

    feats = mods["VIEWS"][view]["features"]
    cols = [mods["ALL_FEATURES"].index(f) for f in feats]
    Xv = X_all[:, :, cols]
    scaler = joblib.load(find_scaler(args.model_root, view))
    Xs = scaler.transform(Xv.reshape(-1, len(cols))).reshape(Xv.shape)
    n, seeds = len(Xs), args.seed_list
    flags = np.zeros((n, len(seeds), len(feats)), dtype=bool)
    ratio = np.zeros((n, len(seeds), len(feats)), dtype=float)
    for j, seed in enumerate(seeds):
        stem = f"{args.model_root}/model_{view}/{MODEL_NAMES[view]}_seed{seed}"
        model = tf.keras.models.load_model(f"{stem}.keras", compile=False)
        with open(f"{stem}_thresholds.json") as fh:
            thr = scale_thresholds(json.load(fh), args.r)
        for a in range(0, n, args.chunk):
            fl = mods["flag_anomalies"](model, Xs[a:a + args.chunk], feats, thr)
            for i, f in enumerate(feats):
                flags[a:a + args.chunk, j, i] = fl[f"{f}_anomaly"].to_numpy(dtype=bool)
                ratio[a:a + args.chunk, j, i] = fl[f"{f}_mean_error"].to_numpy(float) / thr["mean_thr"][f]
        print(f"  {view} seed {seed}: {int(flags[:, j, :].any(axis=1).sum())} of {n} sequences flagged", flush=True)
    voted = flags.sum(axis=1) * 2 > len(seeds)          # majority of the seeds, per feature
    return voted, np.median(ratio, axis=1), flags.any(axis=2).sum(axis=1)


# ---------------------------------------------------------------------------
def main():
    """Command line entry point: load the file, prepare the flights, apply the models, write the three CSV files."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Parquet or CSV file with OpenSky data (1 Hz)")
    ap.add_argument("--out_dir", default="detect_out")
    ap.add_argument("--model_root", default=f"{HERE}/artifacts", help="folder with model_<view>/ and sequences/<view>/scaler.joblib")
    ap.add_argument("--r", type=float, default=1.5, help="threshold scaling (paper: 1, 1.5, 2)")
    ap.add_argument("--seeds", nargs="+", default=["42"], help="e.g. 42 or 'all' (majority vote over 6 seeds)")
    ap.add_argument("--views", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    ap.add_argument("--no_outlier_filter", action="store_true",
                    help="skip the traffic outlier filter (the models then see unfiltered values)")
    ap.add_argument("--max_flights", type=int, default=None, help="only the first N flights (test)")
    ap.add_argument("--chunk", type=int, default=100000, help="sequences per model call")
    args = ap.parse_args()
    args.seed_list = ALL_SEEDS if args.seeds == ["all"] else [int(s) for s in args.seeds]
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "1_data"))

    # heavy imports only here
    from traffic.core import Flight
    from adsb_utils import flag_anomalies
    from adsb_utils import filtering
    from pipeline_common import ALL_FEATURES, GS_DIFF_MAX, GS_MIN, VIEWS, VS_DIFF_MAX, compute_all_features
    from clean_flights import CFG, drop_static_rows, remove_onground_rows, rule_masks
    mods = dict(Flight=Flight, flag_anomalies=flag_anomalies, filtering=filtering, ALL_FEATURES=ALL_FEATURES,
                GS_DIFF_MAX=GS_DIFF_MAX, GS_MIN=GS_MIN, VS_DIFF_MAX=VS_DIFF_MAX, VIEWS=VIEWS,
                compute_all_features=compute_all_features, CFG=CFG, drop_static_rows=drop_static_rows,
                remove_onground_rows=remove_onground_rows, rule_masks=rule_masks)

    # --- Load data ---
    t0 = time.time()
    raw = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        sys.exit(f"Missing columns in the input: {missing}")
    if "flight_id" not in raw.columns:
        d0 = pd.to_datetime(raw["timestamp"], utc=True).dt.strftime("%y/%m/%d")
        cs = raw["callsign"].fillna("") if "callsign" in raw.columns else ""
        raw["flight_id"] = (d0 + "_" + cs.astype(str) + "_" + raw["icao24"].astype(str)) if "icao24" in raw.columns else d0 + "_" + cs.astype(str)
        print("Note: no 'flight_id' column; it was built from date, callsign and icao24.")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    dts = raw.sort_values(["flight_id", "timestamp"]).groupby("flight_id")["timestamp"].diff().dt.total_seconds().dropna()
    if len(dts) and not 0.5 <= float(dts.median()) <= 1.5:
        print(f"WARNING: median time step {dts.median():.1f} s; the models expect 1 Hz (one row per second).")
    groups = list(raw.groupby("flight_id", sort=False))
    if args.max_flights:
        groups = groups[:args.max_flights]
    print(f"{len(raw)} rows, {len(groups)} flights loaded ({time.time() - t0:.0f} s)", flush=True)

    # --- Prepare flights ---
    summaries, seg_meta, X_list = [], [], []
    for n_f, (fid, df) in enumerate(groups, 1):
        info, segs = prepare_flight(str(fid), df, args, mods)
        summaries.append(info)
        for s in segs:
            X_list.append(s["X"])
            for i in range(len(s["X"])):
                seg_meta.append((str(fid), s["seg"], s["start"][i], s["end"][i], s["rules"][i], s["in_scope"]))
        if n_f % 500 == 0:
            print(f"  {n_f}/{len(groups)} flights prepared", flush=True)
    summ = pd.DataFrame(summaries)
    if not X_list:
        summ.to_csv(f"{args.out_dir}/flights_summary.csv", index=False)
        sys.exit("No sequence formed (all flights too short or faulty). See flights_summary.csv.")
    X_all = np.concatenate(X_list)
    meta = pd.DataFrame(seg_meta, columns=["flight_id", "segment", "start", "end", "rules", "in_training_scope"])
    print(f"{len(X_all)} sequences from {int((summ['n_sequences'] > 0).sum())} flights; models: {args.views}, "
          f"Seeds {args.seed_list}, r={args.r}", flush=True)

    # --- Apply models ---
    flagged_rows = []
    for view in args.views:
        voted, ratio, n_seeds = evaluate_view(view, X_all, args, mods)
        feats = mods["VIEWS"][view]["features"]
        hit = np.flatnonzero(voted.any(axis=1))
        for i in hit:
            fl = [f for k, f in enumerate(feats) if voted[i, k]]
            flagged_rows.append(dict(
                flight_id=meta.at[i, "flight_id"], segment=meta.at[i, "segment"], view=view,
                start=pd.Timestamp(meta.at[i, "start"]).tz_localize("UTC"), end=pd.Timestamp(meta.at[i, "end"]).tz_localize("UTC"),
                features=";".join(fl), max_ratio=float(max(ratio[i, k] for k, f in enumerate(feats) if voted[i, k])),
                seeds_flagged=int(n_seeds[i]), rules=meta.at[i, "rules"],
                in_training_scope=meta.at[i, "in_training_scope"]))
    flagged = pd.DataFrame(flagged_rows, columns=["flight_id", "segment", "view", "start", "end", "features",
                                                  "max_ratio", "seeds_flagged", "rules", "in_training_scope"])

    # --- Output ---
    intervals = []
    for (fid, seg, view), g in flagged.groupby(["flight_id", "segment", "view"], sort=False):
        for iv in merge_intervals(g):
            intervals.append(dict(flight_id=fid, segment=seg, view=view, features=";".join(sorted(iv["features"])),
                                  start=iv["start"], end=iv["end"],
                                  duration_s=(iv["end"] - iv["start"]).total_seconds() + 1,
                                  n_sequences=iv["n_sequences"], max_error_ratio=round(iv["max_ratio"], 3),
                                  rule_violations=";".join(sorted(iv["rules"])),
                                  affected_columns=VIEW_COLUMNS[view],
                                  in_training_scope=bool(g["in_training_scope"].iloc[0])))
    out_iv = pd.DataFrame(intervals)
    if len(out_iv):
        out_iv = out_iv.sort_values(["flight_id", "start", "view"]).reset_index(drop=True)
    out_iv.to_csv(f"{args.out_dir}/cleaning_intervals.csv", index=False)
    flagged.to_csv(f"{args.out_dir}/flagged_sequences.csv", index=False)
    for view in args.views:
        summ[f"n_flagged_{view}"] = summ["flight_id"].map(
            flagged.loc[flagged["view"] == view].groupby("flight_id").size()).fillna(0).astype(int)
    summ.to_csv(f"{args.out_dir}/flights_summary.csv", index=False)

    print("\n=== Summary ===")
    print(f"Flights: {len(summ)}, checked: {int((summ['n_sequences'] > 0).sum())}, "
          f"skipped: {int((summ['n_sequences'] == 0).sum())}")
    for view in args.views:
        iv = out_iv[out_iv["view"] == view] if len(out_iv) else out_iv
        n_fl = int((summ[f"n_flagged_{view}"] > 0).sum())
        secs = float(iv["duration_s"].sum()) if len(iv) else 0.0
        print(f"{view:9s}: {int((flagged['view'] == view).sum())} flagged sequences, {n_fl} flights, "
              f"{len(iv)} sections, {secs:.0f} s")
    print(f"Output: {args.out_dir}/cleaning_intervals.csv, flagged_sequences.csv, flights_summary.csv")


if __name__ == "__main__":
    main()