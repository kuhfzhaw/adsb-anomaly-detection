#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_flights.py  -  one pipeline, one place for all exclusions

Order for each flight (per day file, one process per day):
  1. Scope on the raw data:  callsign, required columns (cheap, before any computation)
  2. Cleaning (rows/values only, never whole flights), each step exactly once:
        remove on-ground rows -> remove repeated positions -> outlier filter (traffic "aggressive")
  3. Features: compute_all_features (the same function as in model training, injection and evaluation)
  4. ALL rules on this one feature data frame, ALL are logged:
        scope:         too short, altitude, speed
        continuity:    time gap
        plausibility:  dalt, vertical_rate, gs_jump, track_jump, dist_diff
        training:      gs_diff, vs_diff, turn_intensity
  5. One decision (decide); reason by priority scope > gap > plausibility > training.

Output per day under <OUT>/<day>/:
  flight_log.csv       one row per flight with all flags, group and reason
  violations.parquet   timestamps of the rule violations (at most CAP per rule and flight)
  kept.parquet         cleaned raw data of the flights that passed (input of prepare_sequences.py)
  rejected_A.parquet   cleaned raw data, plausibility rule violated (no gap)
  rejected_B.parquet   cleaned raw data, only a training rule violated
  (rejected_G.parquet  flights with a gap, only with --save_gap)
Rejected flights go through exactly the same steps 1-3 as the accepted ones.

Usage:
  python 1_data/clean_flights.py --day_index 0 --max_flights 100            # quick test on 100 flights
  python 1_data/clean_flights.py --day_index 3 --skip_done                  # one day
  python 1_data/clean_flights.py --combine [--write_final]                  # funnel (and the combined Parquet file)
"""
import os as _os, sys as _sys  # REPO_ROOT_SHIM
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import time
from glob import glob

import numpy as np
import pandas as pd

BASE = os.environ.get("ADSB_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAY_DIR = f"{BASE}/data_fetching2024/preprocessed"
OUT = f"{BASE}/artifacts/pipeline"

DAYS = [
    "2024_01_06", "2024_02_15", "2024_03_21", "2024_04_27", "2024_05_01", "2024_06_17",
    "2024_07_06", "2024_08_12", "2024_09_25", "2024_10_08", "2024_11_30", "2024_12_23",
    "2024_01_15", "2024_02_20", "2024_03_25", "2024_04_12", "2024_05_20", "2024_06_30",
    "2024_07_15", "2024_08_20", "2024_09_07", "2024_10_18", "2024_11_10", "2024_12_05",
]

# ----------------------------------------------------------------------------
# ONE place for all thresholds (the training rules come from pipeline_common, so there is only one source)
# ----------------------------------------------------------------------------
CFG = dict(
    callsign_exclude=("HB", "GLID", "GIZ", "RGA"),
    required_cols=("altitude", "geoaltitude", "vertical_rate", "groundspeed"),
    min_rows=60,                 # rows in the cleaned flight (corresponds to one sequence)
    min_altitude_ft=1000,
    max_static_points=2,
    max_dt=1.0,                  # s
    dalt_mps=800,                # m/s, delta_z is in m/s
    vr_fpm=6000,                 # ft/min
    gs_jump_mps2=30,             # m/s^2
    track_deg_s=20,              # degrees/s
    dist_diff_m=50,              # m
)
CAP = 200
PLAUS_RULES = ["dalt", "vertical_rate", "gs_jump", "track_jump", "dist_diff"]
TRAIN_RULES = ["gs_diff", "vs_diff", "turn_intensity"]


# ----------------------------------------------------------------------------
# Pure logic (no traffic/tensorflow, testable)
# ----------------------------------------------------------------------------
def safe_max(a):
    """Maximum ignoring NaN; NaN if there is no finite value."""
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(a)) if np.isfinite(a).any() else np.nan


def scope_reason_raw(df, callsign):
    """Cheap scope check on the raw data. Returns the reason for exclusion, or '' if the flight is in scope."""
    for col in CFG["required_cols"]:
        if col not in df.columns or df[col].dropna().empty:
            return "missing_columns"
    if callsign is None or str(callsign).strip() == "":
        return "no_callsign"
    if str(callsign).startswith(CFG["callsign_exclude"]):
        return "callsign_prefix"
    return ""


def remove_onground_rows(df):
    """Removes on-ground rows and the first row after take-off. Returns (df, number of removed rows)."""
    if "onground" not in df.columns:
        return df.reset_index(drop=True), 0
    og = df["onground"].fillna(False).astype(bool)
    trans = og.shift(1, fill_value=False) & ~og
    keep = (~og) & (~trans)
    out = df.loc[keep].copy()
    out["onground"] = False
    return out.reset_index(drop=True), int((~keep).sum())


def drop_static_rows(df, max_static_points=2):
    """Removes rows in groups of at least max_static_points identical consecutive positions."""
    lat, lon = df["latitude"].round(6), df["longitude"].round(6)
    change = ((lat.diff().fillna(1) != 0) | (lon.diff().fillna(1) != 0)).astype(int)
    grp = change.cumsum()
    static = (grp.map(grp.value_counts()) >= max_static_points).to_numpy()
    return df.loc[~static].reset_index(drop=True), int(static.sum())


def rule_masks(d):
    """Row masks and maximum values of all rules on the feature data frame d. Returns {rule: (mask, maximum)}."""
    c = CFG
    with np.errstate(invalid="ignore"):
        dt = d["dt"].to_numpy(dtype=float)
        out = {"gap": (dt > c["max_dt"], safe_max(dt))}
        dz = np.abs(d["delta_z"].to_numpy(dtype=float))
        out["dalt"] = (dz > c["dalt_mps"], safe_max(dz))
        vr = np.abs(d["vertical_rate"].to_numpy(dtype=float))
        out["vertical_rate"] = (vr > c["vr_fpm"], safe_max(vr))
        sp = d["speed"].to_numpy(dtype=float)
        dgs = np.r_[np.nan, np.abs(np.diff(sp)) / dt[1:]]
        out["gs_jump"] = (dgs > c["gs_jump_mps2"], safe_max(dgs))
        if "track" in d.columns:
            tr = d["track"].ffill().to_numpy(dtype=float)
            dtr = np.abs(np.diff(tr))
            dtr = np.minimum(dtr, 360 - dtr)
            rate = np.r_[np.nan, dtr / dt[1:]]
            out["track_jump"] = (rate > c["track_deg_s"], safe_max(rate))
        else:
            out["track_jump"] = (np.zeros(len(d), dtype=bool), np.nan)
        dd = np.abs(d["dist_diff"].to_numpy(dtype=float))
        out["dist_diff"] = (dd > c["dist_diff_m"], safe_max(dd))
    return out


def decide(scope_reason, gap, plaus, train):
    """ONE decision. plaus/train: {rule: bool}. Priority: scope > gap > plausibility > training."""
    if scope_reason:
        return "C_scope", scope_reason
    if gap:
        return "G_gap", "gap"
    for r in PLAUS_RULES:
        if plaus.get(r):
            return "A_plausibility", r
    for r in TRAIN_RULES:
        if train.get(r):
            return "B_training", r
    return "ok", "ok"


# ----------------------------------------------------------------------------
# One flight
# ----------------------------------------------------------------------------
def process_flight(f):
    """Returns (log row, violations, cleaned raw data frame or None)."""
    from traffic.core import Flight
    from pipeline_common import (GS_DIFF_MAX, GS_MIN, VS_DIFF_MAX, compute_all_features,
                                 filter_flights_no_high_turn_intensity)
    from adsb_utils import filtering

    df = f.data.copy().reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    fid = f.flight_id
    row = dict(flight_id=fid, callsign=f.callsign, n_rows_raw=len(df))

    def finish(scope, gap=False, plaus=None, train=None, clean=None, extra=None):
        group, reason = decide(scope, gap, plaus or {}, train or {})
        row.update(extra or {})
        row.update(group=group, reason=reason, scope_reason=scope)
        return row, viols, clean

    viols = []
    r = scope_reason_raw(df, f.callsign)
    if r:
        return finish(r)

    # --- Cleaning: each step exactly once ---
    df, n_og = remove_onground_rows(df)
    df, n_static = drop_static_rows(df, CFG["max_static_points"])
    row.update(n_onground_removed=n_og, n_static_rows=n_static, n_rows_clean=len(df))
    if len(df) <= 1:
        return finish("too_short")
    ff = filtering(Flight(df))                                # traffic "aggressive": replaces outlier values
    if ff is None or len(ff.data) == 0:
        return finish("filter_failed")
    clean = ff.data.copy().reset_index(drop=True)
    clean["flight_id"] = fid

    # --- Features: the same function as later in the model ---
    try:
        fe = compute_all_features(Flight(clean.copy()))
    except Exception as e:                                    # noqa: BLE001
        return finish("feature_error", extra=dict(error=repr(e)[:150]))
    if fe is None:
        return finish("feature_error")
    d = fe.data
    row.update(n_rows_final=len(d), duration_s_final=float((d["timestamp"].iloc[-1] - d["timestamp"].iloc[0]).total_seconds())
               if len(d) > 1 else 0.0, max_altitude=safe_max(d["altitude"]), max_groundspeed=safe_max(d["groundspeed"]))

    # --- Scope on the final data frame ---
    if len(d) < CFG["min_rows"]:
        return finish("too_short")
    if row["max_altitude"] < CFG["min_altitude_ft"]:
        return finish("low_altitude")
    if GS_MIN is not None and row["max_groundspeed"] < GS_MIN:
        return finish("low_groundspeed")

    # --- All rules, all logged ---
    ts = pd.to_datetime(d["timestamp"], utc=True)
    masks = rule_masks(d)
    gsd, vsd = safe_max(d["gs_diff"]), safe_max(d["vs_diff"])
    masks["gs_diff"] = (d["gs_diff"].to_numpy(dtype=float) >= GS_DIFF_MAX if GS_DIFF_MAX is not None
                        else np.zeros(len(d), bool), gsd)
    masks["vs_diff"] = (d["vs_diff"].to_numpy(dtype=float) >= VS_DIFF_MAX if VS_DIFF_MAX is not None
                        else np.zeros(len(d), bool), vsd)
    turn_bad = filter_flights_no_high_turn_intensity(fe) is None
    for name, (mask, mx) in masks.items():
        row[f"viol_{name}"], row[f"max_{name}"], row[f"n_viol_{name}"] = bool(mask.any()), mx, int(mask.sum())
        if mask.any():
            for t in ts.iloc[np.flatnonzero(mask)[:CAP]]:
                viols.append(dict(flight_id=fid, rule=name, timestamp=t))
    row["viol_turn_intensity"] = bool(turn_bad)
    plaus = {r: row[f"viol_{r}"] for r in PLAUS_RULES}
    train = {"gs_diff": row["viol_gs_diff"], "vs_diff": row["viol_vs_diff"], "turn_intensity": bool(turn_bad)}
    return finish("", gap=row["viol_gap"], plaus=plaus, train=train, clean=clean)


# ----------------------------------------------------------------------------
# One day
# ----------------------------------------------------------------------------
def process_day(day_index, max_flights=None, skip_done=False, save_gap=False):
    """Processes all flights of one day file and writes the per-day outputs (see module docstring)."""
    from traffic.core import Traffic

    day = DAYS[day_index]
    out = f"{OUT}/{day}"
    os.makedirs(out, exist_ok=True)
    if skip_done and os.path.exists(f"{out}/.complete"):
        print(f"[{day}] already done, skipped", flush=True)
        return
    t = Traffic.from_file(f"{DAY_DIR}/{day}.parquet")
    rows, viols = [], []
    kept, rej = [], {"A": [], "B": [], "G": []}
    t0 = time.time()
    for i, f in enumerate(t):
        if max_flights is not None and i >= max_flights:
            break
        try:
            row, v, clean = process_flight(f)
        except Exception as e:                                # noqa: BLE001
            row, v, clean = dict(flight_id=getattr(f, "flight_id", f"unknown_{i}"), group="E_error",
                                 reason="exception", error=repr(e)[:150]), [], None
        row["tag"] = day
        rows.append(row)
        viols += v
        if clean is not None:
            g = row["group"]
            if g == "ok":
                kept.append(clean)
            elif g == "A_plausibility":
                rej["A"].append(clean)
            elif g == "B_training":
                rej["B"].append(clean)
            elif g == "G_gap" and save_gap:
                rej["G"].append(clean)
        if (i + 1) % 100 == 0:
            print(f"[{day}] {i + 1} flights, {time.time() - t0:.0f} s", flush=True)

    fl = pd.DataFrame(rows)
    fl.to_csv(f"{out}/flight_log.csv", index=False)
    if viols:
        pd.DataFrame(viols).to_parquet(f"{out}/violations.parquet", index=False)
    for name, lst in (("kept", kept), ("rejected_A", rej["A"]), ("rejected_B", rej["B"]), ("rejected_G", rej["G"])):
        if lst:
            pd.concat(lst, ignore_index=True).to_parquet(f"{out}/{name}.parquet", index=False)
    if max_flights is None:
        open(f"{out}/.complete", "w").close()
    print(f"[{day}] done: {len(fl)} flights\n{fl['group'].value_counts().to_string()}", flush=True)


# ----------------------------------------------------------------------------
# Funnel (summary over all days)
# ----------------------------------------------------------------------------
def combine(write_final=False):
    """Summarises the per-day flight logs and optionally writes the combined table t_unified_final.parquet."""
    paths = sorted(glob(f"{OUT}/*/flight_log.csv"))
    fl = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    fl.to_csv(f"{OUT}/flight_log_all.csv", index=False)
    print(f"{len(paths)} days, {len(fl)} flights\n")
    print("=== Groups ===\n" + fl["group"].value_counts().to_string())
    print("\n=== Reason (first by priority) ===")
    print(fl.groupby(["group", "reason"]).size().to_string())
    scoped = fl[fl["group"] != "C_scope"]
    viol_cols = [c for c in fl if c.startswith("viol_")]
    print(f"\n=== Rules among the {len(scoped)} flights in scope (not exclusive) ===")
    print(scoped[viol_cols].fillna(False).astype(bool).sum().sort_values(ascending=False).to_string())
    if write_final:
        kept = [pd.read_parquet(p) for p in sorted(glob(f"{OUT}/*/kept.parquet"))]
        allk = pd.concat(kept, ignore_index=True)
        allk.to_parquet(f"{OUT}/t_unified_final.parquet", index=False)
        print(f"\n{allk['flight_id'].nunique()} flights written to {OUT}/t_unified_final.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day_index", type=int, default=None)
    ap.add_argument("--max_flights", type=int, default=None)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--save_gap", action="store_true", help="also save flights with a time gap")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--write_final", action="store_true")
    a = ap.parse_args()
    if a.combine:
        combine(a.write_final)
    elif a.day_index is not None:
        process_day(a.day_index, a.max_flights, a.skip_done, a.save_gap)
    else:
        ap.error("specify --day_index N or --combine")


if __name__ == "__main__":
    main()