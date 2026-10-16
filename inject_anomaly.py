# inject_anomaly.py
"""
Synthetic anomaly injection test: how many injected anomalies does the matching model detect?

    python inject_anomaly.py --view position --out_dir <dir>
    python inject_anomaly.py --view speed    --out_dir <dir>
    python inject_anomaly.py --view vertical --out_dir <dir>

How it works (all three views)
  1. Load the test flights from the raw table (artifacts/pipeline/t_unified_final.parquet).
  2. Write the anomaly into the raw columns (position: latitude/longitude, speed: groundspeed,
     vertical: altitude/vertical_rate) and recompute the features (compute_all_features).
     Only speed also knows --level feature: injection into the already smoothed base flight.
  3. Scale the sequences, let the model reconstruct them, flag them with the validation thresholds.
  4. Detected = a sequence that contains the anomaly and was NOT flagged in the unchanged flight
     is now flagged (additionally per event: detected_net).

Pairing: the same flights and the same start time t0 (and the same sign/direction) for all
variants. Differences between the cells therefore come only from magnitude and duration.

Groups (--groups)
  position: jumps (default), noise, freeze, drift
  speed:    noise, offset, spike, freeze, drift (default: all)
  vertical: alt_noise, vr_noise, alt_step, alt_spike, vr_offset, vr_spike, vr_sign_flip,
            alt_freeze, vr_freeze, alt_drift (default: all)

The model (--model) defaults to the paper model of the view (seed 42); the foreign models for
cross-specificity (--cross view=<model.keras>) are evaluated on the same injected flights.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from traffic.core import Flight, Traffic

from pipeline_common import compute_all_features
from cross_eval import Det, make_cross, save_cross   # next to this script

_BASE = os.environ.get("ADSB_BASE", os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(_BASE, "artifacts")
RAW_FILE = os.path.join(BASE, "pipeline", "t_unified_final.parquet")

SEQ_LEN, STRIDE = 60, 15
MARGIN_S = 30            # after-effect of the rolling windows after the event
LO, TAIL = 30, 30        # distance of the anomaly from the start / end of the flight (s)
T0_DUR = 60              # choose t0 so that the longest duration fits (same t0 for all variants)
MIN_OVERLAP = 10         # speed/vertical: a sequence counts if at least min(duration, MIN_OVERLAP) s of the anomaly lie in it
PERSIST = "persistent"       # the offset persists until the end of the flight


# ====================================================================== common helpers
def _col(df, c):
    """Float copy of a column."""
    return df[c].to_numpy(dtype=float).copy()


def _sign(rng, signed):
    """Random sign (always drawn, so the rng state does not depend on `signed`)."""
    s = rng.choice([-1.0, 1.0])          # always draw so that the rng state stays the same
    return s if signed else 1.0


def to_sec(ts):
    """Timestamps -> seconds since 1970 (independent of the time unit ns/us and of the time zone)."""
    x = pd.DatetimeIndex(pd.to_datetime(ts))
    if x.tz is not None:
        x = x.tz_convert("UTC").tz_localize(None)
    return x.astype("datetime64[s]").asi8


def load_flights(split_json, split, n_flights, seed):
    """Raw data frames of the flights of the given split (at most n_flights, random subset)."""
    split_of = json.load(open(split_json))
    ids = {fid for fid, s in split_of.items() if s == split}
    t = Traffic.from_file(RAW_FILE)
    raw = [f.data.copy().reset_index(drop=True) for f in t if f.flight_id in ids]
    print(f"{split} flights in the split: {len(ids)}, found in the raw file: {len(raw)}")
    if len(raw) > n_flights:
        idx = np.sort(np.random.default_rng(seed).choice(len(raw), size=n_flights, replace=False))
        raw = [raw[i] for i in idx]
    return raw


def build_bases(raws):
    """Computes the features of every raw flight. Returns (raw flights, feature data frames) for the usable flights."""
    r2, bases = [], []
    for raw in raws:
        p = compute_all_features(Flight(raw.copy()))
        if p is not None and len(p.data) >= SEQ_LEN:
            r2.append(raw)
            bases.append(p.data.reset_index(drop=True))
    return r2, bases


# ====================================================================== POSITION
M_PER_DEG = 111_320.0

# Grid of the jumps
POS_DIST_SHORT = (10, 25, 50, 75)                       # short jumps (m)
POS_DIST_LONG = (100, 250, 500, 1000, 1500, 2000)       # long jumps (m)
POS_DUR_SHORT = (1, 5)                                  # short duration (s)
POS_DUR_LONG = (20, 60)                                 # long duration (s)
POS_DUR_COLS = [f"{d} s" for d in POS_DUR_SHORT + POS_DUR_LONG] + [PERSIST]
POS_GROUPS = ["jumps", "noise", "freeze", "drift"]


def pos_noise(df, t0, t1, rng, sigma_m):
    """Gaussian position noise with standard deviation sigma_m (metres) on [t0, t1)."""
    lat, lon = _col(df, "latitude"), _col(df, "longitude")
    lat[t0:t1] += rng.normal(0, sigma_m, t1 - t0) / M_PER_DEG
    lon[t0:t1] += rng.normal(0, sigma_m, t1 - t0) / (M_PER_DEG * np.cos(np.radians(lat[t0:t1])))
    df["latitude"], df["longitude"] = lat, lon


def pos_jump(df, t0, t1, rng, offset_m, persistent=True):
    """Position jump of offset_m metres in a random direction on [t0, t1) (persistent: until the end of the flight)."""
    b = np.radians(rng.uniform(0, 360))
    end = len(df) if persistent else t1
    lat, lon = _col(df, "latitude"), _col(df, "longitude")
    lat[t0:end] += offset_m * np.cos(b) / M_PER_DEG
    lon[t0:end] += offset_m * np.sin(b) / (M_PER_DEG * np.cos(np.radians(lat[t0:end])))
    df["latitude"], df["longitude"] = lat, lon


def pos_freeze(df, t0, t1, rng):
    """Freezes the position at its value before t0 for [t0, t1)."""
    lat, lon = _col(df, "latitude"), _col(df, "longitude")
    lat[t0:t1], lon[t0:t1] = lat[t0 - 1], lon[t0 - 1]
    df["latitude"], df["longitude"] = lat, lon


def pos_drift(df, t0, t1, rng, rate_m_s):
    """Creeping drift; the offset persists after t1 (no second spike)."""
    b = np.radians(rng.uniform(0, 360))
    lat, lon = _col(df, "latitude"), _col(df, "longitude")
    off = np.minimum(np.arange(len(lat) - t0), t1 - t0 - 1) * rate_m_s
    lat[t0:] += off * np.cos(b) / M_PER_DEG
    lon[t0:] += off * np.sin(b) / (M_PER_DEG * np.cos(np.radians(lat[t0:])))
    df["latitude"], df["longitude"] = lat, lon


def make_variants_position(groups):
    """List of all anomaly variants. groups: subset of jumps, noise, freeze, drift."""
    V = [dict(group="control", anomaly="control", param="offset 0 m, 10 s", dist=0, dur_label="10 s",
              dur_s=10, dur=10, fn=pos_jump, kw=dict(offset_m=0, persistent=False), point_event=False)]
    if "jumps" in groups:
        for d in POS_DIST_SHORT + POS_DIST_LONG:
            for dur in POS_DUR_SHORT + POS_DUR_LONG:
                V.append(dict(group="jumps", anomaly="pos_jump_temporary", param=f"{d} m, {dur} s", dist=d,
                              dur_label=f"{dur} s", dur_s=dur, dur=dur, fn=pos_jump,
                              kw=dict(offset_m=d, persistent=False), point_event=False))
            V.append(dict(group="jumps", anomaly="pos_jump_persistent", param=f"{d} m, {PERSIST}", dist=d,
                          dur_label=PERSIST, dur_s=np.nan, dur=1, fn=pos_jump,
                          kw=dict(offset_m=d, persistent=True), point_event=True))
    if "noise" in groups:
        for s in (2, 3, 5, 7, 10, 30, 100, 250):
            V.append(dict(group="noise", anomaly="pos_noise", param=f"sigma {s} m, 30 s", dist=s,
                          dur_label="30 s", dur_s=30, dur=30, fn=pos_noise, kw=dict(sigma_m=s), point_event=False))
    if "freeze" in groups:
        for d in (1, 5, 10, 30):
            V.append(dict(group="freeze", anomaly="pos_freeze", param=f"{d} s", dist=0, dur_label=f"{d} s",
                          dur_s=d, dur=d, fn=pos_freeze, kw={}, point_event=False))
    if "drift" in groups:
        for r in (3, 5, 7, 10, 15, 20, 25, 50):
            V.append(dict(group="drift", anomaly="pos_drift", param=f"{r} m/s, 40 s", dist=r,
                          dur_label="40 s", dur_s=40, dur=40, fn=pos_drift, kw=dict(rate_m_s=r), point_event=False))
    return V


def build_injected_position(raw_dfs, v, seed):
    """Writes anomaly v into every raw flight and recomputes the features. -> (processed, events)"""
    processed, events = [], {}
    for i, raw in enumerate(raw_dfs):
        n = len(raw)
        hi = n - T0_DUR - TAIL
        if hi <= LO:
            continue
        rng = np.random.default_rng([seed, i])          # the same per flight for all variants (pairing)
        t0 = int(rng.integers(LO, hi))
        t1 = t0 + v["dur"]
        df = raw.copy()
        ts = df["timestamp"]
        ev_start = ts.iloc[t0]
        ev_end = ts.iloc[min(t0 + 1 if v["point_event"] else t1, n - 1)] + pd.Timedelta(seconds=MARGIN_S)
        v["fn"](df, t0, t1, rng, **v["kw"])
        proc = compute_all_features(Flight(df))
        if proc is None or len(proc.data) < SEQ_LEN:
            continue
        processed.append(proc)
        events[proc.flight_id] = (ev_start, ev_end)
    return processed, events


def evaluate_position(processed, events, det, combos):
    """Returns one result dict per (variant, r) combination (sequences are built only once)."""
    res = det.seqs(processed)
    if res is None:
        return None
    X, fids, starts = res
    ts_map = {f.flight_id: f.data["timestamp"].reset_index(drop=True) for f in processed}
    in_win = np.zeros(len(fids), dtype=bool)
    has_onset = np.zeros(len(fids), dtype=bool)      # the sequence contains the onset of the anomaly
    for i, (fid, s) in enumerate(zip(fids, starts)):
        ev = events.get(fid)
        if ev is None:
            continue
        ts = ts_map[fid]
        t_a, t_b = ts.iloc[int(s)], ts.iloc[int(s) + SEQ_LEN - 1]
        in_win[i] = (t_a <= ev[1]) and (t_b >= ev[0])
        has_onset[i] = (t_a <= ev[0]) and (t_b >= ev[0])
    idx = np.where(in_win)[0]
    if len(idx) == 0:
        return None
    out = {}
    for combo in combos:
        fl = det.flags(X[idx], *combo)        # only sequences in the event window
        cf = det.clean_by[combo]
        clean = np.array([cf.get((f, int(s)), False) for f, s in zip(fids[idx], starts[idx])], dtype=bool)
        n_missing = sum((f, int(s)) not in cf for f, s in zip(fids[idx], starts[idx]))
        d = pd.DataFrame(dict(fid=fids[idx], onset=has_onset[idx], raw=fl, new=fl & ~clean, clean=clean))
        e = d.groupby("fid").agg(raw=("raw", "any"), new=("new", "any"), clean=("clean", "any"))
        # Per sequence: of the sequences with anomaly that were NOT flagged without it, how many are flagged now?
        sq = d[d["onset"] & ~d["clean"]]
        out[combo] = dict(n_events=len(e),
                          n_seq=len(sq),
                          detected_seq=100 * sq["raw"].mean() if len(sq) else float("nan"),
                          detected_net=100 * e["new"].mean(),     # newly flagged: this is the detection rate
                          detected_raw=100 * e["raw"].mean(),       # any flag in the window (includes chance hits)
                          chance=100 * e["clean"].mean(),          # the window was already flagged without the anomaly
                          n_no_clean=int(n_missing))
    return out


def run_all_position(raw_dfs, det, cross, variants, seed, combos):
    """Runs all position variants and returns (results, cross-model results) as data frames."""
    clean = []
    for raw in raw_dfs:
        p = compute_all_features(Flight(raw.copy()))
        if p is not None and len(p.data) >= SEQ_LEN:
            clean.append(p)
    for d_ in [det] + cross:
        d_.prepare_clean(clean, combos)

    rows, xrows = [], []
    t_start = time.time()
    for k, v in enumerate(variants, 1):
        processed, events = build_injected_position(raw_dfs, v, seed)
        rr = evaluate_position(processed, events, det, combos) if processed else None
        det.clear()                                    # empty the cache per variant (memory)
        meta = {c: v[c] for c in ("group", "anomaly", "param", "dist", "dur_label", "dur_s")}
        xs = []
        if processed:
            for xd in cross:                           # foreign models on the same injected flights
                rx = evaluate_position(processed, events, xd, combos)
                xd.clear()
                if rx is None:
                    continue
                for combo, r in rx.items():
                    xrows.append({**meta, "model": xd.view, "variant": combo[0], "r": combo[1], **r})
                xs.append(f"{xd.view} {rx[combos[0]]['detected_net']:.0f} %")
        if rr is None:
            print(f"[{k}/{len(variants)}] {v['anomaly']} {v['param']}: no suitable flights")
            continue
        for combo, r in rr.items():
            rows.append({**meta, "variant": combo[0], "r": combo[1], **r})
        r0 = rr[combos[0]]
        print(f"[{k}/{len(variants)}] {v['anomaly']} {v['param']}: {r0['n_events']} events, "
              f"detected {r0['detected_seq']:.0f} % of {r0['n_seq']} sequences (events net {r0['detected_net']:.0f} %) "
              f"[{combos[0]}], {time.time() - t_start:.0f} s")
        if xs:
            print("      Cross (net, events): " + ", ".join(xs))
    return pd.DataFrame(rows), pd.DataFrame(xrows)


def report_position(df, out_dir):
    """Writes results.csv and prints the position report (jump grids, four classes, other anomalies)."""
    df.to_csv(f"{out_dir}/results.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    ctrl = df[df["group"] == "control"]
    if len(ctrl):
        c = float(ctrl["detected_seq"].iloc[0])
        print(f"\nControl (offset 0): net {c:.1f} % " + ("(OK, must be 0)" if c == 0 else
              "<- should be 0! Check the assignment of the sequences."))

    j = df[df["group"] == "jumps"]
    if len(j):
        for val, name, fn in (("detected_seq", "sequences (sequence level)", "jump_grid.csv"),
                              ("detected_net", "events (net, event level)", "jump_grid_net.csv")):
            grid = j.pivot_table(index="dist", columns="dur_label", values=val)
            grid = grid.reindex(columns=[c for c in POS_DUR_COLS if c in grid.columns])
            grid.index.name = "jump distance [m]"
            grid.columns.name = "duration"
            print(f"\nDetected {name} in %, jump distance x duration:")
            print(grid.round(0).to_string())
            grid.to_csv(f"{out_dir}/{fn}")

        t = j[j["anomaly"] == "pos_jump_temporary"].copy()
        t["distance"] = np.where(t["dist"].isin(POS_DIST_SHORT), f"short (<= {max(POS_DIST_SHORT)} m)",
                                 f"long (>= {min(POS_DIST_LONG)} m)")
        t["duration"] = np.where(t["dur_s"].isin(POS_DUR_SHORT), f"short ({min(POS_DUR_SHORT)}-{max(POS_DUR_SHORT)} s)",
                                 f"long ({min(POS_DUR_LONG)}-{max(POS_DUR_LONG)} s)")
        four = t.pivot_table(index="distance", columns="duration", values="detected_seq", aggfunc="mean")
        print("\nFour classes (mean over the cells, only jumps with return):")
        print(four.round(0).to_string())
        four.to_csv(f"{out_dir}/jump_classes.csv")

    other = df[df["group"].isin(["noise", "freeze", "drift"])]
    if len(other):
        print("\nOther anomalies:")
        print(other[["anomaly", "param", "n_events", "detected_seq", "detected_raw", "chance"]].round(1).to_string(index=False))

    miss = int(df["n_no_clean"].sum())
    if miss:
        print(f"\nNote: {miss} sequences without a counterpart in the unchanged flights (counted as not flagged).")


# ====================================================================== SPEED
# Breakdown by covariates (computed from the unchanged flight at the event)
DYN_WIN = 60             # window length (s) for the dynamics; capped at this value for persistent anomalies
SPEED_BINS, SPEED_LABELS = [0, 150, 300, np.inf], ["<150 kt", "150-300 kt", ">300 kt"]
DYN_BINS, DYN_LABELS = [0, 2, 10, np.inf], ["calm <2 kt", "medium 2-10 kt", "dynamic >=10 kt"]

# Grid (kt or kt/s)
SPD_NOISE_SIGMA = (0.5, 1, 1.5, 2, 5, 10, 30, 50, 100)
SPD_NOISE_DUR = 30
SPD_OFFSET_DELTA = (0.5, 1, 2, 5, 7, 10, 30, 60, 100)
SPD_OFFSET_DUR = (10, 30, 60)
SPD_SPIKE_DELTA = (1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 500)
SPD_SPIKE_DUR = (1, 3, 5, 10, 30)
SPD_FREEZE_DUR = (10, 20, 40)
SPD_FREEZE_MIN_CHANGE = (10, 30, 50)       # freeze only where the speed changes by >= x kt during the duration
SPD_DRIFT_RATE = (0.1, 0.25, 0.5, 1.5, 2, 5)
SPD_DRIFT_DUR = 40
SPD_GROUPS = ["noise", "offset", "spike", "freeze", "drift"]


def _put(df, v):
    """Writes the groundspeed back (never negative)."""
    df["groundspeed"] = np.maximum(v, 0.0)


def gs_noise(df, t0, t1, rng, sigma, signed=True):
    """Gaussian groundspeed noise (kt) on [t0, t1)."""
    v = _col(df, "groundspeed")
    v[t0:t1] += rng.normal(0, sigma, t1 - t0)
    _put(df, v)


def gs_offset(df, t0, t1, rng, delta, persistent=False, signed=True):
    """Constant shift: groundspeed + delta for [t0, t1), then back (persistent: until the end of the flight)."""
    s = _sign(rng, signed)
    v = _col(df, "groundspeed")
    v[t0:(len(v) if persistent else t1)] += s * delta
    _put(df, v)


def gs_drift(df, t0, t1, rng, rate, signed=True):
    """Ramp with slope rate [kt/s] over [t0, t1), then a jump back to the original value."""
    s = _sign(rng, signed)
    v = _col(df, "groundspeed")
    v[t0:t1] += s * np.arange(t1 - t0) * rate
    _put(df, v)


def gs_freeze(df, t0, t1, rng, signed=True):
    """Freezes the groundspeed at its value before t0 for [t0, t1)."""
    v = _col(df, "groundspeed")
    v[t0:t1] = v[t0 - 1]
    _put(df, v)


def recompute_gs_features(df):
    """add_delta_gs without the first smoothing (rolling 10 on groundspeed). Rest 1:1 as in adsb_utils."""
    df = df.copy()
    gs = df["groundspeed"]
    df["delta_gs"] = gs.diff().fillna(0.0)
    df["gs_smooth"] = gs.ewm(alpha=0.2).mean()
    df["gs_norm"] = gs / 600.0
    if "dt" in df.columns:
        df["delta_gs"] = (df["delta_gs"] / df["dt"].replace(0, 1)).rolling(window=10, min_periods=1, center=False).mean()
    if "compute_gs" in df.columns:
        df["gs_diff"] = (df["compute_gs"] - df["gs_smooth"]).abs().rolling(window=20, min_periods=1, center=False).mean()
    return df


def _v_speed(group, anomaly, param, fn, kw, mag, dur, dur_label, persistent=False, min_change=None):
    """Builds one speed variant dict."""
    return dict(group=group, anomaly=anomaly, param=param, fn=fn, kw=kw, mag=mag, dur=dur,
                dur_label=dur_label, dur_s=(np.nan if persistent else dur), persistent=persistent,
                min_change=min_change)


def make_variants_speed(groups, signed):
    """List of all speed anomaly variants for the given groups."""
    V = [_v_speed("control", "control", "delta 0 kt, 3 s", gs_offset, dict(delta=0, signed=signed), 0, 3, "3 s")]
    if "noise" in groups:
        for s in SPD_NOISE_SIGMA:
            V.append(_v_speed("noise", "gs_noise", f"sigma {s} kt, {SPD_NOISE_DUR} s", gs_noise,
                              dict(sigma=s, signed=signed), s, SPD_NOISE_DUR, f"{SPD_NOISE_DUR} s"))
    if "offset" in groups:
        for d in SPD_OFFSET_DELTA:
            for dur in SPD_OFFSET_DUR:
                V.append(_v_speed("offset", "gs_offset", f"{d} kt, {dur} s", gs_offset,
                                  dict(delta=d, persistent=False, signed=signed), d, dur, f"{dur} s"))
            V.append(_v_speed("offset", "gs_offset_persistent", f"{d} kt, {PERSIST}", gs_offset,
                              dict(delta=d, persistent=True, signed=signed), d, 1, PERSIST, persistent=True))
    if "spike" in groups:
        for d in SPD_SPIKE_DELTA:
            for dur in SPD_SPIKE_DUR:
                V.append(_v_speed("spike", "gs_spike", f"{d} kt, {dur} s", gs_offset,
                                  dict(delta=d, persistent=False, signed=signed), d, dur, f"{dur} s"))
    if "freeze" in groups:
        for mc in SPD_FREEZE_MIN_CHANGE:
            for dur in SPD_FREEZE_DUR:
                V.append(_v_speed("freeze", "gs_freeze", f"{dur} s, change >= {mc} kt", gs_freeze,
                                  dict(signed=signed), mc, dur, f"{dur} s", min_change=mc))
    if "drift" in groups:
        for r in SPD_DRIFT_RATE:
            V.append(_v_speed("drift", "gs_drift", f"{r} kt/s, {SPD_DRIFT_DUR} s (max {r * (SPD_DRIFT_DUR - 1):.0f} kt)",
                              gs_drift, dict(rate=r, signed=signed), r, SPD_DRIFT_DUR, f"{SPD_DRIFT_DUR} s"))
    return V


def pick_t0_speed(gs, v, rng):
    """Start index in the base flight (the same for all variants); freeze only in dynamic phases."""
    n = len(gs)
    hi = n - T0_DUR - TAIL
    if hi <= LO:
        return None
    if v["min_change"] is None:
        return int(rng.integers(LO, hi))
    cand = [t for t in range(LO, hi) if abs(gs[min(t + v["dur"], n) - 1] - gs[t - 1]) >= v["min_change"]]
    return int(rng.choice(cand)) if cand else None


def build_injected_speed(raws, bases, v, seed, level):
    """Writes anomaly v into every flight. -> (processed, events)"""
    processed, events = [], {}
    for i, (raw, base) in enumerate(zip(raws, bases)):
        n = len(base)
        rng = np.random.default_rng([seed, i])
        gs_base = base["groundspeed"].to_numpy(float)
        t0 = pick_t0_speed(gs_base, v, rng)
        if t0 is None:
            continue
        t1 = n if v["persistent"] else min(t0 + v["dur"], n)
        ts_b = base["timestamp"]
        start_ts, last_ts = ts_b.iloc[t0], ts_b.iloc[t1 - 1]
        if level == "feature":
            df = base.copy()
            v["fn"](df, t0, t1, rng, **v["kw"])
            df = recompute_gs_features(df)
        else:
            r0 = int(raw["timestamp"].searchsorted(start_ts))
            r1 = len(raw) if v["persistent"] else r0 + (t1 - t0)
            df = raw.copy()
            v["fn"](df, r0, r1, rng, **v["kw"])
        proc = Flight(df) if level == "feature" else compute_all_features(Flight(df))
        if proc is None or len(proc.data) < SEQ_LEN:
            continue
        # actual (effective) deviation of the groundspeed in the model input
        a = proc.data.set_index("timestamp")["groundspeed"]
        b = base.set_index("timestamp")["groundspeed"].reindex(a.index)
        eff = float((a - b).abs().max())
        # covariates from the unchanged flight: speed at t0, range around the event
        seg = gs_base[max(0, t0 - SEQ_LEN // 2): min(n, min(t1, t0 + DYN_WIN) + SEQ_LEN // 2)]
        events[proc.flight_id] = dict(start=int(to_sec([start_ts])[0]), last=int(to_sec([last_ts])[0]),
                                      end=int(to_sec([last_ts])[0]) + MARGIN_S, n_aff=t1 - t0, eff=eff,
                                      gs0=float(gs_base[t0]), dyn=float(seg.max() - seg.min()))
        processed.append(proc)
    return processed, events


def evaluate_speed(processed, events, det, combos):
    """Returns one result dict per (variant, r) combination (sequences are built only once)."""
    res = det.seqs(processed)
    if res is None:
        return None
    X, fids, starts = res
    tsec = {f.flight_id: to_sec(f.data["timestamp"]) for f in processed}
    win = np.zeros(len(fids), dtype=bool)
    cover = np.zeros(len(fids), dtype=bool)
    for fid in np.unique(fids):
        ev = events.get(fid)
        if ev is None:
            continue
        m = np.where(fids == fid)[0]
        ta, tb = tsec[fid][starts[m]], tsec[fid][starts[m] + SEQ_LEN - 1]
        win[m] = (ta <= ev["end"]) & (tb >= ev["start"])
        ov = np.minimum(tb, ev["last"]) - np.maximum(ta, ev["start"]) + 1
        cover[m] = ov >= min(ev["n_aff"], MIN_OVERLAP)
    idx = np.where(win)[0]
    if len(idx) == 0:
        return None
    n_lost = sum(det.clean_count.get(f, 0) for f in set(fids)) - len(fids)
    out = {}
    for combo in combos:
        fl = det.flags(X[idx], *combo)       # only sequences in the event window
        cf = det.clean_by[combo]
        clean = np.array([cf.get((f, int(s)), False) for f, s in zip(fids[idx], starts[idx])], dtype=bool)
        n_missing = sum((f, int(s)) not in cf for f, s in zip(fids[idx], starts[idx]))
        d = pd.DataFrame(dict(fid=fids[idx], cover=cover[idx], raw=fl, new=fl & ~clean, clean=clean))
        e = d.groupby("fid").agg(raw=("raw", "any"), new=("new", "any"), clean=("clean", "any"))
        sq = d[d["cover"] & ~d["clean"]]
        sq = sq.assign(gs0=sq["fid"].map(lambda f: events[f]["gs0"]), dyn=sq["fid"].map(lambda f: events[f]["dyn"]))
        out[combo] = dict(n_events=len(e), n_seq=len(sq),
                          detected_seq=100 * sq["raw"].mean() if len(sq) else float("nan"),
                          detected_net=100 * e["new"].mean(),
                          detected_raw=100 * e["raw"].mean(),
                          chance=100 * e["clean"].mean(),
                          eff_dev_kt=float(np.mean([events[f]["eff"] for f in e.index])),
                          n_lost=int(n_lost), n_no_clean=int(n_missing),
                          _sq=sq[["raw", "gs0", "dyn"]])
    return out


def strata_rows(meta, sq):
    """detected_seq per class (speed at t0, dynamics around the event) for one variant."""
    out = []
    if len(sq) == 0:
        return out
    for kind, col, bins, labels in (("speed", "gs0", SPEED_BINS, SPEED_LABELS),
                                    ("dynamics", "dyn", DYN_BINS, DYN_LABELS)):
        cls = pd.cut(sq[col], bins, labels=labels, right=False)
        g = sq.groupby(cls, observed=True)["raw"].agg(["size", "mean"])
        for c, row in g.iterrows():
            out.append(dict(meta, kind=kind, class_label=str(c), n_seq=int(row["size"]),
                            detected_seq=100 * float(row["mean"])))
    return out


def run_all_speed(raws, bases, det, cross, variants, seed, level, combos):
    """Runs all speed variants and returns (results, strata, cross-model results) as data frames."""
    clean_proc = [Flight(b) for b in bases]
    for d_ in [det] + cross:
        d_.prepare_clean(clean_proc, combos)

    rows, strata, xrows, t_start = [], [], [], time.time()
    for k, v in enumerate(variants, 1):
        processed, events = build_injected_speed(raws, bases, v, seed, level)
        rr = evaluate_speed(processed, events, det, combos) if processed else None
        det.clear()                                    # empty the cache per variant (memory)
        meta = {c: v[c] for c in ("group", "anomaly", "param", "mag", "dur_label", "dur_s", "min_change")}
        xs = []
        if processed:
            for xd in cross:                           # foreign models on the same injected flights
                rx = evaluate_speed(processed, events, xd, combos)
                xd.clear()
                if rx is None:
                    continue
                for combo, r in rx.items():
                    r.pop("_sq", None)
                    xrows.append({**meta, "model": xd.view, "variant": combo[0], "r": combo[1], **r})
                xs.append(f"{xd.view} {rx[combos[0]]['detected_net']:.0f} %")
        if rr is None:
            print(f"[{k}/{len(variants)}] {v['anomaly']} {v['param']}: no suitable flights")
            continue
        for combo, r in rr.items():
            cmeta = {**meta, "variant": combo[0], "r": combo[1]}
            strata += strata_rows(cmeta, r.pop("_sq"))
            rows.append({**cmeta, **r})
        r0 = rows[-len(combos)]
        print(f"[{k}/{len(variants)}] {v['anomaly']} {v['param']}: {r0['n_events']} events, "
              f"detected {r0['detected_seq']:.0f} % of {r0['n_seq']} sequences (eff. dev. {r0['eff_dev_kt']:.1f} kt) "
              f"[{combos[0]}], {time.time() - t_start:.0f} s")
        if xs:
            print("      Cross (net, events): " + ", ".join(xs))
    return pd.DataFrame(rows), pd.DataFrame(strata), pd.DataFrame(xrows)


def report_strata(df, strata, out_dir):
    """Compact tables: only variants whose detection is not already ~0 or ~100 %."""
    if strata is None or len(strata) == 0:
        return
    strata.to_csv(f"{out_dir}/strata.csv", index=False)
    key = ["anomaly", "param"]
    ov = df.set_index(key)["detected_seq"]
    s = strata[strata["group"] != "control"].copy()
    s["overall"] = [ov.get((a, p), np.nan) for a, p in zip(s["anomaly"], s["param"])]
    s = s[(s["overall"] >= 5) & (s["overall"] <= 95)]
    s["cell"] = [f"{e:.0f} ({n})" for e, n in zip(s["detected_seq"], s["n_seq"])]
    order = [k for k in zip(df["anomaly"], df["param"])]
    for kind, labels in (("speed", SPEED_LABELS), ("dynamics", DYN_LABELS)):
        t = s[s["kind"] == kind]
        if len(t) == 0:
            continue
        piv = t.pivot_table(index=key, columns="class_label", values="cell", aggfunc="first")
        piv = piv[[c for c in labels if c in piv.columns]]
        piv = piv.reindex([k for k in order if k in piv.index])
        piv.insert(0, "overall", [round(ov[k]) for k in piv.index])
        print(f"\nDetection by {kind} (detected % (n sequences); only variants with 5-95 % overall)")
        print(piv.to_string())


def report_speed(df, out_dir, strata=None):
    """Writes results.csv and prints the speed report (grids per group, strata, warnings)."""
    df.to_csv(f"{out_dir}/results.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    ctrl = df[df["group"] == "control"]
    if len(ctrl):
        c = float(ctrl["detected_seq"].iloc[0])
        print(f"\nControl (delta 0): detected {c:.1f} % " + ("(OK, must be 0)" if c == 0 else
              "<- should be 0! Check the feature recomputation / the assignment of the sequences."))

    cols = ["param", "n_events", "n_seq", "detected_seq", "detected_net", "eff_dev_kt"]
    for g in ("noise", "drift"):
        t = df[df["group"] == g]
        if len(t):
            print(f"\n{g}: detected sequences in % (n_seq = sequences with anomaly, without prior flag)")
            print(t[cols].round(1).to_string(index=False))
            t[cols].to_csv(f"{out_dir}/{g}.csv", index=False)

    for g in ("offset", "spike"):
        t = df[df["group"] == g]
        if len(t):
            grid = t.pivot_table(index="mag", columns="dur_label", values="detected_seq")
            order = [c for c in [f"{d} s" for d in (SPD_OFFSET_DUR if g == "offset" else SPD_SPIKE_DUR)] + [PERSIST]
                     if c in grid.columns]
            grid = grid[order]
            grid.index.name = "delta [kt]"
            print(f"\n{g}: detected sequences in %, delta x duration")
            print(grid.round(0).to_string())
            grid.to_csv(f"{out_dir}/{g}_grid.csv")
            gn = t.pivot_table(index="mag", columns="dur_label", values="detected_net")[order]
            gn.index.name = "delta [kt]"
            print(f"\n{g}: detected events (net, event level) in %, delta x duration")
            print(gn.round(0).to_string())
            gn.to_csv(f"{out_dir}/{g}_grid_net.csv")

    t = df[df["group"] == "freeze"]
    if len(t):
        grid = t.pivot_table(index="min_change", columns="dur_label", values="detected_seq")
        grid.index.name = "change >= [kt]"
        print("\nfreeze: detected sequences in %, dynamics x duration")
        print(grid.round(0).to_string())
        print(t[["param", "n_events", "n_seq", "detected_seq", "detected_net", "eff_dev_kt"]].round(1).to_string(index=False))
        grid.to_csv(f"{out_dir}/freeze_grid.csv")

    report_strata(df, strata, out_dir)

    if (df["n_lost"] > 0).any():
        print(f"\nWARNING: in {int((df['n_lost'] > 0).sum())} variants sequences are missing compared with the "
              "unchanged flights (build_sequences_raw/point_threshold?). Detection there may be underestimated.")
    miss = int(df["n_no_clean"].sum())
    if miss:
        print(f"Note: {miss} sequences without a counterpart in the unchanged flights (counted as not flagged).")


def check_recompute(bases):
    """Zero injection: recomputation on the unchanged base flight must reproduce the stored values."""
    worst, worst_late = 0.0, 0.0
    for b in bases[:50]:
        r = recompute_gs_features(b)
        for c in ("delta_gs", "gs_smooth", "gs_diff"):
            if c in b.columns:
                d = (r[c] - b[c]).abs().to_numpy()
                worst = max(worst, float(np.nanmax(d)))
                worst_late = max(worst_late, float(np.nanmax(d[30:])))
    print(f"Recomputation check (50 flights): max. deviation overall {worst:.2e}, from row 30: {worst_late:.2e}")
    if worst_late > 1e-6:
        print("  <- Deviation! recompute_gs_features does not match add_delta_gs; do not interpret the results.")


# ====================================================================== VERTICAL
# Grid (altitude in ft, vertical_rate in ft/min)
VERT_NOISE_DUR = 30
ALT_NOISE = (5, 10, 50, 100, 200, 300, 500, 1000, 2000)                       # sigma [ft]
VR_NOISE = (50, 100, 150, 200, 250, 300, 400, 750, 1000, 2000)                # sigma [ft/min]
ALT_STEP = (10, 20, 30, 50, 100, 150, 200, 500, 750, 1000, 2000, 5000)        # persistent altitude offset [ft]
ALT_SPIKE = (50, 100, 200, 300, 500, 750, 1000, 2000, 5000)                   # [ft]
VR_OFFSET = (100, 150, 200, 300, 400, 500, 600, 700, 800, 1500, 2000)         # [ft/min]
VR_SPIKE = (150, 300, 400, 500, 600, 750, 1000, 1500, 2000)                   # [ft/min]
VERT_SPIKE_DUR = (3, 5, 10, 30)
VERT_OFFSET_DUR = (20, 40)
FLIP_DUR = (10, 20, 40)                                                       # sign flip of the vertical_rate
ALT_FREEZE_DUR = (20, 40)
VR_FREEZE_DUR = (10, 20, 30, 40, 50)
ALT_DRIFT = (5, 15, 50, 100)                                                  # Rate [ft/s]
VERT_DRIFT_DUR = 40
# Dynamics conditions (anomaly only where it changes something)
FLIP_MIN_VR = 500             # mean |vertical_rate| in the window [ft/min]
ALT_FREEZE_MIN_CHANGE = 200   # altitude change in the window [ft]
VR_FREEZE_MIN_PTP = 300       # range of the vertical_rate in the window [ft/min]
VERT_GROUPS = ["alt_noise", "vr_noise", "alt_step", "alt_spike", "vr_offset", "vr_spike",
               "vr_sign_flip", "alt_freeze", "vr_freeze", "alt_drift"]


def col_noise(df, t0, t1, rng, col, sigma, signed=True):
    """Gaussian noise on column col on [t0, t1)."""
    v = _col(df, col)
    v[t0:t1] += rng.normal(0, sigma, t1 - t0)
    df[col] = v


def col_offset(df, t0, t1, rng, col, delta, persistent=False, signed=True):
    """Constant offset on column col on [t0, t1) (persistent: until the end of the flight)."""
    s = _sign(rng, signed)
    v = _col(df, col)
    v[t0:(len(v) if persistent else t1)] += s * delta
    df[col] = v


def col_scale(df, t0, t1, rng, col, factor, signed=True):
    """Multiplies column col by factor on [t0, t1) (factor -1 flips the sign)."""
    v = _col(df, col)
    v[t0:t1] *= factor
    df[col] = v


def col_freeze(df, t0, t1, rng, col, signed=True):
    """Freezes column col at its value before t0 for [t0, t1)."""
    v = _col(df, col)
    v[t0:t1] = v[t0 - 1]
    df[col] = v


def col_drift(df, t0, t1, rng, col, rate, signed=True):
    """Ramp with slope rate per second over [t0, t1), then a jump back to the original value."""
    s = _sign(rng, signed)
    v = _col(df, col)
    v[t0:t1] += s * np.arange(t1 - t0) * rate
    df[col] = v


def _v_vert(group, param, col, fn, kw, mag, dur, persistent=False, dyn=None):
    """Builds one vertical variant dict."""
    return dict(group=group, anomaly=group, param=param, col=col, fn=fn, kw=kw, mag=mag, dur=dur,
                dur_label=("persistent" if persistent else f"{dur} s"), persistent=persistent, dyn=dyn)


def make_variants_vertical(groups, signed):
    """List of all vertical anomaly variants for the given groups."""
    V = [_v_vert("control", "delta 0, 20 s", "vertical_rate", col_offset,
                 dict(col="vertical_rate", delta=0, signed=signed), 0, 20)]
    for s in ALT_NOISE if "alt_noise" in groups else ():
        V.append(_v_vert("alt_noise", f"sigma {s} ft, {VERT_NOISE_DUR} s", "altitude", col_noise,
                         dict(col="altitude", sigma=s), s, VERT_NOISE_DUR))
    for s in VR_NOISE if "vr_noise" in groups else ():
        V.append(_v_vert("vr_noise", f"sigma {s} ft/min, {VERT_NOISE_DUR} s", "vertical_rate", col_noise,
                         dict(col="vertical_rate", sigma=s), s, VERT_NOISE_DUR))
    for d in ALT_STEP if "alt_step" in groups else ():
        V.append(_v_vert("alt_step", f"{d} ft, persistent", "altitude", col_offset,
                         dict(col="altitude", delta=d, persistent=True, signed=signed), d, 1, persistent=True))
    for d in ALT_SPIKE if "alt_spike" in groups else ():
        for dur in VERT_SPIKE_DUR:
            V.append(_v_vert("alt_spike", f"{d} ft, {dur} s", "altitude", col_offset,
                             dict(col="altitude", delta=d, signed=signed), d, dur))
    for d in VR_OFFSET if "vr_offset" in groups else ():
        for dur in VERT_OFFSET_DUR:
            V.append(_v_vert("vr_offset", f"{d} ft/min, {dur} s", "vertical_rate", col_offset,
                             dict(col="vertical_rate", delta=d, signed=signed), d, dur))
    for d in VR_SPIKE if "vr_spike" in groups else ():
        for dur in VERT_SPIKE_DUR:
            V.append(_v_vert("vr_spike", f"{d} ft/min, {dur} s", "vertical_rate", col_offset,
                             dict(col="vertical_rate", delta=d, signed=signed), d, dur))
    for dur in FLIP_DUR if "vr_sign_flip" in groups else ():
        V.append(_v_vert("vr_sign_flip", f"sign -1, {dur} s (mean |vr| >= {FLIP_MIN_VR})", "vertical_rate",
                         col_scale, dict(col="vertical_rate", factor=-1.0), 0, dur, dyn="flip"))
    for dur in ALT_FREEZE_DUR if "alt_freeze" in groups else ():
        V.append(_v_vert("alt_freeze", f"{dur} s (change >= {ALT_FREEZE_MIN_CHANGE} ft)", "altitude",
                         col_freeze, dict(col="altitude"), 0, dur, dyn="alt_freeze"))
    for dur in VR_FREEZE_DUR if "vr_freeze" in groups else ():
        V.append(_v_vert("vr_freeze", f"{dur} s (range >= {VR_FREEZE_MIN_PTP} ft/min)", "vertical_rate",
                         col_freeze, dict(col="vertical_rate"), 0, dur, dyn="vr_var"))
    for r in ALT_DRIFT if "alt_drift" in groups else ():
        V.append(_v_vert("alt_drift", f"{r} ft/s, {VERT_DRIFT_DUR} s (max {r * (VERT_DRIFT_DUR - 1)} ft)", "altitude",
                         col_drift, dict(col="altitude", rate=r, signed=signed), r, VERT_DRIFT_DUR))
    return V


def pick_t0_vertical(base, v, rng):
    """Start index in the base flight (the same for all variants); dynamics conditions for flip/freeze."""
    n = len(base)
    hi = n - T0_DUR - TAIL
    if hi <= LO:
        return None
    if v["dyn"] is None:
        return int(rng.integers(LO, hi))
    alt = base["altitude"].to_numpy(float)
    vr = base["vertical_rate"].to_numpy(float)
    d = v["dur"]
    cand = []
    for t in range(LO, hi):
        t1 = min(t + d, n)
        if v["dyn"] == "flip":
            ok = np.mean(np.abs(vr[t:t1])) >= FLIP_MIN_VR
        elif v["dyn"] == "alt_freeze":
            ok = abs(alt[t1 - 1] - alt[t - 1]) >= ALT_FREEZE_MIN_CHANGE
        else:
            ok = np.ptp(vr[t:t1]) >= VR_FREEZE_MIN_PTP
        if ok:
            cand.append(t)
    return int(rng.choice(cand)) if cand else None


def build_injected_vertical(raws, bases, v, seed):
    """Writes anomaly v into every raw flight and recomputes the features. -> (processed, events)"""
    processed, events = [], {}
    for i, (raw, base) in enumerate(zip(raws, bases)):
        n = len(base)
        rng = np.random.default_rng([seed, i])
        t0 = pick_t0_vertical(base, v, rng)
        if t0 is None:
            continue
        t1 = n if v["persistent"] else min(t0 + v["dur"], n)
        ts_b = base["timestamp"]
        start_ts, last_ts = ts_b.iloc[t0], ts_b.iloc[t1 - 1]
        r0 = int(raw["timestamp"].searchsorted(start_ts))
        r1 = len(raw) if v["persistent"] else r0 + (t1 - t0)
        df = raw.copy()
        v["fn"](df, r0, r1, rng, **v["kw"])
        proc = compute_all_features(Flight(df))
        if proc is None or len(proc.data) < SEQ_LEN:
            continue
        # actual (effective) deviation of the affected column in the model input
        a = proc.data.set_index("timestamp")[v["col"]]
        b = base.set_index("timestamp")[v["col"]].reindex(a.index)
        eff = float((a - b).abs().max())
        events[proc.flight_id] = dict(start=int(to_sec([start_ts])[0]), last=int(to_sec([last_ts])[0]),
                                      end=int(to_sec([last_ts])[0]) + MARGIN_S, n_aff=t1 - t0, eff=eff)
        processed.append(proc)
    return processed, events


def evaluate_vertical(processed, events, det, combos):
    """Returns one result dict per (variant, r) combination (sequences are built only once)."""
    res = det.seqs(processed)
    if res is None:
        return None
    X, fids, starts = res
    tsec = {f.flight_id: to_sec(f.data["timestamp"]) for f in processed}
    win = np.zeros(len(fids), dtype=bool)
    cover = np.zeros(len(fids), dtype=bool)
    for fid in np.unique(fids):
        ev = events.get(fid)
        if ev is None:
            continue
        m = np.where(fids == fid)[0]
        ta, tb = tsec[fid][starts[m]], tsec[fid][starts[m] + SEQ_LEN - 1]
        win[m] = (ta <= ev["end"]) & (tb >= ev["start"])
        ov = np.minimum(tb, ev["last"]) - np.maximum(ta, ev["start"]) + 1
        cover[m] = ov >= min(ev["n_aff"], MIN_OVERLAP)
    idx = np.where(win)[0]
    if len(idx) == 0:
        return None
    n_lost = sum(det.clean_count.get(f, 0) for f in set(fids)) - len(fids)
    out = {}
    for combo in combos:
        fl = det.flags(X[idx], *combo)       # only sequences in the event window
        cf = det.clean_by[combo]
        clean = np.array([cf.get((f, int(s)), False) for f, s in zip(fids[idx], starts[idx])], dtype=bool)
        n_missing = sum((f, int(s)) not in cf for f, s in zip(fids[idx], starts[idx]))
        d = pd.DataFrame(dict(fid=fids[idx], cover=cover[idx], raw=fl, new=fl & ~clean, clean=clean))
        e = d.groupby("fid").agg(raw=("raw", "any"), new=("new", "any"), clean=("clean", "any"))
        sq = d[d["cover"] & ~d["clean"]]
        out[combo] = dict(n_events=len(e), n_seq=len(sq),
                          detected_seq=100 * sq["raw"].mean() if len(sq) else float("nan"),
                          detected_net=100 * e["new"].mean(),
                          detected_raw=100 * e["raw"].mean(),
                          chance=100 * e["clean"].mean(),
                          eff_dev=float(np.mean([events[f]["eff"] for f in e.index])),
                          n_lost=int(n_lost), n_no_clean=int(n_missing))
    return out


def run_all_vertical(raws, bases, det, cross, variants, seed, combos):
    """Runs all vertical variants and returns (results, cross-model results) as data frames."""
    clean_proc = [Flight(b) for b in bases]
    for d_ in [det] + cross:
        d_.prepare_clean(clean_proc, combos)

    rows, xrows, t_start = [], [], time.time()
    for k, v in enumerate(variants, 1):
        processed, events = build_injected_vertical(raws, bases, v, seed)
        rr = evaluate_vertical(processed, events, det, combos) if processed else None
        det.clear()                                    # empty the cache per variant (memory)
        meta = {c: v[c] for c in ("group", "anomaly", "param", "mag", "dur_label", "col")}
        xs = []
        if processed:
            for xd in cross:                           # foreign models on the same injected flights
                rx = evaluate_vertical(processed, events, xd, combos)
                xd.clear()
                if rx is None:
                    continue
                for combo, r in rx.items():
                    xrows.append({**meta, "model": xd.view, "variant": combo[0], "r": combo[1], **r})
                xs.append(f"{xd.view} {rx[combos[0]]['detected_net']:.0f} %")
        if rr is None:
            print(f"[{k}/{len(variants)}] {v['anomaly']} {v['param']}: no suitable flights")
            continue
        for combo, r in rr.items():
            rows.append({**meta, "variant": combo[0], "r": combo[1], **r})
        r0 = rr[combos[0]]
        print(f"[{k}/{len(variants)}] {v['anomaly']} {v['param']}: {r0['n_events']} events, "
              f"detected {r0['detected_seq']:.0f} % of {r0['n_seq']} sequences (events {r0['detected_net']:.0f} %, "
              f"eff. dev. {r0['eff_dev']:.1f}) [{combos[0]}], {time.time() - t_start:.0f} s")
        if xs:
            print("      Cross (net, events): " + ", ".join(xs))
    return pd.DataFrame(rows), pd.DataFrame(xrows)


def report_vertical(df, out_dir):
    """Writes results.csv and prints the vertical report (one table per group)."""
    df.to_csv(f"{out_dir}/results.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    ctrl = df[df["group"] == "control"]
    if len(ctrl):
        c = float(ctrl["detected_seq"].iloc[0])
        print(f"\nControl (delta 0): detected {c:.1f} % " + ("(OK, must be 0)" if c == 0 else
              "<- should be 0! Check the assignment of the sequences."))

    cols = ["param", "n_events", "n_seq", "detected_seq", "detected_net", "eff_dev"]
    for g in VERT_GROUPS:
        t = df[df["group"] == g]
        if len(t):
            note = "  (persistent offset: detected_net = more meaningful per event)" if g == "alt_step" else ""
            print(f"\n{g}: detected sequences in %{note}")
            print(t[cols].round(1).to_string(index=False))
            t[cols].to_csv(f"{out_dir}/{g}.csv", index=False)

    if (df["n_lost"] > 0).any():
        print(f"\nWARNING: in {int((df['n_lost'] > 0).sum())} variants sequences are missing compared with the "
              "unchanged flights (build_sequences_raw/point_threshold?). Detection there may be underestimated.")
    miss = int(df["n_no_clean"].sum())
    if miss:
        print(f"Note: {miss} sequences without a counterpart in the unchanged flights (counted as not flagged).")


# ====================================================================== Main
VIEW_INFO = {
    "position": dict(groups=POS_GROUPS, default_groups=["jumps"],
                     model=f"{BASE}/model_position/E_pure_mse_seed42.keras"),
    "speed": dict(groups=SPD_GROUPS, default_groups=SPD_GROUPS,
                  model=f"{BASE}/model_speed/I_latent_smoothing_seed42.keras"),
    "vertical": dict(groups=VERT_GROUPS, default_groups=VERT_GROUPS,
                     model=f"{BASE}/model_vertical/I_altitude_latent_smoothing_seed42.keras"),
}


def write_reports(view, df, strata, out_dir, combos):
    """One result folder (a single combination) or one subfolder per (variant, r)."""
    def rep(d, od, st=None):
        if view == "position":
            report_position(d, od)
        elif view == "speed":
            report_speed(d, od, st)
        else:
            report_vertical(d, od)

    if len(combos) == 1:
        rep(df, out_dir, strata)
        return
    df.to_csv(f"{out_dir}/results_all.csv", index=False)
    if strata is not None:
        strata.to_csv(f"{out_dir}/strata_all.csv", index=False)
    for (v, r), sub in df.groupby(["variant", "r"]):
        od = f"{out_dir}/{v}_r{r:g}"
        os.makedirs(od, exist_ok=True)
        st = None
        if strata is not None:
            st = strata[(strata["variant"] == v) & (strata["r"] == r)].reset_index(drop=True) if len(strata) else strata
        print(f"\n######## Variant {v}, r = {r:g} ########")
        rep(sub.reset_index(drop=True), od, st)


def main():
    """Command line entry point: injects the anomalies, evaluates the model(s) and writes the reports."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True, choices=list(VIEW_INFO))
    ap.add_argument("--model", default=None, help="path to the .keras model (default: paper model of the view, seed 42)")
    ap.add_argument("--data_dir", default=None,
                    help="folder with sequences.npz and scaler.joblib (default: artifacts/sequences/<view>)")
    ap.add_argument("--split_json", default=None, help="split.json (default: artifacts/sequences/split.json)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--n_flights", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", nargs="*", default=None, help="anomaly groups, see above (depends on --view)")
    ap.add_argument("--level", default="feature", choices=["feature", "raw"],
                    help="speed only: injection into the features (feature) or into the raw data (raw)")
    ap.add_argument("--sign", default="random", choices=["random", "pos"],
                    help="speed/vertical only: sign of the anomaly")
    ap.add_argument("--r", type=float, nargs="+", default=[1.0], help="scaling of all thresholds (1 = unscaled)")
    ap.add_argument("--variants", nargs="+", default=["mp"], choices=["mp", "mpx"],
                    help="mp: mean or >=2 time steps; mpx: additionally a single time step above the extreme threshold")
    ap.add_argument("--extreme_q", type=float, default=0.9999, help="quantile of the extreme threshold (time step)")
    ap.add_argument("--cross", nargs="*", default=[],
                    help="foreign models for cross-specificity: view=<model.keras> (e.g. position=... vertical=...)")
    ap.add_argument("--cross_data_root", default=f"{BASE}/sequences", help="contains <view>/sequences.npz and scaler.joblib")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    view = args.view
    info = VIEW_INFO[view]
    if args.groups is None:
        args.groups = list(info["default_groups"])
    bad = [g for g in args.groups if g not in info["groups"]]
    if bad:
        ap.error(f"--groups {bad} does not fit --view {view} (allowed: {', '.join(info['groups'])})")
    if args.model is None:
        args.model = info["model"]
    if args.data_dir is None:
        args.data_dir = f"{BASE}/sequences/{view}"
    for path, hint in ((args.model, "Model missing. Train it with `python train_models.py --view {v}` or pass --model."),
                       (RAW_FILE, "Raw table missing. Run `python clean_flights.py --combine --write_final` first "
                                  "(see README, 'Reproducing the paper'); ADSB_BASE sets the base folder.")):
        if not os.path.exists(path):
            sys.exit(f"Not found: {path}\n{hint.format(v=view)}")
    os.makedirs(args.out_dir, exist_ok=True)

    split_json = args.split_json
    if split_json is None:
        for cand in (f"{BASE}/sequences/split.json",):
            if os.path.exists(cand):
                split_json = cand
                break
    if split_json is None or not os.path.exists(split_json):
        sys.exit("split.json not found. Run `python prepare_sequences.py` first or pass --split_json.")
    print("View:", view, "| Model:", args.model)
    print("split.json:", split_json, "(must match the split used for the models)",
          "| Level:", args.level if view == "speed" else "-",
          "| Sign:", args.sign if view != "position" else "-")
    run_info = dict(vars(args), split_json=split_json, model_name=os.path.basename(args.model),
                    started=time.strftime("%Y-%m-%d %H:%M:%S"),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    with open(f"{args.out_dir}/run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    det = Det(view, args.model, args.data_dir, args.extreme_q)
    cross = make_cross(args.cross, args.cross_data_root, args.extreme_q)
    combos = [(v, float(r)) for v in args.variants for r in args.r]
    signed = args.sign == "random"

    strata = None
    if view == "position":
        print("Combinations (variant, r):", combos)
        raw_dfs = load_flights(split_json, args.split, args.n_flights, args.seed)
        df, xdf = run_all_position(raw_dfs, det, cross, make_variants_position(args.groups), args.seed, combos)
    else:
        raws, bases = build_bases(load_flights(split_json, args.split, args.n_flights, args.seed))
        print(f"{len(bases)} flights after feature computation | combinations (variant, r): {combos}")
        if view == "speed":
            check_recompute(bases)
            df, strata, xdf = run_all_speed(raws, bases, det, cross, make_variants_speed(args.groups, signed),
                                            args.seed, args.level, combos)
        else:
            df, xdf = run_all_vertical(raws, bases, det, cross, make_variants_vertical(args.groups, signed),
                                       args.seed, combos)

    write_reports(view, df, strata, args.out_dir, combos)
    save_cross(xdf, df, det.view, args.out_dir)
    print(f"\nResults in {args.out_dir}/")


if __name__ == "__main__":
    main()