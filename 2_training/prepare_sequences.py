#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_sequences.py  -  60 s sequences and the train/val/test split

Input:   <PIPE>/t_unified_final.parquet     cleaned flights that passed ALL rules of the pipeline
         <PIPE>/<day>/rejected_A|B.parquet  rejected flights without a gap (A: plausibility, B: training rule),
                                            cleaned by exactly the same steps
Output:  <OUT>/sequences_all.npz            train/val/test sequences (unscaled, all features)
         <OUT>/split.json                   flight id -> train | val | test
         <OUT>/flight_filter_log.csv        control: check_flight on the input flights (all should pass)
         <OUT>/sequences_excluded.npz       sequences of the rejected flights (A and B), reason = "<group>:<rule>"
         <OUT>/preprocess_meta.json         numbers for the paper

All exclusions happen in clean_flights.py. check_flight only runs here as a control: if a flight
fails it anyway, it is excluded but reported loudly (there should be none).

Usage:
  python 2_training/prepare_sequences.py --max_flights 200 --out artifacts/sequences_test --no_rejected    # test
  python 2_training/prepare_sequences.py                                                               # full run
"""
import os as _os, sys as _sys  # REPO_ROOT_SHIM
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import logging
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from traffic.core import Traffic

from adsb_utils import build_sequences_raw
from pipeline_common import ALL_FEATURES, check_flight, compute_all_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = os.environ.get("ADSB_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPE = f"{BASE}/artifacts/pipeline"
RAW_FILE = f"{PIPE}/t_unified_final.parquet"
OUT = f"{BASE}/artifacts/sequences"
SEQ_LEN, STRIDE, POINT_THRESHOLD = 60, 15, 4.0
SPLIT_SEED = 42


def featurize(traffic, max_flights=None):
    """compute_all_features on all flights (in parallel; with max_flights serially, for tests only)."""
    if max_flights is None:
        t = traffic.iterate_lazy().pipe(compute_all_features).eval(desc="Features")
        return list(t)
    out = []
    for i, f in enumerate(traffic):
        if i >= max_flights:
            break
        g = compute_all_features(f)
        if g is not None:
            out.append(g)
    return out


def check_all(flights):
    """check_flight as a control. Returns (flights that pass, flights that fail with reason, log table)."""
    keep, bad, rows = [], [], []
    for f in flights:
        ok, reason = check_flight(f)
        df = f.data
        rows.append(dict(flight_id=f.flight_id, reason=reason, n_rows=len(df),
                         gs_max=df["groundspeed"].max(),
                         gs_diff_max=df["gs_diff"].max(), gs_diff_p95=df["gs_diff"].quantile(0.95),
                         vs_diff_max=df["vs_diff"].max(),
                         frac_gs_diff_gt10=(df["gs_diff"] > 10).mean(),
                         turn_max=df["turn_intensity"].max()))
        if ok:
            keep.append(f)
        else:
            bad.append((f, reason))
    return keep, bad, pd.DataFrame(rows)


def make_split(ids, seed=SPLIT_SEED):
    """70/10/20 split by flight id (sorted ids, fixed seed). Returns {flight_id: 'train'|'val'|'test'}."""
    ids = np.array(sorted(set(ids)))
    trv, test = train_test_split(ids, test_size=0.2, random_state=seed)
    train, val = train_test_split(trv, test_size=0.125, random_state=seed)
    return {**{i: "train" for i in train}, **{i: "val" for i in val}, **{i: "test" for i in test}}


def sequences_by_split(keep, split_of):
    """Builds the sequences of each split. Returns (dict of arrays for np.savez, counts per split)."""
    arrays, counts = {}, {}
    for name in ("train", "val", "test"):
        flights = [f for f in keep if split_of[f.flight_id] == name]
        X, fids, starts, scores = build_sequences_raw(flights, features=ALL_FEATURES, seq_len=SEQ_LEN,
                                                      stride=STRIDE, point_threshold=POINT_THRESHOLD)
        log.info(f"{name}: {len(flights)} flights, {len(X)} sequences")
        arrays.update({f"X_{name}": X, f"flight_ids_{name}": fids,
                       f"start_idx_{name}": starts, f"turn_scores_{name}": scores})
        counts[name] = dict(flights=len(flights), sequences=int(len(X)))
    return arrays, counts


def rejected_sequences(pipe_dir, max_files=None):
    """Sequences of the rejected flights (A and B) with reason '<group>:<rule>' taken from the pipeline log."""
    logs = [pd.read_csv(p, usecols=["flight_id", "group", "reason"], low_memory=False)
            for p in sorted(glob(f"{pipe_dir}/*/flight_log.csv"))]
    if not logs:
        log.warning("No flight_log.csv of the pipeline found, rejected flights are skipped")
        return None
    info = pd.concat(logs).drop_duplicates("flight_id").set_index("flight_id")
    Xs, Fs, Ss, Rs = [], [], [], []
    files = sorted(glob(f"{pipe_dir}/*/rejected_[AB].parquet"))
    if max_files:
        files = files[:max_files]
    for p in files:
        flights = featurize(Traffic.from_file(p))
        if not flights:
            continue
        X, fids, starts, _ = build_sequences_raw(flights, features=ALL_FEATURES, seq_len=SEQ_LEN, stride=STRIDE,
                                                 point_threshold=POINT_THRESHOLD)
        if len(X) == 0:
            continue
        fids = np.asarray(fids)
        reasons = np.array([f"{info.at[i, 'group']}:{info.at[i, 'reason']}" if i in info.index else "unknown"
                            for i in fids])
        Xs.append(X.astype("float32")); Fs.append(fids); Ss.append(np.asarray(starts)); Rs.append(reasons)
        log.info(f"{os.path.relpath(p, pipe_dir)}: {len(flights)} flights, {len(X)} sequences")
    if not Xs:
        return None
    return dict(X_excluded=np.concatenate(Xs), flight_ids_excluded=np.concatenate(Fs),
                start_idx_excluded=np.concatenate(Ss), reason_excluded=np.concatenate(Rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_FILE)
    ap.add_argument("--pipe", default=PIPE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max_flights", type=int, default=None, help="for tests only")
    ap.add_argument("--no_rejected", action="store_true", help="do not create sequences_excluded.npz")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t = Traffic.from_file(a.raw)
    flights = featurize(t, a.max_flights)
    log.info(f"{len(flights)} flights with features from {a.raw}")

    keep, bad, prot = check_all(flights)
    prot.to_csv(f"{a.out}/flight_filter_log.csv", index=False)
    if bad:
        reasons = pd.Series([r for _, r in bad]).value_counts().to_string()
        log.warning(f"WARNING: {len(bad)} flights fail check_flight and are excluded "
                    f"(expected: 0, pipeline and check_flight disagree):\n{reasons}")
    else:
        log.info("Control: all flights pass check_flight")

    split_of = make_split([f.flight_id for f in keep])
    json.dump(split_of, open(f"{a.out}/split.json", "w"))
    arrays, counts = sequences_by_split(keep, split_of)
    np.savez(f"{a.out}/sequences_all.npz", feature_names=np.array(ALL_FEATURES), **arrays)

    n_rej = 0
    if not a.no_rejected:
        rej = rejected_sequences(a.pipe, max_files=2 if a.max_flights else None)
        if rej is not None:
            np.savez(f"{a.out}/sequences_excluded.npz", **rej)
            n_rej = int(len(rej["X_excluded"]))
            log.info(f"Rejected flights: {n_rej} sequences\n{pd.Series(rej['reason_excluded']).value_counts().to_string()}")

    meta = dict(input=a.raw, flights_in_file=len(flights), flights_failing_check=len(bad), split=counts,
                rejected_sequences=n_rej, seq_len=SEQ_LEN, stride=STRIDE, split_seed=SPLIT_SEED)
    json.dump(meta, open(f"{a.out}/preprocess_meta.json", "w"), indent=2)
    log.info(f"done: {json.dumps(meta)}")


if __name__ == "__main__":
    main()