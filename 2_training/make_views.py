# make_views.py
"""Builds the three feature views (position, speed, vertical) from artifacts/sequences/sequences_all.npz.

For every view: selects its feature columns, balances the training set if configured (VIEWS in pipeline_common),
fits the scaler on the training sequences and writes artifacts/sequences/<view>/sequences.npz and scaler.joblib.
ADSB_BASE sets the base folder (default: folder of this script).

Usage:
  python 2_training/make_views.py
"""
import os as _os, sys as _sys  # REPO_ROOT_SHIM
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys
import joblib
import numpy as np
from pipeline_common import ALL_FEATURES, VIEWS

OUT = os.path.join(os.environ.get("ADSB_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "artifacts", "sequences")


def need(path, hint):
    """Exits with a friendly message if a required file is missing."""
    if not os.path.exists(path):
        sys.exit(f"Missing file: {path}\n{hint}")


def balanced_indices(turn_scores, min_points=3, target_ratio=0.10, oversample_factor=2, seed=42):
    """Indices for a training set with about target_ratio curvy sequences: oversamples the curvy ones, subsamples the straight ones."""
    rng = np.random.default_rng(seed)
    curvy = turn_scores >= min_points
    idx_c, idx_s = np.where(curvy)[0], np.where(~curvy)[0]
    n_target = max(1, int(len(idx_c) / target_ratio) - len(idx_c))
    if len(idx_s) > n_target:
        idx_s = rng.choice(idx_s, size=n_target, replace=False)
    idx = np.concatenate([np.repeat(idx_c, oversample_factor), idx_s])
    rng.shuffle(idx)
    return idx


def main():
    """Writes sequences.npz and scaler.joblib for every view."""
    need(f"{OUT}/sequences_all.npz",
         "Run `python 2_training/prepare_sequences.py` first (see README, 'Reproducing the paper'). ADSB_BASE sets the base folder.")
    d = np.load(f"{OUT}/sequences_all.npz")
    for name, cfg in VIEWS.items():
        cols = [ALL_FEATURES.index(f) for f in cfg["features"]]
        idx = balanced_indices(d["turn_scores_train"]) if cfg["balance"] else np.arange(len(d["X_train"]))
        X_tr = d["X_train"][idx][:, :, cols]
        scaler = cfg["scaler"]()
        res = {"X_train_scaled": scaler.fit_transform(X_tr.reshape(-1, len(cols))).reshape(X_tr.shape),
               "flight_ids_train": d["flight_ids_train"][idx], "start_idx_train": d["start_idx_train"][idx]}
        for s in ("val", "test"):
            X = d[f"X_{s}"][:, :, cols]
            res[f"X_{s}_scaled"] = scaler.transform(X.reshape(-1, len(cols))).reshape(X.shape)
            res[f"flight_ids_{s}"], res[f"start_idx_{s}"] = d[f"flight_ids_{s}"], d[f"start_idx_{s}"]

        ex_path = f"{OUT}/sequences_excluded.npz"
        if os.path.exists(ex_path):
            e = np.load(ex_path)
            Xe = e["X_excluded"][:, :, cols]
            res["X_excluded_scaled"] = scaler.transform(Xe.reshape(-1, len(cols))).reshape(Xe.shape)
            res["flight_ids_excluded"] = e["flight_ids_excluded"]
            res["start_idx_excluded"] = e["start_idx_excluded"]
            res["reason_excluded"] = e["reason_excluded"]
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        np.savez(f"{OUT}/{name}/sequences.npz", **res)
        joblib.dump(scaler, f"{OUT}/{name}/scaler.joblib")
        print(name, {k: v.shape for k, v in res.items() if k.startswith("X_")})


if __name__ == "__main__":
    main()