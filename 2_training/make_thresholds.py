# make_thresholds.py
"""
Computes the detection thresholds of the final models from the validation sequences and
saves them as a small JSON file next to the model (<model>_thresholds.json).

Same computation as in cross_eval.Det:
  mean_thr        99 % quantile of the MSE per sequence and feature         (tau)
  per_ts_thr      99.9 % quantile of the per-step errors per feature        (theta)
  extreme_ts_thr  99.99 % quantile of the per-step errors (variant mpx only)

Usage (compute node, TensorFlow required):
  ADSB_BASE=<project folder containing artifacts/sequences/...> python 2_training/make_thresholds.py [--model_root <folder>]
"""
import os as _os, sys as _sys  # REPO_ROOT_SHIM
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from adsb_utils import compute_feature_thresholds
from pipeline_common import VIEWS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("ADSB_BASE", HERE)
MODELS = {"position": "E_pure_mse",
          "speed": "I_latent_smoothing",
          "vertical": "I_altitude_latent_smoothing"}
SEEDS = [42, 7, 123, 155, 17, 98]
QUANTILES = {"per_ts_thr": 0.999, "mean_thr": 0.99, "extreme_ts_thr": 0.9999}


def main():
    """Computes and writes the threshold JSON file for every requested view and seed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(BASE, "artifacts"),
                    help="contains sequences/<view>/sequences.npz (validation data)")
    ap.add_argument("--model_root", default=os.path.join(HERE, "artifacts"),
                    help="contains model_<view>/*.keras; the JSON files are written there")
    ap.add_argument("--views", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    for view in args.views:
        features = list(VIEWS[view]["features"])
        X_val = np.load(f"{args.data_root}/sequences/{view}/sequences.npz")["X_val_scaled"]
        print(f"{view}: {X_val.shape[0]} validation sequences, features {features}", flush=True)
        for seed in args.seeds:
            name = f"{MODELS[view]}_seed{seed}"
            sweep = f"{args.model_root}/model_{view}"
            model = tf.keras.models.load_model(f"{sweep}/{name}.keras", compile=False)
            thr = compute_feature_thresholds(model, X_val, features)
            out = {"view": view, "features": features, "model": f"{name}.keras",
                   "n_val_sequences": int(X_val.shape[0]), "quantiles": QUANTILES, **thr}
            with open(f"{sweep}/{name}_thresholds.json", "w") as f:
                json.dump(out, f, indent=2)

            ref = f"{args.data_root}/model_{view}/{name}_thresholds.npy"
            note = "no .npy for comparison"
            if os.path.exists(ref):
                r = np.load(ref, allow_pickle=True).item()
                rel = max(abs(r[k][ft] - thr[k][ft]) / max(abs(r[k][ft]), 1e-12)
                          for k in QUANTILES for ft in features)
                note = f"max. relative deviation from .npy: {rel:.2e}" + ("" if rel < 1e-3 else "  <-- CHECK")
            print(f"  {name}: saved, {note}", flush=True)


if __name__ == "__main__":
    main()