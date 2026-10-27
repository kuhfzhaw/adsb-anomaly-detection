"""Share of flagged normal test sequences per model, seed and r (table tab:fpr).

Needs artifacts/sequences/<view>/sequences.npz (run make_views.py) and the trained models in
artifacts/model_<view>/ (shipped with the repository or trained with train_models.py).
ADSB_BASE sets the base folder (default: repository root).

Usage:
  python 4_evaluation/baseline_fpr.py
Output: artifacts/baseline_fpr.csv and a summary (mean/std over seeds) on screen.
"""
import os as _os, sys as _sys  # REPO_ROOT_SHIM
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys
import numpy as np
import pandas as pd
import cross_eval as ce

MODELS = {"position": "E_pure_mse", "speed": "I_latent_smoothing",
          "vertical": "I_altitude_latent_smoothing"}
SEEDS = [42, 7, 123, 155, 17, 98]
RS = [1.0, 1.5, 2.0]
BASE = os.environ.get("ADSB_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def need(path, hint):
    """Exits with a friendly message if a required file is missing."""
    if not os.path.exists(path):
        sys.exit(f"Missing file: {path}\n{hint}")


rows = []
for view, name in MODELS.items():
    ddir = f"{BASE}/artifacts/sequences/{view}"
    need(f"{ddir}/sequences.npz",
         "Run `python 2_training/prepare_sequences.py` and then `python 2_training/make_views.py` first (see README, 'Reproducing the paper').")
    Xt = np.load(f"{ddir}/sequences.npz")["X_test_scaled"]
    for seed in SEEDS:
        mp = f"{BASE}/artifacts/model_{view}/{name}_seed{seed}.keras"
        need(mp, f"Model file not found. Use the models shipped in artifacts/ or train them with `python 2_training/train_models.py --view {view}`.")
        det = ce.Det(view, mp, ddir)
        for r in RS:
            fl = ce.flag_anomalies(det.cm, Xt, det.features, ce.scale_thr(det.thr, r))["any_anomaly"]
            fl = np.asarray(fl.to_numpy() if hasattr(fl, "to_numpy") else fl, dtype=bool)
            rows.append(dict(view=view, seed=seed, r=r, n_seq=len(fl), flagged_pct=100 * fl.mean()))
            print(view, seed, r, len(fl), round(100 * fl.mean(), 2), flush=True)
        det.clear()

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/artifacts/baseline_fpr.csv", index=False)
print(df.groupby(["view", "r"]).flagged_pct.agg(["mean", "std"]).round(2))