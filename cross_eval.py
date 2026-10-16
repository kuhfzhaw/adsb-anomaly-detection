# cross_eval.py
"""
Cross-specificity for injected anomalies: the same injected flights are scored by all three models
(position, speed, vertical), each model with its own features, scaler and thresholds.

Must be located next to eval_excluded.py. Usage in the injection script (inject_anomaly.py):
  det   = Det("speed", args.model, args.data_dir, args.extreme_q)               # own model
  cross = make_cross(args.cross, args.cross_data_root, args.extreme_q)          # e.g. ["position=<path>.keras", ...]
"""
import joblib
import numpy as np
import pandas as pd

from adsb_utils import build_sequences_raw
from adsb_utils import compute_feature_thresholds, flag_anomalies
from eval_excluded import Cached, scale_thr, pick_per_ts, _mul

VIEW_FEATURES = {
    "position": ["dx", "dy", "turn_intensity"],
    "speed": ["groundspeed", "delta_gs", "gs_diff"],
    "vertical": ["altitude", "vertical_rate", "vs_diff"],
}
SEQ_LEN, STRIDE = 60, 15


class Det:
    """One detector: model + scaler + validation thresholds of one view."""

    def __init__(self, view, model_path, data_dir, extreme_q=0.9999, tag=""):
        """Loads the model, the scaler and the validation sequences, and computes the thresholds."""
        import tensorflow as tf
        self.view, self.features, self.tag = view, VIEW_FEATURES[view], tag
        model = tf.keras.models.load_model(model_path, compile=False)
        d = np.load(f"{data_dir}/sequences.npz")
        self.scaler = joblib.load(f"{data_dir}/scaler.joblib")
        self.thr = compute_feature_thresholds(model, d["X_val_scaled"], self.features)
        self.ext = pick_per_ts(compute_feature_thresholds(model, d["X_val_scaled"], self.features,
                                                          timestep_quantile=extreme_q), self.features)
        self.cm = Cached(model)
        self.clean_by, self.clean_count = {}, {}

    def seqs(self, processed):
        """Builds the 60 s sequences of the processed flights. Returns (X, flight_ids, start_idx) or None."""
        X, fids, starts, _ = build_sequences_raw(
            processed, features=self.features, seq_len=SEQ_LEN, stride=STRIDE, point_threshold=4.0)
        if len(X) == 0:
            return None
        return X, np.asarray(fids), np.asarray(starts).astype(int)

    def flags(self, X, variant="mp", r=1.0):
        """r scales all thresholds; mpx additionally allows a single time step above the extreme threshold."""
        Xs = self.scaler.transform(X.astype(float).reshape(-1, X.shape[2])).reshape(X.shape)
        kw = {"extreme_ts_thr": _mul(self.ext, r)} if variant == "mpx" else {}
        out = flag_anomalies(self.cm, Xs, self.features, scale_thr(self.thr, r), **kw)["any_anomaly"]
        return np.asarray(out.to_numpy() if hasattr(out, "to_numpy") else out, dtype=bool)

    def prepare_clean(self, clean_proc, combos):
        """Flags of the unchanged flights (reference for detected_net)."""
        res = self.seqs(clean_proc)
        for combo in combos:
            fl = self.flags(res[0], *combo)
            self.clean_by[combo] = {(f, int(s)): bool(x) for f, s, x in zip(res[1], res[2], fl)}
            print(f"{self.tag}Unchanged flights {combo}: {len(clean_proc)} flights, {len(fl)} sequences, "
                  f"{100 * np.mean(fl):.2f} % flagged")
        self.clean_count = pd.Series(res[1]).value_counts().to_dict()

    def clear(self):
        """Empties the prediction cache (saves memory between variants)."""
        self.cm._c.clear()


def make_cross(specs, data_root, extreme_q):
    """specs: ["position=<model.keras>", "vertical=<model.keras>", ...]; data in <data_root>/<view>/."""
    out = []
    for s in specs or []:
        view, path = s.split("=", 1)
        out.append(Det(view, path, f"{data_root}/{view}", extreme_q, tag=f"[cross {view}] "))
    return out


def save_cross(xdf, own_df, own_view, out_dir):
    """results_cross.csv + short overview: detected_net (events) per model, own model for comparison."""
    if xdf is None or len(xdf) == 0:
        return
    xdf.to_csv(f"{out_dir}/results_cross.csv", index=False)
    own = own_df.copy()
    own["model"] = own_view
    both = pd.concat([own, xdf], ignore_index=True)
    key = ["anomaly", "param"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    for (v, r), sub in both.groupby(["variant", "r"]):
        piv = sub.pivot_table(index=key, columns="model", values="detected_net", aggfunc="first")
        order = list(dict.fromkeys(zip(own_df["anomaly"], own_df["param"])))
        piv = piv.reindex([k for k in order if k in piv.index])
        piv.to_csv(f"{out_dir}/cross_net_{v}_r{r:g}.csv")
        if (v, r) == (both["variant"].iloc[0], both["r"].iloc[0]):
            print(f"\n== Cross-specificity: detected_net [%] (events) per model, {v}, r={r:g} "
                  f"(own model: {own_view}) ==")
            print(piv.round(0).to_string())