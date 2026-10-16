#!/usr/bin/env python
"""Detection at the violation for all three models, with r factor, extreme criterion and fine classes.

For every excluded sequence (reason_excluded = "<group>:<rule>") it is checked whether it contains a
violation of exactly this rule (start_idx <= row < start_idx + 60). The row follows from the timestamp in
artifacts/pipeline/<day>/violations.parquet and the position of the timestamp in the flight
(rejected_A/B.parquet, sorted by time).

Output per reason:
  seq_in     flag rate of the sequences WITH a violation of the rule
  seq_out    flag rate of the remaining sequences of the same flights (without this violation)
  normal     flag rate of normal test sequences
  event      share of the violations where at least one containing sequence is flagged
  event_all  share where all containing sequences are flagged
Confidence intervals: bootstrap over flights (mean over seeds).

Features:
  * all three models in one run (--position_models / --speed_models / --vertical_models), plus the
    union ("union": a sequence is flagged if any of the three models flags it; seeds paired by order)
  * column "own": does the rule belong to the model? (dist_diff/turn_intensity -> position; track_jump belongs to no model,
    gs_jump/gs_diff -> speed, vertical_rate/vs_diff -> vertical)
  * r: all thresholds (mean_thr, per_ts_thr, extreme threshold) are multiplied by r; r=1 is the unscaled operation
  * variant mp (mean OR >=2 time steps above the threshold, the standard detector) and
    mpx (additionally: a single time step above the extreme threshold, quantile --extreme_q)
  * dist_diff in finer classes: <100, 100-200, 200-500, 500-750, 750-1000, >=1000 m
  * flag rate of normal test sequences per model/variant/r (the cost of the sensitivity)

Procedure:
  1. Check:  python eval_excluded.py --check --vertical_models <one model> --out_dir tmp
     (small self-test: r=1 must reproduce the previous flags exactly, mpx >= mp, r=2 <= r=1)
  2. If the structure of the thresholds is not recognised: --inspect prints it.
  3. Full run (e.g. via Slurm).

Example:
  python eval_excluded.py \
    --position_models artifacts/model_position/E_pure_mse_seed{42,7,123}.keras \
    --speed_models    artifacts/model_speed/I_latent_smoothing_seed{42,7,123}.keras \
    --vertical_models artifacts/model_vertical/I_altitude_latent_smoothing_seed{42,7,123}.keras \
    --r 1 1.59 2.5 --variants mp mpx --out_dir artifacts/excluded
"""
import argparse, hashlib, inspect, json, os
import numpy as np
import pandas as pd
from adsb_utils import compute_feature_thresholds, flag_anomalies
import glob
_BASE = os.environ.get("ADSB_BASE", os.path.dirname(os.path.abspath(__file__)))

FEATURES = {
    "position": ["dx", "dy", "turn_intensity"],
    "speed": ["groundspeed", "delta_gs", "gs_diff"],
    "vertical": ["altitude", "vertical_rate", "vs_diff"],
}
SEQ_LEN = 60


def build_events(unified_dir, fids, starts, reasons, shift):
    """Maps the rule violations of the rejected flights to row positions within the flights.

    Returns (viol, rows_by_flight): viol[(flight_id, rule)] = sorted row positions (+ shift), rows_by_flight[flight_id] = number of rows."""
    need = set(np.unique(fids))
    rows_by_flight = {}   # flight_id -> {timestamp: position}
    viol = {}             # (flight_id, rule) -> sorted row positions
    for day in sorted(glob.glob(f"{unified_dir}/*/")):
        v = pd.read_parquet(f"{day}/violations.parquet")
        v = v[v.flight_id.isin(need)]
        parts = []
        for n in ("rejected_A", "rejected_B"):
            p = f"{day}/{n}.parquet"
            if os.path.exists(p):
                r = pd.read_parquet(p, columns=["flight_id", "timestamp"])
                parts.append(r[r.flight_id.isin(need)])
        if not parts:
            continue
        r = pd.concat(parts).sort_values(["flight_id", "timestamp"], kind="stable")
        r["pos"] = r.groupby("flight_id").cumcount()
        pos = {(f, t): p for f, t, p in zip(r.flight_id, r.timestamp, r.pos)}
        for f, rule, t in zip(v.flight_id, v.rule, v.timestamp):
            p = pos.get((f, t))
            if p is not None:
                viol.setdefault((f, rule), []).append(p + shift)
        for f, n in r.groupby("flight_id").size().items():
            rows_by_flight[f] = n
    viol = {k: np.unique(v) for k, v in viol.items()}
    return viol, rows_by_flight

# track_jump is a separate signal and belongs to none of the three models -> own=False, the row stays as context
# turn_intensity (computed from compute_track) belongs to the position model
OWN = {"dist_diff": "position", "turn_intensity": "position",
       "gs_jump": "speed", "gs_diff": "speed",
       "vertical_rate": "vertical", "vs_diff": "vertical"}
DOSE_EDGES = {"gs_diff": [60, 100], "vs_diff": [3000, 5000], "dist_diff": [50, 100, 200, 500, 750, 1000, 1500, 2000]}

# ----------------------------------------------------------------------------- Thresholds
def _mul(v, r):
    """Multiplies every number in a nested structure by r."""
    if isinstance(v, dict):
        return {k: _mul(x, r) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_mul(x, r) for x in v)
    out = v * r
    return float(out) if np.ndim(out) == 0 and not hasattr(out, "index") else out


def scale_thr(obj, r):
    """Multiplies all values under keys ending in '_thr' (mean_thr, per_ts_thr) by r."""
    if isinstance(obj, dict):
        return {k: (_mul(v, r) if isinstance(k, str) and k.endswith("_thr") else scale_thr(v, r))
                for k, v in obj.items()}
    return obj


def count_thr_keys(obj):
    """Counts the keys ending in '_thr' in a nested threshold dict."""
    if isinstance(obj, dict):
        return sum((1 if isinstance(k, str) and k.endswith("_thr") else 0) + count_thr_keys(v)
                   for k, v in obj.items())
    return 0


def describe(obj, ind=0):
    """Prints the structure of a nested dict (for --inspect)."""
    pad = "  " * ind
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                print(f"{pad}{k}: dict")
                describe(v, ind + 1)
            else:
                shp = getattr(v, "shape", None)
                print(f"{pad}{k}: {type(v).__name__}{'' if shp is None else ' ' + str(shp)} "
                      f"{'' if shp not in (None, ()) else v}")
    else:
        print(f"{pad}{type(obj).__name__}")


def pick_per_ts(ext, feats):
    """Extreme threshold in the format of extreme_ts_thr: the per-time-step value of each feature."""
    if isinstance(ext, dict):
        if all(isinstance(ext.get(f), dict) and "per_ts_thr" in ext[f] for f in feats):
            return {f: ext[f]["per_ts_thr"] for f in feats}
        if "per_ts_thr" in ext:
            return ext["per_ts_thr"]
    raise ValueError("Threshold format not recognised; run --inspect and check the output.")


class Cached:
    """Caches model outputs (key = content of the input) so that many (r, variant) combinations
    compute the prediction only once."""

    def __init__(self, model):
        self.__dict__["_m"] = model
        self.__dict__["_c"] = {}
        self.calls = 0
        self.hits = 0

    @staticmethod
    def _key(tag, X):
        a = np.ascontiguousarray(np.asarray(X))
        return (tag, a.shape, str(a.dtype), hashlib.blake2b(a.view(np.uint8).reshape(-1), digest_size=16).hexdigest())

    def _get(self, tag, fn, X, *a, **k):
        key = self._key(tag, X)
        if key in self._c:
            self.hits += 1
        else:
            self._c[key] = fn(X, *a, **k)
            self.calls += 1
        return self._c[key]

    def predict(self, X, *a, **k):
        """Cached model.predict."""
        return self._get("predict", self._m.predict, X, *a, **k)

    def __call__(self, X, *a, **k):
        """Cached model call."""
        return self._get("call", self._m, X, *a, **k)

    def __getattr__(self, n):
        return getattr(self.__dict__["_m"], n)


def run_flags(cm, X, feats, thr0, ext0, variant, r):
    """Flags of all sequences for one variant (mp / mpx) and threshold factor r (bool array)."""
    kw = {}
    if variant == "mpx":
        kw["extreme_ts_thr"] = _mul(ext0, r)
    out = flag_anomalies(cm, X, feats, scale_thr(thr0, r), **kw)
    return np.asarray(out["any_anomaly"].to_numpy() if hasattr(out["any_anomaly"], "to_numpy")
                      else out["any_anomaly"]).astype(bool)


# ----------------------------------------------------------------------------- Evaluation
def boot_ci(code, vals, n_boot, rng):
    """Bootstrap over flights: share of flagged (sequences|events), mean over seeds. Returns the 95 % interval."""
    if len(code) == 0:
        return (np.nan, np.nan)
    u, inv = np.unique(code, return_inverse=True)
    if len(u) < 2:
        return (np.nan, np.nan)
    S = vals.shape[1]
    K = np.stack([np.bincount(inv, weights=vals[:, s].astype(float), minlength=len(u)) for s in range(S)], 1)
    N = np.bincount(inv, minlength=len(u)).astype(float)
    ix = rng.integers(0, len(u), (n_boot, len(u)))
    est = (K[ix].sum(1) / N[ix].sum(1)[:, None]).mean(1)
    return tuple(np.percentile(est, [2.5, 97.5]))


def class_labels(rule, val):
    """Class edges and labels for the dose-response table of a rule."""
    edges = DOSE_EDGES.get(rule)
    if edges is None:
        edges = [float(x) for x in np.unique(np.nanquantile(val, [1 / 3, 2 / 3]))]
    e = [-np.inf] + list(edges) + [np.inf]
    labels = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:g}"]
    return e, labels


class Ctx:
    """Container for the sequence / violation / event arrays shared by the analysis."""
    pass


def event_flags(FE, ctx):
    """Per event: (any containing sequence flagged, all containing sequences flagged)."""
    if len(ctx.ev_ptr) == 0:
        z = np.zeros((0, FE.shape[1]), bool)
        return z, z
    sub = FE[ctx.ev_flat]
    return np.logical_or.reduceat(sub, ctx.ev_ptr, axis=0), np.logical_and.reduceat(sub, ctx.ev_ptr, axis=0)


def analyse(FE, ctx, n_boot, rng, clean=None):
    """FE: (N, S) bool. Returns (summary, dose-response) as lists of rows."""
    S = FE.shape[1]
    nan = np.full(S, np.nan)
    A, L = event_flags(FE, ctx)
    rows, drows = [], []
    for reason in ctx.reason_list:
        m = ctx.reasons == reason
        cm, om = m & ctx.contain, m & ~ctx.contain
        fl_c = FE[cm].mean(0) if cm.any() else nan
        fl_o = FE[om].mean(0) if om.any() else nan
        em = ctx.ev_reason == reason
        ev_any = A[em].mean(0) if em.any() else nan
        ev_all = L[em].mean(0) if em.any() else nan
        ci = boot_ci(ctx.fcode[cm], FE[cm], n_boot, rng) if cm.any() else (np.nan, np.nan)
        ci_ev = boot_ci(ctx.ev_fcode[em], A[em], n_boot, rng) if em.any() else (np.nan, np.nan)
        # Overlap: sequences without a violation of any rule of the evaluated model ("purely foreign")
        if clean is not None and cm.any():
            cc = cm & clean
            seq_pure_n = int(cc.sum())
            seq_pure = np.nanmean(FE[cc].mean(0)) if cc.any() else np.nan
            overlap = 1.0 - cc.sum() / cm.sum()
        else:
            seq_pure_n, seq_pure, overlap = 0, np.nan, np.nan
        rows.append(dict(reason=reason, flights=len(np.unique(ctx.fids[m])), seq_in_n=int(cm.sum()),
                         seq_pure_n=seq_pure_n, seq_pure=seq_pure, overlap=overlap,
                         seq_in=np.nanmean(fl_c) if cm.any() else np.nan, seq_in_std=(np.nanstd(fl_c) if cm.any() and np.size(fl_c) > 1 else np.nan),
                         seq_in_lo=ci[0], seq_in_hi=ci[1], seq_out_n=int(om.sum()),
                         seq_out=np.nanmean(fl_o) if om.any() else np.nan,
                         events=int(em.sum()), event=np.nanmean(ev_any) if em.any() else np.nan,
                         event_lo=ci_ev[0], event_hi=ci_ev[1],
                         event_all=np.nanmean(ev_all) if em.any() else np.nan))
        # Dose-response by size of the exceedance
        rule = reason.split(":", 1)[1]
        col = f"max_{rule}"
        sel = m & ctx.contain
        if col not in ctx.dose.columns or sel.sum() < 20:
            continue
        val = ctx.dose[col].reindex(ctx.fids[sel]).to_numpy(dtype=float)
        e, labels = class_labels(rule, val)
        cat = np.asarray(pd.cut(val, e, right=False, labels=labels).astype(str))
        srate = FE[sel].mean(1)
        sfl = ctx.fids[sel]
        evval = ctx.dose[col].reindex(ctx.ev_flight[em]).to_numpy(dtype=float)
        evcat = np.asarray(pd.cut(evval, e, right=False, labels=labels).astype(str))
        erate = A[em].mean(1)
        for lab in labels:
            k, ke = cat == lab, evcat == lab
            if k.any():
                drows.append(dict(reason=reason, class_label=lab, flights=len(np.unique(sfl[k])), seq=int(k.sum()),
                                  seq_rate=srate[k].mean(), events=int(ke.sum()),
                                  event_rate=erate[ke].mean() if ke.any() else np.nan))
    return rows, drows


# ----------------------------------------------------------------------------- Main program
def main():
    """Command line entry point: loads the models, flags the excluded and normal test sequences and writes the result tables."""
    ap = argparse.ArgumentParser()
    for v in FEATURES:
        ap.add_argument(f"--{v}_models", nargs="*", default=[])
    ap.add_argument("--r", nargs="+", type=float, default=[1.0])
    ap.add_argument("--variants", nargs="+", default=["mp", "mpx"], choices=["mp", "mpx"])
    ap.add_argument("--extreme_q", type=float, default=0.9999, help="quantile of the extreme threshold (time step)")
    ap.add_argument("--data_root", default=os.path.join(_BASE, "artifacts/sequences"))
    ap.add_argument("--unified_dir", default=os.path.join(_BASE, "artifacts/pipeline"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--check", action="store_true", help="self-test with one model on a subset")
    ap.add_argument("--inspect", action="store_true", help="print the structure of the thresholds")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    views = {v: getattr(args, f"{v}_models") for v in FEATURES if getattr(args, f"{v}_models")}
    if not views:
        raise SystemExit("Specify at least one group of models (--position_models ...).")
    data = {v: np.load(f"{args.data_root}/{v}/sequences.npz", allow_pickle=True) for v in views}
    import tensorflow as tf

    if args.inspect or args.check:
        v = next(iter(views))
        d = data[v]
        model = tf.keras.models.load_model(views[v][0], compile=False)
        feats = FEATURES[v]
        thr0 = compute_feature_thresholds(model, d["X_val_scaled"], feats)
        ext_raw = compute_feature_thresholds(model, d["X_val_scaled"], feats, timestep_quantile=args.extreme_q)
        if args.inspect:
            print("flag_anomalies", inspect.signature(flag_anomalies))
            print(inspect.getdoc(flag_anomalies))
            print("\ncompute_feature_thresholds", inspect.signature(compute_feature_thresholds))
            print("\nthresholds:"); describe(thr0)
            print("\nthresholds (extreme quantile):"); describe(ext_raw)
            return
        # --check
        assert "extreme_ts_thr" in inspect.signature(flag_anomalies).parameters, "flag_anomalies without extreme_ts_thr"
        n_thr = count_thr_keys(thr0)
        print("keys with '_thr' in the thresholds:", n_thr)
        assert n_thr > 0, "no '_thr' keys found -> --inspect"
        Xs = d["X_excluded_scaled"][:20000]
        Xt = d["X_test_scaled"][:5000]
        cm = Cached(model)
        ext0 = pick_per_ts(ext_raw, feats)
        ok = True
        for name, X in (("excluded", Xs), ("test", Xt)):
            base = np.asarray(flag_anomalies(model, X, feats, thr0)["any_anomaly"]).astype(bool)
            mp1 = run_flags(cm, X, feats, thr0, ext0, "mp", 1.0)
            mpx1 = run_flags(cm, X, feats, thr0, ext0, "mpx", 1.0)
            mp2 = run_flags(cm, X, feats, thr0, ext0, "mp", 2.0)
            c1 = bool((base == mp1).all()); c2 = bool((mpx1 >= mp1).all()); c3 = bool((mp2 <= mp1).all())
            print(f"{name}: r=1 == previous: {c1} | mpx >= mp: {c2} | r=2 <= r=1: {c3} | "
                  f"flag rate previous {base.mean():.4f}, mpx {mpx1.mean():.4f}, r=2 {mp2.mean():.4f}")
            ok &= c1 and c2 and c3
        print(f"Cache: {cm.calls} predictions, {cm.hits} hits")
        print("SELFCHECK", "OK" if ok else "FAILED")
        return

    # ---- Context: sequences, violations, events
    d0 = data[next(iter(views))]
    ctx = Ctx()
    ctx.fids, starts, ctx.reasons = d0["flight_ids_excluded"], d0["start_idx_excluded"], d0["reason_excluded"]
    for v, d in data.items():
        for k in ("flight_ids_excluded", "start_idx_excluded", "reason_excluded"):
            assert np.array_equal(d[k], d0[k]), f"{k} differs between the views ({v})"
    rules = np.array([r.split(":", 1)[1] for r in ctx.reasons])
    viol, rows_by_flight = build_events(args.unified_dir, ctx.fids, starts, ctx.reasons, args.shift)
    last = pd.DataFrame(dict(f=ctx.fids, s=starts)).groupby("f").s.max() + SEQ_LEN
    rest = pd.Series({f: rows_by_flight.get(f, np.nan) for f in last.index}) + args.shift - last
    print(f"Rows (rejected) + shift({args.shift}) minus end of the last window (expected 0..14):\n", rest.describe().to_string())

    ctx.contain = np.zeros(len(ctx.fids), dtype=bool)
    for i, (f, s, rl) in enumerate(zip(ctx.fids, starts, rules)):
        rr = viol.get((f, rl))
        if rr is not None:
            ctx.contain[i] = ((rr >= s) & (rr < s + SEQ_LEN)).any()
    print("Sequences with a violation of their rule:", int(ctx.contain.sum()), "of", len(ctx.fids))

    # all rules violated in a sequence (not only the reason for exclusion)
    all_rules = sorted({rl for (_, rl) in viol})
    print("Rules in violations.parquet:", all_rules)
    print("OWN rules without a hit in violations:", [k for k in OWN if k not in all_rules])
    ctx.has = {rl: np.zeros(len(ctx.fids), bool) for rl in all_rules}
    by_f = {}
    for (f_, rl), rr in viol.items():
        by_f.setdefault(f_, []).append((rl, rr))
    for i, (f_, s_) in enumerate(zip(ctx.fids, starts)):
        for rl, rr in by_f.get(f_, ()):
            if ((rr >= s_) & (rr < s_ + SEQ_LEN)).any():
                ctx.has[rl][i] = True
    clean = {}
    for g in views:
        own_r = [rl for rl, m_ in OWN.items() if m_ == g and rl in ctx.has]
        clean[g] = ~np.any([ctx.has[rl] for rl in own_r], axis=0) if own_r else np.ones(len(ctx.fids), bool)

    ev_reason, ev_flight, ev_idx = [], [], []
    df = pd.DataFrame(dict(f=ctx.fids, s=starts, rule=rules, reason=ctx.reasons, i=np.arange(len(ctx.fids))))
    for (f, rl), g in df.groupby(["f", "rule"]):
        rr = viol.get((f, rl))
        if rr is None:
            continue
        ss, ii = g.s.to_numpy(), g.i.to_numpy()
        for r_ in rr:
            m = (ss <= r_) & (r_ < ss + SEQ_LEN)
            if m.any():
                ev_reason.append(g.reason.iloc[0]); ev_flight.append(f); ev_idx.append(ii[m])
    ctx.ev_reason, ctx.ev_flight = np.array(ev_reason), np.array(ev_flight)
    ctx.ev_ptr = np.cumsum([0] + [len(x) for x in ev_idx])[:-1].astype(int)
    ctx.ev_flat = np.concatenate(ev_idx) if ev_idx else np.zeros(0, int)
    uniq, ctx.fcode = np.unique(ctx.fids, return_inverse=True)
    ctx.ev_fcode = np.searchsorted(uniq, ctx.ev_flight) if len(ctx.ev_flight) else np.zeros(0, int)
    ctx.reason_list = list(np.unique(ctx.reasons))
    parts = []
    for day in sorted(glob.glob(f"{args.unified_dir}/*/")):
        p = f"{day}/flight_log.csv"
        if os.path.exists(p):
            fl = pd.read_csv(p)
            fl = fl[fl.flight_id.isin(set(ctx.fids))]
            parts.append(fl[["flight_id"] + [c for c in fl.columns if c.startswith("max_")]])
    ctx.dose = pd.concat(parts).drop_duplicates("flight_id").set_index("flight_id")
    print("Events:", len(ev_reason))

    # ---- Flags per group, variant, r (prediction only once per model)
    combos = [(v_, r) for v_ in args.variants for r in args.r]
    FEX = {v: {c: [] for c in combos} for v in views}   # excluded
    FTE = {v: {c: [] for c in combos} for v in views}   # normal test sequences
    for v, mps in views.items():
        d, feats = data[v], FEATURES[v]
        for mp in mps:
            print(f"[{v}] {os.path.basename(mp)}", flush=True)
            model = tf.keras.models.load_model(mp, compile=False)
            cm = Cached(model)
            thr0 = compute_feature_thresholds(model, d["X_val_scaled"], feats)
            assert count_thr_keys(thr0) > 0, "no '_thr' keys -> --inspect"
            ext0 = None
            if "mpx" in args.variants:
                ext0 = pick_per_ts(compute_feature_thresholds(model, d["X_val_scaled"], feats,
                                                              timestep_quantile=args.extreme_q), feats)
            for c in combos:
                FEX[v][c].append(run_flags(cm, d["X_excluded_scaled"], feats, thr0, ext0, *c))
                FTE[v][c].append(run_flags(cm, d["X_test_scaled"], feats, thr0, ext0, *c))
            print(f"   predictions {cm.calls}, cache hits {cm.hits}", flush=True)
            del cm, model

    n_seeds = {v: len(views[v]) for v in views}
    do_union = len(views) > 1 and len(set(n_seeds.values())) == 1 and \
        len({data[v]["X_test_scaled"].shape[0] for v in views}) == 1
    rng = np.random.default_rng(0)
    summ, dose, norm = [], [], []
    for c in combos:
        sets = {v: (np.stack(FEX[v][c], 1), np.stack(FTE[v][c], 1)) for v in views}
        if do_union:
            sets["union"] = (np.logical_or.reduce([sets[v][0] for v in views]),
                             np.logical_or.reduce([sets[v][1] for v in views]))
        for g, (fe, ft) in sets.items():
            nrm = ft.mean(0)
            norm.append(dict(model=g, variant=c[0], r=c[1], normal=nrm.mean(), normal_std=nrm.std()))
            rows, drows = analyse(fe, ctx, args.n_boot, rng, clean.get(g))
            for rw in rows:
                rule = rw["reason"].split(":", 1)[1]
                rw.update(model=g, variant=c[0], r=c[1], normal=nrm.mean(), own=(OWN.get(rule) == g))
            for rw in drows:
                rule = rw["reason"].split(":", 1)[1]
                rw.update(model=g, variant=c[0], r=c[1], own=(OWN.get(rule) == g))
            summ += rows; dose += drows
            print(f"done: {g} {c}", flush=True)
    S_ = pd.DataFrame(summ); D_ = pd.DataFrame(dose); N_ = pd.DataFrame(norm)
    S_.to_csv(f"{args.out_dir}/summary_long.csv", index=False)
    D_.to_csv(f"{args.out_dir}/dose_long.csv", index=False)
    N_.to_csv(f"{args.out_dir}/normal_rates.csv", index=False)
    for v in views:
        for c in combos:
            np.save(f"{args.out_dir}/flags_excluded_{v}_{c[0]}_r{c[1]:g}.npy", np.stack(FEX[v][c], 1))
    np.save(f"{args.out_dir}/contain.npy", ctx.contain)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    fmt = lambda x: f"{x:.3f}"
    print("\n== Flag rate of normal test sequences (cost) ==")
    print(N_.pivot_table(index="model", columns=["variant", "r"], values="normal").to_string(float_format=fmt))
    own = S_[S_.own]
    print("\n== Own model: share of flagged sequences WITH a violation (seq_in) ==")
    print(own.pivot_table(index=["reason", "model"], columns=["variant", "r"], values="seq_in").to_string(float_format=fmt))
    print("\n== Own model: events (at least one containing sequence flagged) ==")
    print(own.pivot_table(index=["reason", "model"], columns=["variant", "r"], values="event").to_string(float_format=fmt))
    c0 = combos[0]
    print(f"\n== Specificity (seq_in per model, variant {c0[0]}, r={c0[1]:g}) ==")
    sp = S_[(S_.variant == c0[0]) & (S_.r == c0[1])]
    print(sp.pivot_table(index="reason", columns="model", values="seq_in").to_string(float_format=fmt))
    print(f"\n== PURE specificity (only sequences WITHOUT a violation of any rule of the evaluated model), {c0[0]}, r={c0[1]:g} ==")
    fr = sp[~sp.own & (sp.model != "union")]
    print(fr.pivot_table(index="reason", columns="model", values="seq_pure").to_string(float_format=fmt))
    print("-- number of pure sequences --")
    print(fr.pivot_table(index="reason", columns="model", values="seq_pure_n").to_string(float_format="{:.0f}".format))
    print("-- share of the foreign sequences that also violate a rule of the model (overlap) --")
    print(fr.pivot_table(index="reason", columns="model", values="overlap").to_string(float_format=fmt))
    print("\n== Dose-response, own model ==")
    dd = D_[D_.own]
    print(dd.pivot_table(index=["reason", "class_label"], columns=["variant", "r"], values="seq_rate").to_string(float_format=fmt))
    with open(f"{args.out_dir}/run_info.json", "w") as f:
        json.dump(dict(vars(args)), f, indent=2, default=str)


if __name__ == "__main__":
    main()