"""Combined evaluation of the injection / cross runs (3 models x seeds) for the paper.

Usage (project folder, CPU is enough):
  python aggregate_cross.py \
      --pos    artifacts/inject_position \
      --speed  artifacts/inject_speed \
      --vertical artifacts/inject_vertical \
      --r 1 1.5 2 --out_dir agg_cross
Expects seed*/results_all.csv (own model) and seed*/results_cross.csv (the other models) in each folder.
Missing folders are skipped. Main metric: detected_net (%, event level), mean over seeds.

Output: report.txt (everything important, ready to copy), plus CSV files:
  own_<view>.csv, limits_<view>.csv, cross_matrix.csv, controls.csv
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd
_BASE = os.environ.get("ADSB_BASE", os.path.dirname(os.path.abspath(__file__)))

METRICS = ["n_events", "n_seq", "detected_seq", "detected_net", "detected_raw", "chance",
           "eff_dev_kt", "eff_dev", "n_lost", "n_no_clean"]
MAGCOL = {"position": "dist", "speed": "mag", "vertical": "mag"}
UNIT = {"position": "m", "speed": "kt", "vertical": "(unit of the column)"}


class Tee:
    """Callable that prints a line and also writes it to the report file."""

    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")


def load(d, fname, say):
    """Loads seed*/<fname> from folder d into one table with a 'seed' column; None if nothing is found."""
    parts = []
    for p in sorted(glob.glob(f"{d}/seed*/{fname}")):
        df = pd.read_csv(p)
        df["seed"] = int(re.search(r"seed(\d+)", p).group(1))
        parts.append(df)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    n = df.groupby("seed").size()
    say(f"  {fname}: seeds {sorted(n.index)}" + ("" if n.nunique() == 1 else f"  WARNING unequal row counts {n.to_dict()}"))
    return df


def agg(df):
    """Mean of the metrics over seeds for each variant; adds the standard deviation of detected_net."""
    k = [c for c in df.columns if c not in METRICS and c != "seed"]
    d = df.copy()
    d[k] = d[k].fillna("")
    mets = [m for m in METRICS if m in d.columns]
    g = d.groupby(k, sort=False)
    out = g[mets].mean()
    out["net_std"] = g["detected_net"].std()
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def limits(a, view):
    """Smallest tested magnitude with net detection >= 50 % and >= 95 % for each anomaly type."""
    mc = MAGCOL[view]
    gk = [c for c in ["group", "anomaly", "col", "dur_label", "min_change", "r"] if c in a.columns]
    rows = []
    for key, g in a[a.group != "control"].groupby(gk):
        g = g.assign(_m=pd.to_numeric(g[mc], errors="coerce")).sort_values("_m")
        rec = dict(zip(gk, key))
        for t in (50, 95):
            ok = g[g.detected_net >= t]
            rec[f"lim{t}"] = ok._m.min() if len(ok) else np.nan
        rec["max_net"] = g.detected_net.max()
        rows.append(rec)
    return pd.DataFrame(rows)


def fmt_rng(x):
    """Formats a range 'a-b' (or a single value); 'never' if there is no value."""
    x = x.dropna()
    if not len(x):
        return "never"
    lo, hi = x.min(), x.max()
    return f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default=os.path.join(_BASE, "artifacts/inject_position"))
    ap.add_argument("--speed", default=os.path.join(_BASE, "artifacts/inject_speed"))
    ap.add_argument("--vertical", default=os.path.join(_BASE, "artifacts/inject_vertical"))
    ap.add_argument("--r", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--out_dir", default="agg_cross")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    say = Tee(f"{args.out_dir}/report.txt")
    pd.set_option("display.width", 220, "display.max_rows", 1000, "display.max_columns", 50)
    dirs = {"position": args.pos, "speed": args.speed, "vertical": args.vertical}

    own, cross, ctrl = {}, {}, []
    for view, d in dirs.items():
        say(f"\n##### Loading {view}: {d}")
        a, c = load(d, "results_all.csv", say), load(d, "results_cross.csv", say)
        if a is None:
            say("  -> not found, skipped")
            continue
        miss = sorted(set(args.r) - set(a.r.unique()))
        if miss:
            say(f"  WARNING: r values missing in the data: {miss} (available: {sorted(a.r.unique())})")
        a = a[a.r.isin(args.r)]
        own[view] = agg(a)
        own[view].to_csv(f"{args.out_dir}/own_{view}.csv", index=False)
        k = own[view][own[view].group == "control"]
        ctrl.append(pd.DataFrame(dict(run=view, r=k.r, param=k.param, net=k.detected_net, raw_chance=k.detected_raw)))
        if c is not None:
            c = c[c.r.isin(args.r)]
            cross[view] = agg(c)
            kc = cross[view][cross[view].group == "control"]
            ctrl.append(pd.DataFrame(dict(run=view + "->" + kc.model, r=kc.r, param=kc.param,
                                          net=kc.detected_net, raw_chance=kc.detected_raw)))
    if not own:
        sys.exit("no data found")
    ctrl = pd.concat(ctrl)
    ctrl.to_csv(f"{args.out_dir}/controls.csv", index=False)

    say("\n\n===== 1. Controls (offset/delta 0): detected_net must be 0 =====")
    say("max detected_net over all controls:", ctrl.net.max())
    say("Chance rate (detected_raw) of the controls = share of sequences flagged anyway, own model:")
    say(ctrl[~ctrl.run.str.contains("->")].pivot_table(index="run", columns="r", values="raw_chance").round(1))

    say("\n\n===== 2. Detection limits per model (smallest tested level with net >= 50 % / >= 95 %) =====")
    say("Range 'a-b' = over the tested durations or min_change; 'never' = threshold not reached at any level.")
    for view, a in own.items():
        lim = limits(a, view)
        lim.to_csv(f"{args.out_dir}/limits_{view}.csv", index=False)
        gk = [c for c in ["group", "anomaly", "col"] if c in lim.columns]
        rows = []
        for key, g in lim.groupby(gk + ["r"]):
            rec = dict(zip(gk + ["r"], key))
            rec[f"50% [{UNIT[view]}]"] = fmt_rng(g.lim50)
            rec[f"95% [{UNIT[view]}]"] = fmt_rng(g.lim95)
            rec["max_net"] = round(g.max_net.max(), 1)
            rows.append(rec)
        say(f"\n--- {view} (magnitude: column '{MAGCOL[view]}') ---")
        say(pd.DataFrame(rows).to_string(index=False))

    say("\n\n===== 3. Net detection per anomaly: mean, minimum and maximum over all levels (%) =====")
    for view, a in own.items():
        b = a[a.group != "control"]
        gk = [c for c in ["group", "anomaly", "col"] if c in b.columns]
        say(f"\n--- {view} ---")
        say(b.groupby(gk + ["r"]).detected_net.agg(mean="mean", minimum="min", maximum="max").round(1).unstack("r").to_string())

    say("\n\n===== 4. Cross matrix: injection x scoring model (mean net detection over all levels, %) =====")
    rows = []
    for view in own:
        parts = [own[view][own[view].group != "control"].assign(model=view)]
        if view in cross:
            parts.append(cross[view][cross[view].group != "control"])
        for df in parts:
            for (grp, mdl, r), g in df.groupby(["group", "model", "r"]):
                rows.append(dict(injection=view, group=grp, model=mdl, r=r,
                                 net_mean=g.detected_net.mean(), net_max=g.detected_net.max()))
    mat = pd.DataFrame(rows)
    mat.to_csv(f"{args.out_dir}/cross_matrix.csv", index=False)
    say(mat.pivot_table(index=["injection", "group", "r"], columns="model", values="net_mean").round(1))
    say("\nMaximum over all levels (other models, i.e. where specificity is violated):")
    foreign = mat[mat.injection != mat.model]
    say(foreign.pivot_table(index=["injection", "r"], columns="model", values="net_max", aggfunc="max").round(1))

    if "position" in cross:
        x = cross["position"]
        x = x[(x.group != "control") & (x.model == "speed")]
        say("\n\n===== 5. Position injection -> speed model (coupling via gs_diff), net % per group/distance/duration =====")
        say(x.pivot_table(index=["group", "dist", "dur_label"], columns="r", values="detected_net").round(1).to_string())

    say(f"\nDone. Report: {args.out_dir}/report.txt, tables: {args.out_dir}/*.csv")


if __name__ == "__main__":
    main()