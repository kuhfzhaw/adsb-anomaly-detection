# ADS-B Anomaly Detection with Feature-Specific LSTM Autoencoders

Code and trained models for the paper *Feature-Specific Anomaly Detection for
ADS-B Trajectory Data Using LSTM Autoencoders* (OpenSky Symposium 2026, ZHAW
Centre for Aviation).

Three LSTM autoencoders, one per feature group, are trained on ADS-B data from
Swiss airspace (OpenSky Network, 24 days of 2024). Each model reconstructs
60 s sequences; large reconstruction errors flag the corresponding feature group.

| Model    | Features                                  | Config                          |
|----------|-------------------------------------------|---------------------------------|
| Position | `dx`, `dy`, `turn_intensity`              | `E_pure_mse`                    |
| Speed    | `groundspeed`, `delta_gs`, `gs_diff`      | `I_latent_smoothing`            |
| Vertical | `altitude`, `vertical_rate`, `vs_diff`    | `I_altitude_latent_smoothing`   |

Six seeds per model: 42, 7, 123, 155, 17, 98.
Config names are internal labels from our experiments.

## Quick start: find anomalies in your own OpenSky data

```bash
pip install -r requirements.txt
python 3_detection/detect.py --input my_flights.parquet --out_dir results --r 1.5
```

Tested with Python 3.13.

**Quick check.**  To try the detector, fetch a single day as described under [Data](#data) and run 
`python 3_detection/detect.py --input <day file> --max_flights 30
--out_dir results` on it.

**Input.** A Parquet file with one row per second and flight, in the format of
OpenSky state vectors as returned by the `traffic` library:

| Column | Unit | Required |
|--------|------|----------|
| `timestamp` | UTC | yes |
| `latitude`, `longitude` | deg | yes |
| `altitude`, `geoaltitude` | ft | yes |
| `vertical_rate` | ft/min | yes |
| `groundspeed` | kt | yes |
| `track` | deg | yes |
| `flight_id` | | recommended (otherwise built from date, callsign and icao24) |
| `onground`, `icao24`, `callsign` | | optional |

The models expect 1 Hz data, as in the training set. `detect.py` warns if the median
time step within flights is not about 1 s.

**Output** (in `--out_dir`):

| File | Content |
|------|---------|
| `cleaning_intervals.csv` | One row per flagged interval: flight, segment, feature group (`position`, `speed`, `vertical`), flagged features, `start`, `end`, affected raw columns, pipeline rules violated in the interval. This is the list of sequences to clean. |
| `flagged_sequences.csv` | Every flagged 60 s sequence individually, with `max_ratio` and the number of seeds that flagged it. |
| `flights_summary.csv` | One row per flight: row counts, segments, skipped flights with reason, flagged sequences per group. |

**Options.** `--r` scales all thresholds (paper: 1, 1.5, 2; default 1.5; lower = more
sensitive). `--seeds 42` uses one model per group, `--seeds all` takes a majority vote over
all six seeds. `--views` selects feature groups. `--no_outlier_filter` skips the `traffic`
outlier filter. `--model_root` points to a different model folder (default: `artifacts/` in the repository root).

**How to read the results**

- **Processing is identical to training.** Each flight is cleaned as in `clean_flights.py`
  (on-ground rows and repeated positions removed, `traffic` filter `aggressive`, feature computation),
  then cut into 60 s sequences with stride 15 s. A flight is split into segments at time gaps
  longer than 1 s; each segment is checked separately. Flights or segments shorter than 60 s
  cannot be checked and appear in `flights_summary.csv` with a reason.
- **Resolution is one sequence.** A flagged interval is at least 60 s long. The method does
  not localise an error to single seconds.
- **`max_ratio` below 1 is normal.** A sequence is flagged if the mean error exceeds
  `r * mean_thr`, or if at least two time steps exceed `r * per_ts_thr`. A ratio (mean error
  over threshold) below 1 means the sequence was flagged by short spikes.
- **A position error can also flag the speed model**, because the position-derived speed enters
  `gs_diff`. Speed and vertical errors flag only their own group. Check both groups for the
  same interval before cleaning.
- **The `traffic` outlier filter runs before the models, as during training.** Very large single-point
  errors may be smoothed out before the models see them; use `--no_outlier_filter` to inspect unfiltered data.
- **`in_training_scope`** is `False` for flights below 1000 ft or never faster than 80 kt.
  Such flights were excluded from training and are not evaluated in the paper.
- **`rule_violations`** lists the rule-based plausibility limits of `clean_flights.py`
  (for example `dist_diff`, `gs_jump`, `gs_diff`, `vs_diff`) that are violated inside the
  interval. This is a hint only; it does not influence which sequences are flagged.

## Detection rule

For every feature f, a threshold pair is computed on the validation set:

- `mean_thr` (tau_f): 99 % quantile of the per-sequence mean squared error
- `per_ts_thr` (theta_f): 99.9 % quantile of the per-time-step squared error

A feature is flagged in a sequence if its mean error exceeds `r * tau_f`, or if at least
two time steps exceed `r * theta_f`. Thresholds are stored next to each model as
`<name>_thresholds.json`.

## Repository layout

The scripts are grouped by pipeline stage. `adsb_utils.py` and `pipeline_common.py` stay in the
repository root and are imported by all scripts. All commands are run from the repository root.

| File | Purpose |
|------|---------|
| `1_data/fetch_opensky.py`, `1_data/preprocess_opensky.py` | Fetch raw OpenSky data and resample it to daily 1 Hz files |
| `1_data/clean_flights.py` | Raw OpenSky day files to one cleaned table (`t_unified_final.parquet`) |
| `2_training/prepare_sequences.py` | Flight filtering, 60 s sequences (stride 15), train/val/test split by flight |
| `2_training/make_views.py` | Per-feature-group views, scaler fitted on train |
| `2_training/train_models.py` | Model training, one run per view and seed (`--view position\|speed\|vertical`) |
| `2_training/make_thresholds.py` | Thresholds from the validation set |
| `3_detection/detect.py` | Apply the trained models to your own OpenSky file |
| `4_evaluation/baseline_fpr.py` | Share of normal test sequences flagged (false-alarm rates) |
| `4_evaluation/inject_anomaly.py` | Synthetic anomaly injection tests (`--view position\|speed\|vertical`) |
| `4_evaluation/cross_eval.py`, `4_evaluation/aggregate_cross.py` | Cross-model evaluation (which model flags which injection) |
| `4_evaluation/eval_excluded.py` | Evaluation on sequences excluded by the quality filter (detection at the rule violation, all three models) |
| `adsb_utils.py` | Shared helpers: filtering, sequence building, anomaly flagging, thresholds |
| `pipeline_common.py` | Feature and view definitions. `altitude_div` and `altitude_mul` are referenced by the saved vertical scaler (`scaler.joblib`); do not rename or move them |
| `artifacts/` | Trained models, scalers, thresholds, split, preprocessing metadata (see below) |

### `artifacts/`
Everything the pipeline reads and writes lives here (default: in the repository root, or under `$ADSB_BASE`).
Included in the repository:

```
artifacts/
  model_position/   E_pure_mse_seed<seed>.keras + _thresholds.json, _log.csv, _history.json, 6 seeds
  model_speed/      I_latent_smoothing_seed<seed>.keras + _thresholds.json, _log.csv, _history.json, 6 seeds
  model_vertical/   I_altitude_latent_smoothing_seed<seed>.keras + _thresholds.json, _log.csv, _history.json, 6 seeds
  sequences/
    split.json     flight id -> train/val/test split
    preprocess_meta.json  dataset statistics: flights and sequences per split, sequence length, stride, split seed
    position/ speed/ vertical/         scaler.joblib of each view (sequences.npz is not included)
```

Created when you run the pipeline yourself (large, not included): `pipeline/` (output of
`clean_flights.py`), `sequences/sequences_all.npz`, `sequences/<view>/sequences.npz`, and the result
folders of the evaluation scripts.

## Data

Raw ADS-B data and the intermediate sequence files are **not** part of this
repository. The data come from the OpenSky Network and are subject to its terms
of use (https://opensky-network.org). The trained models, scalers, thresholds and the
train/val/test split (`artifacts/`) are included, so the detector can be applied to your
own data without retraining.

To obtain data in the format the code expects (requires an OpenSky account with Trino
access, configured for `pyopensky`; credentials stay in the pyopensky config file and are
never part of this repository):

```bash
# 1. hourly raw files for Swiss airspace (lat 44.05-48.92, lon 4.38-11.24)
python 1_data/fetch_opensky.py --dates 2024-02-15 2024-03-01 --out_dir data_fetching2024

# 2. one 1 Hz file per day, clipped to the Swiss FIR (LSASFRA)
python 1_data/preprocess_opensky.py --start 2024-02-15 --end 2024-03-01 --in_dir data_fetching2024
```

The result (`data_fetching2024/preprocessed/<YYYY_MM_DD>.parquet`) is the input of
`detect.py` and of `clean_flights.py`. The days used in the paper are: 2024-01-06, 2024-01-15, 2024-02-15, 2024-02-20, 2024-03-21, 2024-03-25, 2024-04-12, 2024-04-27, 2024-05-01, 2024-05-20, 2024-06-17, 2024-06-30, 2024-07-06, 2024-07-15, 2024-08-12, 2024-08-20, 2024-09-07, 2024-09-25, 2024-10-08, 2024-10-18, 2024-11-10, 2024-11-30, 2024-12-05, 2024-12-23.

If you use OpenSky data, please cite: M. Schäfer, M. Strohmeier, V. Lenders, I. Martinovic,
M. Wilhelm, *Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for Research*,
Proc. 13th IEEE/ACM International Symposium on Information Processing in Sensor Networks
(IPSN), 2014.

## Reproducing the paper

All scripts read the base directory from the environment variable `ADSB_BASE`
(default: the repository root). Raw day files go to `data_fetching2024/preprocessed/`
(one Parquet file per day, same columns as above). `detect.py` is the exception for the
models: it loads them from `artifacts/` in the repository root (or from `--model_root`).

```bash
export ADSB_BASE=/path/to/workdir

# 1. raw day files -> unified table (artifacts/pipeline/t_unified_final.parquet)
for i in $(seq 0 23); do python 1_data/clean_flights.py --day_index $i; done   # one run per day (24 days)
python 1_data/clean_flights.py --combine --write_final

# 2. sequences and split (artifacts/sequences)
python 2_training/prepare_sequences.py

# 3. feature-group views and scalers
python 2_training/make_views.py

# 4. training, one run per view and seed (6 seeds: 42 7 123 155 17 98)
for s in 42 7 123 155 17 98; do
  python 2_training/train_models.py --view position --seed $s     # E_pure_mse
  python 2_training/train_models.py --view speed    --seed $s     # I_latent_smoothing
  python 2_training/train_models.py --view vertical --seed $s     # I_altitude_latent_smoothing
done

# 5. thresholds (needs all six seeds) and false-alarm rates on normal test sequences
python 2_training/make_thresholds.py
python 4_evaluation/baseline_fpr.py

# 6. injection tests (one run per view; see --help for all options), then aggregation
python 4_evaluation/inject_anomaly.py --view position --r 1 1.5 2 --out_dir artifacts/inject_position/seed42
python 4_evaluation/inject_anomaly.py --view speed    --r 1 1.5 2 --out_dir artifacts/inject_speed/seed42
python 4_evaluation/inject_anomaly.py --view vertical --r 1 1.5 2 --out_dir artifacts/inject_vertical/seed42
python 4_evaluation/aggregate_cross.py --r 1 1.5 2 --out_dir artifacts/aggregated
```

Each injection run uses one model seed (default 42 via `--model`); `aggregate_cross.py` expects one `seed*/` subfolder per seed
under each `artifacts/inject_<view>/`. To evaluate the other two models on the same injected flights (cross-model tables), add
`--cross <view>=<model.keras> ...`, for example `--cross speed=artifacts/model_speed/I_latent_smoothing_seed42.keras`.

Check each script's
`--help` for the remaining options.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE`. Copyright ZHAW School of Engineering, Centre for Aviation. You may use, share and adapt the code, models and documentation, also commercially, as long as you give appropriate credit (see `CITATION.cff`) and indicate changes. The license covers the contents of this repository only; the OpenSky data are not included and remain subject to the OpenSky Network terms of use.

## Citation

See `CITATION.cff` (GitHub: "Cite this repository").
