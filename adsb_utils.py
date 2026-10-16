"""Shared helper functions: feature computation (UTM position, speed, vertical consistency, turn
intensity), sequence building, the traffic outlier filter, and the threshold / flagging logic of the
autoencoder detector."""


import numpy as np
import pandas as pd
from pyproj import Transformer
import warnings


# ============================================================================
# Feature computation
# ============================================================================


def compute_features(flight):
    """Adds the basic features: dt, projected positions x/y/z (m), speed (m/s), differences, dx/dy and the
    consistency columns dist_diff and climb_diff. Drops on-ground rows and the first row."""
    df = flight.data.copy()

    # -----------------------------------------
    # Time difference (seconds)
    # -----------------------------------------
    ts = df["timestamp"]

    if hasattr(ts, "dt"):
        dt = ts.diff().dt.total_seconds()
    elif str(ts.dtype).startswith("timestamp"):
        dt = ts.to_pandas().diff().dt.total_seconds()
    else:
        dt = ts.diff()

    df["dt"] = dt.fillna(1.0).replace(0, 1e-6)

    # -----------------------------------------
    # PROJECTION: lat/lon -> metres (UTM)
    # -----------------------------------------
    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32632",
        always_xy=True
    )

    x, y = transformer.transform(
        df["longitude"].to_numpy(),
        df["latitude"].to_numpy()
    )
    df["x"] = x
    df["y"] = y
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    # HARD NORMALISATION (IMPORTANT!)
    df["x"] = df["x"] - df["x"].iloc[0]
    df["y"] = df["y"] - df["y"].iloc[0]

    df["z"] = df["altitude"] * 0.3048

    # -----------------------------------------
    # Speed (in m/s)
    # -----------------------------------------
    if "groundspeed" in df.columns:
        df["speed"] = df["groundspeed"] * 0.514444
    else:
        df["speed"] = np.nan

    # -----------------------------------------
    # Vertical rate (in metres per second)
    # -----------------------------------------
    if "vertical_rate" in df.columns:
        df["vertical_rate_mps"] = pd.to_numeric(
            df["vertical_rate"], errors="coerce"
        ) * 0.00508
    else:
        df["vertical_rate_mps"] = np.nan

    # -----------------------------------------
    # DIFFERENTIAL FEATURES
    # -----------------------------------------
    df["delta_x"] = df["x"].diff() / df["dt"]
    df["delta_y"] = df["y"].diff() / df["dt"]
    df["delta_z"] = df["z"].diff() / df["dt"]

    # do NOT divide dx/dy by dt!
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()

    # -----------------------------------------
    # Consistency features
    # -----------------------------------------
    # horizontal distance (haversine)
    df["horizontal_dist"] = haversine(
        df["latitude"].shift(1),
        df["longitude"].shift(1),
        df["latitude"],
        df["longitude"]
    )
    df.loc[0, "horizontal_dist"] = 0.0

    df["expected_dist"] = df["speed"]

    df["dist_diff"] = df["horizontal_dist"] - df["expected_dist"]
    df["climb_diff"] = df["delta_z"] - df["vertical_rate_mps"]

    # -----------------------------------------
    # Remove onground points and transitions
    # -----------------------------------------
    if "onground" in df.columns:
        og = df["onground"].fillna(False)
        transition = (og.shift(1) != og)
        mask = (~og) & (~transition)
        df = df.loc[mask].reset_index(drop=True)

    # -----------------------------------------
    # Cleanup
    # -----------------------------------------
    df = df.iloc[1:].reset_index(drop=True)

    numeric_cols = [
        "dt", "x", "y", "z",
        "delta_x", "delta_y", "delta_z",
        "speed", "vertical_rate_mps",
        "horizontal_dist", "expected_dist",
        "dist_diff", "climb_diff"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    flight.data = df
    return flight


def add_vs_consistency(flight):
    """Adds vs_diff: smoothed absolute difference between the reported vertical rate and the one derived from altitude."""
    df = flight.data.copy()
    df["altitude_smooth"] = df["altitude"].ewm(alpha=0.2).mean()

    if "dt" in df.columns:
        computed_vs = df["altitude"].diff().fillna(0.0) / df["dt"].replace(0, 1) * 60
    else:
        computed_vs = df["altitude"].diff().fillna(0.0) * 60

    computed_vs = computed_vs.rolling(window=15, min_periods=1, center=False).mean()

    df["vs_diff"] = (df["vertical_rate"] - computed_vs).abs()
    df["vs_diff"] = df["vs_diff"].rolling(window=15, min_periods=1, center=False).mean()

    flight.data = df
    return flight


# ------------------------------------------
# Haversine function for the horizontal distance
# ------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Computes the haversine distance in metres between two points.
    Expects coordinates in degrees.
    """
    R = 6371000  # Earth radius in metres

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# ------------------------------------------
# Heading change over a window of 5 steps (turn intensity)
# ------------------------------------------
def compute_track_delta(flight, window=5):
    """
    Computes the turn intensity: the heading is first smoothed over a window
    to remove GPS jitter in straight flight completely.
    """
    df = flight.data.copy()

    # Force track to numeric
    df["compute_track"] = pd.to_numeric(df["compute_track"], errors="coerce")
    
    # 1) Convert angles to radians
    rad_track = np.radians(df["compute_track"])
    
    # 2) Split into x and y components (avoids problems at the 360° -> 0° wrap)
    df["track_cos"] = np.cos(rad_track)
    df["track_sin"] = np.sin(rad_track)

    # Raw delta between compute_track values (correctly cyclic)
    a_raw = np.radians(df["compute_track"])
    b_raw = np.radians(df["compute_track"].shift(1))

    raw_delta = np.degrees(np.arctan2(np.sin(a_raw - b_raw), np.cos(a_raw - b_raw)))
    df["delta_track_raw"] = raw_delta.fillna(0.0).abs()
    
    # 3) SMOOTH the components (moving average)
    # A larger window (e.g. 15 or 30) removes GPS noise in straight flight completely
    smooth_cos = df["track_cos"].rolling(window=window, min_periods=1, center=False).mean()
    smooth_sin = df["track_sin"].rolling(window=window, min_periods=1, center=False).mean()
    
    # 4) Recompute the smoothed heading
    df["smooth_track"] = np.degrees(np.arctan2(smooth_sin, smooth_cos))
    # Normalise to [0, 360]
    df["smooth_track"] = np.where(df["smooth_track"] >= 0, df["smooth_track"], 360 + df["smooth_track"])
    
    # 5) Only NOW compute the delta of the smoothed heading (step to step)
    a = np.radians(df["smooth_track"])
    b = np.radians(df["smooth_track"].shift(1))
    
    delta = np.degrees(np.arctan2(np.sin(a - b), np.cos(a - b)))
    df["delta_track_smooth"] = delta.fillna(0.0).abs()
    
    # 6) Turn intensity is the sum of the smoothed heading changes
    # With the noise gone, the sum only reacts to real, sustained turns
    df["turn_intensity"] = (
        df["delta_track_smooth"]
        .rolling(window=window, min_periods=1)
        .sum()
    )
    
    # Delete temporary helper columns
    df = df.drop(columns=["track_cos", "track_sin", "smooth_track", "delta_track_smooth"])

    flight.data = df
    return flight

# ------------------------------------------
# Change of the groundspeed (delta_gs) per time step
# ------------------------------------------

def add_delta_gs(flight, smooth_gs=True):
    """Adds delta_gs (smoothed groundspeed change per second) and gs_diff (smoothed difference between the computed
    and the reported groundspeed)."""
    df = flight.data.copy()
    if smooth_gs:
        df["groundspeed"] = df["groundspeed"].rolling(window=10, min_periods=1, center=False).mean()
    df["delta_gs"] = (
        df.groupby("flight_id")["groundspeed"]
          .diff()
          .fillna(0.0)
    )

    df["delta_compute_gs"]  = (
        df.groupby("flight_id")["compute_gs"]
          .diff()
          .fillna(0.0)
    )
    df["gs_smooth"] = df["groundspeed"].ewm(alpha=0.2).mean()
    df["gs_norm"] = df["groundspeed"] / 600.0

    if "dt" in df.columns:
        df["delta_gs"] = df["delta_gs"] / df["dt"].replace(0, 1)
        df["delta_gs"] = df["delta_gs"].rolling(window=10, min_periods=1, center=False).mean()
        df["delta_compute_gs"] = df["delta_compute_gs"] / df["dt"].replace(0, 1)
        df["delta_compute_gs"] = df["delta_compute_gs"].rolling(window=15, min_periods=1, center=False).mean()
    if "compute_gs" in df.columns:
        df["gs_diff"] = (df["compute_gs"] - df["gs_smooth"]).abs()
        df["gs_diff"] = df["gs_diff"].rolling(window=20, min_periods=1, center=False).mean()
    flight.data = df
    return flight


# ============================================================================
# Flight selection and sequence building
# ============================================================================


def filter_flights_no_high_turn_intensity(flight):
    """Returns the flight, or None if its id is on the block list or its maximum |turn_intensity| is >= 40."""
    # List of flight_ids to remove
    blocked_ids = {
        "24/10/08_LAS3_5521",
        "24/06/30_THY5GH_6677",
        "24/09/07_DMWYK_3180",
        "24/04/12_CHX45_2966",
        "24/06/30_CHX45_2669",
        "24/08/12_DKSSB_2884",
        "24/04/12_DRAGO741_1791",
        "24/04/27_REGA15_5308"
    }

    # 1) Remove flights with a blocked flight_id
    if flight.flight_id in blocked_ids:
        return None

    # 2) Remove flights with high turn_intensity
    df = flight.data
    turn = df["turn_intensity"]

    if turn.abs().max() < 40:
        return flight

    return None


def build_sequences_raw(
    flights,
    features,
    seq_len=60,
    stride=15,
    point_threshold=5.0,
):
    """Cuts every flight into windows of seq_len rows with the given stride.

    Returns (X, flight_ids, start_indices, turn_scores); turn_scores counts the steps with turn_intensity > point_threshold.
    """
    X = []
    flight_ids = []
    start_indices = []
    turn_scores = []

    for flight in flights:
        df = flight.data
        arr = df[features].values
        ti = df["turn_intensity"].values
        n = len(arr)

        if n < seq_len:
            continue

        for s in range(0, n - seq_len + 1, stride):
            X.append(arr[s:s+seq_len])
            flight_ids.append(flight.flight_id)
            start_indices.append(s)

            window = ti[s:s+seq_len]
            count_over = np.sum(window > point_threshold)
            turn_scores.append(count_over)

    return (
        np.array(X),
        np.array(flight_ids),
        np.array(start_indices),
        np.array(turn_scores)
    )


# ============================================================================
# Outlier filter
# ============================================================================


warnings.simplefilter(action='ignore', category=FutureWarning)

def filtering(flight):
    """traffic outlier filter ("aggressive"): replaces outlier values. Returns the flight, or None if nothing remains."""
    flight.data["onground"] = flight.data["onground"].fillna(False).astype(bool)
    flight = flight.filter("aggressive")
    if flight is not None and len(flight) > 0:
        return flight
    else:
        return None
    


# ============================================================================
# Detector: targets, thresholds and flagging
# ============================================================================


def altitude_div(x):
    """Altitude in ft -> thousands of ft (a duplicate of pipeline_common.altitude_div; do not remove)."""
    return x / 1000.0
def make_targets(X_scaled, feature_names):
    """One target array per feature, matching the output heads `<feature>_out`."""
    return {f"{name}_out": X_scaled[:, :, i:i + 1] for i, name in enumerate(feature_names)}


def compute_feature_thresholds(model, X_scaled, feature_names,
                                 timestep_quantile=0.999, mean_quantile=0.99,
                                 extreme_timestep_quantile=0.9999,
                                 per_feature_mean_quantile=None):
    """Detection thresholds from the (validation) reconstruction errors, per feature.

    per_ts_thr: timestep_quantile of the per-step squared errors (theta); mean_thr: mean_quantile of the
    per-sequence mean errors (tau); extreme_ts_thr: extreme_timestep_quantile of the per-step errors.
    """
    per_feature_mean_quantile = per_feature_mean_quantile or {}
    y_true = make_targets(X_scaled, feature_names)
    preds = model.predict(X_scaled, batch_size=512, verbose=0)

    per_ts_thr, mean_thr, extreme_ts_thr = {}, {}, {}
    for i, name in enumerate(feature_names):
        errs = (y_true[f"{name}_out"][:, :, 0] - preds[i][:, :, 0]) ** 2
        flat = errs.flatten()
        per_ts_thr[name] = float(np.quantile(flat, timestep_quantile))
        extreme_ts_thr[name] = float(np.quantile(flat, extreme_timestep_quantile))
        q = per_feature_mean_quantile.get(name, mean_quantile)
        mean_thr[name] = float(np.quantile(np.mean(errs, axis=1), q))
    return {"per_ts_thr": per_ts_thr, "mean_thr": mean_thr, "extreme_ts_thr": extreme_ts_thr}


def flag_anomalies(model, X_scaled, feature_names, thresholds, min_points=2, extreme_ts_thr=None):
    """
    Uses BOTH threshold types: a sequence is anomalous if either its mean error
    is above the threshold OR at least min_points time steps individually exceed
    their threshold (also catches short, isolated spikes that would be diluted
    in the mean). With extreme_ts_thr, a single step above the extreme threshold also counts.
    Returns a data frame with one `<feature>_anomaly` column per feature, `any_anomaly` and `<feature>_mean_error`.
    """
    per_ts_thr, mean_thr = thresholds["per_ts_thr"], thresholds["mean_thr"]
    y_true = make_targets(X_scaled, feature_names)
    preds = model.predict(X_scaled, batch_size=512, verbose=0)

    flags = {}
    mean_errors = {}
    for i, name in enumerate(feature_names):
        errs = (y_true[f"{name}_out"][:, :, 0] - preds[i][:, :, 0]) ** 2
        mean_err = np.mean(errs, axis=1)
        mean_errors[name] = mean_err
        mean_flag = mean_err > mean_thr[name]
        points_flag = np.sum(errs > per_ts_thr[name], axis=1) >= min_points
        extreme_flag = np.max(errs, axis=1) > extreme_ts_thr[name] if extreme_ts_thr else False
        flags[f"{name}_anomaly"] = mean_flag | points_flag | extreme_flag

    flags_df = pd.DataFrame(flags)
    flags_df["any_anomaly"] = flags_df.any(axis=1)
    for name in feature_names:
        flags_df[f"{name}_mean_error"] = mean_errors[name]
    return flags_df