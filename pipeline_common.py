# pipeline_common.py
"""Feature list, training-selection rules (check_flight) and the scalers of the three feature groups.

compute_all_features is the single feature function used for training, injection and evaluation.
The scaler helper functions (altitude_div / altitude_mul) are referenced by name inside the saved
scaler.joblib files of the vertical view, so they must stay in this module under these names.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from adsb_utils import compute_features
from adsb_utils import (
    add_delta_gs, add_vs_consistency, compute_track_delta,
    filter_flights_no_high_turn_intensity,
)

ALL_FEATURES = ["dx", "dy", "turn_intensity", "groundspeed", "delta_gs", "gs_diff",
                "altitude", "vertical_rate", "vs_diff"]
GS_MIN = 80.0
GS_DIFF_MAX = 45.0      # None = rule switched off
VS_DIFF_MAX = 2000

def compute_all_features(flight):
    """Computes all features only. Never discards a flight; a feature function that would discard one raises an error."""
    flight = flight.cumulative_distance()
    for step in (compute_features, add_delta_gs, add_vs_consistency, compute_track_delta):
        flight = step(flight)
        if flight is None:
            raise RuntimeError(f"{step.__name__} discarded a flight: move the rule into check_flight")
    df = flight.data.dropna(subset=ALL_FEATURES).reset_index(drop=True)
    df[ALL_FEATURES] = df[ALL_FEATURES].astype(float)
    flight.data = df
    return flight


def check_flight(flight):
    """Returns (ok, reason). Only for the selection of training flights, never for evaluation or injection."""
    df = flight.data
    if len(df) < 60:
        return False, "too_short"
    if df["groundspeed"].max() < GS_MIN:
        return False, "low_groundspeed"
    if GS_DIFF_MAX is not None and df["gs_diff"].max() >= GS_DIFF_MAX:
        return False, "gs_diff_max"
    if VS_DIFF_MAX is not None and df["vs_diff"].max() >= VS_DIFF_MAX:
        return False, "vs_diff_max"
    if filter_flights_no_high_turn_intensity(flight) is None:
        return False, "high_turn_intensity"
    return True, "ok"


# --- Scalers (defined here so that joblib can load them anywhere) ---
def altitude_div(x):
    """Altitude in ft -> thousands of ft. Referenced by the saved vertical scaler; do not rename."""
    return x / 1000.0


def _arcsinh_robust():
    """arcsinh transform followed by a RobustScaler (for signed, heavy-tailed features)."""
    return Pipeline([("arcsinh", FunctionTransformer(np.arcsinh, inverse_func=np.sinh)),
                     ("robust", RobustScaler())])


def _log1p_robust():
    """log1p transform followed by a RobustScaler (for non-negative, skewed features)."""
    return Pipeline([("log1p", FunctionTransformer(np.log1p, inverse_func=np.expm1)),
                     ("robust", RobustScaler())])


def scaler_position():
    """Scaler of the position view: dx, dy standardised; turn_intensity log1p + robust."""
    return ColumnTransformer([("position_std", StandardScaler(), [0, 1]),
                              ("turn_robust", _log1p_robust(), [2])])


def scaler_speed():
    """Scaler of the speed view: groundspeed standardised; delta_gs arcsinh + robust; gs_diff log1p + robust."""
    return ColumnTransformer([("groundspeed", StandardScaler(), [0]),
                              ("delta_gs", _arcsinh_robust(), [1]),
                              ("gs_diff", _log1p_robust(), [2])])


def scaler_vertical():
    """Scaler of the vertical view: altitude /1000 then standardised; vertical_rate standardised; vs_diff log1p + robust."""
    alt = Pipeline([
        ("div", FunctionTransformer(altitude_div, inverse_func=altitude_mul)),
        ("std", StandardScaler())
    ])
    return ColumnTransformer([
        ("altitude", alt, [0]),
        ("vertical_rate", StandardScaler(), [1]),
        ("vs_diff", _log1p_robust(), [2])
    ])


VIEWS = {
    "position": dict(features=["dx", "dy", "turn_intensity"], balance=True, scaler=scaler_position),
    "speed": dict(features=["groundspeed", "delta_gs", "gs_diff"], balance=True, scaler=scaler_speed),
    "vertical": dict(features=["altitude", "vertical_rate", "vs_diff"], balance=False, scaler=scaler_vertical),
}

def altitude_mul(x):
    """Inverse of altitude_div. Referenced by the saved vertical scaler; do not rename."""
    return x * 1000.0