"""Step 2: hourly raw files -> one resampled (1 Hz) Parquet file per day, clipped to the Swiss FIR (LSASFRA).

Usage:
    python preprocess_opensky.py --start 2024-01-01 --end 2024-01-31 [--in_dir DIR] [--out_dir DIR]

Input : <in_dir>/<YYYY-MM-DD>_<HH>-<HH>.parquet (output of fetch_opensky.py)
Output: <out_dir>/<YYYY_MM_DD>.parquet, the input format of clean_flights.py and detect.py.
Defaults: in_dir = $ADSB_BASE/data_fetching2024, out_dir = <in_dir>/preprocessed.
--start and --end are inclusive. The Swiss FIR polygon is loaded by the traffic library (needs internet).
"""
import argparse
import glob
import os
from os.path import join as opj
from traffic.core import Traffic, Flight
from traffic.algorithms.filters import FilterMedian, FilterMean
import pandas as pd
from datetime import datetime, timedelta

from traffic.data import aixm_airspaces
aixm_airspaces.data = aixm_airspaces.data.set_geometry("geometry")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
     

lsasfra = aixm_airspaces["LSASFRA"]

# Removal of short flights
def remove_short(flight):
    """Keeps the flight only if it has more than 20 rows."""
    if len(flight)>20:
        return flight

# Removal of flights with no altitude
def remove_ground_only(flight):
    """Keeps the flight only if it has at least one altitude value."""
    if not flight.data['altitude'].isna().all():
        return flight

# Add typecode to flight data 
def add_typecode(flight):
    """Adds the aircraft typecode as a column ('unknown' if not available)."""
    if flight.typecode is not None:
        flight.data["typecode"] = flight.typecode
    else:
        flight.data["typecode"] = "unknown"
    return flight

# Resampling the flight data
def resample_flight(flight):
    """Resamples the flight to 1 Hz: interpolates numeric columns, forward-fills identifiers, keeps only the new rows within gaps of at most 10 s."""
    df = flight.data

    df["original"] = True
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds()
    df["interpolate_row"] = df["time_diff"] <= 10

    # Resample to 1-second intervals
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")]
    df_resampled = df.resample("1s").asfreq()
    df_resampled["original"] = False

    # Combine the original and resampled, then sort
    df_combined = pd.concat([df, df_resampled[~df_resampled.index.isin(df.index)]])
    df_combined = df_combined.sort_values("timestamp")

    # interpolate and fill
    interpolate_cols = [
        "latitude",
        "longitude",
        "altitude",
        "vertical_rate",
        "geominusbaro",
        "geoaltitude",
        "groundspeed",
        "track",
    ]
    df_combined[interpolate_cols] = df_combined[interpolate_cols].interpolate(
        method="time"
    )
    fill_cols = [
        "icao24",
        "squawk",
        "nic",
        "onground",
        "callsign",
        "flight_id",
        "typecode",
    ]
    df_combined["nic"] = pd.to_numeric(df_combined["nic"], errors="coerce").astype(
        "Int64"
    )
    df_combined = df_combined.astype(
        {
            "icao24": "string",
            "squawk": "string",
            "onground": "boolean",
            "callsign": "string",
            "flight_id": "string",
            "typecode": "string",
        }
    )
    df_combined[fill_cols] = df_combined[fill_cols].ffill().bfill()
    df_combined["interpolate_row"] = df_combined["interpolate_row"].bfill()

    df_combined = df_combined[
        (df_combined.original == False) & (df_combined.interpolate_row == True)
    ]
    df_combined = df_combined.reset_index()
    df_combined = df_combined[
        [
            "timestamp",
            "icao24",
            "latitude",
            "longitude",
            "onground",
            "altitude",
            "nic",
            "vertical_rate",
            "geominusbaro",
            "geoaltitude",
            "groundspeed",
            "track",
            "squawk",
            "callsign",
            "flight_id",
        ]
    ]

    return Flight(df_combined)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last day (inclusive), YYYY-MM-DD")
    ap.add_argument("--in_dir", default=opj(os.environ.get("ADSB_BASE", os.path.dirname(os.path.abspath(__file__))),
                                            "data_fetching2024"))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    trajectory_data_path = args.in_dir
    out_dir = args.out_dir or opj(trajectory_data_path, "preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    for day in days:
        # Load trajectory data for day
        print(f"Processing {day.strftime('%Y-%m-%d')}")
        list_parquet = glob.glob(
            opj(
                trajectory_data_path,
                f"{day.strftime('%Y-%m-%d')}_*.parquet",  # Match hourly files
            )
        )

        # Check if there are files for the day
        if not list_parquet:
            print(f"No data found for {day.strftime('%Y-%m-%d')}. Skipping...")
            continue

        try:
            # Load and concatenate parquet files
            t = Traffic(
                pd.concat(
                    [pd.read_parquet(f) for f in list_parquet],
                    axis=0,
                )
            )

            # Assign IDs
            t_id = (
                t.iterate_lazy(iterate_kw={"by": "30min"})
                .assign_id("{self.start:%y/%m/%d}_{self.callsign}_{idx:>03}")
                .eval(max_workers=1, desc="Assigning")
            )

            # Other processing steps
            t_proc = (
                t_id.iterate_lazy()
                .pipe(remove_ground_only)
                .pipe(add_typecode)
                # NOTE: 'groundspee' (sic) is kept as in the run that produced the paper data.
                .filter(
                    FilterMedian(
                        altitude=3, geoaltitude=3, vertical_rate=3, groundspee=3, track=0
                    )
                )
                .filter(
                    FilterMean(
                        altitude=3, geoaltitude=3, vertical_rate=3, groundspee=3, track=0
                    )
                )
                .clip(lsasfra)
                .pipe(resample_flight)
                .pipe(remove_short)
                .eval(max_workers=1, desc="Processing")
            )

            # Save processed data
            t_proc.to_parquet(
                opj(out_dir, f"{day.strftime('%Y_%m_%d')}.parquet")
            )
            print(f"Finished processing {day.strftime('%Y-%m-%d')}")

        except Exception as e:
            print(f"Error processing {day.strftime('%Y-%m-%d')}: {e}")