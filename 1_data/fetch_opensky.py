"""Step 1: fetch ADS-B data for Swiss airspace from the OpenSky Network (Trino) as hourly Parquet files.

Requires an OpenSky account with Trino access (pyopensky configuration, see
https://mode-s.org/pyopensky/); credentials are read from the pyopensky config file and
never belong in this repository.

Usage:
    python 1_data/fetch_opensky.py --dates 2024-02-15 2024-03-01 [--out_dir DIR]

Output: <out_dir>/<YYYY-MM-DD>_<HH>-<HH>.parquet, one file per hour (existing files are skipped).
Default out_dir: $ADSB_BASE/data_fetching2024 (ADSB_BASE defaults to the folder of this script).
"""
import os as _os, sys as _sys  # REPO_ROOT_SHIM
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import rs1090
import traceback
from pyopensky.trino import Trino
from datetime import datetime, timedelta
import pandas as pd
from pyopensky.schema import PositionData4, VelocityData4, RollcallRepliesData4
from traffic.core import Traffic
from traffic.data import opensky

# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", required=True, help="days to fetch, format YYYY-MM-DD")
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("ADSB_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                      "data_fetching2024"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dates = args.dates
    #start = datetime.strptime("2024-01-15 00:00", "%Y-%m-%d %H:%M")
    #end = datetime.strptime("2024-01-16 00:00", "%Y-%m-%d %H:%M")
    for date in dates:
        day_start = datetime.strptime(date, "%Y-%m-%d")
        day_end = day_start + timedelta(days=1)

        windows = []
        current = day_start
        one_hour = timedelta(hours=1)

        while current < day_end:
            windows.append((current, current + one_hour))
            current += one_hour

    #one_hour = timedelta(hours=1)
    #windows = []
    #current = start
    #while current < end:
    #    windows.append((current, current + one_hour))
    #    current += one_hour

        # %% Iterate through hourly windows
        for hour in windows:
            # Extract and print start and stop time
            starttime = hour[0].strftime("%Y-%m-%d %H:%M")
            stoptime = hour[1].strftime("%Y-%m-%d %H:%M")
            print(f"start:{starttime}, stop:{stoptime}")

            # Only process if file for this hour does not exist yet
            file_path = os.path.join(args.out_dir, f"{starttime[0:10] + '_' + starttime[11:13] + '-' + stoptime[-5:-3]}.parquet")
            if not os.path.exists(file_path):
                try:
                    # Get icaos of flights that crossed rectangle---------------------------
                    icao24s = opensky.rawdata(
                        starttime,
                        stoptime,
                        PositionData4.lat > 44.05,
                        PositionData4.lat < 48.91999390290994,
                        PositionData4.lon > 4.383367679058126,
                        PositionData4.lon < 11.238634434863998,
                        Table=PositionData4,
                    ).data.icao24.unique()

                    # Get position data ----------------------------------------------------
                    # Fetch data from PositionData4 table
                    pos_data = opensky.rawdata(
                        starttime,
                        stoptime,
                        Table=PositionData4,
                        icao24=icao24s,
                        extra_columns=(
                            PositionData4.lat,
                            PositionData4.lon,
                            PositionData4.alt,
                            PositionData4.groundspeed,
                            PositionData4.heading,
                            PositionData4.surface,
                            PositionData4.nic,
                        ),
                    ).data

                    # Convert timestamp
                    pos_data["timestamp"] = pd.to_datetime(
                        pos_data["mintime"], unit="s", errors="coerce"
                    ).dt.tz_localize("UTC")

                    # Rename columns
                    pos_data = pos_data.rename(
                        columns={
                            "lat": "latitude",
                            "lon": "longitude",
                            "surface": "onground",
                            "alt": "altitude",
                            "heading": "track",
                        }
                    )

                    # # Where altitude is 0, replace with nan
                    pos_data["altitude"] = pos_data["altitude"].replace(0, pd.NA)

                    # Convert units (altitude to ft, groundspeed to kts)
                    pos_data["altitude"] = pos_data["altitude"] * 3.28084

                    # Remove duplicates in raw message
                    pos_data = pos_data.drop_duplicates(subset=["rawmsg"])

                    # Remove columns that are not decoded
                    pos_data = pos_data.dropna(
                        subset=["latitude", "longitude", "groundspeed", "track", "altitude", "nic"],
                        how="all",
                    )[
                        [
                            "timestamp",
                            "icao24",
                            "latitude",
                            "longitude",
                            "groundspeed",
                            "track",
                            "onground",
                            "altitude",
                            "nic",
                        ]
                    ]

                    # Reduce to geographic extent
                    pos_data = pos_data.query(
                        "latitude > 44.05 and latitude < 48.91999390290994 and longitude > 4.383367679058126 and longitude < 11.238634434863998"
                    )

                    # Get speed data------------------------------------------------------------
                    # Fetch data from VelocityData4 table
                    speed_data = opensky.rawdata(
                        starttime,
                        stoptime,
                        Table=VelocityData4,
                        icao24=icao24s,
                        extra_columns=(
                            VelocityData4.velocity,
                            VelocityData4.heading,
                            VelocityData4.vertrate,
                        ),
                    ).data

                    # Convert timestamp
                    speed_data["timestamp"] = pd.to_datetime(
                        speed_data["mintime"], unit="s", errors="coerce"
                    ).dt.tz_localize("UTC")

                    # Rename columns
                    speed_data = speed_data.rename(
                        columns={
                            "velocity": "groundspeed",
                            "heading": "track",
                            "vertrate": "vertical_rate",
                        }
                    )

                    # Drop duplicates
                    speed_data = speed_data.drop_duplicates(subset=["rawmsg"])

                    # Convert units (groundspeed to kts, vertical rate to ft/min)
                    speed_data["groundspeed"] = speed_data["groundspeed"] * 1.94384
                    speed_data["vertical_rate"] = speed_data["vertical_rate"] * 196.850394

                    # Extract geo-baro difference
                    diff_binary = (
                        speed_data["rawmsg"]
                        .str[-20:-6]
                        .apply(lambda x: bin(int(x, 16))[2:].zfill(len(x) * 4))
                    )
                    speed_data["s_diff_alt"] = diff_binary.str[-8]
                    speed_data["geominusbaro"] = diff_binary.str[-7:]
                    speed_data["geominusbaro"] = (
                        speed_data["geominusbaro"].apply(lambda x: int(x, 2)) - 1
                    ) * 25
                    speed_data.loc[speed_data["s_diff_alt"] == "1", "geominusbaro"] = -speed_data.loc[
                        speed_data["s_diff_alt"] == "1", "geominusbaro"
                    ]

                    # Remove columns that are not decoded
                    speed_data = speed_data.dropna(
                        subset=["groundspeed", "track", "vertical_rate", "geominusbaro"], how="all"
                    )[["timestamp", "icao24", "groundspeed", "track", "vertical_rate", "geominusbaro"]]

                    # Merge speed data----------------------------------------------------------
                    pos_data = pd.merge_asof(
                        pos_data.sort_values("timestamp"),
                        speed_data.sort_values("timestamp"),
                        on="timestamp",
                        by="icao24",
                        direction="nearest",
                        tolerance=pd.Timedelta(seconds=5),
                    )
                    pos_data["geoaltitude"] = pos_data["altitude"] + pos_data["geominusbaro"]
                    pos_data["groundspeed"] = pos_data["groundspeed_x"].combine_first(
                        pos_data["groundspeed_y"]
                    )
                    pos_data["track"] = pos_data["track_x"].combine_first(pos_data["track_y"])
                    pos_data = pos_data.drop(
                        columns=["groundspeed_x", "groundspeed_y", "track_x", "track_y"]
                    )

                    # Fetch rollcall (squawk) data----------------------------------------------
                    rollcall_data = opensky.rawdata(
                        starttime,
                        stoptime,
                        Table=RollcallRepliesData4,
                        icao24=icao24s,
                        extra_columns=(RollcallRepliesData4.identity,),
                    ).data

                    rollcall_data = rollcall_data.rename(
                        columns={
                            "identity": "squawk",
                        }
                    )

                    rollcall_data["timestamp"] = pd.to_datetime(
                        rollcall_data["mintime"], unit="s", errors="coerce"
                    ).dt.tz_localize("UTC")
                    rollcall_data = rollcall_data.dropna(subset=["squawk"])[
                        ["timestamp", "icao24", "squawk"]
                    ]

                    # Merge rollcall (squawk) data----------------------------------------------
                    pos_data = pd.merge_asof(
                        pos_data.sort_values("timestamp"),
                        rollcall_data.sort_values("timestamp"),
                        on="timestamp",
                        by="icao24",
                        direction="nearest",
                        tolerance=pd.Timedelta(minutes=10),
                    )

                    # Fetch identification (callsign) data--------------------------------------
                    # Fetch raw messages
                    trino = Trino()
                    identification_data = trino.rawdata(starttime, stoptime, icao24=icao24s)

                    # Decode raw messages
                    identification_data = rs1090.decode(identification_data["rawmsg"].tolist(), identification_data["mintime"].tolist())

                    # Reduce to BDS20 messages only and turn into dataframe
                    identification_data = [entry for entry in identification_data if 'bds20' in entry]
                    identification_data = [
                        {
                            'timestamp': item['timestamp'],
                            'icao24': item['icao24'],
                            'callsign': item['bds20']['callsign']
                        }
                        for item in identification_data
                    ]
                    identification_data = pd.DataFrame(identification_data)

                    # Clean up decoding errors
                    identification_data['callsign'] = identification_data['callsign'].str.strip().str.upper()
                    identification_data = identification_data[~identification_data['callsign'].str.contains('#', na=False)]
                    counts = identification_data['callsign'].value_counts()
                    identification_data = identification_data[identification_data['callsign'].isin(counts[counts >= 10].index)]

                    # Set column types
                    identification_data["timestamp"] = pd.to_datetime(
                        identification_data["timestamp"], unit="s", errors="coerce"
                    ).dt.tz_localize("UTC")

                    pos_data["icao24"] = pos_data["icao24"].astype(str).str.lower()
                    identification_data["icao24"] = identification_data["icao24"].astype(str).str.lower()

                    # Merge identification (callsign) data--------------------------------------
                    pos_data = pd.merge_asof(
                        pos_data.sort_values("timestamp"),
                        identification_data.sort_values("timestamp"),
                        on="timestamp",
                        by="icao24",
                        direction="nearest",
                        tolerance=pd.Timedelta(minutes=10),
                    )

                    # Crop to rectangle---------------------------------------------------------
                    # Function to filter each group
                    def filter_group(group):
                        # Find rows where latitude is not null
                        valid_rows = group[group["latitude"].notna()]

                        if valid_rows.empty:
                            return pd.DataFrame()  # Return empty if no valid latitude

                        # Get the time range
                        start_time = valid_rows["timestamp"].iloc[0]
                        end_time = valid_rows["timestamp"].iloc[-1]

                        # Filter rows in that time range
                        return group[
                            (group["timestamp"] >= start_time) & (group["timestamp"] <= end_time)
                        ]

                    # Apply per group
                    pos_data = (
                        pos_data.groupby("icao24", group_keys=False)
                        .apply(filter_group)
                        .sort_values(by=["timestamp"])
                    )

                    # Turn into traffic---------------------------------------------------------
                    t = Traffic(pos_data)

                    # Save
                    t.to_parquet(file_path)
                except Exception as e:
                    print(f"{file_path} failed — {type(e).__name__}: {e}")
                    traceback.print_exc()