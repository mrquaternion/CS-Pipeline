# carbonpipeline/cli.py
import argparse
from collections import defaultdict
from carbonpipeline.processing_utils import *
from carbonpipeline.api_request import APIRequest
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime

MIN_PATTERN_SIZE = 1000

def processing(file: str, lat: float, lon: float):
    """
    This function is intended to be used as the following.

    The attribute `file` in the argument of the CLI is CSV file. The latitude/longitude 
    attributs are meant to represent the coordinates of the EC tower the dataset is from.

    The process is to find all the missing values for a column X and to obtain the corresponding timestamp.
    From there, we have, for each column, each timestamp where there's a missing value. We then use a data
    structure called disjoint set with the union-find algorithm to group all the column who share a row with
    missing values. This is all done to optimize API calls. See the markdown file in the `research/` dir for more info.
    """
    column_mapping = {
        'CO2_F_MDS': 'CO2',
        'G_F_MDS': 'G',
        'H_F_MDS': 'H',
        'LE_F_MDS': 'LE',
        'LW_IN_F': 'LW_IN',
        'LW_OUT': 'LW_OUT',
        'NETRAD': 'NETRAD',
        'PA_F': 'PA',
        'PPFD_IN': 'PPFD_IN',
        'PPFD_OUT': 'PPFD_OUT',
        'P_F': 'P',
        'RH': 'RH',
        'SW_IN_F': 'SW_IN',
        'SW_OUT': 'SW_OUT',
        'TA_F': 'TA',
        'USTAR': 'USTAR',
        'VPD_F': 'VPD',
        'WD': 'WD',
        'WS_F': 'WS',
        'timestamp': 'timestamp'
    }

    cs_dataset = pd.read_csv(file)
    df = cs_dataset.copy()

    filtered_df = filtered_and_renamed_columns(df, column_mapping)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])

    vars_ = filtered_df.columns.drop('timestamp')

    # keep only rows with any missing
    miss = filtered_df[filtered_df[vars_].isnull().any(axis=1)].copy()

    miss['year']  = miss['timestamp'].dt.year
    miss['month'] = miss['timestamp'].dt.month
    miss['day']   = miss['timestamp'].dt.day

    times = [f"{h:02d}:00" for h in range(24)]
    days = [f"{d:02d}" for d in range(1, 32)]

    # group by (year, month)
    for (year, month), _ in miss.groupby(['year', 'month']):
        print(f"\nRequesting {year}-{month:02d}")

        request = APIRequest(
            year=[str(year)],
            month=[f"{month:02d}"],
            day=days,
            time=times,
            lat=lat,
            lon=lon
        )
        request.query()

    # dataset = xr.open_dataset(file, engine="cfgrib")


def parse_timestamp(ts: str):
    dt = datetime.fromisoformat(ts)
    return dt.date(), dt.strftime("%H:%M")


def filtered_and_renamed_columns(df: pd.DataFrame, column_mapping: dict):
    """
    This function serves as a renaming process to ease the reading of the dataset.
    """
    filtered_columns = [col for col in column_mapping.keys() if col in df.columns]

    return df[filtered_columns].rename(columns=column_mapping)


def main():
    parser = argparse.ArgumentParser(description="Pipeline to fill missing values in CarbonSense dataset.")
    parser.add_argument("--file", required=True, type=str, help="Path to the dataset file")
    parser.add_argument("--lat", required=True, type=int, help="Latitude coordinate")
    parser.add_argument("--lon", required=True, type=int, help="Longitude coordinate")
    args = parser.parse_args()

    print(f"File: {args.file}")
    print(f"Lat: {args.lat}, Lon: {args.lon}")

    processing(args.file, args.lat, args.lon)

if __name__ == "__main__":
    main()
