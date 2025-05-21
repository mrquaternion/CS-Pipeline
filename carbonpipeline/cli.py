# carbonpipeline/cli.py
import argparse
import os

from tqdm import tqdm
from carbonpipeline.processing_utils import *
from carbonpipeline.api_request import APIRequest
import xarray as xr
import pandas as pd
from datetime import datetime
import zipfile



def processing(file: str, lat: float, lon: float):
    """
    This function is intended to be used as the following.

    The attribute `file` in the argument of the CLI is CSV file. The latitude/longitude 
    attributs are meant to represent the coordinates of the EC tower the dataset is from.

    The process is to find all the rows with missing values. Some datasets have all there 
    rows with at least one missing value, others don't.
    """
    column_mapping = {
        'CO2_F_MDS': 'CO2',
        'G_F_MDS':   'G',
        'H_F_MDS':   'H',
        'LE_F_MDS':  'LE',
        'LW_IN_F':   'LW_IN',
        'LW_OUT':    'LW_OUT',
        'NETRAD':    'NETRAD',
        'PA_F':      'PA',
        'PPFD_IN':   'PPFD_IN',
        'PPFD_OUT':  'PPFD_OUT',
        'P_F':       'P',
        'RH':        'RH',
        'SW_IN_F':   'SW_IN',
        'SW_OUT':    'SW_OUT',
        'TA_F':      'TA',
        'USTAR':     'USTAR',
        'VPD_F':     'VPD',
        'WD':        'WD',
        'WS_F':      'WS',
        'timestamp': 'timestamp'
    }

    cs_dataset = pd.read_csv(file)
    df = cs_dataset.copy()

    filtered_df = filtered_and_renamed_columns(df, column_mapping)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])

    # keep only rows with any missing
    miss = filtered_df[filtered_df[filtered_df.columns.drop('timestamp')].isnull().any(axis=1)].copy()

    # create new columns for year & month
    miss['year']  = miss['timestamp'].dt.year
    miss['month'] = miss['timestamp'].dt.month

    days = [f"{d:02d}" for d in range(1, 32)]
    times = [f"{h:02d}:00" for h in range(24)]

    # group by year
    for (year, month), _ in miss.groupby(['year', 'month']):
        print(f"\n############ Requesting {year}-{month:02d} ############")
        request_id = query(year, month, days, times, lat, lon)

        print(f"############ Unzipping the file ############")
        out_zip_path = f"./datasets/zip/{request_id}.zip"
        out_unzip_path = f"./datasets/unzip/data_{year}-{month:02d}"
        os.makedirs(out_unzip_path, exist_ok=True)

        unzip_grib(out_zip_path, out_unzip_path)
        
    
    # dataset = xr.open_dataset(file, engine="cfgrib")

def query(year: str, month: int, days: list[int], times: list[int], lat: float, lon: float):
    request = APIRequest(
            year=[str(year)],
            month=[f"{month:02d}"],
            day=days,
            time=times,
            lat=lat,
            lon=lon
        )
    return request.get_id_and_download()


def unzip_grib(zip_path: str, unzip_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            try:
                zip_ref.extractall(unzip_path)
            except zipfile.error as e:
                pass


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
