# carbonpipeline/cli.py
import argparse
import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
from typing import Tuple

import cfgrib
from carbonpipeline.processing_utils import *
from carbonpipeline.api_request import APIRequest
import xarray as xr
import pandas as pd
from datetime import datetime



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

    # create new columns for year, month & day
    miss['year']  = miss['timestamp'].dt.year
    miss['month'] = miss['timestamp'].dt.month
    miss['day'] = miss['timestamp'].dt.day

    times = [f"{h:02d}:00" for h in range(24)]

    # multiprocessing for faster download
    nb_of_hours = 47
    temp = miss.loc[:nb_of_hours] # ONLY THERE FOR TEST (1 day worth of data)
    groups = list(temp.groupby(['year', 'month', 'day']))
    query_partial = partial(query, times=times, lat=lat, lon=lon)

    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(query_partial, groups)
    end = time.time()

    delta = end - start
    print(f"Time taken to download {(nb_of_hours + 1)/24} day worth of data: {delta} seconds")
        
    df = merge_datasets()


def rename_combined(df: pd.DataFrame) -> pd.DataFrame:
    """
    To improve readability, the `shortName` columns in the original GRIB files are being renamed to 
    their expanded names.
    """
    short_to_full = {
        '10u': '10m_u_component_of_wind',
        '10v': '10m_v_component_of_wind',
        '2t': '2m_temperature',
        '2d': '2m_dewpoint_temperature',
        'sp': 'surface_pressure',
        'tp': 'total_precipitation',
        'avg_sdlwrf': 'mean_surface_downward_long_wave_radiation_flux',
        'avg_sdlwrfcs': 'mean_surface_downward_long_wave_radiation_flux_clear_sky',
        'avg_sdswrf': 'mean_surface_downward_short_wave_radiation_flux',
        'avg_sdswrfcs': 'mean_surface_downward_short_wave_radiation_flux_clear_sky',
        'ishf': 'instantaneous_surface_sensible_heat_flux',
        'slhf': 'surface_latent_heat_flux',
        'sshf': 'surface_sensible_heat_flux',
        'stl1': 'soil_temperature_level_1',
        'stl2': 'soil_temperature_level_2',
        'stl3': 'soil_temperature_level_3',
        'swvl1': 'volumetric_soil_water_layer_1',
        'swvl2': 'volumetric_soil_water_layer_2',
        'swvl3': 'volumetric_soil_water_layer_3',
        'fal': 'forecast_albedo',
        'zust': 'friction_velocity'
    }
    return df.rename(columns=short_to_full)


def merge_datasets() -> pd.DataFrame:
    """
    Instead of querying for the whole time interval of the `predictors.csv` file, we queried for each day.

    Now, merging every GRIB file together by `time`.
    """
    dataframes = []
    directory = "./datasets/"
    for name in os.listdir(directory): # source: https://www.geeksforgeeks.org/python-loop-through-folders-and-files-in-directory/
        if name.endswith(".grib"):
            path_to_file = os.path.join(directory, name)
            ds_list = cfgrib.open_datasets(path_to_file, decode_timedelta=True)
            ds = xr.merge(ds_list, compat="override")
            df = ds.to_dataframe()
            dataframes.append(df)

    combined = pd.concat(dataframes, axis=1)
    return rename_combined(combined)

    

def query(groupby_df: tuple, times: list[int], lat: float, lon: float):
    """
    Prepare the API request with args formatting and fetches the download.
    """
    (year, month, day), _ = groupby_df
    request = APIRequest(
            year=str(year),
            month=f"{month:02d}",
            day=f"{day:02d}",
            time=times,
            lat=lat,
            lon=lon
        )
    
    print(f"############ Download started for {year}-{month:02d}-{day:02d} ############")
    request.fetch_download()
    print(f"############ Download finished for {year}-{month:02d}-{day:02d} ############")


def parse_timestamp(ts: str) -> Tuple[datetime.date, str]:
    dt = datetime.fromisoformat(ts)
    return dt.date(), dt.strftime("%H:%M")


def filtered_and_renamed_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
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
