# carbonpipeline/cli.py
import argparse
import os, shutil, time

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple

import cfgrib
from carbonpipeline.processing_utils import *
from carbonpipeline.api_request import APIRequest
import xarray as xr
import pandas as pd
from datetime import datetime
from carbonpipeline.constants import *


def processing(file: str, lat: float, lon: float, preds: list[str]):
    """
    This function is intended to be used as the following.

    The attribute `file` in the argument of the CLI is CSV file. The latitude/longitude 
    attributs are meant to represent the coordinates of the EC tower the dataset is from.
    """

    cs_dataset = pd.read_csv(file)
    df = cs_dataset.copy()

    filtered_df = filtered_and_renamed_columns(df, COLUMN_NAME_MAPPING)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])

    miss = missing_rows(filtered_df.copy()) # Keep only rows with any missing

    # -------------------------------------------------------------------------------------------------
    nb_of_hours = 47                                        # ONLY THERE FOR TEST (1 day worth of data)
    temp = miss.loc[:nb_of_hours]                           # ONLY THERE FOR TEST (1 day worth of data)
    groups = list(temp.groupby(['year', 'month', 'day']))   # ONLY THERE FOR TEST (1 day worth of data)
    # -------------------------------------------------------------------------------------------------
    
    times = [f"{h:02d}:00" for h in range(24)]

    # Clean the folder before new query (CHANGE LATER ON TO CHECK INSTEAD IF THE DATA ALREADY HAS BEEN DOWNLOADED i.e. *with download logs*)
    dir_ = "./datasets/"
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_, exist_ok=True)
    query_partial = partial(query, dir_=dir_, times=times, lat=lat, lon=lon, preds=preds) # Initialize the the arguments for the query
    # groups = list(miss.groupby(['year', 'month', 'day']))

    # Multiprocessing for faster download
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(query_partial, groups)
    end = time.time()
    delta = end - start
    print(f"Time taken to download {(nb_of_hours + 1)/24} day worth of data: {delta} seconds")
        
    df = merge_datasets()   
    


def missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps rows with any cell with missing value and creates new columns (year, month & day) for easier request access.
    """
    miss = df[df[df.columns.drop('timestamp')].isnull().any(axis=1)]

    miss['year']  = miss['timestamp'].dt.year
    miss['month'] = miss['timestamp'].dt.month
    miss['day'] = miss['timestamp'].dt.day
    return miss

def parse_timestamp(ts: str) -> Tuple[datetime.date, str]:
    dt = datetime.fromisoformat(ts)
    return dt.date(), dt.strftime("%H:%M")


def filtered_and_renamed_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    This function serves as a renaming process to ease the reading of the dataset.
    """
    filtered_columns = [col for col in column_mapping.keys() if col in df.columns]
    return df[filtered_columns].rename(columns=column_mapping)


def rename_combined_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improve readability by renaming columns under the `shortName` naming given in the original GRIB file. Renamed to their expanded names.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame before processing.
    
    Returns:
        pd.DataFrame. The DataFrame with renamed columns.
    """
    return df.rename(columns=SHORTNAME_TO_FULLNAME)


def merge_datasets() -> pd.DataFrame:
    """
    Merges GRIB files after the download. Proceed with transforming them into DataFrames and renaming the column
    for easier readibility.

    Returns:
        pd.DataFrame. The final DataFrame after merging the xr.DataSet objects.
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
    return rename_combined_dataframe(combined)


def query(groupby_df: tuple, dir_: str, times: list[str], lat: float, lon: float, preds: list[str]):
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
            lon=lon,
            preds=preds
        )
    
    print(f"############ Download started for {year}-{month:02d}-{day:02d} ############")
    request.fetch_download(dir_)
    print(f"############ Download finished for {year}-{month:02d}-{day:02d} ############")


def main():
    parser = argparse.ArgumentParser(description="Pipeline to fill missing values in CarbonSense dataset.")
    parser.add_argument("--file", required=True, type=str, help="Path to the dataset file")
    parser.add_argument("--lat", required=True, type=int, help="Latitude coordinate")
    parser.add_argument("--lon", required=True, type=int, help="Longitude coordinate")
    parser.add_argument("--preds", nargs='*', help="(Optional) List of predictors to use (e.g., --preds CO2 TA RH)")
    args = parser.parse_args()

    if args.preds is not None:
        invalid = []
        VALID_PREDICTORS = list(VARIABLES_FOR_PREDICTOR.keys())
        for pred in args.preds:
            if pred not in VALID_PREDICTORS:
                invalid.append(pred)
        if invalid:
            parser.error(f"\nInvalid predictor(s): {', '.join(invalid)} \nValid options are: {', '.join(VALID_PREDICTORS)}")

    my_set = set()
    if args.preds is None:
        vars_ = ERA5_VARIABLES
    else:
        for pred in args.preds:
            for var in VARIABLES_FOR_PREDICTOR[pred]:
                my_set.add(var) # Avoid duplicates
        vars_ = list(my_set)

    print(f"File: {args.file}")
    print(f"Latitude: {args.lat}, Longitude: {args.lon}")
    print(f"Predictors: {vars_}")

    processing(args.file, args.lat, args.lon, vars_)

if __name__ == "__main__":
    main()
