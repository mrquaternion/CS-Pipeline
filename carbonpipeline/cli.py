# carbonpipeline/cli.py
import argparse
import os, shutil, time

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Tuple
import zipfile

from tqdm import tqdm
from carbonpipeline.processing_utils import *
from carbonpipeline.api_request import APIRequest
import xarray as xr
import pandas as pd
from datetime import datetime
from carbonpipeline.constants import *


def processing(file: str, lat: float, lon: float, preds: list[str]):
    """
    This function is intended to be used as the following.

    The attribute `file` in the argument of the CLI is CSV file (tested with CarbonSense V1 DDCFM 
    `predictors.csv` files). The latitude/longitude attributs are meant to represent the coordinates 
    of the EC tower the dataset is from.
    """

    cs_dataset = pd.read_csv(file)
    df = cs_dataset.copy()

    filtered_df = filtered_and_renamed_columns(df, COLUMN_NAME_MAPPING)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])

    miss = missing_rows(filtered_df.copy()) # Keep only rows with any missing

    # ---------------------- Testing  # of hours to pull ------------------------
    nb_of_hours = 47                                      
    temp = miss.loc[:nb_of_hours]                           
    groups = list(temp.groupby(['year', 'month', 'day']))   
    # ---------------------------------------------------------------------------
    
    times = [f"{h:02d}:00" for h in range(24)]

    # Clean the folder before new query (CHANGE LATER ON TO CHECK INSTEAD IF THE DATA ALREADY HAS BEEN DOWNLOADED i.e. *with download logs*)
    ZIP_DIR   = "./datasets/zip"
    UNZIP_DIR = "./datasets/unzip"

    for d in (ZIP_DIR, UNZIP_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    # groups = list(miss.groupby(['year', 'month', 'day']))

    # Multiprocessing for faster download
    start = time.time()

    unzip_sub_fldrs = [] 
    with ProcessPoolExecutor(max_workers=4) as executor:
        for filename in executor.map(
            partial( # Initialize the the arguments for the query
                query_and_download, 
                dir_=ZIP_DIR, 
                times=times, 
                lat=lat, 
                lon=lon, 
                preds=preds
            ), 
            groups
        ):
            print(f"Unzipping the file ")
            zip_path = f"./datasets/zip/{filename}"
            unzip_path = f"./datasets/unzip/{filename.split('.')[0]}"
            unzip_sub_fldrs.append(unzip_path)
            unzip_grib(zip_path, unzip_path)
        
    end = time.time()
    delta = end - start
    print(f"Time taken to download {(nb_of_hours + 1)/24} day worth of data: {delta} seconds")
        
    df = merge_datasets(unzip_sub_fldrs)  
    print(f"The final dataframe:\n{df}") 

    """ The area the query is being done is limited by 4 equidistant grid points. This
    yield that each point has the same 'weight', thus letting us do a simple average
    of the 4 corners. """

    df = df.groupby(['valid_time']).mean()
    print(f"Average of 4 corners:\n{df}")

    dataframe_restructuration(miss, preds)


def dataframe_restructuration(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    renamed_preds = map(lambda var: "CS, " + var, preds) # Adding prefix to predictor's name 
    renamed_df = df.rename(columns=dict(zip(preds, renamed_preds)))
    renamed_df[[("ERA5, " + var) for var in preds]] = np.nan

    raw_cols = renamed_df.columns.tolist()

    levels = []
    for col in raw_cols:
        parts = col.split(', ')        
        if len(parts) == 2:
            src, var = parts           # Inputs vars => has both CS and ERA5 levels
        else:
            src, var = 'CS', parts[0]  # Vars not needed to be requested => doesn't have ERA5 level
        levels.append((var, src))

    renamed_df.columns = pd.MultiIndex.from_tuples(levels, names=['variable', 'source']) # `variable` for outer level, `source` for inner level

    return renamed_df.sort_index(axis=1, level='variable') # Group by variable

    
    
def unzip_grib(zip_path: str, unzip_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            try:
                zip_ref.extractall(unzip_path)
            except zipfile.error as e:
                pass
            

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


def merge_datasets(sub_fldrs: list[str]) -> pd.DataFrame:
    """
    Merges NetCDF files after the download. Proceed with transforming them into DataFrames and renaming the column
    for easier readibility.

    Returns:
        pd.DataFrame. The final DataFrame after merging the xr.DataSet objects.
    """
    dfs = []
    for fldr in sub_fldrs:
        sub_dfs = []
        for name in os.listdir(fldr): # source: https://www.geeksforgeeks.org/python-loop-through-folders-and-files-in-directory/
            path_to_file = os.path.join(fldr, name)
            ds = xr.open_dataset(path_to_file, engine='netcdf4', decode_timedelta=True)
            df = ds.to_dataframe()
            sub_dfs.append(df)
        dfs.append(pd.concat(sub_dfs, axis=1))

    combined = pd.concat(dfs, axis=0)
    return rename_combined_dataframe(combined).drop(columns=['number', 'expver'])


def query_and_download(groupby_df: tuple, dir_: str, times: list[str], lat: float, lon: float, preds: list[str]):
    """
    Prepare the API request with args formatting and fetches the download.
    """
    year, month, day = groupby_df[0]
    request = APIRequest(
            year=str(year),
            month=f"{month:02d}",
            day=f"{day:02d}",
            time=times,
            lat=lat,
            lon=lon,
            preds=preds
        )
    return request.fetch_download(dir_)



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
