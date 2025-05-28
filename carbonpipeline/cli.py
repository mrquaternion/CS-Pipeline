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
    Processes a CSV file containing climate or environmental data, identifies missing data points,
    and downloads the corresponding ERA5 datasets for those points using multiprocessing.

    Steps
    -----
    1. Reads the input CSV file and applies column filtering and renaming.
    2. Converts the 'timestamp' column to datetime objects.
    3. Identifies rows with missing data.
    4. Groups missing data by year, month, and day.
    5. Cleans and prepares directories for downloading and unzipping datasets.
    6. Uses multiprocessing to download and unzip ERA5 data for each group of missing data.
    7. Merges the downloaded datasets.
    8. Averages data over the four nearest grid points for each valid time.
    9. Restructures the final dataframe for further processing.

    Parameters
    ----------
    file : str 
        Path to the input CSV file.
    lat : float 
        Latitude coordinate for data extraction.
    lon : float 
        Longitude coordinate for data extraction.
    preds : list[str]
        List of predictor variable names to query and process.
    """

    cs_dataset = pd.read_csv(file)
    df = cs_dataset.copy()

    filtered_df = filtered_and_renamed_columns(df, COLUMN_NAME_MAPPING)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])

    miss = missing_rows(filtered_df.copy()) # Keep only rows with any missing

    # ---------------------- Testing  # of hours to pull ------------------------
    # nb_of_hours = 47                                      
    # temp = miss.loc[:nb_of_hours]                           
    # groups = list(temp.groupby(['year', 'month', 'day']))   
    # ---------------------------------------------------------------------------
    
    times = [f"{h:02d}:00" for h in range(24)]

    # Clean the folder before new query (CHANGE LATER ON TO CHECK INSTEAD IF THE DATA ALREADY HAS BEEN DOWNLOADED i.e. *with download logs*)
    ZIP_DIR   = "./datasets/zip"
    UNZIP_DIR = "./datasets/unzip"

    for d in (ZIP_DIR, UNZIP_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    groups = list(miss.groupby(['year', 'month', 'day']))

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
        
    df = merge_datasets(unzip_sub_fldrs)  

    """ The area the query is being done is limited by 4 equidistant grid points. This
    yield that each point has the same 'weight', thus letting us do a simple average
    of the 4 corners. """

    df = df.groupby(['valid_time']).mean()

    dataframe_restructuration(miss, preds)


def dataframe_restructuration(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    """
    Restructures a DataFrame by renaming specified predictor columns with a "CS, " prefix,
    adds corresponding "ERA5, " columns filled with NaN, and sets a MultiIndex on columns
    with levels ['variable', 'source'].

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing predictor columns to be restructured.
    preds : list[str]
        List of predictor column names to be renamed and paired with new ERA5 columns.

    Returns
    -------
    pd.DataFrame
        The restructured DataFrame with MultiIndex columns, where each predictor has
        both "CS" and "ERA5" sources, and columns are grouped by variable name.
    """

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
    """
    Extracts all files from a ZIP archive to a specified directory, displaying a progress bar.

    Parameters
    ----------
    zip_path : str 
        The path to the ZIP file to be extracted.
    unzip_path : str
        The directory where the contents will be extracted.

    Notes
    -----
    - Uses tqdm to display extraction progress. 
    - Ignores extraction errors for individual files.
    """

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            try:
                zip_ref.extractall(unzip_path)
            except zipfile.error as e:
                pass
            

def missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and return rows in the DataFrame that contain missing (NaN) values, excluding the 'timestamp' 
    column. Adds 'year', 'month', and 'day' columns extracted from the 'timestamp' column for each missing row.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'timestamp' column and other columns to check for missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rows with missing values (excluding 'timestamp'), with additional 'year', 
        'month', and 'day' columns derived from 'timestamp'.
    """

    miss = df[df[df.columns.drop('timestamp')].isnull().any(axis=1)]

    miss['year']  = miss['timestamp'].dt.year
    miss['month'] = miss['timestamp'].dt.month
    miss['day'] = miss['timestamp'].dt.day
    return miss


def parse_timestamp(ts: str) -> Tuple[datetime.date, str]:
    """
    Parses an ISO formatted timestamp string and returns the date and time components.

    Parameters
    ----------
    ts : str
        The timestamp string in ISO format (e.g., 'YYYY-MM-DDTHH:MM:SS').

    Returns
    -------
    Tuple[datetime.date, str]
        A tuple containing the date part as a `datetime.date` object and the time part as a 
        string in 'HH:MM' format.

    Raises
    ------
    ValueError
        If the input string is not in a valid ISO format.
    """

    dt = datetime.fromisoformat(ts)
    return dt.date(), dt.strftime("%H:%M")


def filtered_and_renamed_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    Filters the columns of a DataFrame based on a given mapping and renames them.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to filter and rename columns.
    column_mapping : dict 
        A dictionary mapping original column names to new names.

    Returns
    -------
    pd.DataFrame 
        A DataFrame containing only the filtered columns, renamed according to the mapping.
    """

    filtered_columns = [col for col in column_mapping.keys() if col in df.columns]
    return df[filtered_columns].rename(columns=column_mapping)


def rename_combined_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of the given DataFrame using the SHORTNAME_TO_FULLNAME mapping.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with columns to be renamed.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with columns renamed according to SHORTNAME_TO_FULLNAME.
    """

    return df.rename(columns=SHORTNAME_TO_FULLNAME)


def ameriflux_to_era5(df: pd.DataFrame, pred: str) -> np.array:
    """
    Converts AmeriFlux DataFrame columns to ERA5 predictor values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing AmeriFlux data.
    pred : str
        The predictor variable name for which ERA5 values are required.

    Returns
    -------
    np.array 
        Array of ERA5 predictor values computed from the DataFrame.

    Notes
    -----
    - The columns required for the predictor are determined by VARIABLES_FOR_PREDICTOR[pred].
    - If a processing function is defined in PROCESSORS for the predictor, it is applied row-wise.
    - If no processing function is found, the first relevant column is returned as a NumPy array.
    """

    cols = VARIABLES_FOR_PREDICTOR[pred]     
    func = PROCESSORS.get(pred)

    if func is None:
        return df[cols[0]].to_numpy()

    return df[cols].apply(lambda row: func(*row), axis=1).to_numpy()


def merge_datasets(sub_fldrs: list[str]) -> pd.DataFrame:
    """
    Merges NetCDF files from multiple subfolders into a single pandas DataFrame.

    This function searches through each provided subfolder for files with a `.nc` extension,
    opens each NetCDF file as an xarray Dataset, converts it to a pandas DataFrame, and
    concatenates all DataFrames from each subfolder along the columns. Then, it concatenates
    the resulting DataFrames from all subfolders along the rows. The final DataFrame is
    renamed for readability and specific columns ('number', 'expver') are dropped.

    Parameters
    ----------
    sub_fldrs : list[str]
        List of paths to subfolders containing NetCDF files.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame after processing all NetCDF files.
    """

    dfs = []
    for fldr in sub_fldrs:
        sub_dfs = []
        for name in os.listdir(fldr): # source: https://www.geeksforgeeks.org/python-loop-through-folders-and-files-in-directory/
            if name.endswith('.nc'):
                path_to_file = os.path.join(fldr, name)
                ds = xr.open_dataset(path_to_file, engine='netcdf4', decode_timedelta=True)
                df = ds.to_dataframe()
                sub_dfs.append(df)
        dfs.append(pd.concat(sub_dfs, axis=1))

    combined = pd.concat(dfs, axis=0)
    return rename_combined_dataframe(combined).drop(columns=['number', 'expver'])


def query_and_download(groupby_df: tuple, dir_: str, times: list[str], lat: float, lon: float, preds: list[str]):
    """
    Queries data for a specific date and location, then downloads the results.

    Parameters
    ----------
        groupby_df : tuple
            A tuple containing (year, month, day) to specify the date for the query.
        dir\_ : str
            The directory path where the downloaded data will be saved.
        times : list[str]
            List of time strings (e.g., ["00:00", "12:00"]) to include in the query.
        lat : float 
            Latitude coordinate for the data query.
        lon : float
            Longitude coordinate for the data query.
        preds : list[str]
            List of predictor variable names to request.
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
