# carbonpipeline/cli.py
import argparse
import os, shutil

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import zipfile

from tqdm import tqdm
from carbonpipeline.processing_utils import *
from carbonpipeline.api_request import APIRequest
import xarray as xr
import pandas as pd
from carbonpipeline.constants import *


def process_missing_data(file_path: str, lat: float, lon: float, start: str, end: str, preds: list[str], vars: list[str]):
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
    start: str
        Desired start timestamp of the dataset.
    end : str
        Desired end timestamp of the dataset.
    preds : list[str]
        List of predictor variable names to query and process.
    """
    df = load_and_filter_dataframe(file_path, COLUMN_NAME_MAPPING, start, end)
    
    """ Clean the folder before new query (change later on to check instead if the data has already 
    been downloaded i.e. with download logs) """
    ZIP_DIR   = './datasets/zip'
    UNZIP_DIR = './datasets/unzip'
    setup_directories(ZIP_DIR, UNZIP_DIR)
    
    unzip_sub_fldrs = [] 
    groups = list(df.groupby(['year', 'month', 'day']))
    with ProcessPoolExecutor(max_workers=4) as executor:    
        for filename in executor.map(partial(
            prepare_download_request, 
            dir_=ZIP_DIR, 
            times=generate_hourly_times(), 
            lat=lat, 
            lon=lon, 
            vars=vars
        ), groups):
            zip_path = f"./datasets/zip/{filename}"
            unzip_path = f"./datasets/unzip/{filename.split('.')[0]}"
            unzip_sub_fldrs.append(unzip_path)
            extract_zip(zip_path, unzip_path)

    """ The area the query is being done is limited by 4 equidistant grid points. This
    yield that each point has the same 'weight', thus letting us do a simple average
    of the 4 corners. """
    dfpp = postprocess_era5_data(df.drop(columns=['year', 'month', 'day']), preds, unzip_sub_fldrs)
    print(dfpp)


def load_and_filter_dataframe(path: str, column_mapping: dict, start: str, end: str):
    df = pd.read_csv(path)
    df = df[df['timestamp'].between(start, end)]

    filtered_df = filter_and_rename_columns(df, column_mapping)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])

    return find_missing_rows(filtered_df.copy())


def filter_and_rename_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
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


def find_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
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


def setup_directories(zip_dir: str, unzip_dir: str):
    for d in (zip_dir, unzip_dir): 
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)


def prepare_download_request(groupby_df: tuple, dir_: str, times: list[str], lat: float, lon: float, vars: list[str]):
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
    vars : list[str]
        List of variable names to request.
    """
    year, month, day = groupby_df[0]
    request = APIRequest(
            year=str(year),
            month=f"{month:02d}",
            day=f"{day:02d}",
            time=times,
            lat=lat,
            lon=lon,
            vars=vars
        )
    
    return request.fetch_download(dir_)


def generate_hourly_times():
    return [f"{h:02d}:00" for h in range(24)]


def extract_zip(zip_path: str, unzip_path: str):
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
    print(f"Unzipping the file")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
            try:
                zip_ref.extractall(unzip_path)
            except zipfile.error as e:
                pass


def postprocess_era5_data(df: pd.DataFrame, preds: list[str], unzip_dirs: list):
    """
    Post-processes ERA5 data by merging datasets, grouping by valid time, and updating a multi-index DataFrame.

    Parameters
    ----------
    df : pd.DataFrame 
        The input DataFrame containing predictions and their origins.
    preds : list[str] 
        List of prediction variable names to process.
    unzip_dirs : list 
        List of directories containing unzipped ERA5 datasets.

    Notes
    -----
    - Merges datasets from the provided directories.
    - Groups merged data by 'valid_time' and computes the mean.
    - Builds a multi-index DataFrame for the predictions.
    - For each prediction with origin containing 'ERA', updates the corresponding values using ERA5 data.
    """
    dfm = merge_datasets(unzip_dirs)  
    dfg = dfm.groupby(['valid_time']).mean()
    print(f"DF: {df}")
    print(f"Preds: {preds}")
    dfr = build_multiindex_dataframe(df, preds)
    print(f"DFR: {dfr}")
    for pred, origin in dfr.columns:
        if 'ERA' in origin:
            dfr.loc[:, (pred, 'ERA5')] = convert_ameriflux_to_era5(dfg, pred)

    return dfr

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
        for name in os.listdir(fldr):                                                       # source: https://www.geeksforgeeks.org/python-loop-through-folders-and-files-in-directory/
            if name.endswith('.nc'):
                path_to_file = os.path.join(fldr, name)
                ds = xr.open_dataset(path_to_file, engine='netcdf4', decode_timedelta=True)
                df = ds.to_dataframe()
                sub_dfs.append(df)
        dfs.append(pd.concat(sub_dfs, axis=1))

    combined = pd.concat(dfs, axis=0)

    return apply_column_rename(combined).drop(columns=['number', 'expver'])


def apply_column_rename(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=SHORTNAME_TO_FULLNAME)


def build_multiindex_dataframe(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
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
    cs_cols   = [f"CS, {p}"   for p in preds]
    era5_cols = [f"ERA5, {p}" for p in preds]
    renamed = df.rename(columns=dict(zip(preds, cs_cols))).assign(**{col: np.nan for col in era5_cols})

    tuples = []
    for col in renamed.columns:
        if ", " in col:
            src, var = col.split(", ", 1)
        else:
            src, var = "CS", col
        tuples.append((var, src))

    renamed.columns = pd.MultiIndex.from_tuples(tuples, names=["variable", "source"])

    return renamed.sort_index(axis=1, level="variable")


def convert_ameriflux_to_era5(df: pd.DataFrame, pred: str) -> np.ndarray:
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


def main():
    parser = argparse.ArgumentParser(description="Pipeline to fill missing values in CarbonSense dataset.")
    parser.add_argument("--file", required=True, type=str, help="Path to the dataset file")
    parser.add_argument("--lat", required=True, type=int, help="Latitude coordinate")
    parser.add_argument("--lon", required=True, type=int, help="Longitude coordinate")
    parser.add_argument("--start", required=True, type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, type=str, help="End date in YYYY-MM-DD format")
    parser.add_argument("--preds", nargs='*', help="(Optional) List of supported predictors to use (e.g., --preds CO2 TA RH)")
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
        args.preds = list(VARIABLES_FOR_PREDICTOR.keys())
    else:
        for pred in args.preds:
            for var in VARIABLES_FOR_PREDICTOR[pred]:
                my_set.add(var)                         # Avoid duplicates
        vars_ = list(my_set)

    print(f"File: {args.file}")
    print(f"Latitude: {args.lat}, Longitude: {args.lon}")
    print(f"Start date: {args.start}, End date: {args.end}")
    print(f"Predictors: {args.preds}")

    process_missing_data(args.file, args.lat, args.lon, args.start, args.end, args.preds, vars_)

if __name__ == "__main__":
    main()
