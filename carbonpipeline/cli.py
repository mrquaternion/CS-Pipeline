# carbonpipeline/cli.py
import argparse
import json
import re
import sys
import os, shutil, time, glob
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import zipfile

from tqdm import tqdm
from carbonpipeline.processing_utils import *
from carbonpipeline.api_request import APIRequest
import xarray as xr
import pandas as pd
from carbonpipeline.constants import *


MAX_WORKERS = 4
ZIP_DIR = "./datasets/zip"
UNZIP_DIR = "./datasets/unzip"
DATETIME_FORMAT = r"\d{4}-\d{2}-\d{2} \d{2}:00:00"
OUTPUT_MANIFEST = "./manifest.json"


def run_point_pipeline(file_path: str, coords: list[float], start: str, end: str, preds: list[str], vars_: list[str]):
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
    coords : list[float] 
        Latitude and longitude coordinates.
    start: str
        Desired start date of the dataset.
    end : str
        Desired end date of the dataset.
    preds : list[str]
        List of the AmeriFlux predictor names to process.
    vars\_ : list[str]
        List of the corresponding ERA5 variables to query.
    """
    df = load_and_filter_dataframe(file_path, start, end)
    
    groups = [group for group, _ in df.groupby(['year', 'month', 'day', 'time'])]
    unzip_sub_fldrs = multiprocessing_download(groups, vars_, coords)

    dfpp = postprocess_era5_data(df.drop(columns=['year', 'month', 'day', 'time']), preds, unzip_sub_fldrs)
    print(f"\n{dfpp}")

    return dfpp


def run_area_download(coords: list[float], start: str, end: str, preds: list[str], vars_: list[str]):
    """
    Downloads ERA5 datasets based on a geographical bounding box and creates a CSV file that contains
    the desired predictors computed from the requested variables.

    Steps
    -----
    1. Cleans and prepares directories for downloading and unzipping datasets.
    2. Uses multiprocessing to download and unzip ERA5 data for each group of missing data.
    3. Merges the downloaded datasets.
    4. Restructures the final dataframe for further processing.

    Parameters
    ----------
    coords : list[float]
        North, West, South and East coordinates.
    start: str
        Desired start date of the dataset.
    end : str
        Desired end date of the dataset.
    preds : list[str]
        List of the AmeriFlux predictor names to process.
    vars\_ : list[str]
        List of the corresponding ERA5 variables to query.
    """
    
    hourly_range = pd.date_range(start=start, end=end, freq='h')
    splitted_ts = list(map(lambda x: x + ':00', np.datetime_as_string(hourly_range, unit='h')))
    groups = [re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}:00)", ts).groups() for ts in splitted_ts]

    start_d = time.time()
    unzip_sub_fldrs = multiprocessing_download(groups, vars_, coords)
    end_d = time.time()
    print(f"\nTime taken to download: {end_d - start_d}")

    os.makedirs(os.path.dirname(OUTPUT_MANIFEST), exist_ok=True)
    with open(OUTPUT_MANIFEST, 'w') as fp:
        json.dump(
            {
                "preds": preds,
                "unzip_sub_folders": unzip_sub_fldrs
            },
            fp,
            indent=2
        )


def run_area_process() -> pd.DataFrame:
    with open(OUTPUT_MANIFEST, "r") as fp:
        manifest = json.load(fp)

    preds = manifest["preds"]
    unzip_sub_fldrs = manifest["unzip_sub_folders"]

    start_m = time.time()
    dfm = merge_datasets(unzip_sub_fldrs) 
    end_m = time.time()
    print(f"\n{dfm}")
    print(f"Time taken to merge the NetCDF files: {end_m - start_m}")

    start_c = time.time()
    d = { pred: convert_ameriflux_to_era5(dfm, pred) for pred in preds }
    out_df = pd.DataFrame(d, index=dfm.index)
    end_c = time.time()
    print(f"\n{out_df}")
    print(f"Time taken to create the new dataframe: {end_c - start_c}")

    return out_df


def multiprocessing_download(groups: list[tuple], vars_: list[str], coords: list[float]) -> list[str]:
    # Clean the folder before new query (change later on to check instead if the data has already 
    # been downloaded i.e. with download logs)
    setup_directories(ZIP_DIR, UNZIP_DIR)
    
    fldrs = [] 
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:    
        for filename in executor.map(partial(
            prepare_download_request, 
            dir_=ZIP_DIR, 
            coords=coords, 
            vars=vars_
        ), groups):
            zip_path = f"./datasets/zip/{filename}"
            unzip_path = f"./datasets/unzip/{filename.split('.')[0]}"
            fldrs.append(unzip_path)
            extract_zip(zip_path, unzip_path)

    return fldrs


def load_and_filter_dataframe(path: str, start: str, end: str):
    df = pd.read_csv(path)

    # Apply date format
    df['timestamp'] = df['timestamp'].apply(lambda row: validate_date_format(row))                     
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Remove all the dates that are not HH:00:00
    df = df[
        (df['timestamp'].dt.minute == 0) &
        (df['timestamp'].dt.second == 0)
    ]

    # Filter the dates
    df = df[df['timestamp'].between(pd.to_datetime(start), pd.to_datetime(end))]

    return find_missing_rows(df)


def validate_date_format(timestamp: str | int) -> str:
    # Check if already a string
    if isinstance(timestamp, str):                                      
        res = True
        try:
            res = bool(datetime.strptime(timestamp, DATETIME_FORMAT))
        except ValueError:
            res = False
        if res:
            return timestamp

    regex = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})'

    # Pattern match to extract year, month, day, hour, minute values
    timestamp_split = re.split(regex, str(timestamp))                   
    timestamp = ' '.join((
        '-'.join(timestamp_split[1:4]), 
        ':'.join(timestamp_split[4:6] + ['00'])
    ))

    return timestamp


def find_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and return rows in the DataFrame that contain missing (NaN) values, excluding the 'timestamp' 
    column. Adds 'year', 'month', 'day' and time columns extracted from the 'timestamp' column for each missing row.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'timestamp' column and other columns to check for missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rows with missing values (excluding 'timestamp').
    """
    miss = df[df[df.columns.drop('timestamp')].isnull().any(axis=1)]

    miss['year']  = miss['timestamp'].dt.year
    miss['month'] = miss['timestamp'].dt.month
    miss['day'] = miss['timestamp'].dt.day
    miss['time'] = miss['timestamp'].dt.time.astype(str)

    return miss


def setup_directories(zip_dir: str, unzip_dir: str):
    for d in (zip_dir, unzip_dir): 
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)


def prepare_download_request(group: tuple, dir_: str, coords: list[float], vars: list[str]):
    """
    Queries data for a specific date and location, then downloads the results.

    Parameters
    ----------
    groupby_df : tuple
        A tuple containing (year, month, day) to specify the date for the query.
    dir\_ : str
        The directory path where the downloaded data will be saved.
    coords : list[float]
        Coordinates for the data query.
    vars : list[str]
        List of variable names to request.
    """
    year, month, day, time = group
    time = re.search(r'\d{2}:00', time).group()
    request = APIRequest(
        year=str(year),
        month=f"{int(month):02d}",
        day=f"{int(day):02d}",
        time=time,
        coords=coords,
        vars=vars
    )
    
    return request.query(dir_)


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
    dfr = build_multiindex_dataframe(df, preds)
    for pred, origin in dfr.columns:
        if 'ERA' in origin:
            dfr.loc[:, (pred, 'ERA5')] = convert_ameriflux_to_era5(dfm, pred)

    ts = dfr.pop('timestamp')
    dfr.insert(0, 'timestamp', ts)

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
    paths = []
    for fldr in sub_fldrs:
        paths.extend(glob.glob(os.path.join(fldr, "*.nc")))

    ds = xr.open_mfdataset(
        paths,
        engine="netcdf4",
        combine="by_coords",
        chunks={"time": 1}
    )
    print(f"ITS NOT THE MERGING WHO FAILS, BUT THE CONVERSION TO DATAFRAME.")

    return apply_column_rename(ds.to_dataframe()).drop(columns=['number', 'expver'])


def apply_column_rename(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=SHORTNAME_TO_FULLNAME)


def build_multiindex_dataframe(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    """
    Restructures a DataFrame by renaming specified predictor columns with a "AMF, " prefix,
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
        both "AMERIFLUX" and "ERA5" sources, and columns are grouped by variable name.
    """
    ameriflux_cols = [f"AMF, {p}" for p in preds]
    era5_cols = [f"ERA5, {p}" for p in preds]
    renamed = df.rename(columns=dict(zip(preds, ameriflux_cols))).assign(**{col: np.nan for col in era5_cols})

    tuples = []
    for col in renamed.columns:
        if ", " in col:
            src, var = col.split(", ", 1)
        else:
            src, var = "AMF", col
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

    arr = df[cols].to_numpy(dtype=float)

    if func is None:
        return arr[:, 0]

    return func(*[arr[:, i] for i in range(arr.shape[1])])


def main():
    parser = argparse.ArgumentParser(
        prog="carbonpipeline",
        description="Pipeline to retrieve and compute AmeriFlux variables based on ERA5 variables.",
        epilog="More information available on GitHub."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------
    # Subcommand: point
    # -----------------
    point_parser = subparsers.add_parser(
        name="point", description="Run pipeline for a specific point (lat/lon)"
    )
    point_parser.add_argument("--output-format", required=True, choices=['csv', 'netcdf'], help="Desired output format")
    point_parser.add_argument("--file", required=True, type=str, help="Path to the dataset file")
    point_parser.add_argument("--coords", required=True, nargs=2, type=float, metavar=("LAT", "LON"), help="Latitude and longitude coordinates")
    point_parser.add_argument("--start", required=True, type=str, help="Start date (YYYY-MM-DDTHH:MM:SS)")
    point_parser.add_argument("--end", required=True, type=str, help="End date (YYYY-MM-DDTHH:MM:SS)")
    point_parser.add_argument("--preds", required=False, nargs='*', help="List of predictors (e.g., TA RH CO2)")

    # ----------------
    # Subcommand: area
    # ----------------
    area_parser = subparsers.add_parser(
        name="area",
        help="Work with data over a bounding box in two stages (download / process)",
        epilog="There is two options here because the process can sometimes fail if large datasets were downloaded."
    )
    area_actions = area_parser.add_subparsers(
        dest="action",
        required=True,
        help="Choose either 'download' or 'process' for area"
    )
    # --------------------------
    # Subcommand: area, download
    # --------------------------
    area_download = area_actions.add_parser(
        "download",
        help="Download ERA5 files for a bounding box (N W S E)"
    )
    area_download.add_argument("--coords", required=True, nargs=4, type=float, metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'), help="Geographical bounding box")
    area_download.add_argument("--start", required=True, type=str, help="Start date (YYYY-MM-DDTHH:MM:SS)")
    area_download.add_argument("--end", required=True, type=str, help="End date (YYYY-MM-DDTHH:MM:SS)")
    area_download.add_argument("--preds", required=False, nargs='*', help="List of predictors (e.g., TA RH CO2)")
    # -------------------------
    # Subcommand: area, process
    # -------------------------
    area_process = area_actions.add_parser(
        "process",
        help="Process previously downloaded ERA5 folders for a bounding box"
    )
    area_process.add_argument("--output-format", required=True, choices=['csv', 'netcdf'], help="Desired output format")

    args = parser.parse_args()

    if not args.action == 'process':
        args.start = ' '.join(args.start.split('T'))
        args.end = ' '.join(args.end.split('T'))

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
                    my_set.add(var)                             
            vars_ = list(my_set)

        if args.command == 'point':
            print(f"------------------- Inputs -------------------")
            print(f"File: {args.file}")
            print(f"Latitude/longitude: {args.coords}")
            print(f"Start date: {args.start}, End date: {args.end}")
            print(f"Predictors: {args.preds}")
            print(f"-----------------------------------------------\n")
            df = run_point_pipeline(args.file, args.coords, args.start, args.end, args.preds, vars_)
        elif args.command == 'area':
            print(f"------------------- Inputs -------------------")
            print(f"Area: {args.coords}")
            print(f"Start date: {args.start}, End date: {args.end}")
            print(f"Predictors: {args.preds}")
            print(f"-----------------------------------------------\n")
            run_area_download(args.coords, args.start, args.end, args.preds, vars_)
            print(f"The downloads have been done and the manifest have been written, you can now proceed to the process phase.")
            sys.exit(0)
    elif args.action == 'process':
        df = run_area_process()

    start = time.time()

    if args.output_format == 'csv':
        df.to_csv('out.csv')
    elif args.output_format == 'netcdf':
        ds = df.to_xarray()
        ds.to_netcdf('out.nc', format='NETCDF4', engine='netcdf4')

    end = time.time()
    print(f"Time taken to write in the {args.output_format.upper()} file: {end - start}")



if __name__ == "__main__":
    main()
