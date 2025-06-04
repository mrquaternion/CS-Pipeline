# carbonpipeline/cli.py
import os, shutil, glob, re, json, argparse, zipfile
from datetime import datetime
from timezonefinder import TimezoneFinder
from tqdm import tqdm

from carbonpipeline.processing_utils import *
from carbonpipeline.constants import *
from carbonpipeline.api_request import APIRequest
import xarray as xr
import pandas as pd


ZIP_DIR         = "./datasets/zip"
UNZIP_DIR       = "./datasets/unzip"
DATETIME_FMT    = r"\d{4}-\d{2}-\d{2} \d{2}:00:00"
OUTPUT_MANIFEST = "./manifest.json"
TZ_FINDER       = TimezoneFinder()


def run_point_download(file_path: str, format: str,
                       coords: list[float], start: str, end: str, 
                       preds: list[str], vars_: list[str]):
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
    format : str
        Desired format for the output file.
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
    df         = _load_and_filter_dataframe(file_path, start, end)
    dftz       = _adjust_timezone(df, coords)
    groups     = _missing_groups(dftz, coords)
    unzip_dirs = _download_groups(groups, vars_, coords)
    df_out     = _run_point_process(df.drop(columns=["year", "month", "day", "time"]), 
                                    preds, unzip_dirs)
    _save_output(format, df_out)


def run_area_download(coords: list[float], 
                      start: str, end: str, 
                      preds: list[str], vars_: list[str]):
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
    groups = _hourly_groups(start, end)
    unzip_dirs = _download_groups(groups, vars_, coords)

    os.makedirs(os.path.dirname(OUTPUT_MANIFEST), exist_ok=True)
    with open(OUTPUT_MANIFEST, 'w') as fp:
        json.dump({"preds": preds, "unzip_sub_folders": unzip_dirs},
                  fp, indent=2)


def run_area_process(outfile_name: str):
    preds, unzip_dirs = _load_manifest()
    merged_ds         = _merge_unzipped(unzip_dirs)

    tmp_dir = _write_chunks(merged_ds, preds)
    _concat_chunks(tmp_dir, outfile_name + ".nc")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    

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
    arr  = df[cols].to_numpy(dtype=float)

    if func is None:
        return arr[:, 0]
    return func(*[arr[:, i] for i in range(arr.shape[1])])
    

def _load_and_filter_dataframe(path: str, start: str, end: str) -> pd.DataFrame:
    df              = pd.read_csv(path)
    df["timestamp"] = df["timestamp"].apply(_validate_date_format).pipe(pd.to_datetime)         
    df              = df[(df["timestamp"].dt.minute == 0) & (df["timestamp"].dt.second == 0)]

    min_ts, max_ts   = df["timestamp"].min(), df["timestamp"].max()
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

    if start_ts < min_ts or end_ts > max_ts:
        msg = (f"The requested interval [{start_ts} -> {end_ts}] "
               f"is out of bound for the given CSV file [{min_ts} -> {max_ts}].")
        raise ValueError(msg)
    return _find_missing_rows(df[df["timestamp"].between(pd.to_datetime(start), pd.to_datetime(end))])


def _validate_date_format(ts: str | int) -> str:
    if isinstance(ts, str):                                      
        try:
            datetime.strptime(ts, DATETIME_FMT)
            return ts
        except ValueError: 
            pass

    regex = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})'
    Y, M, D, h, m = re.split(regex, str(ts))[1:6]
    return f"{Y}-{M}-{D} {h}:{m}:00"


def _find_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and return rows in the DataFrame that contain missing (NaN) values, excluding the 'timestamp' 
    column. Adds 'year', 'month', 'day' and 'time' columns extracted from the 'timestamp' column for each missing row.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'timestamp' column and other columns to check for missing values.

    Returns
    -------
    pd.DataFrame
        DataFrame containing rows with missing values (excluding 'timestamp').
    """
    miss          = df[df[df.columns.drop("timestamp")].isnull().any(axis=1)]
    miss["year"]  = miss["timestamp"].dt.year
    miss["month"] = miss["timestamp"].dt.month
    miss["day"]   = miss["timestamp"].dt.day
    miss["time"]  = miss["timestamp"].dt.time.astype(str)
    return miss


def _adjust_timezone(df: pd.DataFrame, coords: list[float]):
    dftz              = df.copy()
    tz                = TZ_FINDER.timezone_at(lat=coords[0], lng=coords[1])
    dftz["timestamp"] = dftz["timestamp"].dt.tz_localize("UTC")
    dftz["timestamp"] = dftz["timestamp"].dt.tz_convert(tz)
    return dftz


def _missing_groups(df: pd.DataFrame, coords: list[float]) -> list[tuple]:
    return [g for g, _ in df.groupby(["year", "month", "day", "time"])]


def _hourly_groups(start: str, end: str) -> list[tuple]:
    hrs = pd.date_range(start=start, end=end, freq="h")
    ts  = (np.datetime_as_string(hrs, unit="h") + ":00")
    return [re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}:00)", t).groups() for t in ts]


def _download_groups(groups: list[tuple], vars_: list[str], 
                     coords: list[float]) -> list[str]:
    _setup_dirs(ZIP_DIR, UNZIP_DIR)

    fldrs = [] 
    for group in tqdm(groups, desc="Number of hours downloaded", unit="hours", colour="green"):
        fname    = _prepare_request(group, ZIP_DIR, coords, vars_)
        zip_fp   = f"{ZIP_DIR}/{fname}"
        unzip_fp = f"{UNZIP_DIR}/{fname.split('.')[0]}"
        fldrs.append(unzip_fp)
        _extract_zip(zip_fp, unzip_fp)
    return fldrs


def _setup_dirs(*dirs):
    for d in dirs: 
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)


def _prepare_request(group: tuple, dir_: str, 
                     coords: list[float], vars_: list[str]) -> str:
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
    Y, M, D, t = group
    request = APIRequest(year=str(Y), month=f"{int(M):02d}", day=f"{int(D):02d}",
                         time=re.search(r"\d{2}:00", t).group(),
                         coords=coords, vars_=vars_)
    return request.query_era5(dir_)


def _extract_zip(zip_fp: str, unzip_fp: str):
    """
    Extracts all files from a ZIP archive to a specified directory, displaying a progress bar.

    Parameters
    ----------
    zip_path : str 
        The path to the ZIP file to be extracted.
    unzip_path : str
        The directory where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_fp, "r") as z:
        try: 
            z.extractall(unzip_fp)
            os.remove(zip_fp)
        except zipfile.error as e: 
            print(f"Failed to extract {zip_fp}: {e}")


def _run_point_process(df: pd.DataFrame, preds: list[str], 
                       unzip_dirs: list[str]) -> pd.DataFrame:
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
    dfm = _apply_column_rename(_merge_unzipped(unzip_dirs)).to_dataframe()
    dfr = _build_multiindex_dataframe(df, preds)
    for pred, origin in dfr.columns:
        if "ERA" in origin:
            dfr.loc[:, (pred, "ERA5")] = convert_ameriflux_to_era5(dfm, pred)
    ts = dfr.pop("timestamp")
    dfr.insert(0, "timestamp", ts)
    return dfr


def _load_manifest() -> tuple[list[str], list[str]]:
    with open(OUTPUT_MANIFEST, "r") as fp:
        m = json.load(fp)
    return m["preds"], m["unzip_sub_folders"]


def _merge_unzipped(dirs: list[str]) -> xr.Dataset:
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
    paths = [
        p 
        for d in dirs 
        for p in glob.glob(os.path.join(d, "*.nc"))
    ]
    return xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords",
                             chunks={"time": "auto"}, drop_variables=["number", "expver"])


def _save_output(format: str, df: pd.DataFrame):
    if format == "csv":
        df.to_csv("out.csv")
    else:
        (df.to_xarray()
           .to_netcdf("out.nc", format="NETCDF4", engine="netcdf4"))


def _write_chunks(ds: xr.Dataset, preds: list[str]) -> str:
    tmp_dir = "./outputs"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    for i in tqdm(range(ds.sizes['valid_time'])):
        chunk_df = ds.isel(valid_time=i).to_dataframe()
        lookup   = {p: convert_ameriflux_to_era5(chunk_df, p) for p in preds}
        (pd.DataFrame(lookup, index=chunk_df.index)
           .to_xarray()
           .to_netcdf(f"{tmp_dir}/CHUNK_{i}.nc", mode="w",
                      format="NETCDF4", engine="netcdf4"))
    return tmp_dir


def _concat_chunks(tmp_dir: str, final_out: str):
    paths = glob.glob(os.path.join(tmp_dir, "*.nc"))
    xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords", 
                      chunks={"time": "auto"}, drop_variables=["number", "expver"]
    ).to_netcdf(final_out, mode="w", format="NETCDF4", engine="netcdf4")


def _apply_column_rename(obj: pd.DataFrame | xr.Dataset) -> pd.DataFrame | xr.Dataset:
    return obj.rename(name_dict=SHORTNAME_TO_FULLNAME)


def _build_multiindex_dataframe(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carbonpipeline",
        description="Pipeline to retrieve and compute AmeriFlux variables based on ERA5 variables.",
        epilog="More information available on GitHub."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_point_subparser(subparsers)
    _add_area_subparser(subparsers)
    
    return parser


def _add_point_subparser(subparsers):
    p = subparsers.add_parser(name="point", help="Work with data for a specific point (lat/lon)")
    p.add_argument("--output-format", required=True, choices=['csv', 'netcdf'], help="Desired output format")
    p.add_argument("--file", required=True, type=str, help="Path to the dataset file")
    p.add_argument("--coords", required=True, nargs=2, type=float, metavar=("LAT", "LON"), help="Latitude and longitude coordinates")
    p.add_argument("--start", required=True, type=str, help="Start date (YYYY-MM-DDTHH:MM:SS)")
    p.add_argument("--end", required=True, type=str, help="End date (YYYY-MM-DDTHH:MM:SS)")
    p.add_argument("--preds", required=False, nargs='*', help="List of predictors (e.g., TA RH CO2)")


def _add_area_subparser(subparsers):
    a = subparsers.add_parser(name="area", help="Work with data over a bounding box in two stages (download / process)", 
                                  epilog="There is two options here because the process can sometimes fail if large datasets were downloaded.")
    actions = a.add_subparsers(dest="action", required=True)

    dl = actions.add_parser("download", help="Download ERA5 files for a bounding box")
    dl.add_argument("--coords", required=True, nargs=4, type=float, metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'), help="Geographical bounding box")
    dl.add_argument("--start", required=True, type=str, help="Start date (YYYY-MM-DDTHH:MM:SS)")
    dl.add_argument("--end", required=True, type=str, help="End date (YYYY-MM-DDTHH:MM:SS)")
    dl.add_argument("--preds", required=False, nargs='*', help="List of predictors (e.g., TA RH CO2)")

    proc = actions.add_parser("process", help="Process previously downloaded ERA5 folders for a bounding box", 
                              epilog="The output format can only be NetCDF for compressibility reasons.")
    proc.add_argument("--name", required=True, type=str, help="Name of the output file. Overwrite if already exists within the directory")


def _validate_and_prepare(args, parser):
    args.start = ' '.join(args.start.split("T"))
    args.end   = ' '.join(args.end.split("T"))

    needs_wtd = False
    needs_co2 = False
    for pred in args.preds:
        if pred == "WTD": # Only variable that don't need ERA5 variables
            args.preds.remove("WTD")
            needs_wtd = True
        if pred == "CO2":
            needs_co2 = True

    if args.preds is not None:
        invalid = [
            p 
            for p in args.preds 
            if p not in VARIABLES_FOR_PREDICTOR
        ]
        if invalid:
            parser.error(
                f"\nInvalid predictor(s): {', '.join(invalid)}\n"
                f"Valid options are: {', '.join(VARIABLES_FOR_PREDICTOR)}"
            )

        needed = {
            var
            for pred in args.preds
            for var in VARIABLES_FOR_PREDICTOR[pred]
        }
        vars_ = list(needed)  
    else:
        vars_ = ERA5_VARIABLES
        args.preds = list(VARIABLES_FOR_PREDICTOR)

    return vars_, needs_wtd, needs_co2 


def _pretty_print_inputs(title: str, **fields):
    print(f"------------------- {title} -------------------")
    for k, v in fields.items():
        print(f"{k}: {v}")
    print("-----------------------------------------------\n")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "area" and args.action == "process":
        run_area_process(args.name)
        return
        
    vars_, needs_wtd, needs_co2 = _validate_and_prepare(args, parser)

    if args.command == "point":
        _pretty_print_inputs(
            "Inputs", File=args.file, Coordinates=args.coords,
            Start = args.start, End = args.end, Predictors=args.preds
        )
        run_point_download(args.file, args.output_format, args.coords, args.start, args.end, args.preds, vars_)
    elif args.command == "area" and args.action == "download":
        _pretty_print_inputs(
            "Inputs", Area=args.coords, Start = args.start, 
            End = args.end, Predictors=args.preds
        )
        run_area_download(args.coords, args.start, args.end, args.preds, vars_)
        print("The downloads are complete and the manifest is written; you can now proceed to the process phase.")


if __name__ == "__main__":
    main()
