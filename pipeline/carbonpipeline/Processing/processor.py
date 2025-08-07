# carbonpipeline/processor.py
from datetime import datetime
import re
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from .constants import VARIABLES_FOR_PREDICTOR
from .processing_utils import PROCESSORS
from ..config import CarbonPipelineConfig


class DataProcessor:
    """Handles data processing operations for climate and environmental data."""
    
    def __init__(self, config: CarbonPipelineConfig):
        self.config = config
        self.tz_finder = TimezoneFinder()
    
    def convert_ameriflux_to_era5(self, df: pd.DataFrame, pred: str) -> np.ndarray:
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
    
    def load_and_filter_dataframe(self, path: str, start: str, end: str) -> pd.DataFrame:
        """Load and filter dataframe based on time range."""
        df = pd.read_csv(path, on_bad_lines='skip') # Added to handle potential bad lines
        df["timestamp"] = df["timestamp"].apply(self._validate_date_format).pipe(pd.to_datetime)         
        df = df[(df["timestamp"].dt.minute == 0) & (df["timestamp"].dt.second == 0)]

        min_ts, max_ts = df["timestamp"].min(), df["timestamp"].max()
        start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

        if start_ts < min_ts or end_ts > max_ts:
            msg = (f"The requested interval [{start_ts} -> {end_ts}] "
                   f"is out of bound for the given CSV file [{min_ts} -> {max_ts}].")
            raise ValueError(msg)
        
        filtered_df = df[df["timestamp"].between(start_ts, end_ts)].copy()
        return self._find_missing_rows(filtered_df)

    def _validate_date_format(self, ts: str | int) -> str:
        """Validate and format date strings."""
        if isinstance(ts, str):                                      
            try:
                # Attempt to parse with the main format
                datetime.strptime(ts, self.config.DATETIME_FMT)
                return ts
            except ValueError: 
                # If it fails, try to parse the other common format
                try:
                    regex = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})'
                    Y, M, D, h, m = re.split(regex, str(ts))[1:6]
                    return f"{Y}-{M}-{D} {h}:{m}:00"
                except (ValueError, IndexError):
                     return pd.NaT # Return Not a Time for unparseable dates
        
        # Handle integer timestamps
        try:
            regex = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})'
            Y, M, D, h, m = re.split(regex, str(ts))[1:6]
            return f"{Y}-{M}-{D} {h}:{m}:00"
        except (ValueError, IndexError):
            return pd.NaT

    def _find_missing_rows(self, df: pd.DataFrame) -> pd.DataFrame:
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
        miss = df[df.drop(columns="timestamp").isnull().any(axis=1)].copy()
        miss.loc[:, "year"] = miss["timestamp"].dt.year
        miss.loc[:, "month"] = miss["timestamp"].dt.month
        miss.loc[:, "day"] = miss["timestamp"].dt.day
        miss.loc[:, "time"] = miss["timestamp"].dt.strftime('%H:%M:%S')
        return miss

    def adjust_timezone(self, df: pd.DataFrame, coords: list[float]) -> pd.DataFrame:
        """Adjust DataFrame timezone based on coordinates."""
        dftz = df.copy()
        tz = self.tz_finder.timezone_at(lat=coords[0], lng=coords[1])
        dftz["timestamp"] = dftz["timestamp"].dt.tz_localize("UTC")
        dftz["timestamp"] = dftz["timestamp"].dt.tz_convert(tz)
        return dftz

    def get_missing_groups(self, df: pd.DataFrame) -> list[tuple]:
        """Get groups of missing data."""
        return [g for g, _ in df.groupby(["year", "month", "day", "time"])]

    def get_hourly_groups(self, start: str, end: str) -> list[tuple]:
        """Generate hourly groups for the given time range."""
        hrs = pd.date_range(start=start, end=end, freq="h")
        return [(d.year, d.month, d.day, d.strftime('%H:%M:%S')) for d in hrs]