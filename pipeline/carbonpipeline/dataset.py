# carbonpipeline/dataset.py
import glob
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import rioxarray as rxr

from .api_request import CO2_FOLDERNAME
from .Processing.constants import SHORTNAME_TO_FULLNAME
from .Processing.processor import DataProcessor
from .config import CarbonPipelineConfig


class DatasetManager:
    """Manages dataset operations including merging, processing, and saving."""
    
    def __init__(self, config: CarbonPipelineConfig):
        self.config = config
        
    def merge_unzipped(self, dirs: list[str]) -> xr.Dataset:
        """
        Merges NetCDF files from multiple subfolders into a single xarray Dataset.
        """
        paths = [
            p 
            for d in dirs 
            for p in glob.glob(os.path.join(d, "*.nc"))
        ]
        
        if not paths:
            return None

        return xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords",
                                chunks={"time": "auto"}, drop_variables=["number", "expver"])

    def add_co2_column(self, ds_era5: xr.Dataset, ds_co2: xr.Dataset) -> xr.Dataset:
        """Prepare CO2 dataset for merging."""

        # Rename CO2 indexes so it matches ERA5 indexes
        ds_co2_renamed = ds_co2.rename({"time": "valid_time", "lat": "latitude", "lon": "longitude"})

        # Add column year_month to both dataset
        ds_co2_renamed = self._add_year_month(ds_co2_renamed, "valid_time")
        ds_era5_renamed = self._add_year_month(ds_era5, "valid_time")
     
        ds_co2_monthly = ds_co2_renamed.groupby('year_month').mean(dim='valid_time')
        
        # Cut for dates for which we only queried through ERA5
        unique_months_era5 = np.unique(ds_era5_renamed.year_month.values)
        ds_co2_monthly_cut = ds_co2_monthly.sel(year_month=unique_months_era5)
        ds_co2_sortby = ds_co2_monthly_cut.sortby(['latitude', 'longitude'], ascending=[False, False])

        """print(ds_co2_sortby.to_dataframe())"""

        ds_era5_coord_reajusted = self._assign_closest_lat_lon(ds_era5, ds_co2_monthly_cut, "latitude", "longitude")
        ds_era5_sortby = ds_era5_coord_reajusted.sortby(['lat', 'lon'], ascending=[False, False])

        """print(ds_era5_sortby.to_dataframe())"""

        ##### EVERYTHING IS GOOD FROM HERE

        co2_selected = ds_co2_sortby["xco2"].sel(
            year_month=ds_era5_sortby["year_month"],
            latitude=ds_era5_sortby["lat"],
            longitude=ds_era5_sortby["lon"]
        )

        ds_era5_sortby["xco2"] = (("valid_time", "latitude", "longitude"), co2_selected.data)

        return ds_era5_sortby
        
    def add_wtd_column(self, ds_era5: xr.Dataset, ds_wtd: xr.Dataset) -> xr.Dataset:
        """Add WTD column to ERA5 dataset."""

        # Remove unwanted columns
        ds_wtd = ds_wtd.drop_vars("spatial_ref")

        # Rename WTD indexes so it matches ERA5 indexes
        ds_wtd_renamed = ds_wtd.rename({"time": "valid_time", "y": "latitude", "x": "longitude"})

        # Add column year_month to both dataset
        ds_wtd_renamed = self._add_year_month(ds_wtd_renamed, "valid_time")
        ds_era5 = self._add_year_month(ds_era5, "valid_time")
        
        ds_wtd_monthly = ds_wtd_renamed.groupby('year_month').mean(dim='valid_time')
        ds_wtd_sortby = ds_wtd_monthly.sortby(['latitude', 'longitude'], ascending=[False, False])

        """print(ds_era5.isel(valid_time=slice(0, 5)).to_dataframe())"""

        ds_wtd_coord_reajusted = self._assign_closest_lat_lon(ds_wtd_sortby, ds_era5, "lat", "lon")

        # Reconstructing the good index
        ds_wtd_coord_reajusted = ds_wtd_coord_reajusted.set_index({
            'year_month': 'year_month',
            'latitude': 'lat',
            'longitude': 'lon'
        })

        """print(ds_wtd_coord_reajusted.to_dataframe())"""

        # Manipulate the index
        df = ds_wtd_coord_reajusted.to_dataframe().reset_index()

        # Delete duplicates
        df = df.drop_duplicates(subset=["year_month", "latitude", "longitude"])
        ds_wtd_coord_reajusted_clean = df.set_index(["year_month", "latitude", "longitude"]).to_xarray()

        wtd_selected = ds_wtd_coord_reajusted_clean["wtd"].sel(
            year_month=ds_era5["year_month"],
            latitude=ds_era5["lat"],
            longitude=ds_era5["lon"]
        )

        ds_era5["wtd"] = (("valid_time", "latitude", "longitude"), wtd_selected.data)

        """print(ds_era5.isel(valid_time=slice(0, 5)).to_dataframe())"""
        """print(np.isnan(ds_era5["xco2"].values).all())"""
        """print(np.isnan(ds_era5["wtd"].values).all())"""

        return ds_era5.drop(["year_month", "lat", "lon"])

    def _add_year_month(self, ds: xr.Dataset, time_coord: str) -> xr.Dataset:
        """Add year_month coordinate as datetime64[M] (truncated to month)."""
        year_month_periods = pd.to_datetime(ds[time_coord].values).to_period('M')
        ds['year_month'] = (time_coord, year_month_periods)
        return ds

    def _assign_closest_lat_lon(
            self, 
            ds_projected_on: xr.Dataset, 
            ds_projecting: xr.Dataset,
            lat_name: str,
            lon_name: str
    ) -> xr.Dataset:
        """Assign closest lat/lon coordinates."""
        b_lats = np.unique(ds_projecting[lat_name].values)
        b_lons = np.unique(ds_projecting[lon_name].values)
        
        return ds_projected_on.assign_coords(
            lat=("latitude", self._match_to_closest(ds_projected_on["latitude"].values, b_lats)),
            lon=("longitude", self._match_to_closest(ds_projected_on["longitude"].values, b_lons)),
        )

    def load_and_clean_co2_dataset(self) -> xr.Dataset:
        """Load and clean CO2 dataset."""
        co2_files = glob.glob(os.path.join(self.config.UNZIP_DIR, CO2_FOLDERNAME, "*.nc"))
        if not co2_files:
            return None
        ds = xr.open_dataset(co2_files[0])
        ds["xco2"] = ds["xco2"].where(ds["xco2"] < 1e10, np.nan)

        return ds[["xco2"]]

    def load_and_clean_wtd_dataset(self, start: str, end: str) -> xr.Dataset:
        """Load and clean WTD dataset."""
        wtd_dir_name = "_".join(["WTD", pd.to_datetime(start).strftime("%Y-%m"), pd.to_datetime(end).strftime("%Y-%m")])
        wtd_full_path = os.path.join(self.config.UNZIP_DIR, wtd_dir_name)

        wtd_files = glob.glob(os.path.join(wtd_full_path, "*.tif"))
        if not wtd_files:
            return None
        
        datasets = []
        for fn in wtd_files:
            da = rxr.open_rasterio(fn, masked=True).squeeze("band", drop=True)
            nb_to_agg = int(np.ceil(self.config.ERA5_RES/self.config.WTD_RES))
            da_coarse = da.coarsen(x=nb_to_agg, y=nb_to_agg, boundary="trim").mean()
            ds = da_coarse.to_dataset(name="wtd")
            # Extract date from filename and set as time coordinate
            date_str = os.path.basename(fn).split('-')[2].split('.')[0]
            time_val = pd.to_datetime(date_str, format='%Y%m%d')
            ds = ds.expand_dims(time=[time_val])
            datasets.append(ds)
        
        return xr.concat(datasets, dim="time") if datasets else None

    def _match_to_closest(self, values, reference_points) -> None:
        """Match values to closest reference points."""
        reference_points = np.asarray(reference_points)
        return np.array([reference_points[np.abs(reference_points - v).argmin()] for v in values])

    def apply_column_rename(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply column renaming to dataset."""
        rename_dict = {
            k: v 
            for k, v in SHORTNAME_TO_FULLNAME.items() 
            if k in ds.data_vars
        }
        return ds.rename(rename_dict)

    def build_multiindex_dataframe(self, df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
        """
        Restructures a DataFrame to have a MultiIndex for AmeriFlux vs ERA5 data.
        """
        ameriflux_cols = {p: f"AMF, {p}" for p in preds if p in df.columns}
        renamed_df = df.rename(columns=ameriflux_cols)
        
        for p in preds:
            era5_col_name = f"ERA5, {p}"
            if era5_col_name not in renamed_df.columns:
                 renamed_df[era5_col_name] = np.nan

        tuples = []
        for col in renamed_df.columns:
            if ", " in col:
                src, var = col.split(", ", 1)
            else:
                src, var = "AMF", col
            tuples.append((var, src))

        renamed_df.columns = pd.MultiIndex.from_tuples(tuples, names=["variable", "source"])
        return renamed_df.sort_index(axis=1, level="variable")

    def write_chunks(self, ds: xr.Dataset, preds: list[str]) -> str:
        """Write dataset in chunks."""
        tmp_dir = "./outputs_tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        processor = DataProcessor(self.config)
        # Process in larger time chunks for efficiency
        for i in tqdm(range(ds.sizes['valid_time']), desc="Processing and writing chunks"):
            chunk_ds = ds.isel(valid_time=slice(i, i + 1))
            chunk_df = chunk_ds.to_dataframe()
            
            # Ensure index is consistent
            chunk_df = chunk_df.reset_index().set_index(['valid_time', 'latitude', 'longitude'])

            lookup = {p: processor.convert_ameriflux_to_era5(chunk_df, p) for p in preds}
            
            (pd.DataFrame(lookup, index=chunk_df.index)
               .to_xarray()
               .to_netcdf(os.path.join(tmp_dir, f"CHUNK_{i}.nc"), mode="w",
                          format="NETCDF4", engine="netcdf4"))
        
        return tmp_dir

    def concat_chunks(self, tmp_dir: str, out_name: str) -> None:
        """Concatenate chunks into final output."""
        paths = glob.glob(os.path.join(tmp_dir, "*.nc"))
        if not paths:
            print("No chunks found to concatenate.")
            return

        out_dir = os.path.join(self.config.OUTPUT_PROCESSED_DIR, f"{out_name}.nc")

        ds = xr.open_mfdataset(paths, engine="netcdf4", combine="by_coords", chunks={"valid_time": "auto"}, parallel=True)
        ds.to_netcdf(out_dir, mode="w", format="NETCDF4", engine="netcdf4")
        print(f"Final dataset saved to {out_dir}\n")

    def save_output(self, df: pd.DataFrame, out_name: str) -> None:
        """Save output in specified format."""
        out_dir = os.path.join(self.config.OUTPUT_PROCESSED_DIR, f"{out_name}.nc")
        (df.to_xarray()
           .to_netcdf(out_dir, format="NETCDF4", engine="netcdf4"))
        print(f"Output saved to {out_name}.{format}")