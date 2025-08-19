# carbonpipeline/core.py
import json
import os
from pathlib import Path
import shutil
import xarray as xr

import pandas as pd

from .Geometry.geometry import Geometry
from .config import CarbonPipelineConfig
from .Processing.processor import DataProcessor
from .downloader import DataDownloader
from .dataset import DatasetManager


class CarbonPipeline:
    """Main pipeline orchestrator for carbon data processing."""
    
    def __init__(self):
        self.config = CarbonPipelineConfig()
        self.processor = DataProcessor(self.config)
        self.downloader = DataDownloader(self.config)
        self.dataset_manager = DatasetManager(self.config)

    async def run_download(
        self,
        coords: list[float],
        region_id: str,
        geometry: Geometry,
        start: str,
        end: str,
        preds: list[str],
        vars_: list[str],
        threshold_activated: bool,
        rect_regions: list[list[float]]
    ) -> None:
        """
        Downloads ERA5 datasets for a specified area and time range.
        """
        start_adj, end_adj = self.processor.adjust_timezone_str(coords, start, end)
        groups = self.processor.get_hourly_groups(start_adj, end_adj)
        unzip_dirs = await self.downloader.download_groups_async(groups, vars_, coords, region_id)

        manifest_data = {
            "region_id": region_id,
            "preds": preds, 
            "unzip_sub_folders": unzip_dirs,
            "start_date": start, 
            "end_date": end,
            "geometry": geometry.geom_type.value,
            "threshold_activated": threshold_activated,
            "rect_regions": rect_regions
        }

        manifest_path = Path(self.config.OUTPUT_MANIFEST)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        if manifest_path.is_file():
            with open(manifest_path, 'r') as fp:
                try:
                    manifest = json.load(fp)
                    if not isinstance(manifest, dict) or "features" not in manifest:
                        manifest = {"features": []}
                except json.JSONDecodeError:
                    manifest = {"features": []}
        else:
            manifest = {"features": []}

        manifest['features'].append(manifest_data)

        with open(manifest_path, 'w') as fp:
            json.dump(manifest, fp, indent=2)

        print(f"Appended new entry to manifest at {manifest_path}")

    def run_area_process(
        self,
        merged_ds: xr.Dataset,
        preds: list[str],
        start: str,
        end: str,
        output_name: str
    ) -> None:
        """Process area data from manifest."""
        print(f"For {output_name}:")
        merged_ds = self.dataset_manager.apply_column_rename(merged_ds)
        #print("ERA5:")
        #print(merged_ds.isel(valid_time=slice(0, 5)).to_dataframe())

        # Handle CO2 data
        ds_co2 = self.dataset_manager.load_and_clean_co2_dataset()
        if ds_co2 is not None:
            print("➕ Adding CO2 column...")
            merged_ds = self.dataset_manager.add_co2_column(merged_ds, ds_co2)
            #print("After CO2 addition:")
            #print(merged_ds.isel(valid_time=slice(0, 5)).to_dataframe())

        # Handle WTD data
        ds_wtd = self.dataset_manager.load_and_clean_wtd_dataset(start, end)
        if ds_wtd is not None:
            print("➕ Adding WTD column...")
            merged_ds = self.dataset_manager.add_wtd_column(merged_ds, ds_wtd)
            #print("After WTD addition:")
            #print(merged_ds.isel(valid_time=slice(0, 5)).to_dataframe())

        tmp_dir = self.dataset_manager.write_chunks(merged_ds, preds)
        self.dataset_manager.concat_chunks(tmp_dir, output_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def run_point_process(
        self,
        data_fp: str,
        preds: list[str],
        merged_ds: xr.Dataset,
        start: str,
        end: str,
        output_name: str
    ) -> None:
        """
        Post-processes downloaded data for a single point.
        """
        df = self.processor.load_and_filter_dataframe(data_fp, start, end)
        if df.empty:
            print("No missing data found in the specified time range. Nothing to do.")
            return None

        first_lat = merged_ds.latitude.values[0]
        first_lon = merged_ds.longitude.values[0]
        dftz = self.processor.adjust_timezone_df(df, [first_lat, first_lon])

        # Handle CO2 data (similar to area processing)
        ds_co2 = self.dataset_manager.load_and_clean_co2_dataset()
        if ds_co2 is not None:
            print("Adding CO2 column...")
            merged_ds = self.dataset_manager.add_co2_column(merged_ds, ds_co2)

        # Handle WTD data (similar to area processing)
        ds_wtd = self.dataset_manager.load_and_clean_wtd_dataset(start, end)
        if ds_wtd is not None:
            print("Adding WTD column...")
            merged_ds = self.dataset_manager.add_wtd_column(merged_ds, ds_wtd)
                
        dfm = self.dataset_manager.apply_column_rename(merged_ds).to_dataframe()
        dfr = self.dataset_manager.build_multiindex_dataframe(dftz, preds)
        
        for pred in preds:
            if pred in dfr.columns.get_level_values('variable'):
                era5_values = self.processor.convert_ameriflux_to_era5(dfm, pred)
                dfr.loc[:, (pred, "ERA5")] = era5_values

        ts = dfr.pop(("timestamp", "AMF"))
        dfr.insert(0, "timestamp", ts.droplevel('source'))

        if dfr is not None:
            self.dataset_manager.save_output(dfr, output_name)

    def load_features_from_manifest(self) -> tuple[list[str], list[str], str, str]:
        """Load manifest file"""
        with open(self.config.OUTPUT_MANIFEST, "r") as fp:
            content = json.load(fp)
        return content["features"]

    def setup_manifest_and_dirs(self, manifest, *dirs) -> None:
        """Setup directories by removing and recreating them."""
        if manifest:
            manifest_path = Path(manifest)
            if manifest_path.exists():
                manifest_path.unlink() # deletes the manifest at each run

        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)