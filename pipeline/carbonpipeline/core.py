# carbonpipeline/core.py
import json
import os
from pathlib import Path
import shutil
from typing import Union

import pandas as pd
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
    
    async def run_point_download(
            self, 
            file_path: str, 
            format: str,
            coords: list[float], 
            start: str, 
            end: str, 
            preds: list[str], 
            vars_: list[str], 
            needs_wtd: bool, 
            needs_co2: bool
    ) -> None:
        """
        Main workflow for point-based data processing.
        """
        df = self.processor.load_and_filter_dataframe(file_path, start, end)
        if df.empty:
            print("No missing data found in the specified time range. Nothing to do.")
            return

        dftz = self.processor.adjust_timezone(df, coords)
        groups = self.processor.get_missing_groups(dftz)
        unzip_dirs = await self.downloader.download_groups_async(groups, vars_, coords)
        
        df_out = await self._run_point_process(df.drop(columns=["year", "month", "day", "time"]), 
                                              preds, unzip_dirs, needs_wtd, needs_co2)
        
        if df_out is not None:
            self.dataset_manager.save_output(format, df_out, "point_output")

    async def run_download(
            self, 
            coords: list[float], 
            region_id: str,
            start: str, 
            end: str, 
            preds: list[str], 
            vars_: list[str]
    ) -> None:
        """
        Downloads ERA5 datasets for a specified area and time range.
        """
        groups = self.processor.get_hourly_groups(start, end)
        unzip_dirs = await self.downloader.download_groups_async(groups, vars_, coords, region_id)

        manifest_data = {
            "preds": preds, 
            "unzip_sub_folders": unzip_dirs, 
            "start_date": start, 
            "end_date": end,
            "region_id": region_id
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

    def run_area_process(self, outfile_name: str) -> None:
        """Process area data from manifest."""
        preds, unzip_dirs, start, end = self.load_manifest()
        merged_ds = self.dataset_manager.merge_unzipped(unzip_dirs)

        if merged_ds is None:
            print("No data to process. Exiting.")
            return
        
        # STOP HERE
        
        merged_ds = self.dataset_manager.apply_column_rename(merged_ds)
        print("ERA5:")
        print(merged_ds.isel(valid_time=slice(0, 5)).to_dataframe())

        # Handle CO2 data
        ds_co2 = self.dataset_manager.load_and_clean_co2_dataset()
        if ds_co2 is not None:
            print("Adding CO2 column...")
            merged_ds = self.dataset_manager.add_co2_column(merged_ds, ds_co2)
            print("After CO2 addition:")
            print(merged_ds.isel(valid_time=slice(0, 5)).to_dataframe())

        # Handle WTD data
        ds_wtd = self.dataset_manager.load_and_clean_wtd_dataset(start, end)
        if ds_wtd is not None:
            print("Adding WTD column...")
            merged_ds = self.dataset_manager.add_wtd_column(merged_ds, ds_wtd)
            print("After WTD addition:")
            print(merged_ds.isel(valid_time=slice(0, 5)).to_dataframe())

        tmp_dir = self.dataset_manager.write_chunks(merged_ds, preds)
        self.dataset_manager.concat_chunks(tmp_dir, outfile_name + ".nc")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_point_process(
            self, 
            df: pd.DataFrame, 
            preds: list[str], 
            unzip_dirs: list[str], 
            needs_wtd: bool, 
            needs_co2: bool
    ) -> pd.DataFrame:
        """
        Post-processes downloaded data for a single point.
        """
        merged_ds = self.dataset_manager.merge_unzipped(unzip_dirs)
        if merged_ds is None:
            print("No downloaded data to process.")
            return None
        
        # Handle CO2 data (similar to area processing)
        if needs_co2:
            ds_co2 = self.dataset_manager.load_and_clean_co2_dataset()
            if ds_co2 is not None:
                print("Adding CO2 column...")
                merged_ds = self.dataset_manager.add_co2_column(merged_ds, ds_co2)

        # Handle WTD data (similar to area processing)  
        if needs_wtd:
            # Extract start and end dates from the dataframe
            start_date = df['timestamp'].min().strftime('%Y-%m-%d')
            end_date = df['timestamp'].max().strftime('%Y-%m-%d')
            
            ds_wtd = self.dataset_manager.load_and_clean_wtd_dataset(start_date, end_date)
            if ds_wtd is not None:
                print("Adding WTD column...")
                merged_ds = self.dataset_manager.add_wtd_column(merged_ds, ds_wtd)
                
        dfm = self.dataset_manager.apply_column_rename(merged_ds).to_dataframe()
        dfr = self.dataset_manager.build_multiindex_dataframe(df, preds)
        
        for pred in preds:
            if pred in dfr.columns.get_level_values('variable'):
                era5_values = self.processor.convert_ameriflux_to_era5(dfm, pred)
                dfr.loc[:, (pred, "ERA5")] = era5_values

        ts = dfr.pop(("timestamp", "AMF"))
        dfr.insert(0, "timestamp", ts.droplevel('source'))
        return dfr

    def load_features_from_manifest(self) -> tuple[list[str], list[str], str, str]:
        """Load manifest file"""
        with open(self.config.OUTPUT_MANIFEST, "r") as fp:
            content = json.load(fp)
        return content["features"]

    def setup_manifest_and_dirs(self, manifest, *dirs) -> None:
        """Setup directories by removing and recreating them."""
        manifest_path = Path(manifest)
        if manifest_path.exists():
            manifest_path.unlink() # deletes the manifest at each run
        for d in dirs: 
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)