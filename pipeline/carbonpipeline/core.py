# carbonpipeline/core.py
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from .Geometry.geometry import Geometry
from .config import CarbonPipelineConfig
from .Processing.processor import DataProcessor
from .downloader import DataDownloader
from .dataset import DatasetManager
from .Processing.processing_utils import AGG_SCHEMA


class CarbonPipeline:
    """Main pipeline orchestrator for carbon data processing."""
    
    def __init__(self):
        self.config = CarbonPipelineConfig()
        self.processor = DataProcessor(self.config)
        self.downloader = DataDownloader(self.config)
        self.dataset_manager = DatasetManager(self.config)

    async def run_download(
        self,
        coords_to_download: list[float],
        region_id: str,
        geometry: Geometry,
        start: str,
        end: str,
        preds: list[str],
        vrs: list[str],
        regions_to_process: dict[str | int, list[float]],
        processing_type: str,
        aggregation_type: str
    ) -> None:
        """
        Downloads ERA5 datasets for a specified area and time range.
        """
        start_adj = pd.to_datetime(start, errors="coerce")
        end_adj = pd.to_datetime(end, errors="coerce")
        if pd.isna(start_adj) or pd.isna(end_adj):
            raise ValueError(f"Invalid dates: start={start}, end={end}")

        groups = self.processor.get_request_groups(start_adj, end_adj, aggregation_type == "MONTHLY")
        unzip_dirs = await self.downloader.download_groups_async(groups, vrs, coords_to_download, aggregation_type == "MONTHLY", region_id)

        feature_entry = {
            "region_id": region_id,
            "start_date": start,
            "end_date": end,
            "geometry": geometry.geom_type.value,
            "unzip_sub_folders": unzip_dirs,
            "preds": preds,
            "rect_regions": regions_to_process,
        }

        manifest_path = Path(self.config.OUTPUT_MANIFEST)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or init manifest
        if manifest_path.is_file():
            with open(manifest_path, "r") as fp:
                try:
                    manifest = json.load(fp)
                    if not isinstance(manifest, dict):
                        manifest = {}
                except json.JSONDecodeError:
                    manifest = {}
        else:
            manifest = {}

        # Clean old per-feature keys (optional)
        for f in manifest.get("features", []):
            f.pop("processing_type", None)
            f.pop("aggregation_type", None)

        # Rebuild the object with desired key order:
        features = manifest.get("features", [])
        features.append(feature_entry)

        ordered_manifest = {
            "processing_type": processing_type,
            "aggregation_type": aggregation_type,
            "features": features
        }

        with open(manifest_path, 'w') as fp:
            json.dump(ordered_manifest, fp, indent=2)

        print(f"Appended new entry to manifest at {manifest_path}")

    def run_area_process(
        self,
        merged_ds: xr.Dataset,
        preds: list[str],
        start: str,
        end: str,
        rect_regions: dict[str | int, list[float]],
        output_name: str,
        processing_type: str,
        aggregation_type: str
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

        if processing_type == "BoundingBox":
            merged_ds = self.dataset_manager.filter_coordinates(ds=merged_ds, regions=rect_regions)
        else:
            merged_df = merged_ds.to_dataframe().reset_index()
            merged_df["region_id"] = merged_df
            merged_df = (
                merged_df
                .set_index(["region_id", "latitude", "longitude", "valid_time"])
                .sort_index()
            )
            merged_ds = merged_df.to_xarray()

        index = ['region_id', 'latitude', 'longitude', 'valid_time']
        tmp_dir = self.dataset_manager.write_chunks(merged_ds, preds, index, processing_type)
        self.dataset_manager.concat_chunks(tmp_dir, output_name)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Aggregation --> not available for global option because too much data --> not optimized with chunk loading
        resample_methods = {"DAILY": "1D", "MONTHLY": "1ME"}
        if aggregation_type in resample_methods.keys():
            while True:
                user_input = input("\nDo you want to delete the original file after aggregation? (Y/n): ").strip()
                if user_input.upper() == "Y":
                    delete_source = True
                    break
                elif user_input.lower() == "n":
                    delete_source = False
                    break
                else:
                    print("Invalid input: please enter 'Y' to delete them, or 'n' to keep them.")

            ds = self.open_nc(output_name)

            variables = list(ds.data_vars.keys())
            filtered_agg_schema = {key: AGG_SCHEMA[key] for key in variables if key in AGG_SCHEMA}
            agg_ds = xr.Dataset({
                name: getattr(
                    ds[pred].resample(valid_time=resample_methods[aggregation_type]),
                    func
                )()
                for pred, agg_types in filtered_agg_schema.items()
                for agg_dict in [agg_types.get(aggregation_type.lower(), {})]
                if agg_dict != "DROP"
                for name, func in agg_dict.items()
            })
            if aggregation_type == "MONTHLY":
                agg_ds["valid_time"] = agg_ds["valid_time"].to_index().to_period("M")

            save_path = self.write_aggregated_ds(
                agg_ds=agg_ds,
                output_name=output_name,
                aggregation_type=aggregation_type,
                delete_source=delete_source
            )
            print(f"✅ Aggregation saved to {save_path}")

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
        dfr = self.dataset_manager.build_multiindex_dataframe(df, preds)
        
        for pred in preds:
            if pred in dfr.columns.get_level_values('variable'):
                era5_values = self.processor.convert_ameriflux_to_era5(dfm, pred)
                dfr.loc[:, (pred, "ERA5")] = era5_values

        ts = dfr.pop(("timestamp", "AMF"))
        dfr.insert(0, "timestamp", ts.droplevel('source'))

        if dfr is not None:
            self.dataset_manager.save_output(dfr, output_name)

    def load_features_from_manifest(self):
        """Load manifest file"""
        with open(self.config.OUTPUT_MANIFEST, "r") as fp:
            content = json.load(fp)
        return content

    def open_nc(self, output_name: str) -> xr.Dataset:
        fname = f"{output_name}.nc"
        path = Path(self.config.OUTPUT_PROCESSED_DIR) / fname
        ds = xr.open_dataset(path, decode_times=True).load()
        return ds

    def write_aggregated_ds(
        self,
        agg_ds: xr.Dataset,
        output_name: str,
        aggregation_type: str,
        delete_source: bool,
    ) -> Path:
        path = Path(self.config.OUTPUT_PROCESSED_DIR) / f"{output_name}_{aggregation_type.lower()}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Overwrite if exists
        if path.exists():
            print(f"⚠️ Overwriting existing aggregated file: {path}")
            path.unlink()

        if "valid_time" in agg_ds.coords:
            agg_ds = agg_ds.assign_coords(
                valid_time=("valid_time", np.array(agg_ds["valid_time"].values, dtype="datetime64[ns]"))
            )

        encoding = {}
        for v in agg_ds.data_vars:
            enc = {"zlib": True, "complevel": 4}
            # If float64 isn't required, store as float32 to cut size in half
            if str(agg_ds[v].dtype).startswith("float64"):
                enc["dtype"] = np.float32
            encoding[v] = enc

        agg_ds.to_netcdf(path, encoding=encoding, engine="netcdf4")

        if delete_source:
            src = Path(self.config.OUTPUT_PROCESSED_DIR) / f"{output_name}.nc"
            try:
                src.unlink()
            except FileNotFoundError:
                pass

        return path

    @staticmethod
    def setup_manifest_and_dirs(manifest, *dirs) -> None:
        """Setup directories by removing and recreating them."""
        manifest_path = Path(manifest)
        if manifest_path.exists():
            manifest_path.unlink() # deletes the manifest at each run

        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)