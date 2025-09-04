# carbonpipeline/core.py
import json
import os
import glob
import re
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})

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

    async def run_download_point(
        self,
        coords_to_download: list[float],
        region_id: str,
        geometry: Geometry,
        start: str,
        end: str,
        preds: list[str],
        vrs: list[str],
        gapfilling: bool,
        data_file: str
    ) -> None:
        """
        Downloads ERA5 datasets for a specified EC station and time range.
        """
        start_adj = pd.to_datetime(start, errors="coerce")
        end_adj = pd.to_datetime(end, errors="coerce")
        if pd.isna(start_adj) or pd.isna(end_adj):
            raise ValueError(f"Invalid dates: start={start}, end={end}")

        self.processor.check_data_file_time_range(data_file, start, end)

        groups = self.processor.get_request_groups(start_adj, end_adj, False)
        unzip_dirs = await self.downloader.download_groups_async(groups, vrs, coords_to_download,False, region_id)

        feature_entry = {
            "region_id": region_id,
            "data_file": data_file,
            "start_date": start,
            "end_date": end,
            "geometry": geometry.geom_type.value,
            "unzip_sub_folders": unzip_dirs,
            "preds": preds
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
            f.pop("gapfilling", None)

        # Rebuild the object with desired key order:
        features = manifest.get("features", [])
        features.append(feature_entry)

        ordered_manifest = {
            "gapfilling": gapfilling,
            "features": features
        }

        with open(manifest_path, 'w') as fp:
            json.dump(ordered_manifest, fp, indent=2)

        print(f"Appended new entry to manifest at {manifest_path}")


    async def run_download_area(
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
        print(f"Processing {output_name}...")
        merged_ds = self.dataset_manager.apply_column_rename(merged_ds)

        # Handle CO2 data
        ds_co2 = self.dataset_manager.load_and_clean_co2_dataset()
        if ds_co2 is not None:
            print("➕ Adding CO2 column...")
            merged_ds = self.dataset_manager.add_co2_column(merged_ds, ds_co2)

        # Handle WTD data
        ds_wtd = self.dataset_manager.load_and_clean_wtd_dataset(start, end)
        if ds_wtd is not None:
            print("➕ Adding WTD column...")
            merged_ds = self.dataset_manager.add_wtd_column(merged_ds, ds_wtd)

        if processing_type == "BoundingBox":
            all_dss = self.dataset_manager.filter_coordinates(ds=merged_ds, regions=rect_regions)
        else:
            merged_df = merged_ds.to_dataframe().reset_index()
            merged_df["region_id"] = list(rect_regions.keys())[0]
            merged_df = (
                merged_df
                .set_index(["region_id", "latitude", "longitude", "valid_time"])
                .sort_index()
            )
            all_dss = [merged_df.to_xarray()]

        # Conversion to AMF predictors and intelligent chunk writing
        index = ['region_id', 'latitude', 'longitude', 'valid_time']
        tmp_dirs = self.dataset_manager.write_chunks(all_dss, preds, index)

        # Reopen the chunks for each region and create the NetCDF files
        region_dsets = self.dataset_manager.concat_chunks(tmp_dirs)

        # Aggregation --> not available for global option because too much data --> not optimized with chunk loading
        resample_methods = {"DAILY": "1D", "MONTHLY": "1ME"}
        if aggregation_type in resample_methods.keys():
            while True:
                user_input = input("\nDo you want to delete the original files after aggregation? (Y/n): ").strip()
                if user_input.upper() == "Y":
                    delete_source = True
                    break
                elif user_input.lower() == "n":
                    delete_source = False
                    break
                else:
                    print("Invalid input: please enter 'Y' to delete them, or 'n' to keep them.")

            for rid, ds in region_dsets.items():
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

                print(f"✅ Aggregation done for region {rid}")

                save_path = self.write_aggregated_ds(
                    agg_ds=agg_ds,
                    output_name=f"{output_name}_{rid}",
                    aggregation_type=aggregation_type,
                    delete_source=delete_source
                )

                print(f"✅ Aggregation saved to {save_path}")

    def run_point_process(
        self,
        data_file: str,
        merged_ds: xr.Dataset,
        preds: list[str],
        start: str,
        end: str,
        region_id: str,
        gapfilling: bool,
        output_name: str
    ) -> None:
        """
        Post-processes downloaded data for a single point.
        """
        df_og = self.processor.load_and_filter_dataframe(data_file, start, end)
                
        dsm = self.dataset_manager.apply_column_rename(merged_ds)
        dfm = (dsm.to_dataframe()
               .droplevel("latitude")
               .droplevel("longitude")
        )

        if gapfilling:
            dfr = self.dataset_manager.build_multiindex_dataframe(df_og, preds)
            for pred in preds:
                if pred in dfr.columns.get_level_values('variable'):
                    era5_values = self.processor.convert_ameriflux_to_era5(dfm, pred)
                    dfr.loc[:, (pred, "ERA5")] = era5_values

            cand = ("timestamp", "AMF")
            if cand in dfr.columns:
                ts = pd.to_datetime(dfr.pop(cand), errors="coerce")
                dfr.insert(0, "timestamp", ts)  # put it first as a plain column
                dfr = dfr.set_index("timestamp")  # make it the index
            dfr = dfr.drop(columns=["year", "month", "day", "time"])

            self.dataset_manager.save_output(dfr, output_name)
        else:
            dsm = dsm.drop_vars(["year_month", "lat", "lon"])

            output_name = f"{output_name}_{region_id}"
            save_path = self.write_aggregated_ds(dsm, output_name)
            print(f"✅ File saved to {save_path}")

    def load_features_from_manifest(self):
        """Load manifest file"""
        with open(self.config.OUTPUT_MANIFEST, "r") as fp:
            content = json.load(fp)
        return content

    def open_nc_all(self, output_name: str) -> dict[str, xr.Dataset]:
        """
        Open all NetCDF files for the given output_name (one per region).
        Returns a dict {region_id: Dataset}.
        """
        pattern = str(Path(self.config.OUTPUT_PROCESSED_DIR) / f"{output_name}_*.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found for {output_name} in {self.config.OUTPUT_PROCESSED_DIR}")

        dsets = {}
        for f in files:
            region_id = Path(f).stem.split("_")[-1]  # ex: output_name_region_1 -> "1"
            dsets[region_id] = xr.open_dataset(f, decode_times=True).load()
        return dsets

    def write_aggregated_ds(
        self,
        agg_ds: xr.Dataset,
        output_name: str,
        aggregation_type: str | None = None,
        delete_source: bool | None = None,
    ) -> Path:
        if aggregation_type:
            filename = f"{output_name}_{aggregation_type.lower()}.nc"
        else:
            filename = f"{output_name}.nc"

        path = Path(self.config.OUTPUT_PROCESSED_DIR) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        # Overwrite if exists
        if path.exists():
            print(f"⚠️ Overwriting existing aggregated file: {path}")
            path.unlink()

        # Ensure valid_time is datetime64[ns]
        if "valid_time" in agg_ds.coords:
            agg_ds = agg_ds.assign_coords(
                valid_time=("valid_time", np.array(agg_ds["valid_time"].values, dtype="datetime64[ns]"))
            )

        # Encoding: compress + use float32 instead of float64 where possible
        encoding = {}
        for v in agg_ds.data_vars:
            enc = {"zlib": True, "complevel": 4}
            if str(agg_ds[v].dtype).startswith("float64"):
                enc["dtype"] = np.float32
            encoding[v] = enc

        agg_ds.to_netcdf(path, encoding=encoding, engine="netcdf4")

        # Handle None for delete_source (only act if explicitly True)
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