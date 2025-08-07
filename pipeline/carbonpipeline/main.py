# carbonpipeline/cli.py
import asyncio
import json
import os
from pathlib import Path
from typing import Union 

from .argparser import ArgumentParserManager
from .Geometry.geometry_processor import GeometryProcessor
from .Geometry.geometry import Geometry, GeometryType
from .Processing.constants import *
from .core import CarbonPipeline


class PipelineError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message
    

class SpecialPredictors:
    def __init__(self, predictors: list[str]):
        self.requires_wtd_data = "WTD" in predictors
        self.requires_co2_data = "CO2" in predictors

    async def download_required_data(self, pipeline, start, end):
        tasks = []

        if self.requires_co2_data:
            print("⬇️ Downloading CO2 data...")
            tasks.append(asyncio.create_task(
                pipeline.downloader.download_co2_data()
            ))
        
        if self.requires_wtd_data:
            print("⬇️ Download WTD data...")
            tasks.append(asyncio.create_task(
                pipeline.downloader.download_wtd_data(start, end)
            ))

        return tasks


class CommandExecutor:
    def __init__(self, config_dict: dict):
        self.pipeline = CarbonPipeline()

        self.action = config_dict.get("action")
        self.output_filename = config_dict.get("output-filename")
        self.data_file = config_dict.get("data-file")
        self.coords_file = config_dict.get("coords-file")
        self.start = config_dict.get("start")
        self.end = config_dict.get("end")
        self.preds = config_dict.get("preds")

        self.geometry_struct = None # Geometry object
        self.rect_regions = None # List of [float] || [int]
        self.special_preds = None # SpecialPredictors object
        self.vars = None # List of variables to download from ERA5

    # Only callable function
    async def run(self):
        match self.action:
            case "download":
                self._prepare_download_inputs()
                ArgumentParserManager.pretty_print_inputs(
                    "Downloading Data Step", 
                    GeometryType=self.geometry_struct.geom_type.value, 
                    StartDate=self.start, 
                    EndDate=self.end, 
                    AMFPredictors=self.preds
                )
                await self._downloading_step()
            case "process":
                ArgumentParserManager.pretty_print_inputs(
                    "Processing Data Step", 
                    OutputDirectory=f"{self.output_filename}.nc", 
                    DataFile=self.data_file
                )
                self._processing_step()
            case _:
                raise ValueError(f"Unknown action: {self.action}")

    async def _downloading_step(self):
        """
        Logic for the downloading step.
        """
        self.pipeline.setup_manifest_and_dirs(
            self.pipeline.config.OUTPUT_MANIFEST, 
            self.pipeline.config.ZIP_DIR, 
            self.pipeline.config.UNZIP_DIR
        )
        regions_list = self._get_regions_list()

        # Download WTD/CO2 data ONCE at the beginning (global datasets)
        global_tasks = await self.special_preds.download_required_data(
            self.pipeline, self.start, self.end
        )
        if global_tasks:
            await asyncio.gather(*global_tasks)

        # Download ERA5 data sequentially for each region (to avoid CDS conflicts)
        for region_idx, region in enumerate(regions_list):
            print(f"REGION: {self.rect_regions}")
            region_id = self._generate_region_id(region_idx, region)
            await self._download_for_region(region, region_id)

    def _processing_step(self):
        """
        Logic for the processing step.
        """
        all_features = self.pipeline.load_features_from_manifest()
        for i in range(len(all_features)):
            unzip_dirs = all_features[i]["unzip_sub_folders"]

    def _postdownloading_step(self):
        """
        If the polygon is non-rectangular, then we need to filter the original coordinates from the downloaded data 
        """
        if self.geometry_struct.geom_type != GeometryType.POINT:
            # Remove all the coordinates we are not interested in
            all_features = self.pipeline.load_features_from_manifest()
            for i in range(len(all_features)):
                unzip_dirs = all_features[i]["unzip_sub_folders"]
                poly = self.geometry_struct.data[i]
                ds = self.pipeline.dataset_manager.remove_coordinates(unzip_dirs, poly)

    def _get_regions_list(self) -> list:
        """
        Convert geometry to a consistent list of regions.
        """
        match self.geometry_struct.geom_type:
            case GeometryType.POINT | GeometryType.POLYGON:
                return [self.rect_regions]
            case GeometryType.MULTIPOLYGON:
                return list(self.rect_regions.values())
            case _:
                raise ValueError(f"Unsupported geometry type: {self.geometry_struct.geom_type}")

    def _generate_region_id(self, region_idx: int, region: list[float]) -> str:
        """
        Generate a unique identifier for each region.
        """
        match self.geometry_struct.geom_type:
            case GeometryType.POINT:
                return f"point_{region[0]:.3f}_{region[1]:.3f}"
            case GeometryType.POLYGON:
                lat_range = f"{region[2]:.1f}to{region[0]:.1f}"
                lon_range = f"{region[1]:.1f}to{region[3]:.1f}" 
                return f"polygon_{lat_range}_{lon_range}"
            case GeometryType.MULTIPOLYGON:
                return f"region_{region_idx}"
            case _:
                return f"region_{region_idx}"
            
    def _prepare_download_inputs(self):
        """
        Cleaning, parsing, filling all variables passed through the config file.
        """
        if self.coords_file is None:
            bounding_box = [90, -180, -90, 180]
            self.geometry_struct, self.rect_regions = Geometry(data=bounding_box), bounding_box # Geometry.data == self.rect_regions
        else:
            coords_raw = self._parse_coords_file(self.coords_file)
            self.geometry_struct, self.rect_regions = GeometryProcessor.process_geometry(coords_raw)

        # Clean the dates
        self.start = self.start.strftime("%Y-%m-%d %H:%M:%S")
        self.end = self.end.strftime("%Y-%m-%d %H:%M:%S")

        # Make a list out of the predictors
        current_preds = list(self.preds or [])

        # Check if any is not supported
        invalid = [p for p in current_preds if p not in VARIABLES_FOR_PREDICTOR]
        if invalid:
            raise ValueError(f"Invalid predictors: {invalid}")
        
        if not current_preds: # Case if no predictors has been specified in the config file
            self.vars = ERA5_VARIABLES
            self.preds = list(VARIABLES_FOR_PREDICTOR)
        else:
            self.vars = list({var for pred in current_preds for var in VARIABLES_FOR_PREDICTOR[pred]})
            self.preds = current_preds

        self.special_preds = SpecialPredictors(predictors=self.preds)

        if "xco2" in self.vars:
            self.vars.remove("xco2") # If we don't do that, the ERA5 request will fail because xco2 doesn't exist within this particular dataset
        if "wtd" in self.vars:
            self.vars.remove("wtd") # Same here

    def _parse_coords_file(self, pathStr: str) -> list:
        path = Path(pathStr)
        with open(path, "r") as f:
            if path.suffix == ".geojson":
                json_dict = json.load(f)
                
                # For now, handle only the first feature
                if not json_dict.get("features"):
                    raise ValueError("No features found in GeoJSON file")
                
                feature = json_dict["features"][0]
                geometry_type = feature["geometry"]["type"]
                coordinates = feature["geometry"]["coordinates"]
                
                print(f"Geometry type: {geometry_type}")
                print(f"Coordinates structure: {type(coordinates)} with {len(coordinates)} elements")
                
                return coordinates
            else:
                raise ValueError(f"❌ Unsupported file format: {path.suffix}")
            
    async def _download_for_region(self, region, region_id: str):
        """Download ERA5 data for a single region. Runs sequentially to avoid CDS conflicts."""
        print(f"⬇️ Downloading ERA5 data for {region_id}...")
        await self.pipeline.run_download(
            region, region_id, self.start, self.end,
            self.preds, self.vars
        )
                

async def main():
    parser = ArgumentParserManager.build_parser()
    args = parser.parse_args()

    config = ArgumentParserManager.load_yaml_config(args.config)
    ce = CommandExecutor(config_dict=config)

    if ce.action == "process":
        unzip_dir = ce.pipeline.config.UNZIP_DIR
        if not os.path.exists((unzip_dir)):
            raise PipelineError(f"Unzip directory does not exist: {unzip_dir}. Please download data first.")
        if not os.listdir(unzip_dir):
            raise PipelineError("No downloads found in the unzip directory. Please download data first.")

    await ce.run()


def run():
    """Synchronous entry point that runs the main async function."""
    try:
        asyncio.run(main())
    except (ValueError, FileNotFoundError) as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")


if __name__ == "__main__":
    run()