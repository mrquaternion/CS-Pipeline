# carbonpipeline/cli.py
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from .argparser import ArgumentParserManager
from .Geometry.geometry_processor import GeometryProcessor
from .Geometry.geometry import Geometry, GeometryType
from .Processing.constants import *
from .core import CarbonPipeline


FEATURES_LENGTH_THRESHOLD = 1000


class CommandExecutorError(Exception):
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
        self.output_suffix = config_dict.get("output-filename")
        self.data_file = config_dict.get("data-file")
        self.coords_file = config_dict.get("coords-file")
        self.start = config_dict.get("start")
        self.end = config_dict.get("end")
        self.preds = config_dict.get("preds")
        self.geometries: list[Geometry] = []
        self.global_geometry: [Geometry] = None
        self.special_preds: SpecialPredictors | None = None # SpecialPredictors object
        self.vars: list[str] | None = None # List of variables to download from ERA5
        self.is_features_threshold_toggled = False

    @property
    def number_requests_per_region(self):
        return (self.end - self.start).total_seconds() / 3600

    # Only callable function
    async def run(self):
        match self.action:
            case "download":
                self._prepare_download_inputs()
                ArgumentParserManager.pretty_print_inputs(
                    "Downloading Data Step",
                    StartDate=self.start, 
                    EndDate=self.end, 
                    AMFPredictors=self.preds
                )
                await self._downloading_step()
            case "process":
                ArgumentParserManager.pretty_print_inputs(
                    "Processing Data Step", 
                    OutputDirectory=self.pipeline.config.OUTPUT_PROCESSED_DIR,
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

        # Download WTD/CO2 data ONCE at the beginning (global datasets)
        global_tasks = await self.special_preds.download_required_data(
            self.pipeline, self.start, self.end
        )
        if global_tasks:
            await asyncio.gather(*global_tasks)

        geometries = self.geometries if self.global_geometry is None else self.global_geometry
        enum_geometries = enumerate(geometries)
        # Download ERA5 data sequentially for each region (to avoid CDS conflicts)
        for geometry_idx, geometry in enum_geometries:
            for region in geometry.rect_regions:
                region_id = CommandExecutor._generate_region_id(region, geometry_idx)
                await self._download_for_region(geometry, region, region_id)

    def _processing_step(self):
        """
        Logic for the processing step.
        """
        self.pipeline.setup_manifest_and_dirs(None, self.pipeline.config.OUTPUT_PROCESSED_DIR)

        all_features = self.pipeline.load_features_from_manifest()
        for i in range(len(all_features)):
            preds = all_features[i]["preds"]
            start = all_features[i]["start_date"]
            end = all_features[i]["end_date"]
            geometry = all_features[i]["geometry"]
            unzip_dirs = all_features[i]["unzip_sub_folders"]
            region_id = all_features[i]["region_id"]
            ds = self.pipeline.dataset_manager.merge_unzipped(unzip_dirs)

            if not self.output_suffix:
                self.output_suffix = "output"
            output_name = "_".join([self.output_suffix , region_id])

            match geometry:
                case GeometryType.POINT.value:
                    if self.data_file:
                        self.pipeline.run_point_process(self.data_file, ds, preds, start, end, output_name)
                    else:
                        # fallback if the client doesn't want gap-filling to the given dataset
                        self.pipeline.run_area_process(ds, preds, start, end, output_name)
                case _:
                    self.pipeline.run_area_process(ds, preds, start, end, output_name)

    @staticmethod
    def _generate_region_id(region: list[float], geometry_idx: int) -> str:
        """
        Generate a unique identifier for each region.
        """
        lat_range = f"{region[2]:.1f}to{region[0]:.1f}"
        lon_range = f"{region[1]:.1f}to{region[3]:.1f}"
        return f"r{geometry_idx}_{lat_range}_{lon_range}"
            
    def _prepare_download_inputs(self):
        """
        Cleaning, parsing, filling all variables passed through the config file.
        """
        # Obtain the rectangular regions for ERA5
        if self.coords_file is None:
            global_earth_bounding_box = [90, -180, -90, 180]
            geometry = Geometry()
            geometry.rect_regions = [global_earth_bounding_box]
            self.geometries = [geometry]
        else:
            self.geometries = self._parse_geojsons()
            rect_regions_total = 0
            for geo in self.geometries:
                geo.rect_regions.extend(GeometryProcessor.process_geometry(geo))
                rect_regions_total += len(geo.rect_regions)
            """ Case where there is too many polygons: IMPORTANT """

            if (rect_regions_total * self.number_requests_per_region) > FEATURES_LENGTH_THRESHOLD:
                print(f"Number of total requests: {rect_regions_total * self.number_requests_per_region}")
                geometry = Geometry()
                geometry.rect_regions = [self._find_global_covering_region()]
                self.global_geometry = [geometry]
                self.is_features_threshold_toggled = True # let know the pipeline to download in the manifest

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

    def _parse_geojsons(self) -> list:
        """
        Parse the coordinate input provided by the user.
        """
        path = Path(self.coords_file)

        # If the provided path is a directory
        geometry_collection: list = []
        if path.is_dir():
            for file_path in sorted(path.iterdir()):
                # Only consider files with recognised extensions
                if file_path.suffix not in (".geojson", ".json"):
                    continue
                with open(file_path, "r") as f:
                    json_dict = json.load(f)
                    if not json_dict.get("features"):
                        raise ValueError(f"No features found in GeoJSON file: {file_path}")
                    features = json_dict["features"]
                    print(f"\nNumber of features?: {len(features)}")

                    for feature in features:
                        geometry_type = feature["geometry"]["type"]
                        coordinates = feature["geometry"]["coordinates"]

                        geometry = Geometry(data=coordinates)
                        geometry.validate_coordinates()
                        geometry_collection.append(geometry)

                        print(f"\nGeoJSON type: {geometry_type}")
                        print(f"Processed type: {geometry.geom_type}, type signature: {geometry.type_signature}")

            if not geometry_collection:
                raise ValueError(f"No valid GeoJSON files found in directory: {path}")
        return geometry_collection

    def _find_global_covering_region(self) -> list[float]:
        rects = self._all_rect_regions()
        if not rects:
            raise ValueError("No rect regions available to build a global covering region.")
        max_lat = max(r[0] for r in rects)  # N
        min_lon = min(r[1] for r in rects)  # W
        min_lat = min(r[2] for r in rects)  # S
        max_lon = max(r[3] for r in rects)  # E
        return [max_lat, min_lon, min_lat, max_lon]

    def _all_rect_regions(self) -> list[list[float]]:
        return [
            region
            for g in self.geometries
            for region in getattr(g, "rect_regions", []) or []
        ]

    async def _download_for_region(self, geometry, region, region_id: str):
        """Download ERA5 data for a single region. Runs sequentially to avoid CDS conflicts."""
        print(f"⬇️ Downloading ERA5 data for {region_id}...")
        await self.pipeline.run_download(
            region, region_id, geometry, self.start, self.end,
            self.preds, self.vars, self.is_features_threshold_toggled, self._all_rect_regions()
        )
                

async def main():
    parser = ArgumentParserManager.build_parser()
    args = parser.parse_args()

    config = ArgumentParserManager.load_yaml_config(args.config)
    ce = CommandExecutor(config_dict=config)

    if ce.action == "process":
        unzip_dir = ce.pipeline.config.UNZIP_DIR
        if not os.path.exists(unzip_dir):
            raise CommandExecutorError(f"Unzip directory does not exist: {unzip_dir}. Please download data first.")
        if not os.listdir(unzip_dir):
            raise CommandExecutorError("No downloads found in the unzip directory. Please download data first.")

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