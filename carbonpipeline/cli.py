# carbonpipeline/cli.py
import argparse
import asyncio

from .constants import *
from .core import CarbonPipeline


class ArgumentParserManager: 
    """Handles command line argument parsing."""
    
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Build the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="carbonpipeline",
            description="Pipeline to retrieve and compute AmeriFlux variables based on ERA5 variables.",
            epilog="More information available on GitHub."
        )
        subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
        ArgumentParserManager._add_point_subparser(subparsers)
        ArgumentParserManager._add_area_subparser(subparsers)
        return parser

    @staticmethod
    def _add_point_subparser(subparsers):
        """Add point subparser."""
        p = subparsers.add_parser("point", help="Work with data for a specific point (lat/lon)")
        p.add_argument("--output-format", required=True, choices=['csv', 'netcdf'], help="Desired output format")
        p.add_argument("--file", required=True, type=str, help="Path to the dataset file")
        p.add_argument("--coords", required=True, nargs=2, type=float, metavar=("LAT", "LON"), help="Latitude and longitude coordinates")
        p.add_argument("--start", required=True, type=str, help="Start date (YYYY-MM-DDTHH:MM:SS)")
        p.add_argument("--end", required=True, type=str, help="End date (YYYY-MM-DDTHH:MM:SS)")
        p.add_argument("--preds", required=False, nargs='*', default=[], help="List of predictors (e.g., TA RH CO2)")

    @staticmethod
    def _add_area_subparser(subparsers):
        a = subparsers.add_parser("area", help="Work with data over a bounding box in two stages (download / process)", epilog="Two stages are provided because downloading large datasets can be time-consuming.")
        actions = a.add_subparsers(dest="action", required=True)

        dl = actions.add_parser("download", help="Download ERA5 files for a bounding box")
        dl.add_argument("--coords", required=False, nargs=4, type=float, metavar=('NORTH', 'WEST', 'SOUTH', 'EAST'), help="Geographical bounding box")
        dl.add_argument("--start", required=True, type=str, help="Start date (YYYY-MM-DDTHH:MM:SS)")
        dl.add_argument("--end", required=True, type=str, help="End date (YYYY-MM-DDTHH:MM:SS)")
        dl.add_argument("--preds", required=False, nargs='*', default=[], help="List of predictors (e.g., TA RH CO2)")

        proc = actions.add_parser("process", help="Process previously downloaded ERA5 folders for a bounding box", epilog="The output format can only be NetCDF for compressibility reasons.")
        proc.add_argument("--name", required=True, type=str, help="Name of the output file (without extension). Overwrites if it already exists.")

    @staticmethod
    def validate_and_prepare(args, parser):
        args.start = ' '.join(args.start.replace("T", " ").split())
        args.end = ' '.join(args.end.replace("T", " ").split())

        current_preds = list(args.preds)
        needs_wtd = "WTD" in current_preds
        needs_co2 = "CO2" in current_preds
        
        # CO2 is both a predictor from ERA5 and from its own source, so we don't remove it from the list for ERA5 processing.

        invalid = [p for p in current_preds if p not in VARIABLES_FOR_PREDICTOR]
        if invalid:
            parser.error(
                f"\nInvalid predictor(s): {', '.join(invalid)}\n"
                f"Valid options are: {', '.join(VARIABLES_FOR_PREDICTOR)}"
            )

        if not current_preds and not needs_wtd:
            print("No predictors specified, processing all available ERA5 predictors.")
            vars_ = ERA5_VARIABLES
            args.preds = list(VARIABLES_FOR_PREDICTOR)
        else:
            needed = {var for pred in current_preds for var in VARIABLES_FOR_PREDICTOR[pred]}
            vars_ = list(needed)
            args.preds = current_preds

        if args.command == "area" and args.coords is None:
            args.coords = [90, -180, -90, 180] # Default to global

        if "xco2" in vars_:
            vars_.remove("xco2") # else the ERA5 request will fail because xco2 doesn't exist within the dataset

        if "wtd" in vars_:
            vars_.remove("wtd") # the variables still need to be included in the manifest.json

        return vars_, needs_wtd, needs_co2 
    
    @staticmethod
    def pretty_print_inputs(title: str, **fields):
        print(f"\n------------------- {title.upper()} -------------------")
        for k, v in fields.items():
            print(f"- {k:<15}: {v}")
        print("--------------------------------------------------\n")


async def main():
    """The main asynchronous logic of the pipeline."""
    pipeline = CarbonPipeline()

    parser = ArgumentParserManager.build_parser()
    args = parser.parse_args()

    if args.command == "area" and args.action == "process":
        ArgumentParserManager.pretty_print_inputs("Processing Area Data", OutputFile=f"{args.name}.nc")
        pipeline.run_area_process(args.name)
        return
        
    vars_, needs_wtd, needs_co2 = ArgumentParserManager.validate_and_prepare(args, parser)

    pipeline.setup_dirs(pipeline.config.ZIP_DIR, pipeline.config.UNZIP_DIR)
    
    main_tasks = []
    if needs_co2:
        await pipeline.downloader.download_co2_data() # Can't be done async or else ERA5 will fail
    if needs_wtd:
        main_tasks.append(pipeline.downloader.web_scraping_wtd(args.start, args.end))

    if args.command == "point":
        ArgumentParserManager.pretty_print_inputs(
            "Point Command Inputs", File=args.file, Coordinates=args.coords,
            Start=args.start, End=args.end, Predictors=args.preds, ERA5_Vars=vars_
        )
        main_tasks.append(pipeline.run_point_download(
            args.file, args.output_format, args.coords, args.start, args.end, 
            args.preds, vars_, needs_wtd, needs_co2
        ))
    elif args.command == "area" and args.action == "download":
        ArgumentParserManager.pretty_print_inputs(
            "Area Download Inputs", Area=args.coords, Start=args.start, 
            End=args.end, Predictors=args.preds, ERA5_Vars=vars_
        )
        main_tasks.append(pipeline.run_area_download(
            args.coords, args.start, args.end, args.preds, vars_
        ))

    await asyncio.gather(*main_tasks)

    print("\n✅ All tasks completed.")


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