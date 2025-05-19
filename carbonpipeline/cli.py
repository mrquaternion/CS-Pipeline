# carbonpipeline/cli.py

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline to fill missing values in CarbonSense dataset."
    )
    parser.add_argument("--file", required=True, type=str, help="Path to the dataset file")
    parser.add_argument("--lat", required=True, type=int, help="Latitude coordinate")
    parser.add_argument("--lon", required=True, type=int, help="Longitude coordinate")
    args = parser.parse_args()

    print(f"File: {args.file}")
    print(f"Lat: {args.lat}, Lon: {args.lon}")

if __name__ == "__main__":
    main()
