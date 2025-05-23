# carbonpipeline/api_request.py
import os
import cdsapi
from carbonpipeline.constants import *


class APIRequest:
    """
    Represents a request to the ERA5 dataset for a specific date, location and set of variables.

    Parameters:
        year (str): Year of the request.
        month (str): Month of the request.
        day (str): Day of the request.
        time (list[str]): Time in "HH:MM" format.
        lat (float): Latitude of the site.
        lon (float): Longitude of the site.
        preds (list[str) | None): Optional list of high-level predictors under AMERIFLUX naming.
    """
    def __init__(self, year: str, month: str, day: str, time: list[str], lat: float, lon: float, preds: list[str]):
        self.year = year
        self.month = month
        self.day = day
        self.time = time
        self.lat = lat
        self.lon = lon
        self.preds = preds

    def fetch_download(self, dir_: str):
        """
        Constructs and submits a download request to the CDS API for ERA5 single-level reanalysis data.

        Returns:
            None. The file is downloaded directly to disk, in the `./datasets/` directory.
        """
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type":    ["reanalysis"],
            "variable":        self.preds,
            "year":            [self.year],
            "month":           [self.month],
            "day":             [self.day],
            "time":            self.time,
            "area":            [self.lat + 0.125, self.lon - 0.125, self.lat - 0.125, self.lon + 0.125],
            "data_format":     "grib",
            "download_format": "unarchived"
        }

        client = cdsapi.Client(wait_until_complete=False, delete=False)
        result = client.retrieve(dataset, request)

        end_dir_ = f"data_{self.year}-{self.month}-{self.day}.grib"
        out_path = os.path.join(dir_, end_dir_)

        result.download(out_path)


