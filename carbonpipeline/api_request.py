# carbonpipeline/api_request.py
import os
import cdsapi
from carbonpipeline.constants import *

class APIRequest:
    """
    Represents a request to the ERA5 dataset for a specific date, location and set of variables.

    Parameters
    ----------
    year : str 
        Year of the request.
    month : str 
        Month of the request.
    day : str 
        Day of the request.
    time : list[str] 
        Time in "HH:MM" format.
    coords : list[float]
        Coordinates of the request.
    vars : list[str) | None 
        Optional list of high-level variables under ERA5 long version naming.
    """
    def __init__(self, year: str, month: str, day: str, time: str, coords: list[float], vars_: list[str]):
        self.year = year
        self.month = month
        self.day = day
        self.time = time
        self.coords = coords
        self.vars = vars_
        self.area = None

    def query(self, ZIP_DIR: str):
        """
        Constructs and submits a download request to the CDS API for ERA5 single-level reanalysis data.
        """
        if len(self.coords) == 2:
            self.area = [self.coords[0], self.coords[1], self.coords[0], self.coords[1]]
        elif len(self.coords) == 4:
            self.area = self.coords

        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type":    ["reanalysis"],
            "variable":        self.vars,
            "year":            [self.year],
            "month":           [self.month],
            "day":             [self.day],
            "time":            [self.time],
            "area":            self.area,
            "data_format":     "netcdf",
            "download_format": "zip"
        }

        client = cdsapi.Client(wait_until_complete=False, delete=False)
        result = client.retrieve(dataset, request)

        filename = f"ERA5_{self.year}-{self.month}-{self.day}T{self.time}.zip"
        target = os.path.join(ZIP_DIR, filename)
        
        print(f"Starting download for {self.year}-{self.month}-{self.day}T{self.time} -> {target}")
        result.download(target)
        print(f"Finished download for {self.year}-{self.month}-{self.day}T{self.time}")

        return filename




