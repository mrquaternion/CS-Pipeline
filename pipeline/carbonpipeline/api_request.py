# carbonpipeline/api_request.py
import os
import cdsapi
from .Processing.constants import *


CO2_FOLDERNAME = "CO2_2003-2022"

class APIRequest:
    """
    Represents a request to the ERA5 dataset for a specific date, location and set of variables.

    Parameters
    ----------
    year : str 
        Year of the request.
    month : list[str]
        Month of the request.
    day : list[str]
        Day of the request.
    time : list[str] 
        Time in "HH:MM" format.
    coords : list[float]
        Coordinates of the request.
    vars : list[str] | None
        Optional list of high-level variables under ERA5 long version naming.
    """

    def __init__(self, year: str, month: str | list[str], day: str | list[str], time: str | list[str],
        coords: list[float], vars_: list[str]):
        self.year = year
        self.month = month if isinstance(month, list) else [month]
        self.day = day if isinstance(day, list) else [day]
        self.time = time if isinstance(time, list) else [time]
        self.coords = coords
        self.vars = vars_
        self.area = None

    def query_era5(self, zip_dir: str) -> str:
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
            "month":           self.month,
            "day":             self.day,
            "time":            self.time,
            "area":            self.area,
            "data_format":     "netcdf",
            "download_format": "zip"
        }

        client = cdsapi.Client(wait_until_complete=False, delete=False)
        result = client.retrieve(dataset, request)

        filename = self._filename_logic()
        target = os.path.join(zip_dir, filename)

        print(f"\nStarting download for {filename} -> {target}")
        result.download(target)
        print(f"\nFinished download for {filename}")

        return filename

    def _filename_logic(self):
        if len(self.month) == 12 and all(m in [f"{i:02d}" for i in range(1, 13)] for m in self.month):
            if self.day == [f"{d:02d}" for d in range(1, 32)] and self.time == [f"{h:02d}:00" for h in range(24)]:
                filename = f"ERA5_{self.year}_full-year.zip"
            else:
                filename = f"ERA5_{self.year}_custom.zip"

        elif len(self.day) > 1:
            if self.day == [f"{d:02d}" for d in range(1, len(self.day) + 1)] and self.time == [f"{h:02d}:00" for h in
                                                                                               range(24)]:
                filename = f"ERA5_{self.year}-{self.month[0]}_full-month.zip"
            else:
                filename = f"ERA5_{self.year}-{self.month[0]}_days{self.day[0]}to{self.day[-1]}.zip"

        elif len(self.time) == 24:
            filename = f"ERA5_{self.year}-{self.month[0]}-{self.day[0]}_full-day.zip"
        elif len(self.time) > 1:
            filename = f"ERA5_{self.year}-{self.month[0]}-{self.day[0]}T{self.time[0]}to{self.time[-1]}.zip"
        else:
            filename = f"ERA5_{self.year}-{self.month[0]}-{self.day[0]}T{self.time[0]}.zip"
        return filename

    @classmethod
    def query_co2(self, zip_dir: str) -> None:
        dataset = "satellite-carbon-dioxide"
        request = {
            "processing_level": ["level_3"],
            "variable": "xco2",
            "sensor_and_algorithm": "merged_obs4mips",
            "version": ["4_5"]
        }

        client = cdsapi.Client()
        result = client.retrieve(dataset, request)

        filename = f"{CO2_FOLDERNAME}.zip"
        target = os.path.join(zip_dir, filename)

        result.download(target)




