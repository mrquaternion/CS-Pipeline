# carbonpipeline/api_request.py
import os
import cdsapi

ERA5_PREDICTORS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_downward_long_wave_radiation_flux_clear_sky",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "instantaneous_surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "forecast_albedo",
    "friction_velocity"
]

class APIRequest:
    def __init__(self, year: str, month: str, day: str, time: list[int], lat: float, lon: float):
        self.year = year
        self.month = month
        self.day = day
        self.time = time
        self.lat = lat
        self.lon = lon

    def fetch_download(self):
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type":    ["reanalysis"],
            "variable":        ERA5_PREDICTORS,
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

        dir_ = "./datasets/"
        os.makedirs(dir_, exist_ok=True)
        end_dir_ = f"data_{self.year}-{self.month}-{self.day}.grib"
        out_path = os.path.join(dir_, end_dir_)

        result.download(out_path)


