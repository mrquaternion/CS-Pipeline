# carbonpipeline/constants.py

from carbonpipeline.processing_utils import *

ZERO_C_IN_K = 273.15

DRY_AIR_MOLE_FRACTION_N2 = 0.7808
DRY_AIR_MOLE_FRACTION_O2 = 0.2095
DRY_AIR_MOLE_FRACTION_AR = 0.0093

ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_downward_short_wave_radiation_flux_clear_sky",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_latent_heat_flux",
    "mean_surface_sensible_heat_flux",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "forecast_albedo",
    "friction_velocity"
]

SHORTNAME_TO_FULLNAME = {
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    't2m': '2m_temperature',
    'd2m': '2m_dewpoint_temperature',
    'sp': 'surface_pressure',
    'tp': 'total_precipitation',
    'avg_sdlwrf': 'mean_surface_downward_long_wave_radiation_flux',
    'avg_sdswrf': 'mean_surface_downward_short_wave_radiation_flux',
    'avg_sdswrfcs': 'mean_surface_downward_short_wave_radiation_flux_clear_sky',
    'avg_snlwrf': 'mean_surface_net_long_wave_radiation_flux',
    'avg_slhtf': 'mean_surface_latent_heat_flux',
    'avg_ishf': 'mean_surface_sensible_heat_flux',
    'stl1': 'soil_temperature_level_1',
    'stl2': 'soil_temperature_level_2',
    'stl3': 'soil_temperature_level_3',
    'swvl1': 'volumetric_soil_water_layer_1',
    'swvl2': 'volumetric_soil_water_layer_2',
    'swvl3': 'volumetric_soil_water_layer_3',
    'fal': 'forecast_albedo',
    'zust': 'friction_velocity'
}