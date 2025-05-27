# carbonpipeline/constants.py

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

VARIABLES_FOR_PREDICTOR = {
    "TA":        ['2m_temperature'],
    "P":         ['total_precipitation'],
    "RH":        ['2m_temperature', '2m_dewpoint_temperature'],
    "VPD":       ['2m_temperature', '2m_dewpoint_temperature'],
    "PA":        ['surface_pressure'],
    "CO2":       ['2m_temperature', '2m_dewpoint_temperature', 'surface_pressure'], # Make another request to https://cds.climate.copernicus.eu/datasets/satellite-carbon-dioxide?tab=overview
    "SW_IN":     ['mean_surface_downward_short_wave_radiation_flux'],
    "SW_IN_POT": ['mean_surface_downward_short_wave_radiation_flux_clear_sky'],
    "SW_OUT":    ['mean_surface_downward_short_wave_radiation_flux', 'forecast_albedo'],
    "LW_IN":     ['mean_surface_downward_long_wave_radiation_flux'],
    "LW_OUT":    ['mean_surface_downward_long_wave_radiation_flux', 'forecast_albedo'],
    "NETRAD":    ['mean_surface_downward_short_wave_radiation_flux', 'mean_surface_downward_long_wave_radiation_flux', 'forecast_albedo'],
    "WS":        ['10m_u_component_of_wind', '10m_v_component_of_wind'],
    "WD":        ['10m_u_component_of_wind', '10m_v_component_of_wind'],
    "USTAR":     ['friction_velocity'],
    "SWC_1":     ['volumetric_soil_water_layer_1'],
    "SWC_2":     ['volumetric_soil_water_layer_1'],
    "SWC_3":     ['volumetric_soil_water_layer_2'],
    "SWC_4":     ['volumetric_soil_water_layer_2'],
    "SWC_5":     ['volumetric_soil_water_layer_3'],
    "TS_1":      ['soil_temperature_level_1'],
    "TS_2":      ['soil_temperature_level_1'],
    "TS_3":      ['soil_temperature_level_2'],
    "TS_4":      ['soil_temperature_level_2'],
    "TS_5":      ['soil_temperature_level_3'],
    "WTD":       [''], # Make another request to https://github.com/UU-Hydro/GLOBGM 
    "G":         ['mean_surface_sensible_heat_flux', 'mean_surface_latent_heat_flux', 'mean_surface_downward_short_wave_radiation_flux', 'mean_surface_downward_long_wave_radiation_flux', 'forecast_albedo'],
    "H":         ['mean_surface_sensible_heat_flux'],
    "LE":        ['mean_surface_latent_heat_flux']
}

COLUMN_NAME_MAPPING = {
    'CO2_F_MDS': 'CO2',
    'G_F_MDS':   'G',
    'H_F_MDS':   'H',
    'LE_F_MDS':  'LE',
    'LW_IN_F':   'LW_IN',
    'LW_OUT':    'LW_OUT',
    'NETRAD':    'NETRAD',
    'PA_F':      'PA',
    'PPFD_IN':   'PPFD_IN',
    'PPFD_OUT':  'PPFD_OUT',
    'P_F':       'P',
    'RH':        'RH',
    'SW_IN_F':   'SW_IN',
    'SW_OUT':    'SW_OUT',
    'TA_F':      'TA',
    'USTAR':     'USTAR',
    'VPD_F':     'VPD',
    'WD':        'WD',
    'WS_F':      'WS',
    'timestamp': 'timestamp'
}

SHORTNAME_TO_FULLNAME = {
    '10u': '10m_u_component_of_wind',
    '10v': '10m_v_component_of_wind',
    '2t': '2m_temperature',
    '2d': '2m_dewpoint_temperature',
    'sp': 'surface_pressure',
    'tp': 'total_precipitation',
    'avg_sdlwrf': 'mean_surface_downward_long_wave_radiation_flux',
    'avg_sdswrf': 'mean_surface_downward_short_wave_radiation_flux',
    'avg_sdswrfcs': 'mean_surface_downward_short_wave_radiation_flux_clear_sky',
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