# carbonpipeline/processing_utils.py
import numpy as np
from carbonpipeline.constants import *

# -------------- Conversion --------------
def kelvin_to_celsius(T_K):
    return T_K - ZERO_C_IN_K


def pa_to_kpa(p_pa):
    return p_pa / 1000


def kpa_to_pa(p_kpa):
    return p_kpa * 1000


def kpa_to_hpa(p_kpa):
    return p_kpa * 10


def volumetric_soil_water(SWC_decimal):
    return SWC_decimal * 100

# -------------- Variable processing --------------
def wind_speed_magnitude(u, v):
    return np.hypot(u, v)


def wind_speed_direction(u, v):
    return np.arctan2(v, u)


def relative_humidity(T_air_C, T_dew_C):
    """
    Source: https://arc.net/l/quote/lrazgyii
    """
    a, b = 17.625, 243.04
    gamma_air = (a * T_air_C)  / (b + T_air_C)
    gamma_dew = (a * T_dew_C)  / (b + T_dew_C)
    return 100 * np.exp(gamma_dew - gamma_air)


def vapor_pressure_deficit(RH_percent, T_air_K):
    RH = RH_percent / 100
    es_kpa = saturated_vapor_pressure(kelvin_to_celsius(T_air_K))
    vpd_kpa = es_kpa * (1 - RH)
    return kpa_to_hpa(vpd_kpa)


def saturated_vapor_pressure(T_air_C):
    """
    Tetens formula: https://en.wikipedia.org/wiki/Tetens_equation
    """
    a = np.where(T_air_C >= 0, 17.27, 21.875)
    b = np.where(T_air_C >= 0, 237.3, 265.5)
    return 0.61078 * np.exp(a * T_air_C / (T_air_C + b))


def shortwave_out(SW_in, albedo):
    return SW_in * albedo


def longwave_out(LW_in, albedo):
    return LW_in * albedo


def net_radiation(SW_in, LW_in, SW_out, LW_out):
    return SW_in + LW_in - SW_out - LW_out


def dry_to_wet_co2_fraction(T_air_K, RH_percent, p_air_pa, XCO2_dry):
    RH = RH_percent / 100
    T_air_C = kelvin_to_celsius(T_air_K)
    es_pa = kpa_to_pa(saturated_vapor_pressure(T_air_C))

    xH2O_wet = RH * es_pa / p_air_pa 
    xdry_wet = 1 - xH2O_wet
    xH2O_dry = xH2O_wet / xdry_wet

    n_tot = (DRY_AIR_MOLE_FRACTION_N2 
             + DRY_AIR_MOLE_FRACTION_O2 
             + DRY_AIR_MOLE_FRACTION_AR 
             + XCO2_dry / 1e6
             + xH2O_dry)
    
    return XCO2_dry / n_tot

