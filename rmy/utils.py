import os
import os.path
from os import path
#!pip install pandas==1.3.4  # Replace with the appropriate version as needed- this is the one that is compatible
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import genextreme as gev

import matplotlib.font_manager as fm
from scipy.stats import genpareto

# === Cell Separator ===

import os
import pandas as pd

import pandas as pd

# === Shared EPW column names ===
epw_columns = [
    'year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air',
    'temp_dew', 'relative_humidity', 'atmospheric_pressure', 'etr', 'etrn',
    'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
    'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance',
    'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
    'visibility', 'ceiling_height', 'present_weather_observation',
    'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth',
    'snow_depth', 'days_since_last_snowfall', 'albedo',
    'liquid_precipitation_depth', 'liquid_precipitation_quantity'
]

def load_epw(epw_path):
    return pd.read_csv(epw_path, skiprows=8, header=None, names=epw_columns)

def load_epw_daily_stats(epw_path):
    df = pd.read_csv(epw_path, skiprows=8, header=None)
    df.columns = ['year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air',
                  'temp_dew', 'relative_humidity', 'atmospheric_pressure', 'etr', 'etrn',
                  'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
                  'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance',
                  'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
                  'visibility', 'ceiling_height', 'present_weather_observation',
                  'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth',
                  'snow_depth', 'days_since_last_snowfall', 'albedo',
                  'liquid_precipitation_depth', 'liquid_precipitation_quantity']

    df['temp_air'] = pd.to_numeric(df['temp_air'], errors='coerce')
    daily = df.groupby(['year', 'month', 'day'])['temp_air'].agg(['max', 'min']).reset_index()

    max_temp = daily['max'].max()
    min_temp = daily['min'].min()
    days_above_35 = (daily['max'] > 30).sum()
    days_below_0 = (daily['min'] < 0).sum()

    return {
        'max_temp': max_temp,
        'min_temp': min_temp,
        'days_above_35C': days_above_35,
        'days_below_0C': days_below_0
    }


# === Cell Separator ===

