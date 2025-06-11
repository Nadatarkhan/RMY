
import pandas as pd

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
    try:
        df = pd.read_csv(epw_path, skiprows=8, header=None, names=epw_columns)
        return df
    except Exception as e:
        print(f"Could not read EPW file {epw_path}: {e}")
        return pd.DataFrame()

def save_csv_safe(df, filename):
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
    else:
        print(f"Skipped empty file: {filename}")
