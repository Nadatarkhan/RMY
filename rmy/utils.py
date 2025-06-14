import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

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

def read_epw_file(epw_file_path):
    return pd.read_csv(epw_file_path, header=None, skiprows=8, sep=',', names=[
        'year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air', 'temp_dew', 'relative_humidity',
        'atmospheric_pressure', 'etr', 'etrn', 'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
        'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance', 'wind_direction', 'wind_speed',
        'total_sky_cover', 'opaque_sky_cover', 'visibility', 'ceiling_height', 'present_weather_observation',
        'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth', 'snow_depth',
        'days_since_last_snowfall', 'albedo', 'liquid_precipitation_depth', 'liquid_precipitation_quantity'
    ])


def load_epw(path):
    return pd.read_csv(path, skiprows=8, header=None, names=epw_columns)

def extract_original_header(epw_path):
    with open(epw_path, 'r') as f:
        return [next(f) for _ in range(8)]

def save_epw(df, original_header_lines, output_path):
    with open(output_path, 'w') as f:
        for line in original_header_lines:
            f.write(line)
        df.to_csv(f, index=False, header=False)

def calculate_heat_index(t, rh):
    hi = 0.5 * (t + 61.0 + ((t-68.0)*1.2) + (rh*0.094))
    return hi

def match_extreme_days(event_df, base_df, n_hours=24):
    matched_days = []
    for _, row in event_df.iterrows():
        day = pd.to_datetime(row['begin_date'], format='%d/%m/%Y')
        match = base_df[
            (base_df['month'] == day.month) &
            (base_df['hour'] == 12)
        ].copy()
        match['abs_diff'] = (match['temp_air'] - base_df.loc[
            (base_df['month'] == day.month) & (base_df['hour'] == 12),
            'temp_air'
        ].mean()).abs()
        if not match.empty:
            best_match = match.sort_values('abs_diff').iloc[0]
            matched_day = base_df[
                (base_df['year'] == best_match['year']) &
                (base_df['month'] == best_match['month']) &
                (base_df['day'] == best_match['day'])
            ]
            matched_days.append(matched_day)
    if matched_days:
        return pd.concat(matched_days)
    return pd.DataFrame(columns=base_df.columns)

def smooth_transition(df, window=3):
    return df.rolling(window=window, min_periods=1, center=True).mean()

def replace_event_days(base_df, insert_df, event_dates):
    new_df = base_df.copy()
    insert_dates = pd.to_datetime(event_dates, format='%d/%m/%Y')
    for insert_date in insert_dates:
        mask = (new_df['month'] == insert_date.month) & (new_df['day'] == insert_date.day)
        if mask.sum() == 24:
            new_df.loc[mask, :] = insert_df.loc[mask, :].values
    return new_df

def seasonal_adjustment(final_df, original_df, variable='temp_air'):
    for month in range(1, 13):
        original_avg = original_df[original_df['month'] == month][variable].mean()
        final_avg = final_df[final_df['month'] == month][variable].mean()
        diff = original_avg - final_avg
        final_df.loc[final_df['month'] == month, variable] += diff
    return final_df
