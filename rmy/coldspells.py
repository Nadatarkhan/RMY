
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.stats import genpareto

def calculate_daily_tmin_tmax(epw_data):
    epw_data['temp_air'] = pd.to_numeric(epw_data['temp_air'], errors='coerce')
    epw_data['wind_speed'] = pd.to_numeric(epw_data['wind_speed'], errors='coerce')
    daily = epw_data.groupby(['year', 'month', 'day']).agg({
        'temp_air': ['min', 'max'],
        'wind_speed': 'mean'
    }).reset_index()
    daily.columns = ['YEAR', 'MONTH', 'DAY', 'Tmin (oC)', 'Tmax (oC)', 'Wind Speed (m/s)']
    return daily

def identify_cold_spells(df, method='ensemble', z_thresh=0.7, tmin_thres=-7, wind_speed_thres=2, window=30, max_duration=21):
    if method == 'temperature':
        condition = df['Tmin (oC)'] < tmin_thres
    elif method == 'percentile':
        condition = df['Tmin (oC)'] < df['Tmin (oC)'].quantile(0.1)
    elif method == 'ensemble':
        cond1 = df['Tmin (oC)'] < tmin_thres
        cond2 = df['Tmin (oC)'] < df['Tmin (oC)'].quantile(0.1)
        condition = cond1 | cond2
    elif method == 'gnn':
        df['z'] = (df['Tmin (oC)'] - df['Tmin (oC)'].rolling(window).mean()) / df['Tmin (oC)'].rolling(window).std()
        df['z'] = df['z'].fillna(0)
        df['is_winter'] = df['MONTH'].isin([12, 1, 2])
        condition = (df['z'] < -z_thresh) & (df['Wind Speed (m/s)'] > wind_speed_thres) & df['is_winter']
    else:
        raise ValueError("Invalid method")

    spells = []
    grouped = condition.groupby((condition != condition.shift()).cumsum()).cumsum()
    for start in grouped[grouped == 1].index:
        end = start
        while end + 1 < len(grouped) and grouped[end + 1] > 0:
            end += 1
        duration = end - start + 1
        if 3 <= duration <= max_duration:
            spells.append((start, end))

    return pd.DataFrame(spells, columns=['Start', 'End']) if spells else pd.DataFrame(columns=['Start', 'End'])

def identify_evt_cold_extremes(df, quantile_threshold=0.05, min_duration=3):
    tmin_series = df['Tmin (oC)'].dropna()
    threshold = tmin_series.quantile(quantile_threshold)
    excesses = -(tmin_series[tmin_series < threshold] - threshold)
    if len(excesses) < 5:
        return pd.DataFrame(columns=['Start', 'End'])
    c, loc, scale = genpareto.fit(excesses, floc=0)
    return_level = threshold - genpareto.ppf(0.98, c, loc=0, scale=scale)
    condition = df['Tmin (oC)'] < return_level
    condition = condition.fillna(False)
    spells = []
    grouped = condition.groupby((condition != condition.shift()).cumsum()).cumsum()
    for start in grouped[grouped == 1].index:
        end = start
        while end + 1 < len(grouped) and condition.iloc[end + 1]:
            end += 1
        duration = end - start + 1
        if duration >= min_duration:
            spells.append((start, end))
    return pd.DataFrame(spells, columns=['Start', 'End']) if spells else pd.DataFrame(columns=['Start', 'End'])
