import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
from .utils import load_epw, calculate_heat_index, smooth_transition, match_extreme_days

import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy.stats import genpareto

warnings.filterwarnings("ignore", category=UserWarning, message="Parsing dates.*")

def calculate_daily_tmin_tmax(epw_data):
    epw_data['temp_air'] = pd.to_numeric(epw_data['temp_air'], errors='coerce')
    epw_data['relative_humidity'] = pd.to_numeric(epw_data['relative_humidity'], errors='coerce')
    daily = epw_data.groupby(['year', 'month', 'day']).agg({
        'temp_air': ['min', 'max'],
        'relative_humidity': 'mean'
    }).reset_index()
    daily.columns = ['YEAR', 'MONTH', 'DAY', 'Tmin (oC)', 'Tmax (oC)', 'Humidity (%)']
    return daily

def calculate_heat_index(temp_air, rh):
    T_F = (temp_air * 9/5) + 32
    hi = (
        -42.379 + 2.04901523*T_F + 10.14333127*rh - 0.22475541*T_F*rh
        - 6.83783e-3*T_F**2 - 5.481717e-2*rh**2
        + 1.22874e-3*T_F**2*rh + 8.5282e-4*T_F*rh**2
        - 1.99e-6*T_F**2*rh**2
    )
    return hi

def identify_heatwaves_static(df, method='ensemble', tmin_th=20, tmax_th=35, max_duration=21):
    if method == 'ensemble':
        df1 = identify_heatwaves_static(df, method='percentile', max_duration=max_duration)
        df2 = identify_heatwaves_static(df, method='temperature', tmax_th=tmax_th, max_duration=max_duration)
        combined = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        return combined
    elif method == 'percentile':
        tmin_q = df['Tmin (oC)'].quantile(0.4)
        tmax_q = df['Tmax (oC)'].quantile(0.9)
        condition = (df['Tmin (oC)'] > tmin_q) & (df['Tmax (oC)'] > tmax_q)
    elif method == 'temperature':
        condition = df['Tmax (oC)'] > tmax_th
    else:
        raise ValueError("Invalid method")

    series = condition.groupby((condition != condition.shift()).cumsum()).cumsum()
    starts = series[series == 1].index
    waves = []

    for start in starts:
        count = 1
        for next_day in range(start + 1, len(df)):
            if series[next_day] > count:
                count += 1
                end = next_day
                while end + 1 < len(df) and series[end + 1] > 0:
                    end += 1
                duration = end - start + 1
                if 3 <= duration <= max_duration:
                    waves.append((start, end))
                break
            else:
                break

    return pd.DataFrame(waves, columns=['Start', 'End']) if waves else pd.DataFrame(columns=['Start', 'End'])


def identify_evt_extremes(df, quantile_threshold=0.95, min_duration=3):
    """
    Identify extreme heatwave events using EVT (Peaks Over Threshold).
    """
    tmax_series = df['Tmax (oC)'].dropna()
    threshold = tmax_series.quantile(quantile_threshold)
    excesses = tmax_series[tmax_series > threshold] - threshold

    if len(excesses) < 5:
        return pd.DataFrame(columns=['Start', 'End'])  # Not enough data to fit GPD

    # Fit Generalized Pareto Distribution to excesses
    c, loc, scale = genpareto.fit(excesses, floc=0)

    # Define a high return level (e.g., 95th percentile of fitted GPD)
    return_level = threshold + genpareto.ppf(0.95, c, loc=0, scale=scale)

    # Flag days exceeding return level
    condition = df['Tmax (oC)'] > return_level
    condition = condition.fillna(False)

    # Group consecutive days into events
    series = condition.groupby((condition != condition.shift()).cumsum()).cumsum()
    starts = series[series == 1].index
    events = []

    for start in starts:
        end = start
        while end + 1 < len(df) and condition.iloc[end + 1]:
            end += 1
        duration = end - start + 1
        if duration >= min_duration:
            events.append((start, end))

    return pd.DataFrame(events, columns=['Start', 'End']) if events else pd.DataFrame(columns=['Start', 'End'])


def apply_gnn_detection(df):
    df.columns = list(range(df.shape[1]))
    df = df.rename(columns={0: 'year', 1: 'month', 2: 'day', 3: 'hour', 6: 'temp_air'})
    df = df[['year', 'month', 'day', 'hour', 'temp_air']].dropna()
    df = df[df['temp_air'] != -9999]
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    daily_max = df.groupby(df['datetime'].dt.date)['temp_air'].max()
    daily_z = (daily_max - daily_max.mean()) / daily_max.std()
    threshold = 2.5
    extreme_days = daily_z[daily_z > threshold]
    return list(pd.to_datetime(extreme_days.index).year)

def calculate_event_stats(heatwave_df, base_df):
    if heatwave_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    events = []
    stats = {}

    for _, row in heatwave_df.iterrows():
        s_idx, e_idx = row['Start'], row['End']
        spell = base_df.iloc[s_idx:e_idx+1]
        year = int(spell.iloc[0]['YEAR'])
        duration = e_idx - s_idx + 1
        avg_tmax = round(spell['Tmax (oC)'].mean(), 1)
        std_tmax = round(spell['Tmax (oC)'].std(), 1)
        max_tmax = round(spell['Tmax (oC)'].max(), 1)
        avg_hi = round(calculate_heat_index(spell['Tmax (oC)'], spell['Humidity (%)']).mean(), 1)

        bdate = datetime(year, int(spell.iloc[0]['MONTH']), int(spell.iloc[0]['DAY']))
        edate = datetime(year, int(spell.iloc[-1]['MONTH']), int(spell.iloc[-1]['DAY']))

        events.append({
            'begin_date': bdate.strftime('%d/%m/%Y'),
            'end_date': edate.strftime('%d/%m/%Y'),
            'duration': duration,
            'avg_tmax': avg_tmax,
            'std_tmax': std_tmax,
            'max_tmax': max_tmax,
            'avg_heat_index': avg_hi,
            'year': year
        })

        if year not in stats:
            stats[year] = {'hwn': 0, 'hwf': 0, 'hwd': 0, 'hwdm': [], 'hwaa': -np.inf}
        stats[year]['hwn'] += 1
        stats[year]['hwf'] += duration
        stats[year]['hwd'] = max(stats[year]['hwd'], duration)
        stats[year]['hwdm'].append(duration)
        stats[year]['hwaa'] = max(stats[year]['hwaa'], max_tmax)

    for y in stats:
        stats[y]['hwdm'] = np.mean(stats[y]['hwdm'])

    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['hwn', 'hwf', 'hwd', 'hwdm', 'hwaa']).reset_index().rename(columns={'index': 'year'}).sort_values(by='year')
    stats_df['year'] = stats_df['year'].astype(int)
    stats_df['severity'] = stats_df['hwf'] * stats_df['hwaa']
    events_df = pd.DataFrame(events).sort_values(by='begin_date')

    peak_year = int(stats_df.loc[stats_df['severity'].idxmax()]['year'])
    #print(f" Highest severity year: {peak_year}")

    return stats_df, events_df, peak_year

def process_file(epw_path, gnn_years=None, apply_gnn=True):
    #print(f" Processing file: {epw_path}")
    epw_df = pd.read_csv(epw_path, skiprows=8, header=None, names=[
        'year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air',
        'temp_dew', 'relative_humidity', 'atmospheric_pressure', 'etr', 'etrn',
        'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
        'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance',
        'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
        'visibility', 'ceiling_height', 'present_weather_observation',
        'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth',
        'snow_depth', 'days_since_last_snowfall', 'albedo',
        'liquid_precipitation_depth', 'liquid_precipitation_quantity'
    ])

    daily_df = calculate_daily_tmin_tmax(epw_df)
    daily_df = daily_df[(daily_df["Tmin (oC)"] != -9999) & (daily_df["Tmax (oC)"] != -9999)]
    daily_df['heat_index'] = calculate_heat_index(daily_df['Tmax (oC)'], daily_df['Humidity (%)'])

    # Static detection
    static_events = identify_heatwaves_static(daily_df, method='ensemble')
    static_stats, static_df, _ = calculate_event_stats(static_events, daily_df)

    # GNN filtering
    if apply_gnn and gnn_years:
        filtered_df = daily_df[daily_df['YEAR'].isin(gnn_years)]
        #print(f"🔎 GNN-filtered years: {gnn_years}")
    else:
        #print("⚠️ No GNN years or GNN skipped.")
        filtered_df = daily_df.iloc[0:0]

    gnn_events = identify_heatwaves_static(filtered_df, method='temperature')
    gnn_stats, gnn_df, _ = calculate_event_stats(gnn_events, filtered_df)

    # EVT detection
    evt_events = identify_evt_extremes(daily_df)
    evt_stats, evt_df, _ = calculate_event_stats(evt_events, daily_df)

    # Combine all
    all_events = pd.concat([static_df, gnn_df, evt_df]).drop_duplicates()
    if 'begin_date' in all_events.columns and not all_events.empty:
        all_events = all_events.sort_values('begin_date').reset_index(drop=True)
    else:
        all_events = pd.DataFrame(columns=[
            'begin_date', 'end_date', 'duration', 'avg_tmax', 'std_tmax',
            'max_tmax', 'avg_heat_index', 'year'
        ])

    all_stats = pd.concat([static_stats, gnn_stats, evt_stats])
    if all_stats.empty:
        print(f" No valid stats found in {epw_path}. Skipping.")
        return all_stats, all_events, pd.DataFrame(), pd.DataFrame()

    all_stats = all_stats.groupby('year').agg({
        'hwn': 'sum', 'hwf': 'sum', 'hwd': 'max',
        'hwdm': 'mean', 'hwaa': 'max', 'severity': 'max'
    }).reset_index().sort_values('year')
    all_stats['severity'] = all_stats['hwf'] * all_stats['hwaa']

    if not all_stats.empty and 'severity' in all_stats.columns:
        peak_year = all_stats.loc[all_stats['severity'].idxmax()]['year']
        peak_events = all_events[all_events['year'] == peak_year]
        peak_stats = all_stats[all_stats['year'] == peak_year]
        #print(f" Highest severity year: {peak_year}")
    else:
        peak_events = pd.DataFrame()
        peak_stats = pd.DataFrame()

    return all_stats, all_events, peak_stats, peak_events, static_stats, static_df



def save_final_outputs(output_dir, base_stats, base_events, all_stats, all_events, peak_stats, peak_events):
    os.makedirs(output_dir, exist_ok=True)
    base_events.to_csv(os.path.join(output_dir, "heatwave_events_base.csv"), index=False)
    base_stats.to_csv(os.path.join(output_dir, "heatwave_stats_base.csv"), index=False)
    all_events.to_csv(os.path.join(output_dir, "heatwave_events.csv"), index=False)
    all_stats.to_csv(os.path.join(output_dir, "heatwave_stats.csv"), index=False)
    peak_events.to_csv(os.path.join(output_dir, "heatwave_events_peak.csv"), index=False)
    peak_stats.to_csv(os.path.join(output_dir, "heatwave_stats_peak.csv"), index=False)

def run_full_pipeline(epw_dir, base_dir, output_dir):
    print(f" Starting hybrid method on EPWs in {epw_dir}")

    # Load and process base EPW
    base_files = [f for f in os.listdir(base_dir) if f.endswith('.epw')]
    if not base_files:
        raise FileNotFoundError("No base EPW file found in base folder.")
    base_epw_path = os.path.join(base_dir, base_files[0])
    #print(f" Processing base file: {base_epw_path}")

    # Read for GNN detection
    df_base = pd.read_csv(base_epw_path, skiprows=8, header=None, names=[
        'year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air',
        'temp_dew', 'relative_humidity', 'atmospheric_pressure', 'etr', 'etrn',
        'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
        'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance',
        'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
        'visibility', 'ceiling_height', 'present_weather_observation',
        'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth',
        'snow_depth', 'days_since_last_snowfall', 'albedo',
        'liquid_precipitation_depth', 'liquid_precipitation_quantity'
    ])
    gnn_years = apply_gnn_detection(df_base)

    # Process base file
    base_stats, base_events, _, _, _, _ = process_file(
        base_epw_path, gnn_years, apply_gnn=True
    )
    if base_stats.empty:
        print(f"No valid stats found in {base_epw_path}. Skipping.")
        base_stats, base_events = pd.DataFrame(), pd.DataFrame()

    # Process all EPW files in the EPWs folder
    all_stats_df = []
    all_events_df = []

    for epw_file in sorted(os.listdir(epw_dir)):
        if not epw_file.endswith(".epw"):
            continue
        epw_path = os.path.join(epw_dir, epw_file)
        print(f" Processing file: {epw_path}")

        try:
            stats_df, events_df, _, _, _, _ = process_file(
                epw_path, gnn_years=None, apply_gnn=False
            )
            if not stats_df.empty:
                all_stats_df.append(stats_df)
            if not events_df.empty:
                all_events_df.append(events_df)
        except Exception as e:
            print(f"Error processing {epw_path}: {e}")

    # Combine all stats and events
    epw_stats = pd.concat(all_stats_df, ignore_index=True) if all_stats_df else pd.DataFrame()
    epw_events = pd.concat(all_events_df, ignore_index=True) if all_events_df else pd.DataFrame()

    if not epw_stats.empty:
        epw_stats['severity'] = epw_stats['hwf'] * epw_stats['hwaa']
        print("\n📈 Severity scores across all years:")
        for _, row in epw_stats[['year', 'severity']].sort_values('year').iterrows():
            print(f"   - Year {int(row['year'])}: Severity = {row['severity']:.2f}")

        peak_year = int(epw_stats.loc[epw_stats['severity'].idxmax()]['year'])
        peak_events = epw_events[epw_events['year'] == peak_year]
        peak_stats = epw_stats[epw_stats['year'] == peak_year]

        print("\n Ensemble method run complete.")
        print(f" Peak severity year identified: {peak_year}")
        print("\n Event Stats for Peak Year:")
        for col in ['hwn', 'hwf', 'hwd', 'hwdm', 'hwaa', 'severity']:
            val = peak_stats.iloc[0][col]
            print(f"   - {col}: {val:.2f}" if isinstance(val, (float, int)) else f"   - {col}: {val}")
        print()

    else:
        print(" No heatwaves detected in EPWs.")
        peak_stats = pd.DataFrame()
        peak_events = pd.DataFrame()


    save_final_outputs(output_dir, base_stats, base_events, epw_stats, epw_events, peak_stats, peak_events)
    print("✅ Saved all 6 heat wave output CSVs")


run_full_pipeline('/content/EPWs', '/content/base', '/content/hotspells')


import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

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

# === DAILY AGGREGATION ===
def calculate_daily_tmin_tmax(epw_data):
    epw_data['temp_air'] = pd.to_numeric(epw_data['temp_air'], errors='coerce')
    epw_data['wind_speed'] = pd.to_numeric(epw_data['wind_speed'], errors='coerce')
    daily = epw_data.groupby(['year', 'month', 'day']).agg({
        'temp_air': ['min', 'max'],
        'wind_speed': 'mean'
    }).reset_index()
    daily.columns = ['YEAR', 'MONTH', 'DAY', 'Tmin (oC)', 'Tmax (oC)', 'Wind Speed (m/s)']
    return daily

# === DETECTION METHOD ===
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

from scipy.stats import genpareto

def identify_evt_cold_extremes(df, quantile_threshold=0.05, min_duration=3):
    """
    Identify cold spells using EVT (POT) approach based on Tmin.
    """
    tmin_series = df['Tmin (oC)'].dropna()
    threshold = tmin_series.quantile(quantile_threshold)
    excesses = -(tmin_series[tmin_series < threshold] - threshold)  # flip tail for cold

    if len(excesses) < 5:
        return pd.DataFrame(columns=['Start', 'End'])  # Not enough data

    c, loc, scale = genpareto.fit(excesses, floc=0)

    # Define return level (e.g., 98th percentile in lower tail)
    return_level = threshold - genpareto.ppf(0.98, c, loc=0, scale=scale)

    # Flag days below return level
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



# === EVENT + STATS ===
def calculate_coldspell_stats(events_df, base_df):
    stats, events = {}, []
    for _, row in events_df.iterrows():
        s_idx, e_idx = row['Start'], row['End']
        period = base_df.iloc[s_idx:e_idx + 1]
        year = int(period.iloc[0]['YEAR'])
        duration = e_idx - s_idx + 1
        avg_tmin = round(period['Tmin (oC)'].mean(), 1)
        std_tmin = round(period['Tmin (oC)'].std(), 1)
        min_tmin = round(period['Tmin (oC)'].min(), 1)
        avg_ws = round(period['Wind Speed (m/s)'].mean(), 1)
        bdate = datetime(year, int(period.iloc[0]['MONTH']), int(period.iloc[0]['DAY']))
        edate = datetime(year, int(period.iloc[-1]['MONTH']), int(period.iloc[-1]['DAY']))
        events.append({
            'begin_date': bdate.strftime('%d/%m/%Y'),
            'end_date': edate.strftime('%d/%m/%Y'),
            'duration': duration,
            'avg_tmin': avg_tmin,
            'std_tmin': std_tmin,
            'min_tmin': min_tmin,
            'avg_wind_speed': avg_ws,
            'year': year
        })
        if year not in stats:
            stats[year] = {'cwn': 0, 'cwf': 0, 'cwd': 0, 'cwdm': [], 'cwaa': np.inf, 'avg_wind_speed': []}
        stats[year]['cwn'] += 1
        stats[year]['cwf'] += duration
        stats[year]['cwd'] = max(stats[year]['cwd'], duration)
        stats[year]['cwdm'].append(duration)
        stats[year]['cwaa'] = min(stats[year]['cwaa'], min_tmin)
        stats[year]['avg_wind_speed'].append(avg_ws)
    for y in stats:
        stats[y]['cwdm'] = np.mean(stats[y]['cwdm'])
        stats[y]['avg_wind_speed'] = np.mean(stats[y]['avg_wind_speed'])
    stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index().rename(columns={'index': 'year'})
    stats_df['severity'] = stats_df['cwf'] * stats_df['cwaa']
    events_df_final = pd.DataFrame(events).sort_values(by='begin_date')
    return stats_df, events_df_final

# === PROCESS FILE ===
def process_file_cold(epw_path, methods=['ensemble', 'gnn', 'evt']):
    try:
        epw_df = pd.read_csv(epw_path, skiprows=8, header=None, names=epw_columns)
    except Exception as e:
        print(f" Could not read EPW file {epw_path}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    daily_df = calculate_daily_tmin_tmax(epw_df)
    daily_df = daily_df[(daily_df["Tmin (oC)"] != -9999) & (daily_df["Tmax (oC)"] != -9999)]

    if daily_df.empty:
        print(f" No valid daily data in {epw_path}. Skipping.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    combined_events = []
    all_stats = []

    for method in methods:
        try:
            if method == 'evt':
                events_df = identify_evt_cold_extremes(daily_df.copy())
            else:
                events_df = identify_cold_spells(daily_df.copy(), method=method)

            if events_df.empty:
                continue

            stats_df, events_full = calculate_coldspell_stats(events_df, daily_df.copy())
            combined_events.append(events_full)
            all_stats.append(stats_df)

        except Exception as e:
            print(f" Error in method '{method}' for {epw_path}: {e}")

    if not combined_events or not all_stats:
        print(f" No cold spells detected using any method in {epw_path}.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_events_df = pd.concat(combined_events).drop_duplicates().sort_values('begin_date').reset_index(drop=True)

    all_stats_df = pd.concat(all_stats).groupby('year').agg({
        'cwn': 'sum',
        'cwf': 'sum',
        'cwd': 'max',
        'cwdm': 'mean',
        'cwaa': 'min',
        'avg_wind_speed': 'mean',
        'severity': 'min'
    }).reset_index()

    if all_stats_df.empty:
        print(f" Cold spell stats were empty after aggregation in {epw_path}.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    peak_year = all_stats_df.loc[all_stats_df['severity'].idxmin()]['year']
    peak_events = all_events_df[all_events_df['year'] == peak_year]
    peak_stats = all_stats_df[all_stats_df['year'] == peak_year]

    return all_stats_df, all_events_df, peak_stats, peak_events



# === SAVE OUTPUTS ===

def save_final_outputs_cold_safe(output_dir, base_stats, base_events, all_stats, all_events, peak_stats, peak_events):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def safe_save(df, filename):
        path = Path(output_dir) / filename
        if df is not None and not df.empty:
            df.to_csv(path, index=False)
            #print(f" Saved: {path}")
        else:
            print(f" Skipped empty file: {path}")

    safe_save(base_events, "coldspells_events_base.csv")
    safe_save(base_stats, "coldspells_stats_base.csv")
    safe_save(all_events, "coldspells_events.csv")
    safe_save(all_stats, "coldspells_stats.csv")
    safe_save(peak_events, "coldspells_events_peak.csv")
    safe_save(peak_stats, "coldspells_stats_peak.csv")


# === MAIN DRIVER ===
def run_full_pipeline_cold(epw_dir, base_dir, output_dir):
    print(f" Starting cold spell detection on EPWs in {epw_dir}")

    base_files = [f for f in os.listdir(base_dir) if f.endswith('.epw')]
    if not base_files:
        raise FileNotFoundError("No base EPW file found.")
    base_path = os.path.join(base_dir, base_files[0])
    print(f" Processing base EPW file: {base_path}")
    try:
        base_stats, base_events, _, _ = process_file_cold(base_path, methods=['temperature'])
    except Exception as e:
        print(f" Base file processing failed: {e}")
        base_stats, base_events = pd.DataFrame(), pd.DataFrame()

    all_stats_list = []
    all_events_list = []

    for epw_file in sorted(os.listdir(epw_dir)):
        if not epw_file.endswith(".epw"):
            continue
        epw_path = os.path.join(epw_dir, epw_file)
        print(f" Processing file: {epw_path}")
        try:
            stats_df, events_df, _, _ = process_file_cold(epw_path, methods=['ensemble', 'gnn', 'evt'])
            if not stats_df.empty:
                all_stats_list.append(stats_df)
            if not events_df.empty:
                all_events_list.append(events_df)
        except Exception as e:
            print(f" Failed on {epw_file}: {e}")

    epw_stats = pd.concat(all_stats_list, ignore_index=True) if all_stats_list else pd.DataFrame()
    epw_events = pd.concat(all_events_list, ignore_index=True) if all_events_list else pd.DataFrame()

    if not epw_stats.empty:
        #Recomputing for consistency
        epw_stats['severity'] = epw_stats['cwf'] * epw_stats['cwaa']
        peak_year = int(epw_stats.loc[epw_stats['severity'].idxmin()]['year'])
        peak_events = epw_events[epw_events['year'] == peak_year]
        peak_stats = epw_stats[epw_stats['year'] == peak_year]

        print("\n Severity scores across all years:")
        for _, row in epw_stats[['year', 'severity']].sort_values('year').iterrows():
            print(f"   - Year {int(row['year'])}: Severity = {row['severity']:.2f}")

        print("\n Ensemble method run complete.")
        print(f" Peak severity year identified: {peak_year}")

        print("\n Event Stats for Peak Year:")
        for col in ['cwn', 'cwf', 'cwd', 'cwdm', 'cwaa', 'severity']:
            val = peak_stats.iloc[0][col]
            print(f"   - {col}: {val:.2f}" if isinstance(val, (float, int)) else f"   - {col}: {val}")
        print()

    else:
        print(" No cold spells detected in EPWs.")
        peak_stats = pd.DataFrame()
        peak_events = pd.DataFrame()


    save_final_outputs_cold_safe(output_dir, base_stats, base_events, epw_stats, epw_events, peak_stats, peak_events)
    print("✅ Saved all 6 cold spell output CSVs.")


run_full_pipeline_cold('/content/EPWs', '/content/base', '/content/coldspells')
