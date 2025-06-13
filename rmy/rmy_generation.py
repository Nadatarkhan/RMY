from heatwaves import run_full_pipeline
from coldspells import run_full_pipeline_cold
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




# === Cell Separator ===

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




# === Cell Separator ===

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Setup paths ---
base_epw_folder = Path('/content/base')
all_epw_folder = Path('/content/EPWs')
output_epw_path = Path('/content/final/RMY.epw')
coldspell_stats_path = Path(output_dir) / 'coldspells' / 'coldspells_stats_peak.csv'
heatwave_stats_path = Path(output_dir) / 'hotspells' / 'heatwave_stats_peak.csv'

def read_epw_file(epw_file_path):
    return pd.read_csv(epw_file_path, header=None, skiprows=8, sep=',', names=[
        'year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air', 'temp_dew', 'relative_humidity',
        'atmospheric_pressure', 'etr', 'etrn', 'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
        'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance', 'wind_direction', 'wind_speed',
        'total_sky_cover', 'opaque_sky_cover', 'visibility', 'ceiling_height', 'present_weather_observation',
        'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth', 'snow_depth',
        'days_since_last_snowfall', 'albedo', 'liquid_precipitation_depth', 'liquid_precipitation_quantity'
    ])

def get_peak_year(stats_csv):
    return int(pd.read_csv(stats_csv).iloc[0]['year'])

def find_epw_by_year(year, epw_folder):
    for fname in os.listdir(epw_folder):
        if fname.endswith('.epw') and str(year) in fname:
            return os.path.join(epw_folder, fname)
    raise FileNotFoundError(f"No EPW found for year {year} in {epw_folder}")

def interpolate_smooth_transitions(df, indices, columns):
    for col in columns:
        series = df.loc[indices, col].infer_objects(copy=False)
        df.loc[indices, col] = series.interpolate()
    return df

def match_events(base_df, peak_df):
    matched, unmatched = [], peak_df.copy()
    if base_df.empty: return matched, unmatched
    for _, base in base_df.iterrows():
        best, min_diff = None, float('inf')
        for idx, peak in unmatched.iterrows():
            if base['begin_date'].month == peak['begin_date'].month and base['begin_date'].day == peak['begin_date'].day:
                diff = abs(base['duration'] - peak['duration'])
                if diff < min_diff:
                    best, min_diff = idx, diff
        if best is not None:
            matched.append((base, unmatched.loc[best]))
            unmatched = unmatched.drop(best)
    return matched, unmatched

def integrate_events(df, matched, unmatched, peak_epw, label):
    all_events = matched + [(None, e) for e in unmatched.itertuples()]
    replaced = set()
    for base, peak in all_events:
        for d in pd.date_range(start=peak.begin_date, end=peak.end_date, freq='D'):
            slice_peak = peak_epw[(peak_epw['month'] == d.month) & (peak_epw['day'] == d.day)].copy()
            if slice_peak.empty: continue
            idx = df[(df['month'] == d.month) & (df['day'] == d.day)].index
            smoothing_idx = list(range(max(0, idx.min()-8), min(len(df), idx.max()+9)))
            df = interpolate_smooth_transitions(df, smoothing_idx, slice_peak.columns.drop(['year', 'month', 'day', 'hour', 'minute']))
            for _, row in slice_peak.iterrows():
                row_idx = df[(df['month'] == row['month']) & (df['day'] == row['day']) & (df['hour'] == row['hour'])].index
                df.loc[row_idx] = row.values
                replaced.add(f"{int(row['month']):02d}-{int(row['day']):02d}")
            df = interpolate_smooth_transitions(df, smoothing_idx, slice_peak.columns.drop(['year', 'month', 'day', 'hour', 'minute']))
    print(f"{label} events integrated. Days replaced: {len(replaced)}")
    return df, replaced

def calculate_monthly_avg_conditions(df, months):
    return df[df['month'].isin(months)].groupby('month')[['temp_air', 'relative_humidity']].mean()

def find_days_to_adjust_avg(epw_files, targets, months, cond):
    days, sources = {m: [] for m in months}, {m: [] for m in months}
    for epw in epw_files:
        df = read_epw_file(epw)
        for m in months:
            for d in range(1, 32):
                day = df[(df['month'] == m) & (df['day'] == d)]
                if not day.empty and cond(day, targets.loc[m]):
                    days[m].append(day)
                    sources[m].append(epw)
                if sum(len(v) for v in days.values()) >= 40:
                    return days, sources
    return days, sources

def integrate_days(df, days, replaced, sources, targets, months):
    inserted = set()
    for m in months:
        for i, day_df in enumerate(days[m]):
            if len(inserted) >= 30: break
            for _, row in day_df.iterrows():
                tag = f"{int(row['month']):02d}-{int(row['day']):02d}"
                if tag in replaced or tag in inserted: continue
                idx = df[(df['month'] == row['month']) & (df['day'] == row['day'])].index
                smoothing_idx = list(range(max(0, idx.min()-8), min(len(df), idx.max()+9)))
                df = interpolate_smooth_transitions(df, smoothing_idx, day_df.columns.drop(['year', 'month', 'day', 'hour', 'minute']))
                df.loc[idx] = day_df.values
                inserted.add(tag)
    return df, inserted

def safe_load_events(path, cols):
    try: return pd.read_csv(path, parse_dates=['begin_date', 'end_date'], dayfirst=True)
    except: return pd.DataFrame(columns=cols)

# --- Begin process ---
base_epw_path = list(base_epw_folder.glob('*.epw'))[0]
#header = [next(open(base_epw_path)) for _ in range(8)]
# Read and store header lines safely
with open(base_epw_path, 'r') as f:
    header = [f.readline() for _ in range(8)]


base_epw = read_epw_file(base_epw_path)
epw_list = list(all_epw_folder.glob('*.epw'))

heat_peak = find_epw_by_year(get_peak_year(heatwave_stats_path), all_epw_folder)
cold_peak = find_epw_by_year(get_peak_year(coldspell_stats_path), all_epw_folder)
peak_heat = read_epw_file(heat_peak)
peak_cold = read_epw_file(cold_peak)

heat_base = safe_load_events(str(Path(output_dir) / 'hotspells' / 'heatwave_events_base.csv', ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
heat_peak_df = safe_load_events(str(Path(output_dir) / 'hotspells' / 'heatwave_events_peak.csv', ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
cold_base = safe_load_events(str(Path(output_dir) / 'coldspells' / 'base/coldspells_events_base.csv', ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])
cold_peak_df = safe_load_events(str(Path(output_dir) / 'coldspells' / 'coldspells_events_peak.csv', ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])

mh, uh = match_events(heat_base, heat_peak_df)
mc, uc = match_events(cold_base, cold_peak_df)

#old
final, rh = integrate_events(base_epw.copy(), mh, uh, peak_heat, 'Heatwave')
final, rc = integrate_events(final, mc, uc, peak_cold, 'Coldspell')

#NEW
# Step 1: Integrate Heatwaves on the base EPW
#heat_intermediate, rh = integrate_events(base_epw.copy(), mh, uh, peak_heat, 'Heatwave')

# Step 2: Use heat-adjusted file as input for coldspell integration
#final, rc = integrate_events(heat_intermediate.copy(), mc, uc, peak_cold, 'Coldspell')


# Adjust monthly averages***************************************************

#Southern Hemisphere
#summer = [12, 1, 2]
#winter = [6, 7, 8]

#Northern Hemisphere
summer = [6, 7, 8]
winter = [12, 1, 2]


base_summer = calculate_monthly_avg_conditions(base_epw, summer)
base_winter = calculate_monthly_avg_conditions(base_epw, winter)

s_condition = lambda df, t: df['temp_air'].mean() < t['temp_air'] and df['relative_humidity'].mean() < t['relative_humidity']
w_condition = lambda df, t: df['temp_air'].mean() > t['temp_air'] and df['relative_humidity'].mean() > t['relative_humidity']

sdays, sfiles = find_days_to_adjust_avg(epw_list, base_summer, summer, s_condition)
wdays, wfiles = find_days_to_adjust_avg(epw_list, base_winter, winter, w_condition)

final, ins_s = integrate_days(final, sdays, rh, sfiles, base_summer, summer)
final, ins_w = integrate_days(final, wdays, rc, wfiles, base_winter, winter)

# Write final EPW
with open(output_epw_path, 'w', newline='') as file:
    for line in header:
        file.write(line if line.endswith('\n') else line + '\n')
    final.to_csv(file, index=False, header=False, float_format='%g', na_rep='999')


print(f"✅ Final RMY saved to: {output_epw_path}")
print(f"📂 Heatwave Peak EPW used: {heat_peak}")
print(f"📂 Coldspell Peak EPW used: {cold_peak}")


# === Cell Separator ===

import os
import pandas as pd

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

def compare_tmy_rmy(base_folder='/content/base', rmy_path='/content/final/RMY.epw'):
    # Find TMY file in base folder
    tmy_path = next((os.path.join(base_folder, f) for f in os.listdir(base_folder) if f.endswith('.epw')), None)

    if not tmy_path or not os.path.exists(rmy_path):
        print("❌ Missing TMY or RMY file.")
        return

    print(f"🔎 Comparing:\n- TMY: {tmy_path}\n- RMY: {rmy_path}")

    tmy_stats = load_epw_daily_stats(tmy_path)
    rmy_stats = load_epw_daily_stats(rmy_path)

    print("\n📊 Comparison Summary:")
    for key in tmy_stats:
        print(f"{key}:")
        print(f"   TMY: {tmy_stats[key]}")
        print(f"   RMY: {rmy_stats[key]}")
        print(f"   Δ   : {rmy_stats[key] - tmy_stats[key]}\n")

# Run the comparison
compare_tmy_rmy()


# === RMY Wrapper ===
def run_full_rmy_pipeline(epw_dir, base_dir, output_dir):
    os.makedirs(epw_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("\n🚀 Running full RMY pipeline...")
    heat_output = os.path.join(output_dir, "hotspells")
    cold_output = os.path.join(output_dir, "coldspells")

    run_full_pipeline(epw_dir, base_dir, heat_output)
    run_full_pipeline_cold(epw_dir, base_dir, cold_output)

    print("\n✅ RMY pipeline complete.")
