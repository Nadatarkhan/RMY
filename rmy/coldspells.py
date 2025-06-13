
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
def run_full_pipeline_cold(epw_dir, base_dir, output_dir):
    print(f" Starting cold spell detection on EPWs in {base_dir}")

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

if __name__ == "__main__":
    run_full_pipeline_cold('/content/EPWs', '/content/base', '/content/coldspells')



# === Cell Separator ===

