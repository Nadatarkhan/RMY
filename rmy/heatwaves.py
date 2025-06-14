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
        for i, df in enumerate(epw_data.values()):
            flagged = flag_percentile_events(df, percentile=percentile, min_duration=min_duration)
            percentile_detected.append(flagged)
    elif method == 'temperature':
        for i, df in enumerate(epw_data.values()):
            flagged = flag_temperature_events(df, threshold=threshold_temp)
            temp_detected.append(flagged)
        for i, df in enumerate(epw_data.values()):
            flagged = flag_temperature_events(df, threshold=threshold_temp)
            temp_detected.append(flagged)
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
    threshold = tmax_series.quantile(quantile_threshold)
    excesses = tmax_series[tmax_series > threshold] - threshold
    if len(excesses) < 5:
        return pd.DataFrame(columns=['Start', 'End'])  # Not enough data to fit GPD
    # Fit Generalized Pareto Distribution to excesses
    c, loc, scale = genpareto.fit(excesses, floc=0)
    # Define a high return level (e.g., 95th percentile of fitted GPD)
    return_level = threshold + genpareto.ppf(0.95, c, loc=0, scale=scale)
    # Flag days exceeding return level
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
    df = df[['year', 'month', 'day', 'hour', 'temp_air']].dropna()
    daily_z = (daily_max - daily_max.mean()) / daily_max.std()
    threshold = 2.5
    extreme_days = daily_z[daily_z > threshold]
    return list(pd.to_datetime(extreme_days.index).year)
def calculate_event_stats(heatwave_df, base_df):
        return pd.DataFrame(), pd.DataFrame(), None
    events = []
    stats = {}
        s_idx, e_idx = row['Start'], row['End']
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
    events_df = pd.DataFrame(events).sort_values(by='begin_date')
    #print(f" Highest severity year: {peak_year}")
    return stats_df, events_df, peak_year
def process_file(epw_path, gnn_years=None, apply_gnn=True):
    #print(f" Processing file: {epw_path}")
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
    # Static detection
    static_events = identify_heatwaves_static(daily_df, method='ensemble')
    static_stats, static_df, _ = calculate_event_stats(static_events, daily_df)
    # GNN filtering
    if apply_gnn and gnn_years:
        #print(f"🔎 GNN-filtered years: {gnn_years}")
    else:
        #print("⚠️ No GNN years or GNN skipped.")
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
import pandas as pd
# Function to create custom colormaps for heatwaves and cold spells
def create_custom_colormap():
    heatwave_colors = ['#F7DDE1', '#EBAFB9', '#DC7284', '#CF3952', '#B22C42']  # Heatwaves color scheme
    coldspell_colors = ['#b3e7f2', '#80d2e6', '#4dbeda', '#1aaacb', '#0086b3']  # Custom blue shades for cold spells
    heatwave_cmap = mcolors.LinearSegmentedColormap.from_list('heatwave_cmap', heatwave_colors)
