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


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# Function to create custom colormaps for heatwaves and cold spells
def create_custom_colormap():
    heatwave_colors = ['#F7DDE1', '#EBAFB9', '#DC7284', '#CF3952', '#B22C42']  # Heatwaves color scheme
    coldspell_colors = ['#b3e7f2', '#80d2e6', '#4dbeda', '#1aaacb', '#0086b3']  # Custom blue shades for cold spells
    heatwave_cmap = mcolors.LinearSegmentedColormap.from_list('heatwave_cmap', heatwave_colors)
    coldspell_cmap = mcolors.LinearSegmentedColormap.from_list('coldspell_cmap', coldspell_colors)
    return heatwave_cmap, coldspell_cmap

# Function to visualize heatwave and cold spell events on a timeline
def visualize_extreme_events(heatwave_csv, coldspell_csv):
    # Read the CSV files for heatwaves and cold spells
    heatwave_df = pd.read_csv(heatwave_csv)
    coldspell_df = pd.read_csv(coldspell_csv)

    # Convert 'begin_date' to datetime format
    heatwave_df['begin_date'] = pd.to_datetime(heatwave_df['begin_date'], format='%d/%m/%Y')
    coldspell_df['begin_date'] = pd.to_datetime(coldspell_df['begin_date'], format='%d/%m/%Y')

    # Extract year, month, and day for plotting
    heatwave_df['year'] = heatwave_df['begin_date'].dt.year
    heatwave_df['month'] = heatwave_df['begin_date'].dt.month
    heatwave_df['day'] = heatwave_df['begin_date'].dt.day

    coldspell_df['year'] = coldspell_df['begin_date'].dt.year
    coldspell_df['month'] = coldspell_df['begin_date'].dt.month
    coldspell_df['day'] = coldspell_df['begin_date'].dt.day

    # Create custom colormaps for heatwaves and cold spells
    heatwave_cmap, coldspell_cmap = create_custom_colormap()

    fig, ax = plt.subplots(figsize=(20, 15))  # Updated from plt.figure()

    # Get unique years
    unique_years = sorted(set(heatwave_df['year'].unique()) | set(coldspell_df['year'].unique()))

    # Plot horizontal lines for each year
    for i, year in enumerate(unique_years):
        ax.hlines(y=i, xmin=1, xmax=13, color='gray', alpha=0.5, linestyle='--')

    # Add vertical dashed lines between months and after December
    for month in range(1, 14):  # Include 13 to add line after December
        ax.axvline(x=month, color='lightgray', linestyle='--', linewidth=0.7)

    # Adjust the normalization to better reflect the range of the data in the colormap
    #heatwave_norm = mcolors.Normalize(vmin=heatwave_df['avg_heat_index'].min(), vmax=heatwave_df['avg_heat_index'].max())
    # Set a fixed normalization range for heat index (85°C to 125°C)
    heatwave_norm = mcolors.Normalize(vmin=85, vmax=125)

    heatwave_df['clipped_index'] = heatwave_df['avg_heat_index'].clip(85, 125)

    if 'avg_wind_chill' in coldspell_df.columns:
        coldspell_norm = mcolors.Normalize(vmin=coldspell_df['avg_wind_chill'].min(), vmax=coldspell_df['avg_wind_chill'].max())
        wind_chill_column = 'avg_wind_chill'
    else:
        coldspell_norm = mcolors.Normalize(vmin=coldspell_df['avg_wind_speed'].min(), vmax=coldspell_df['avg_wind_speed'].max())
        wind_chill_column = 'avg_wind_speed'

    circle_size_factor = 50


    for i, row in heatwave_df.iterrows():
        year_index = unique_years.index(row['year'])
        month_day = row['month'] + row['day'] / 30
        ax.scatter(month_day, year_index,
                   s=row['duration'] * circle_size_factor,
                   #color=heatwave_cmap(heatwave_norm(row['avg_heat_index'])),
                   # Cap heat index normalization between 85 and 125
                   color=heatwave_cmap(heatwave_norm(row['clipped_index'])),
                   alpha=0.7, edgecolors='black', linewidth=0.5)

    for i, row in coldspell_df.iterrows():
        year_index = unique_years.index(row['year'])
        month_day = row['month'] + row['day'] / 30
        ax.scatter(month_day, year_index,
                   s=row['duration'] * circle_size_factor,
                   color=coldspell_cmap(coldspell_norm(row[wind_chill_column])),
                   alpha=0.7, edgecolors='black', linewidth=0.5)

    heatwave_sm = plt.cm.ScalarMappable(cmap=heatwave_cmap, norm=heatwave_norm)
    coldspell_sm = plt.cm.ScalarMappable(cmap=coldspell_cmap, norm=coldspell_norm)
    heatwave_sm.set_array([])
    coldspell_sm.set_array([])

    # Shift heatwave bar closer to coldspell bar by reducing pad
    cbar_heatwave = fig.colorbar(heatwave_sm, ax=ax, orientation='horizontal', pad=0.0001, shrink=0.8, aspect=40)
    cbar_coldspell = fig.colorbar(coldspell_sm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, aspect=40)

    cbar_heatwave.set_label('Heatwave Average Heat Index (°C)',fontsize=14)
    cbar_coldspell.set_label('Cold Spell Wind Chill (°C)' if 'avg_wind_chill' in coldspell_df.columns else 'Cold Spell Wind Speed (m/s)',fontsize=14)

    sizes = [1, 3, 5, 7]
    labels = [f'{size} days' for size in sizes]
    handles = [ax.scatter([], [], s=size * circle_size_factor, color='gray', alpha=0.5, edgecolors='black', label=label)
               for size, label in zip(sizes, labels)]

    # Removed bounding box from legend
    legend = ax.legend(
        handles=handles,
        scatterpoints=1,
        frameon=False,
        labelspacing=1,
        title='Event Duration',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=14  # label font size
    )

    # Now set the title font size explicitly
    legend.get_title().set_fontsize(16)


    ax.set_xlim(0.8, 13.2)
    ax.set_ylim(-1, len(unique_years))
    ax.set_yticks(ticks=range(len(unique_years)))
    ax.set_yticklabels(unique_years, fontsize=16)
    ax.set_xticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=16)

    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Year', fontsize=16)
    ax.set_title('Extreme Event Explorer: Heatwaves and Cold Spells Timeline')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

# Example usage:
heatwave_csv = '/content/hotspells/heatwave_events.csv'
coldspell_csv = '/content/coldspells/coldspells_events.csv'

visualize_extreme_events(heatwave_csv, coldspell_csv)
