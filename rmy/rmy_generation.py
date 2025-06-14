
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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

def run_full_rmy_pipeline(epw_dir, base_dir, output_dir):
    print(f"📂 Running full RMY pipeline:")
    print(f"   AMY Folder: {epw_dir}")
    print(f"   Base File: {base_dir}")
    print(f"   Output Folder: {output_dir}")

    base_epw_path = list(Path(base_dir).glob('*.epw'))[0]
    all_epw_folder = Path(epw_dir)
    output_epw_path = Path(output_dir) / f"RMY_{base_epw_path.name}"
    coldspell_stats_path = Path(output_dir) / 'coldspells/coldspells_stats_peak.csv'
    heatwave_stats_path = Path(output_dir) / 'hotspells/heatwave_stats_peak.csv'

    with open(base_epw_path, 'r') as f:
        header = [f.readline() for _ in range(8)]

    base_epw = read_epw_file(base_epw_path)
    epw_list = list(all_epw_folder.glob('*.epw'))

    heat_peak = find_epw_by_year(get_peak_year(heatwave_stats_path), epw_dir)
    cold_peak = find_epw_by_year(get_peak_year(coldspell_stats_path), epw_dir)
    peak_heat = read_epw_file(heat_peak)
    peak_cold = read_epw_file(cold_peak)

    heat_base = safe_load_events(output_dir + '/hotspells/heatwave_events_base.csv', ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
    heat_peak_df = safe_load_events(output_dir + '/hotspells/heatwave_events_peak.csv', ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
    cold_base = safe_load_events(output_dir + '/coldspells/base/coldspells_events_base.csv', ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])
    cold_peak_df = safe_load_events(output_dir + '/coldspells/coldspells_events_peak.csv', ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])

    mh, uh = match_events(heat_base, heat_peak_df)
    mc, uc = match_events(cold_base, cold_peak_df)

    final, rh = integrate_events(base_epw.copy(), mh, uh, peak_heat, 'Heatwave')
    final, rc = integrate_events(final, mc, uc, peak_cold, 'Coldspell')

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

    os.makedirs(output_epw_path.parent, exist_ok=True)
    with open(output_epw_path, 'w', newline='') as file:
        for line in header:
            file.write(line if line.endswith('\n') else line + '\n')
        final.to_csv(file, index=False, header=False, float_format='%g', na_rep='999')

    print(f"✅ Final RMY saved to: {output_epw_path}")
    print(f"📂 Heatwave Peak EPW used: {heat_peak}")
    print(f"📂 Coldspell Peak EPW used: {cold_peak}")
