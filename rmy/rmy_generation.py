
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

from rmy.heatwaves import run_full_pipeline as run_heatwaves
from rmy.coldspells import run_full_pipeline_cold as run_coldspells
from rmy.utils import load_epw, save_epw, extract_original_header

warnings.filterwarnings("ignore", category=FutureWarning)

def get_peak_year(stats_csv):
    return int(pd.read_csv(stats_csv).iloc[0]['year'])

def find_epw_by_year(year, epw_folder):
    for fname in os.listdir(epw_folder):
        if fname.endswith('.epw') and str(year) in fname:
            return os.path.join(epw_folder, fname)
    raise FileNotFoundError(f"No EPW found for year {year} in {epw_folder}")

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

def seasonal_adjustment(final, base_epw, epw_list):
    def calculate_monthly_avg_conditions(df, months):
        return df[df['month'].isin(months)].groupby('month')[['temp_air', 'relative_humidity']].mean()

    def find_days_to_adjust_avg(epw_files, targets, months, cond):
        days, sources = {m: [] for m in months}, {m: [] for m in months}
        for epw in epw_files:
            df = load_epw(epw)
            for m in months:
                for d in range(1, 32):
                    day = df[(df['month'] == m) & (df['day'] == d)]
                    if not day.empty and cond(day, targets.loc[m]):
                        days[m].append(day)
                        sources[m].append(epw)
                    if sum(len(v) for v in days.values()) >= 40:
                        return days, sources
        return days, sources

    def interpolate_smooth_transitions(df, indices, columns):
        for col in columns:
            series = df.loc[indices, col].infer_objects(copy=False)
            df.loc[indices, col] = series.interpolate()
        return df

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

    summer = [6, 7, 8]
    winter = [12, 1, 2]

    base_summer = calculate_monthly_avg_conditions(base_epw, summer)
    base_winter = calculate_monthly_avg_conditions(base_epw, winter)

    s_condition = lambda df, t: df['temp_air'].mean() < t['temp_air'] and df['relative_humidity'].mean() < t['relative_humidity']
    w_condition = lambda df, t: df['temp_air'].mean() > t['temp_air'] and df['relative_humidity'].mean() > t['relative_humidity']

    sdays, sfiles = find_days_to_adjust_avg(epw_list, base_summer, summer, s_condition)
    wdays, wfiles = find_days_to_adjust_avg(epw_list, base_winter, winter, w_condition)

    final, _ = integrate_days(final, sdays, set(), sfiles, base_summer, summer)
    final, _ = integrate_days(final, wdays, set(), wfiles, base_winter, winter)
    return final

def run_full_rmy_pipeline(epw_dir, base_dir, output_dir):
    print(f"📂 Running full RMY pipeline:")
    print(f"   AMY Folder: {epw_dir}")
    print(f"   Base File: {base_dir}")
    print(f"   Output Folder: {output_dir}")

    hot_output = os.path.join(output_dir, "hotspells")
    cold_output = os.path.join(output_dir, "coldspells")
    os.makedirs(hot_output, exist_ok=True)
    os.makedirs(cold_output, exist_ok=True)

    run_heatwaves(epw_dir, base_dir, hot_output)
    run_coldspells(epw_dir, base_dir, cold_output)

    base_file = [f for f in os.listdir(base_dir) if f.endswith('.epw')][0]
    base_epw_path = os.path.join(base_dir, base_file)
    base_df = load_epw(base_epw_path)
    header = extract_original_header(base_epw_path)
    epw_list = list(Path(epw_dir).glob("*.epw"))

    heatwave_peak = get_peak_year(os.path.join(hot_output, "heatwave_stats_peak.csv"))
    coldspell_peak = get_peak_year(os.path.join(cold_output, "coldspells_stats_peak.csv"))

    peak_heat = load_epw(find_epw_by_year(heatwave_peak, epw_dir))
    peak_cold = load_epw(find_epw_by_year(coldspell_peak, epw_dir))

    heat_base = pd.read_csv(os.path.join(hot_output, "heatwave_events_base.csv"), parse_dates=['begin_date', 'end_date'], dayfirst=True)
    heat_peak_df = pd.read_csv(os.path.join(hot_output, "heatwave_events_peak.csv"), parse_dates=['begin_date', 'end_date'], dayfirst=True)
    cold_base = pd.read_csv(os.path.join(cold_output, "coldspells_events_base.csv"), parse_dates=['begin_date', 'end_date'], dayfirst=True)
    cold_peak_df = pd.read_csv(os.path.join(cold_output, "coldspells_events_peak.csv"), parse_dates=['begin_date', 'end_date'], dayfirst=True)

    mh, uh = match_events(heat_base, heat_peak_df)
    mc, uc = match_events(cold_base, cold_peak_df)

    # Stub integration logic
    final = base_df.copy()
    # You can add your `integrate_events` logic here
    # final, rh = integrate_events(final, mh, uh, peak_heat, 'Heatwave')
    # final, rc = integrate_events(final, mc, uc, peak_cold, 'Coldspell')

    final = seasonal_adjustment(final, base_df, epw_list)

    final_path = os.path.join(output_dir, f"RMY_{base_file}")
    save_epw(final, header, final_path)
    print(f"✅ Final RMY saved to: {final_path}")
