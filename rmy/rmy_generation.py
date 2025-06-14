
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

from rmy.utils import (
    read_epw_file, save_epw, extract_original_header,
    interpolate_smooth_transitions
)
from rmy.heatwaves import run_full_pipeline as run_heatwaves
from rmy.coldspells import run_full_pipeline_cold as run_coldspells

warnings.filterwarnings("ignore", category=FutureWarning)

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
                df.loc[idx] = row.values
                inserted.add(tag)
    return df, inserted

def construct_final_rmy(base_epw_path, hot_events_path, cold_events_path, output_path):
    all_epw_folder = Path(base_epw_path).parent.parent / 'EPWs'
    base_epw = read_epw_file(base_epw_path)
    epw_list = list(all_epw_folder.glob('*.epw'))

    def get_peak_year(stats_csv): return int(pd.read_csv(stats_csv).iloc[0]['year'])
    def find_epw_by_year(year): 
        for fname in os.listdir(all_epw_folder):
            if fname.endswith('.epw') and str(year) in fname:
                return os.path.join(all_epw_folder, fname)
        raise FileNotFoundError(f"No EPW for {year}")

    heat_peak_epw = read_epw_file(find_epw_by_year(get_peak_year(hot_events_path)))
    cold_peak_epw = read_epw_file(find_epw_by_year(get_peak_year(cold_events_path)))

    def safe_load_events(path, cols):
        try: return pd.read_csv(path, parse_dates=['begin_date', 'end_date'], dayfirst=True)
        except: return pd.DataFrame(columns=cols)

    heat_base = safe_load_events(hot_events_path.replace('peak','base'), ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
    heat_peak = safe_load_events(hot_events_path, ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
    cold_base = safe_load_events(cold_events_path.replace('peak','base'), ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])
    cold_peak = safe_load_events(cold_events_path, ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])

    mh, uh = match_events(heat_base, heat_peak)
    mc, uc = match_events(cold_base, cold_peak)

    final, rh = integrate_events(base_epw.copy(), mh, uh, heat_peak_epw, 'Heatwave')
    final, rc = integrate_events(final, mc, uc, cold_peak_epw, 'Coldspell')

    summer = [6, 7, 8]
    winter = [12, 1, 2]
    s_avg = calculate_monthly_avg_conditions(base_epw, summer)
    w_avg = calculate_monthly_avg_conditions(base_epw, winter)

    s_cond = lambda df, t: df['temp_air'].mean() < t['temp_air'] and df['relative_humidity'].mean() < t['relative_humidity']
    w_cond = lambda df, t: df['temp_air'].mean() > t['temp_air'] and df['relative_humidity'].mean() > t['relative_humidity']

    s_days, s_files = find_days_to_adjust_avg(epw_list, s_avg, summer, s_cond)
    w_days, w_files = find_days_to_adjust_avg(epw_list, w_avg, winter, w_cond)

    final, _ = integrate_days(final, s_days, rh, s_files, s_avg, summer)
    final, _ = integrate_days(final, w_days, rc, w_files, w_avg, winter)

    header = extract_original_header(base_epw_path)
    save_epw(final, output_path, header)
    print(f"✅ Final RMY saved to: {output_path}")

def run_full_rmy_pipeline(epw_dir, base_dir, output_dir):
    run_heatwaves(epw_dir, base_dir, os.path.join(output_dir, "hotspells"))
    run_coldspells(epw_dir, base_dir, os.path.join(output_dir, "coldspells"))
    base_epw = [f for f in os.listdir(base_dir) if f.endswith('.epw')][0]
    base_path = os.path.join(base_dir, base_epw)
    output_path = os.path.join(output_dir, f"RMY_{base_epw}")
    construct_final_rmy(
        base_epw_path=base_path,
        hot_events_path=os.path.join(output_dir, "hotspells", "heatwave_events_peak.csv"),
        cold_events_path=os.path.join(output_dir, "coldspells", "coldspells_events_peak.csv"),
        output_path=output_path
    )
