import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os

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
    return pd.read_csv(epw_file_path, header=None, skiprows=8, sep=',', names=epw_columns)


def find_epw_by_year(year, epw_folder):
    """
    Locate the EPW file that contains the specified year in its filename.
    """
    for fname in os.listdir(epw_folder):
        if fname.endswith('.epw') and str(year) in fname:
            return os.path.join(epw_folder, fname)
    raise FileNotFoundError(f"No EPW found for year {year} in {epw_folder}")



def load_epw(path):
    return pd.read_csv(path, skiprows=8, header=None, names=epw_columns)

def safe_load_events(path, cols):
    try:
        return pd.read_csv(path, parse_dates=['begin_date', 'end_date'], dayfirst=True)
    except:
        return pd.DataFrame(columns=cols)


def get_peak_year(stats_csv):
    """Extract the peak year from the event stats CSV."""
    return int(pd.read_csv(stats_csv).iloc[0]['year'])


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

def match_events(base_df, peak_df):
    matched, unmatched = [], peak_df.copy()
    if base_df.empty:
        return matched, unmatched
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
            df.loc[smoothing_idx] = smooth_transition(df.loc[smoothing_idx])
            for _, row in slice_peak.iterrows():
                row_idx = df[
                    (df['month'] == row['month']) & 
                    (df['day'] == row['day']) & 
                    (df['hour'] == row['hour'])
                ].index
                df.loc[row_idx] = row.values
                replaced.add(f"{int(row['month']):02d}-{int(row['day']):02d}")
            df.loc[smoothing_idx] = smooth_transition(df.loc[smoothing_idx])
    #print(f"{label} events integrated. Days replaced: {len(replaced)}")
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
    df_cleaned = df.copy()

    # Ensure all values are numeric, coercing errors to NaN
    for col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    return df_cleaned.rolling(window=window, min_periods=1, center=True).mean()


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
                df.loc[smoothing_idx] = smooth_transition(df.loc[smoothing_idx])
                df.loc[idx] = row.values
                inserted.add(tag)
    return df, inserted

