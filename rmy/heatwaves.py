import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from .utils import load_epw, calculate_heat_index, smooth_transition, match_extreme_days

def calculate_event_stats(heatwave_df, base_df):
    if heatwave_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    heatwave_df['duration'] = (pd.to_datetime(heatwave_df['end_date'], format='%d/%m/%Y') -
                               pd.to_datetime(heatwave_df['begin_date'], format='%d/%m/%Y')).dt.days + 1

    stats = heatwave_df.groupby('year').agg(
        hwn=('duration', 'count'),
        hwf=('duration', lambda x: (x > 1).sum()),
        hwd=('duration', 'sum'),
        hwdm=('duration', 'max'),
        hwaa=('avg_tmax', 'mean'),
        severity=('avg_heat_index', 'mean')
    ).reset_index()

    stats['year'] = stats['year'].astype(int)
    stats['peak'] = stats['severity'].idxmax()
    stats = stats.sort_values('year')

    if not base_df.empty:
        base_stats = base_df.groupby('year').agg(
            hwn=('duration', 'count'),
            hwf=('duration', lambda x: (x > 1).sum()),
            hwd=('duration', 'sum'),
            hwdm=('duration', 'max'),
            hwaa=('avg_tmax', 'mean'),
            severity=('avg_heat_index', 'mean')
        ).reset_index()
        base_stats['year'] = base_stats['year'].astype(int)
        base_stats['peak'] = base_stats['severity'].idxmax()
    else:
        base_stats = pd.DataFrame()

    return stats, base_stats, stats.loc[stats['severity'].idxmax(), 'year']

def process_file(epw_path, gnn_years=None, apply_gnn=True):
    df = load_epw(epw_path)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    df['heat_index'] = calculate_heat_index(df['temp_air'], df['relative_humidity'])
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['date'] = df['datetime'].dt.date

    df['is_extreme'] = ((df['temp_air'] > 35) & (df['heat_index'] > 38)).astype(int)
    df['event_id'] = (df['is_extreme'].diff(1) != 0).astype(int).cumsum()
    heatwave_events = df[df['is_extreme'] == 1].groupby('event_id').agg(
        begin_date=('date', 'first'),
        end_date=('date', 'last'),
        year=('year', 'first'),
        avg_tmax=('temp_air', 'mean'),
        max_tmax=('temp_air', 'max'),
        avg_heat_index=('heat_index', 'mean'),
    ).reset_index()

    peak_year = heatwave_events.groupby('year')['avg_heat_index'].mean().idxmax()

    return heatwave_events, heatwave_events[heatwave_events['year'] == peak_year], peak_year

def run_full_pipeline(epw_dir, base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    base_file = next(Path(base_dir).glob("*.epw"), None)
    if base_file is None:
        raise FileNotFoundError("No base EPW file found in base folder.")

    base_events, base_events_peak, peak_year = process_file(base_file)
    base_stats, base_stats_peak, _ = calculate_event_stats(base_events, base_events)

    all_stats, all_events = [], []
    for epw_file in Path(epw_dir).glob("*.epw"):
        events, events_peak, _ = process_file(epw_file)
        stats, _, _ = calculate_event_stats(events, base_events)
        stats['source'] = epw_file.name
        events['source'] = epw_file.name
        all_stats.append(stats)
        all_events.append(events)

    pd.concat(all_stats).to_csv(Path(output_dir) / "heatwave_stats.csv", index=False)
    pd.concat(all_events).to_csv(Path(output_dir) / "heatwave_events.csv", index=False)
    base_events.to_csv(Path(output_dir) / "heatwave_events_base.csv", index=False)
    base_stats.to_csv(Path(output_dir) / "heatwave_stats_base.csv", index=False)
    base_events_peak.to_csv(Path(output_dir) / "heatwave_events_peak.csv", index=False)
    base_stats_peak.to_csv(Path(output_dir) / "heatwave_stats_peak.csv", index=False)