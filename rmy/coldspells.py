import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from .utils import load_epw

def calculate_event_stats(coldspell_df, base_df):
    if coldspell_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    coldspell_df['duration'] = (pd.to_datetime(coldspell_df['end_date'], format='%d/%m/%Y') -
                                pd.to_datetime(coldspell_df['begin_date'], format='%d/%m/%Y')).dt.days + 1

    stats = coldspell_df.groupby('year').agg(
        csn=('duration', 'count'),
        csf=('duration', lambda x: (x > 1).sum()),
        csd=('duration', 'sum'),
        csdm=('duration', 'max'),
        csaa=('avg_tmin', 'mean'),
        severity=('min_tmin', 'mean')
    ).reset_index()

    stats['year'] = stats['year'].astype(int)
    stats['peak'] = stats['severity'].idxmin()
    stats = stats.sort_values('year')

    if not base_df.empty:
        base_stats = base_df.groupby('year').agg(
            csn=('duration', 'count'),
            csf=('duration', lambda x: (x > 1).sum()),
            csd=('duration', 'sum'),
            csdm=('duration', 'max'),
            csaa=('avg_tmin', 'mean'),
            severity=('min_tmin', 'mean')
        ).reset_index()
        base_stats['year'] = base_stats['year'].astype(int)
        base_stats['peak'] = base_stats['severity'].idxmin()
    else:
        base_stats = pd.DataFrame()

    return stats, base_stats, stats.loc[stats['severity'].idxmin(), 'year']

def process_file(epw_path):
    df = load_epw(epw_path)
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df['date'] = df['datetime'].dt.date
    df['is_extreme'] = (df['temp_air'] < -5).astype(int)
    df['event_id'] = (df['is_extreme'].diff(1) != 0).astype(int).cumsum()

    coldspell_events = df[df['is_extreme'] == 1].groupby('event_id').agg(
        begin_date=('date', 'first'),
        end_date=('date', 'last'),
        year=('year', 'first'),
        avg_tmin=('temp_air', 'mean'),
        min_tmin=('temp_air', 'min')
    ).reset_index()

    peak_year = coldspell_events.groupby('year')['min_tmin'].mean().idxmin()
    return coldspell_events, coldspell_events[coldspell_events['year'] == peak_year], peak_year

def run_full_pipeline_cold(epw_dir, base_dir, output_dir):
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

    pd.concat(all_stats).to_csv(Path(output_dir) / "coldspells_stats.csv", index=False)
    pd.concat(all_events).to_csv(Path(output_dir) / "coldspells_events.csv", index=False)
    base_events.to_csv(Path(output_dir) / "coldspells_events_base.csv", index=False)
    base_stats.to_csv(Path(output_dir) / "coldspells_stats_base.csv", index=False)
    base_events_peak.to_csv(Path(output_dir) / "coldspells_events_peak.csv", index=False)
    base_stats_peak.to_csv(Path(output_dir) / "coldspells_stats_peak.csv", index=False)