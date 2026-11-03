"""Cold-spell detection (ensemble of thresholds, a stage-2 filter and EVT).

Ported from the canonical notebook, section "1.2 Cold spells".
Importing this module has no side effects; call `run_full_pipeline_cold`.

Known gaps, kept here on purpose and handled in later phases:
- The 'gnn' method is a rolling z-score, not the paper's GNN.
- Runs longer than `max_duration` (21 days) are dropped, not clipped.
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from .seasons import season_plan_for_epw
from .utils import EPW_COLUMNS


# === DAILY AGGREGATION ===
def calculate_daily_tmin_tmax_cold(epw_data):
    epw_data['temp_air'] = pd.to_numeric(epw_data['temp_air'], errors='coerce')
    epw_data['wind_speed'] = pd.to_numeric(epw_data['wind_speed'], errors='coerce')
    daily = epw_data.groupby(['year', 'month', 'day']).agg({
        'temp_air': ['min', 'max'],
        'wind_speed': 'mean'
    }).reset_index()
    daily.columns = ['YEAR', 'MONTH', 'DAY', 'Tmin (oC)', 'Tmax (oC)', 'Wind Speed (m/s)']
    return daily


# === DETECTION METHOD ===
def identify_cold_spells(df, method='ensemble', z_thresh=0.7, tmin_thres=-7, wind_speed_thres=2, window=30, max_duration=21,
                         winter_months=(12, 1, 2)):
    """Flag cold-spell runs. `winter_months` limits the 'gnn' method to the
    local winter (all 12 months at equatorial sites); see rmy.seasons."""
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

        df['is_winter'] = df['MONTH'].isin(list(winter_months))

        # Wind stays in the condition
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
            stats[year] = {'cwn': 0, 'cwf': 0, 'cwd': 0, 'cwdm': [], 'cwaa': np.inf, 'avg_wind_speed': [], 'tmin_all': []}
        stats[year]['cwn'] += 1
        stats[year]['cwf'] += duration
        stats[year]['cwd'] = max(stats[year]['cwd'], duration)
        stats[year]['cwdm'].append(duration)
        # accumulate all Tmin values across all cold-spell days for the year
        stats[year]['tmin_all'].extend(period['Tmin (oC)'].tolist())
        stats[year]['avg_wind_speed'].append(avg_ws)
    for y in stats:
        stats[y]['cwdm'] = np.mean(stats[y]['cwdm'])
        stats[y]['avg_wind_speed'] = np.mean(stats[y]['avg_wind_speed'])
        # cwaa = average Tmin across all cold-spell days in the year
        stats[y]['cwaa'] = float(np.mean(stats[y]['tmin_all'])) if stats[y].get('tmin_all') else np.nan
        # remove temp list to avoid adding a new column
        stats[y].pop('tmin_all', None)
    stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index().rename(columns={'index': 'year'})
    # keep severity column as-is (negative), selection will use positive scoring later
    stats_df['severity'] = stats_df['cwf'] * stats_df['cwaa']
    events_df_final = pd.DataFrame(events).sort_values(by='begin_date')
    return stats_df, events_df_final


def resolve_strict_overlaps_cold(events_df):
    """
    Remove strictly-overlapping cold-spell events.
    Keep the longer event; tie-breakers: colder event (lower min_tmin/avg_tmin/avg_wind_chill), then earlier start.
    Expects columns: begin_date, end_date, duration, and ideally min_tmin (or avg_tmin / avg_wind_chill).
    """
    if events_df is None or events_df.empty or 'begin_date' not in events_df.columns or 'end_date' not in events_df.columns:
        return events_df

    df = events_df.copy()
    df['begin_dt'] = pd.to_datetime(df['begin_date'], format='%d/%m/%Y', errors='coerce')
    df['end_dt'] = pd.to_datetime(df['end_date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['begin_dt', 'end_dt'])

    # Prefer colder metric if available (lower is more severe)
    if 'min_tmin' in df.columns:
        intensity_col = 'min_tmin'
        sort_cols, ascending = ['duration', intensity_col, 'begin_dt'], [False, True, True]
    elif 'avg_tmin' in df.columns:
        intensity_col = 'avg_tmin'
        sort_cols, ascending = ['duration', intensity_col, 'begin_dt'], [False, True, True]
    elif 'avg_wind_chill' in df.columns:
        intensity_col = 'avg_wind_chill'
        sort_cols, ascending = ['duration', intensity_col, 'begin_dt'], [False, True, True]
    else:
        sort_cols, ascending = ['duration', 'begin_dt'], [False, True]

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    selected = []
    for i, row in df.iterrows():
        keep = True
        for j in selected:
            s = df.loc[j]
            # strict overlap: (a_begin < b_end) and (b_begin < a_end)
            if (row['begin_dt'] < s['end_dt']) and (s['begin_dt'] < row['end_dt']):
                keep = False
                break
        if keep:
            selected.append(i)

    out = df.loc[selected].copy()
    out = out.sort_values('begin_dt').reset_index(drop=True)
    return out.drop(columns=['begin_dt', 'end_dt'], errors='ignore')


# === PROCESS FILE ===
def process_file_cold(epw_path, methods=('ensemble', 'gnn', 'evt'), winter_months=(12, 1, 2)):
    try:
        epw_df = pd.read_csv(epw_path, skiprows=8, header=None, names=EPW_COLUMNS)
    except Exception as e:
        print(f" Could not read EPW file {epw_path}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    daily_df = calculate_daily_tmin_tmax_cold(epw_df)
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
                events_df = identify_cold_spells(daily_df.copy(), method=method,
                                                 winter_months=winter_months)

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

    # Resolve overlaps within this file's combined events
    if not all_events_df.empty and 'end_date' in all_events_df.columns:
        all_events_df = resolve_strict_overlaps_cold(all_events_df)

    all_stats_df = pd.concat(all_stats).groupby('year').agg({
        'cwn': 'sum',
        'cwf': 'sum',
        'cwd': 'max',
        'cwdm': 'mean',
        'cwaa': 'min',  # placeholder aggregation; selection uses score below
        'avg_wind_speed': 'mean',
        'severity': 'min'
    }).reset_index()

    if all_stats_df.empty:
        print(f" Cold spell stats were empty after aggregation in {epw_path}.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Robust peak-year selector even when cwaa is positive
    K = all_stats_df['cwaa'].max()
    _score = all_stats_df['cwf'] * (K - all_stats_df['cwaa'])
    peak_year = int(all_stats_df.loc[_score.idxmax(), 'year'])
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
        else:
            print(f" Skipped empty file: {path}")

    safe_save(base_events, "coldspells_events_base.csv")
    safe_save(base_stats, "coldspells_stats_base.csv")
    safe_save(all_events, "coldspells_events.csv")
    safe_save(all_stats, "coldspells_stats.csv")
    safe_save(peak_events, "coldspells_events_peak.csv")
    safe_save(peak_stats, "coldspells_stats_peak.csv")


# === MAIN DRIVER ===
def run_full_pipeline_cold(epw_dir, base_dir, output_dir, winter_months=None):
    """Detect cold spells in every AMY in `epw_dir` and in the base TMY.

    Writes up to six CSVs to `output_dir` (empty tables are skipped):
    events and stats for the base file, for all AMY years, and for the
    peak-severity year.

    `winter_months` defaults to the local winter derived from the base EPW's
    latitude (DJF north, JJA south, all months near the equator).
    """
    print(f" Starting cold spell detection on EPWs in {epw_dir}")

    base_files = sorted(f for f in os.listdir(base_dir) if f.endswith('.epw'))
    if not base_files:
        raise FileNotFoundError("No base EPW file found.")
    base_path = os.path.join(base_dir, base_files[0])
    if winter_months is None:
        plan = season_plan_for_epw(base_path)
        winter_months = plan.cold_months
        print(f" Seasons: {plan.describe()}")
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
            stats_df, events_df, _, _ = process_file_cold(epw_path, methods=['ensemble', 'gnn', 'evt'],
                                                          winter_months=winter_months)
            if not stats_df.empty:
                all_stats_list.append(stats_df)
            if not events_df.empty:
                all_events_list.append(events_df)
        except Exception as e:
            print(f" Failed on {epw_file}: {e}")

    epw_stats = pd.concat(all_stats_list, ignore_index=True) if all_stats_list else pd.DataFrame()
    epw_events = pd.concat(all_events_list, ignore_index=True) if all_events_list else pd.DataFrame()

    if not epw_stats.empty:
        if not epw_events.empty:
            # De-dup, sort, and resolve overlaps PER YEAR
            ev = epw_events.drop_duplicates(subset=[
                'begin_date', 'end_date', 'duration', 'avg_tmin', 'min_tmin', 'avg_wind_speed'
            ]).copy()
            ev['begin_dt'] = pd.to_datetime(ev['begin_date'], format='%d/%m/%Y', errors='coerce')
            ev = ev.sort_values(['year', 'begin_dt']).drop(columns='begin_dt')

            ev_resolved = (
                ev.groupby('year', group_keys=False)
                  .apply(lambda g: resolve_strict_overlaps_cold(g).assign(year=g.name), include_groups=False)
                  .reset_index(drop=True)
            )

            # Recompute per-year stats FROM resolved events
            # cwaa = weighted average of avg_tmin across all cold-spell days in the year
            grp = ev_resolved.groupby('year')
            cwn = grp['duration'].size().rename('cwn')
            cwf = grp['duration'].sum().rename('cwf')
            cwd = grp['duration'].max().rename('cwd')
            cwdm = grp['duration'].mean().rename('cwdm')
            avg_ws = grp['avg_wind_speed'].mean().rename('avg_wind_speed')
            cwaa_series = (grp.apply(lambda g: (g['avg_tmin'] * g['duration']).sum() / g['duration'].sum(), include_groups=False)
                              .rename('cwaa'))

            epw_stats = pd.concat([cwn, cwf, cwd, cwdm, cwaa_series, avg_ws], axis=1).reset_index().sort_values('year')

            # keep severity column (negative product)
            epw_stats['severity'] = epw_stats['cwf'] * epw_stats['cwaa']

            # Robust peak-year selector
            K = epw_stats['cwaa'].max()
            _score = epw_stats['cwf'] * (K - epw_stats['cwaa'])
            peak_year = int(epw_stats.loc[_score.idxmax(), 'year'])
            peak_events = ev_resolved[ev_resolved['year'] == peak_year].reset_index(drop=True)

            print("\n Severity scores across all years:")
            for _, row in epw_stats[['year', 'severity']].iterrows():
                print(f"   - Year {int(row['year'])}: Severity = {row['severity']:.2f}")

            print("\n Ensemble method run complete.")
            print(f" Peak severity year identified: {peak_year}")

            print("\n Event Stats for Peak Year:")
            # Peak stats computed consistently with the cwaa definition (weighted avg Tmin)
            _cwf = int(peak_events['duration'].sum()) if not peak_events.empty else 0
            _cwn = int(len(peak_events)) if not peak_events.empty else 0
            _cwd = int(peak_events['duration'].max()) if not peak_events.empty else 0
            _cwdm = float(peak_events['duration'].mean()) if not peak_events.empty else np.nan
            _cwaa = float((peak_events['avg_tmin'] * peak_events['duration']).sum() / peak_events['duration'].sum()) if not peak_events.empty else np.nan
            _avg_ws = float(peak_events['avg_wind_speed'].mean()) if not peak_events.empty else np.nan
            _sev = float(_cwf * _cwaa) if not np.isnan(_cwaa) else np.nan

            peak_stats = pd.DataFrame([{
                'year': peak_year,
                'cwn': _cwn,
                'cwf': _cwf,
                'cwd': _cwd,
                'cwdm': _cwdm,
                'cwaa': _cwaa,
                'avg_wind_speed': _avg_ws,
                'severity': _sev
            }])
            for col in ['cwn', 'cwf', 'cwd', 'cwdm', 'cwaa', 'severity']:
                val = peak_stats.iloc[0][col]
                try:
                    print(f"   - {col}: {val:.2f}")
                except Exception:
                    print(f"   - {col}: {val}")
            print()
        else:
            print(" No cold spells detected in EPWs.")
            peak_stats = pd.DataFrame()
            peak_events = pd.DataFrame()

    else:
        print(" No cold spells detected in EPWs.")
        peak_stats = pd.DataFrame()
        peak_events = pd.DataFrame()

    save_final_outputs_cold_safe(output_dir, base_stats, base_events, epw_stats, epw_events, peak_stats, peak_events)
    print("✅ Saved all 6 cold spell output CSVs.")
