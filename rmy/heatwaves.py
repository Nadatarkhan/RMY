"""Heat-wave detection (ensemble of static thresholds, a stage-2 filter and EVT).

Ported from the canonical notebook, section "1.1 Heatwaves v2 (feb26)".
Importing this module has no side effects; call `run_full_pipeline` to run.

Known gaps, kept here on purpose and handled in later phases:
- `apply_gnn_detection` is a z-score prefilter, not the paper's GNN.
- Runs longer than `max_duration` (21 days) are dropped, not clipped.
- Hemisphere is not handled; the warm-season filter is inferred per file.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from .utils import calculate_heat_index, read_epw_raw

warnings.filterwarnings("ignore", category=UserWarning, message="Parsing dates.*")


def calculate_daily_tmin_tmax(epw_data):
    epw_data = epw_data.copy()

    epw_data['temp_air'] = pd.to_numeric(epw_data['temp_air'], errors='coerce')
    epw_data['relative_humidity'] = pd.to_numeric(epw_data['relative_humidity'], errors='coerce')

    epw_data = epw_data[
        (epw_data['temp_air'].notna()) &
        (epw_data['relative_humidity'].notna()) &
        (epw_data['temp_air'] != -9999) &
        (epw_data['relative_humidity'] != -9999)
    ]

    daily = epw_data.groupby(['year', 'month', 'day'], as_index=False).agg({
        'temp_air': ['min', 'max'],
        'relative_humidity': 'mean'
    })

    daily.columns = ['YEAR', 'MONTH', 'DAY', 'Tmin (oC)', 'Tmax (oC)', 'Humidity (%)']
    daily['date'] = pd.to_datetime(
        dict(year=daily['YEAR'], month=daily['MONTH'], day=daily['DAY']),
        errors='coerce'
    )

    daily = daily.dropna(subset=['date']).reset_index(drop=True)
    return daily


def infer_warm_season_months(df, warm_month_window_c=1.0):
    """
    Infer the warm-season months from the file itself.
    Keeps months whose mean Tmax is within warm_month_window_c of the hottest month.
    For Nairobi this usually yields Jan-Feb-Mar and sometimes Oct.
    """
    if df.empty:
        return None

    monthly_means = df.groupby('MONTH')['Tmax (oC)'].mean()
    if monthly_means.empty:
        return None

    hottest_mean = monthly_means.max()
    warm_months = monthly_means[
        monthly_means >= (hottest_mean - warm_month_window_c)
    ].index.tolist()

    return warm_months


def build_condition_runs(condition, min_duration=3, max_duration=21):
    """
    Convert a boolean day-level condition into start/end event indices.
    """
    condition = pd.Series(condition).fillna(False).astype(bool).reset_index(drop=True)
    groups = (condition != condition.shift()).cumsum()

    events = []
    for _, idx in condition.groupby(groups).groups.items():
        idx = list(idx)

        if not condition.iloc[idx[0]]:
            continue

        start, end = idx[0], idx[-1]
        duration = end - start + 1

        if duration >= min_duration and (max_duration is None or duration <= max_duration):
            events.append((start, end))

    if events:
        return pd.DataFrame(events, columns=['Start', 'End'])
    return pd.DataFrame(columns=['Start', 'End'])


def identify_heatwaves_static(
    df,
    method='ensemble',
    tmin_q=0.4,
    tmax_q=0.9,
    tmax_th=None,
    min_duration=3,
    max_duration=21,
    warm_season_only=True,
    warm_month_window_c=1.0
):
    """
    Static detector:
    - percentile: annual percentile thresholds
    - temperature: fixed or adaptive Tmax threshold
    - ensemble: union of both

    Nairobi-oriented defaults:
    - annual Tmax percentile works well
    - fixed Tmax threshold of 29C is more realistic than 35C
    - warm-season filtering prevents relative anomalies in cool months
    """
    if df.empty:
        return pd.DataFrame(columns=['Start', 'End'])

    warm_months = (
        infer_warm_season_months(df, warm_month_window_c=warm_month_window_c)
        if warm_season_only else None
    )

    if method == 'ensemble':
        df1 = identify_heatwaves_static(
            df,
            method='percentile',
            tmin_q=tmin_q,
            tmax_q=tmax_q,
            min_duration=min_duration,
            max_duration=max_duration,
            warm_season_only=warm_season_only,
            warm_month_window_c=warm_month_window_c
        )

        df2 = identify_heatwaves_static(
            df,
            method='temperature',
            tmax_th=tmax_th,
            min_duration=min_duration,
            max_duration=max_duration,
            warm_season_only=warm_season_only,
            warm_month_window_c=warm_month_window_c
        )

        combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates().reset_index(drop=True)
        return combined

    elif method == 'percentile':
        tmin_threshold = df['Tmin (oC)'].quantile(tmin_q)
        tmax_threshold = df['Tmax (oC)'].quantile(tmax_q)

        condition = (
            (df['Tmin (oC)'] >= tmin_threshold) &
            (df['Tmax (oC)'] >= tmax_threshold)
        )

    elif method == 'temperature':
        if tmax_th is None:
            # adaptive fallback if no fixed threshold is provided
            tmax_threshold = df['Tmax (oC)'].quantile(0.90)
        else:
            tmax_threshold = tmax_th

        condition = df['Tmax (oC)'] >= tmax_threshold

    else:
        raise ValueError("Invalid method")

    if warm_months:
        condition = condition & df['MONTH'].isin(warm_months)

    return build_condition_runs(
        condition,
        min_duration=min_duration,
        max_duration=max_duration
    )


def identify_evt_extremes(
    df,
    quantile_threshold=0.95,
    min_duration=3,
    warm_season_only=True,
    warm_month_window_c=1.0
):
    """
    Identify extreme heatwave events using EVT (Peaks Over Threshold).
    Returns events above a fitted return level.

    Notes:
    - fixed the original bug where EVT never actually ran
    - uses an empirical floor to avoid unstable fits on short series
    """
    if df.empty:
        return pd.DataFrame(columns=['Start', 'End'])

    tmax_series = df['Tmax (oC)'].dropna()
    if tmax_series.empty:
        return pd.DataFrame(columns=['Start', 'End'])

    threshold = tmax_series.quantile(quantile_threshold)
    excesses = tmax_series[tmax_series > threshold] - threshold

    empirical_floor = tmax_series.quantile(0.97)

    if len(excesses) >= 5 and excesses.std(ddof=0) > 0:
        c, loc, scale = genpareto.fit(excesses, floc=0)

        # slightly less aggressive than a second 95th on the GPD tail
        return_level = threshold + genpareto.ppf(0.80, c, loc=0, scale=scale)

        if not np.isfinite(return_level):
            return_level = empirical_floor
        else:
            return_level = max(return_level, empirical_floor)
    else:
        return_level = empirical_floor

    condition = df['Tmax (oC)'] >= return_level

    if warm_season_only:
        warm_months = infer_warm_season_months(df, warm_month_window_c=warm_month_window_c)
        if warm_months:
            condition = condition & df['MONTH'].isin(warm_months)

    return build_condition_runs(
        condition,
        min_duration=min_duration,
        max_duration=None
    )


def apply_gnn_detection(df, z_thresh=2.5):
    """
    Robust GNN-like year prefilter.
    Works whether columns are already named or still positional.
    Uses daily max temp z-scores by year, but avoids datetime issues with EPW hour formatting.
    """
    lower_map = {str(c).lower(): c for c in df.columns}
    needed = {}

    for key in ('year', 'month', 'day', 'temp_air'):
        if key in lower_map:
            needed[key] = lower_map[key]

    if len(needed) == 4:
        df_tmp = df.rename(columns={
            needed['year']: 'year',
            needed['month']: 'month',
            needed['day']: 'day',
            needed['temp_air']: 'temp_air'
        }).copy()
    else:
        df_tmp = df.copy()
        df_tmp.columns = list(range(df_tmp.shape[1]))
        fallback = {0: 'year', 1: 'month', 2: 'day', 6: 'temp_air'}
        df_tmp = df_tmp.rename(columns={k: v for k, v in fallback.items() if k in df_tmp.columns})

    missing = [c for c in ('year', 'month', 'day', 'temp_air') if c not in df_tmp.columns]
    if missing:
        return []

    df_tmp = df_tmp[['year', 'month', 'day', 'temp_air']].dropna()
    df_tmp = df_tmp[df_tmp['temp_air'] != -9999]

    if df_tmp.empty:
        return []

    daily_max = (
        df_tmp.groupby(['year', 'month', 'day'])['temp_air']
        .max()
        .reset_index()
    )

    if len(daily_max) < 5 or daily_max['temp_air'].std(ddof=0) == 0:
        return []

    z_scores = (
        (daily_max['temp_air'] - daily_max['temp_air'].mean()) /
        daily_max['temp_air'].std(ddof=0)
    )

    extreme_years = daily_max.loc[z_scores > z_thresh, 'year'].astype(int).unique().tolist()
    return sorted(extreme_years)


def calculate_event_stats(heatwave_df, base_df):
    if heatwave_df.empty:
        return pd.DataFrame(), pd.DataFrame(), None

    events = []
    stats = {}

    for _, row in heatwave_df.iterrows():
        s_idx = int(row['Start'])
        e_idx = int(row['End'])

        spell = base_df.iloc[s_idx:e_idx + 1].copy()
        if spell.empty:
            continue

        year = int(spell.iloc[0]['YEAR'])
        duration = e_idx - s_idx + 1
        avg_tmax = round(float(spell['Tmax (oC)'].mean()), 1)
        std_tmax = round(float(spell['Tmax (oC)'].std(ddof=0)), 1)
        max_tmax = round(float(spell['Tmax (oC)'].max()), 1)
        avg_hi = round(float(calculate_heat_index(spell['Tmax (oC)'], spell['Humidity (%)']).mean()), 1)

        bdate = pd.Timestamp(
            year=year,
            month=int(spell.iloc[0]['MONTH']),
            day=int(spell.iloc[0]['DAY'])
        )
        edate = pd.Timestamp(
            year=year,
            month=int(spell.iloc[-1]['MONTH']),
            day=int(spell.iloc[-1]['DAY'])
        )

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
            stats[year] = {
                'hwn': 0,
                'hwf': 0,
                'hwd': 0,
                'hwdm': [],
                'hwaa': -np.inf
            }

        stats[year]['hwn'] += 1
        stats[year]['hwf'] += duration
        stats[year]['hwd'] = max(stats[year]['hwd'], duration)
        stats[year]['hwdm'].append(duration)
        stats[year]['hwaa'] = max(stats[year]['hwaa'], max_tmax)

    if not stats:
        return pd.DataFrame(), pd.DataFrame(), None

    for y in stats:
        stats[y]['hwdm'] = float(np.mean(stats[y]['hwdm']))

    stats_df = (
        pd.DataFrame.from_dict(stats, orient='index')
        .reset_index()
        .rename(columns={'index': 'year'})
        .sort_values(by='year')
        .reset_index(drop=True)
    )

    stats_df['year'] = stats_df['year'].astype(int)
    stats_df['severity'] = stats_df['hwf'] * stats_df['hwaa']

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df['begin_dt'] = pd.to_datetime(events_df['begin_date'], format='%d/%m/%Y', errors='coerce')
        events_df = events_df.sort_values(by='begin_dt').drop(columns='begin_dt').reset_index(drop=True)

    peak_year = int(stats_df.loc[stats_df['severity'].idxmax(), 'year'])
    return stats_df, events_df, peak_year


def resolve_strict_overlaps(events_df):
    """
    Remove strictly-overlapping events. Keep the longer event.
    Tie-breakers: higher max_tmax, then earlier begin_date.
    Strict overlap means (a_begin < b_end) and (b_begin < a_end).
    """
    if events_df is None or events_df.empty or 'begin_date' not in events_df.columns:
        return events_df

    df = events_df.copy()

    df['begin_dt'] = pd.to_datetime(df['begin_date'], format='%d/%m/%Y', errors='coerce')
    df['end_dt'] = pd.to_datetime(df['end_date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['begin_dt', 'end_dt'])

    df = df.sort_values(
        ['duration', 'max_tmax', 'begin_dt'],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    selected_idxs = []

    for i, row in df.iterrows():
        keep = True
        for j in selected_idxs:
            s = df.loc[j]
            if (row['begin_dt'] < s['end_dt']) and (s['begin_dt'] < row['end_dt']):
                keep = False
                break
        if keep:
            selected_idxs.append(i)

    out = df.loc[selected_idxs].copy()
    out = out.sort_values('begin_dt').reset_index(drop=True)
    return out.drop(columns=['begin_dt', 'end_dt'], errors='ignore')


def recompute_stats_from_events(events_df):
    """
    Recompute yearly summary stats from the final overlap-resolved event list.
    This avoids double-counting when the same event is found by multiple methods.
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=['year', 'hwn', 'hwf', 'hwd', 'hwdm', 'hwaa', 'severity'])

    stats = (
        events_df.groupby('year')
        .agg(
            hwn=('duration', 'size'),
            hwf=('duration', 'sum'),
            hwd=('duration', 'max'),
            hwdm=('duration', 'mean'),
            hwaa=('max_tmax', 'max')
        )
        .reset_index()
        .sort_values('year')
    )

    stats['severity'] = stats['hwf'] * stats['hwaa']
    return stats


def process_file(epw_path, gnn_years=None, apply_gnn=True):
    epw_df = read_epw_raw(epw_path)

    daily_df = calculate_daily_tmin_tmax(epw_df)

    if daily_df.empty:
        return (
            pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame()
        )

    daily_df['heat_index'] = calculate_heat_index(
        daily_df['Tmax (oC)'],
        daily_df['Humidity (%)']
    )

    # Static detection
    static_events = identify_heatwaves_static(
        daily_df,
        method='ensemble',
        tmax_th=30.0,
        min_duration=3,
        max_duration=21,
        warm_season_only=True,
        warm_month_window_c=1.0
    )
    static_stats, static_df, _ = calculate_event_stats(static_events, daily_df)

    # GNN filtering (optional)
    if apply_gnn and gnn_years:
        filtered_df = daily_df[daily_df['YEAR'].isin(gnn_years)].reset_index(drop=True)
    else:
        filtered_df = pd.DataFrame(columns=daily_df.columns)

    if not filtered_df.empty:
        gnn_events = identify_heatwaves_static(
            filtered_df,
            method='temperature',
            tmax_th=30.0,
            min_duration=3,
            max_duration=21,
            warm_season_only=True,
            warm_month_window_c=1.0
        )
        gnn_stats, gnn_df, _ = calculate_event_stats(gnn_events, filtered_df)
    else:
        gnn_stats, gnn_df = pd.DataFrame(), pd.DataFrame()

    # EVT detection
    evt_events = identify_evt_extremes(
        daily_df,
        quantile_threshold=0.95,
        min_duration=3,
        warm_season_only=True,
        warm_month_window_c=1.0
    )
    evt_stats, evt_df, _ = calculate_event_stats(evt_events, daily_df)

    # Combine events
    all_events = pd.concat([static_df, gnn_df, evt_df], ignore_index=True).drop_duplicates()

    if 'begin_date' in all_events.columns and not all_events.empty:
        all_events['begin_dt'] = pd.to_datetime(all_events['begin_date'], format='%d/%m/%Y', errors='coerce')
        all_events = all_events.sort_values('begin_dt').drop(columns='begin_dt').reset_index(drop=True)
        all_events = resolve_strict_overlaps(all_events)
    else:
        all_events = pd.DataFrame(columns=[
            'begin_date', 'end_date', 'duration', 'avg_tmax', 'std_tmax',
            'max_tmax', 'avg_heat_index', 'year'
        ])

    # Recompute final stats from resolved events
    all_stats = recompute_stats_from_events(all_events)

    if all_stats.empty:
        return (
            all_stats, all_events,
            pd.DataFrame(), pd.DataFrame(),
            static_stats, static_df
        )

    peak_year = int(all_stats.loc[all_stats['severity'].idxmax(), 'year'])
    peak_events = all_events[all_events['year'] == peak_year].reset_index(drop=True)
    peak_stats = all_stats[all_stats['year'] == peak_year].reset_index(drop=True)

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
    """Detect heat waves in every AMY in `epw_dir` and in the base TMY.

    Writes six CSVs to `output_dir`: events and stats for the base file
    (`*_base.csv`), for all AMY years, and for the peak-severity year
    (`*_peak.csv`).
    """
    print(f"Starting hybrid method on EPWs in {epw_dir}")

    # Load and process base EPW
    base_files = sorted(f for f in os.listdir(base_dir) if f.endswith('.epw'))
    if not base_files:
        raise FileNotFoundError("No base EPW file found in base folder.")

    base_epw_path = os.path.join(base_dir, base_files[0])

    # Read base file for GNN prefilter
    df_base = read_epw_raw(base_epw_path)

    try:
        gnn_years = apply_gnn_detection(df_base)
    except Exception as e:
        print(f"⚠️ GNN prefilter failed on base EPW ({e}). Continuing without GNN.")
        gnn_years = []

    # Process base file
    base_stats, base_events, _, _, _, _ = process_file(
        base_epw_path,
        gnn_years=gnn_years,
        apply_gnn=True
    )

    if base_stats.empty:
        print(f"No valid stats found in {base_epw_path}.")
        base_stats, base_events = pd.DataFrame(), pd.DataFrame()

    # Process all EPW files in EPWs folder
    all_stats_df = []
    all_events_df = []

    for epw_file in sorted(os.listdir(epw_dir)):
        if not epw_file.endswith(".epw"):
            continue

        epw_path = os.path.join(epw_dir, epw_file)
        print(f"Processing file: {epw_path}")

        try:
            stats_df, events_df, _, _, _, _ = process_file(
                epw_path,
                gnn_years=None,
                apply_gnn=False
            )

            if not stats_df.empty:
                all_stats_df.append(stats_df)

            if not events_df.empty:
                all_events_df.append(events_df)

        except Exception as e:
            print(f"Error processing {epw_path}: {e}")

    epw_events = pd.concat(all_events_df, ignore_index=True) if all_events_df else pd.DataFrame()

    if not epw_events.empty:
        ev = epw_events.drop_duplicates(subset=[
            'begin_date', 'end_date', 'duration', 'avg_tmax', 'max_tmax', 'avg_heat_index', 'year'
        ]).copy()

        ev['begin_dt'] = pd.to_datetime(ev['begin_date'], format='%d/%m/%Y', errors='coerce')
        ev = ev.sort_values(['year', 'begin_dt']).drop(columns='begin_dt').reset_index(drop=True)

        resolved_list = []
        for year, g in ev.groupby('year'):
            g_res = resolve_strict_overlaps(g).copy()
            g_res['year'] = year
            resolved_list.append(g_res)

        ev_resolved = pd.concat(resolved_list, ignore_index=True) if resolved_list else pd.DataFrame()

        epw_stats = recompute_stats_from_events(ev_resolved)

        print("\n📈 Severity scores across all years:")
        for _, row in epw_stats[['year', 'severity']].iterrows():
            print(f"   - Year {int(row['year'])}: Severity = {row['severity']:.2f}")

        peak_year = int(epw_stats.loc[epw_stats['severity'].idxmax(), 'year'])
        peak_events = ev_resolved[ev_resolved['year'] == peak_year].reset_index(drop=True)
        peak_events = resolve_strict_overlaps(peak_events)

        peak_stats = recompute_stats_from_events(peak_events)
        if not peak_stats.empty:
            peak_stats['year'] = peak_year

        print("\nEnsemble method run complete.")
        print(f"Peak severity year identified: {peak_year}")

        if not peak_stats.empty:
            print("\nEvent Stats for Peak Year:")
            for col in ['hwn', 'hwf', 'hwd', 'hwdm', 'hwaa', 'severity']:
                val = peak_stats.iloc[0][col]
                try:
                    print(f"   - {col}: {float(val):.2f}")
                except (TypeError, ValueError):
                    print(f"   - {col}: {val}")
            print()

        epw_events = ev_resolved

    else:
        print("No heatwaves detected in EPWs.")
        epw_stats = pd.DataFrame()
        peak_stats = pd.DataFrame()
        peak_events = pd.DataFrame()

    save_final_outputs(
        output_dir,
        base_stats,
        base_events,
        epw_stats,
        epw_events,
        peak_stats,
        peak_events
    )

    print("✅ Saved all 6 heat wave output CSVs")
