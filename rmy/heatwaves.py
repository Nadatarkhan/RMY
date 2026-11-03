"""Heat-wave detection: thresholds, the paper's GNN, and EVT (paper Section 2.3).

Ported from the canonical notebook, section "1.1 Heatwaves v2 (feb26)", with
the notebook's z-score stand-in replaced by the GNN in rmy.gnn. Events from
all three methods are pooled and overlaps resolved ("methods are applied in
sequence but also contribute additively to the final event set").
Importing this module has no side effects; call `run_full_pipeline` to run.

EVT uses the heat index and a POT threshold selected on all AMYs pooled
(rmy.evt), as in the paper.

There is no upper limit on event length (MAX_EVENT_DAYS). The notebook
dropped runs longer than 21 days entirely (review note item 1).
"""
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evt import select_pot_threshold
from .gnn import GNNConfig, detect_gnn_anomalies, gnn_flags_for
from .seasons import season_plan_for_epw
from .utils import calculate_heat_index, list_epws, read_epw_raw, report_runner_up

warnings.filterwarnings("ignore", category=UserWarning, message="Parsing dates.*")

# Longest event kept, in days. None keeps runs of any length.
MAX_EVENT_DAYS = None

# Fixed Tmax threshold (C) of the 'temperature' method. The notebook lowered
# the paper's 35 C to 30 C for Nairobi-like climates; it is raised to the
# site's own Tmax percentile where that is higher (see heat_thresholds).
FIXED_TMAX_C = 30.0


@dataclass(frozen=True)
class HeatThresholds:
    """Heat-wave thresholds from all AMY years pooled (see heat_thresholds)."""
    tmin_pct: float          # 'percentile' method: Tmin at or above this
    tmax_pct: float          # 'percentile' method: Tmax at or above this
    tmax_abs: float          # 'temperature' method: Tmax at or above this
    warm_months: tuple       # heat waves are only counted in these months

    def describe(self):
        return (f'Heat thresholds (all AMY years): percentile Tmin >= {self.tmin_pct:.1f} and '
                f'Tmax >= {self.tmax_pct:.1f}; temperature Tmax >= {self.tmax_abs:.1f}; '
                f'months {list(self.warm_months)}')


def heat_thresholds(pooled, summer_months=(), tmin_q=0.4, tmax_q=0.9,
                    fixed_tmax=FIXED_TMAX_C, warm_month_window_c=1.0):
    """Thresholds from the daily data of all AMY years together.

    - Percentiles are of all days of all years (the paper's "historical
      temperature data"), so a hot year has more heat-wave days than a mild
      one. Per-year percentiles gave every year its own top 10 %.
    - The fixed threshold is raised to the Tmax percentile where that is
      higher. In Phoenix every summer day exceeds 30 C, which turned whole
      months into one event; in cooler climates 30 C stays the threshold.
    - Months: the hemisphere's summer months plus any month whose mean Tmax
      is within `warm_month_window_c` of the hottest month (all years). The
      notebook inferred this per year, often leaving a single month and
      missing events such as Seattle's June 2021 heat dome. At equatorial
      sites (no summer months) only the inferred warm months are used.
    """
    tmin_pct = float(pooled['Tmin (oC)'].quantile(tmin_q))
    tmax_pct = float(pooled['Tmax (oC)'].quantile(tmax_q))
    warm = infer_warm_season_months(pooled, warm_month_window_c=warm_month_window_c) or []
    months = tuple(sorted(set(summer_months) | set(int(m) for m in warm)))
    return HeatThresholds(tmin_pct, tmax_pct, max(fixed_tmax, tmax_pct), months)


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


def build_condition_runs(condition, min_duration=3, max_duration=None):
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
    max_duration=None,
    warm_season_only=True,
    warm_month_window_c=1.0,
    thresholds=None
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

    With `thresholds` (HeatThresholds from all AMY years), its values and
    months are used; otherwise thresholds are computed from `df` alone.
    """
    if df.empty:
        return pd.DataFrame(columns=['Start', 'End'])

    if not warm_season_only:
        warm_months = None
    elif thresholds is not None:
        warm_months = list(thresholds.warm_months)
    else:
        warm_months = infer_warm_season_months(df, warm_month_window_c=warm_month_window_c)

    if method == 'ensemble':
        df1 = identify_heatwaves_static(
            df,
            method='percentile',
            tmin_q=tmin_q,
            tmax_q=tmax_q,
            min_duration=min_duration,
            max_duration=max_duration,
            warm_season_only=warm_season_only,
            warm_month_window_c=warm_month_window_c,
            thresholds=thresholds
        )

        df2 = identify_heatwaves_static(
            df,
            method='temperature',
            tmax_th=tmax_th,
            min_duration=min_duration,
            max_duration=max_duration,
            warm_season_only=warm_season_only,
            warm_month_window_c=warm_month_window_c,
            thresholds=thresholds
        )

        combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates().reset_index(drop=True)
        return combined

    elif method == 'percentile':
        if thresholds is not None:
            tmin_threshold, tmax_threshold = thresholds.tmin_pct, thresholds.tmax_pct
        else:
            tmin_threshold = df['Tmin (oC)'].quantile(tmin_q)
            tmax_threshold = df['Tmax (oC)'].quantile(tmax_q)

        condition = (
            (df['Tmin (oC)'] >= tmin_threshold) &
            (df['Tmax (oC)'] >= tmax_threshold)
        )

    elif method == 'temperature':
        if thresholds is not None:
            tmax_threshold = thresholds.tmax_abs
        elif tmax_th is None:
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
    evt=None,
    min_duration=3,
    warm_season_only=True,
    warm_month_window_c=1.0,
    warm_months=None
):
    """
    EVT-POT heat waves (paper Section 2.2.3): runs of at least `min_duration`
    days with the heat index above the POT threshold.

    `evt` is a rmy.evt.POTThreshold, normally selected on all AMYs pooled.
    Without it, the threshold is selected on this file alone.
    """
    if df.empty or 'heat_index' not in df.columns:
        return pd.DataFrame(columns=['Start', 'End'])
    if evt is None:
        evt = select_pot_threshold(df['heat_index'], df['date'], tail='upper')
        if evt is None:
            return pd.DataFrame(columns=['Start', 'End'])

    condition = pd.Series(evt.flags(df['heat_index']), index=df.index)

    if warm_season_only:
        if warm_months is None:
            warm_months = infer_warm_season_months(df, warm_month_window_c=warm_month_window_c)
        if warm_months:
            condition = condition & df['MONTH'].isin(list(warm_months))

    return build_condition_runs(
        condition,
        min_duration=min_duration,
        max_duration=None
    )


def pooled_daily(epw_paths):
    """Daily Tmin/Tmax/humidity/heat index for all files, for pooled thresholds."""
    frames = []
    for p in epw_paths:
        d = calculate_daily_tmin_tmax(read_epw_raw(p))
        d['heat_index'] = calculate_heat_index(d['Tmax (oC)'], d['Humidity (%)']).to_numpy()
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def exceedance_severity(events_df, reference, value_col='avg_tmax', sign=1):
    """Cumulative exceedance per year: sum over events of (excess beyond `reference`) x duration.

    Paper Eq. 4, S = sum(event amplitude x event duration), with the amplitude
    measured beyond the detection threshold rather than from 0 C. This is the
    idea of the Heat Wave Magnitude Index (Russo et al.) and of Meteo-France's
    heat-wave intensity (Ouzeau et al. 2016): a year scores by how far and for
    how long it went past the threshold, so one extreme event can outrank many
    mild ones. `sign` is 1 for heat (above) and -1 for cold (below).
    """
    excess = (sign * (events_df[value_col] - reference)).clip(lower=0)
    return (excess * events_df['duration']).groupby(events_df['year']).sum()


def recompute_stats_from_events(events_df, reference=None):
    """
    Recompute yearly summary stats from the final overlap-resolved event list.
    This avoids double-counting when the same event is found by multiple methods.

    `severity` is the cumulative exceedance above `reference` (C x days; see
    exceedance_severity) when a reference is given, otherwise the notebook's
    total heat-wave days x hottest Tmax.
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

    if reference is None:
        stats['severity'] = stats['hwf'] * stats['hwaa']
    else:
        stats['severity'] = stats['year'].map(exceedance_severity(events_df, reference)).fillna(0.0)
    return stats


def process_file(epw_path, gnn_daily=None, evt=None, thresholds=None):
    """Detect heat waves in one EPW.

    `gnn_daily` is the output of rmy.gnn.detect_gnn_anomalies for the AMY set
    this file belongs to; its hot-anomaly days become GNN events. Leave it as
    None for files outside that set (such as the base TMY). `evt` is the
    pooled EVT threshold (rmy.evt.POTThreshold); without it EVT selects a
    threshold on this file alone. `thresholds` (HeatThresholds) sets the
    static thresholds and the months for all three methods; without it they
    come from this file alone, as in the notebook.
    """
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
        tmax_th=FIXED_TMAX_C,
        min_duration=3,
        max_duration=MAX_EVENT_DAYS,
        warm_season_only=True,
        warm_month_window_c=1.0,
        thresholds=thresholds
    )
    static_stats, static_df, _ = calculate_event_stats(static_events, daily_df)

    # GNN: runs of days the graph autoencoder flags as hot anomalies
    if gnn_daily is not None:
        condition = pd.Series(gnn_flags_for(daily_df, gnn_daily, 'hot'))
        warm_months = (list(thresholds.warm_months) if thresholds is not None
                       else infer_warm_season_months(daily_df, warm_month_window_c=1.0))
        if warm_months:
            condition &= daily_df['MONTH'].isin(warm_months).to_numpy()
        gnn_events = build_condition_runs(condition, min_duration=3, max_duration=MAX_EVENT_DAYS)
        gnn_stats, gnn_df, _ = calculate_event_stats(gnn_events, daily_df)
    else:
        gnn_stats, gnn_df = pd.DataFrame(), pd.DataFrame()

    # EVT detection
    evt_events = identify_evt_extremes(
        daily_df,
        evt=evt,
        warm_months=thresholds.warm_months if thresholds is not None else None,
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
    all_stats = recompute_stats_from_events(
        all_events, thresholds.tmax_abs if thresholds is not None else None)

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


def run_full_pipeline(epw_dir, base_dir, output_dir, use_gnn=True, gnn_config=None):
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

    # Thresholds and months from all AMY years pooled
    pooled = pooled_daily(list_epws(epw_dir))
    if pooled.empty:
        raise ValueError(f"No usable AMY data in {epw_dir}")
    thresholds = heat_thresholds(pooled, season_plan_for_epw(base_epw_path).summer_months)
    print(thresholds.describe())

    # Select the EVT-POT threshold on all AMY days pooled
    evt = select_pot_threshold(pooled['heat_index'], pooled['date'], tail='upper')
    print(evt.describe('heat index') if evt else "EVT: no valid heat-index threshold; EVT skipped.")

    # Process base file (not part of the AMY set the GNN is trained on)
    base_stats, base_events, _, _, _, _ = process_file(base_epw_path, thresholds=thresholds)

    # Train the GNN once on all AMYs (paper: "across the 15 years analyzed")
    gnn_daily = None
    if use_gnn:
        try:
            gnn_daily = detect_gnn_anomalies(list_epws(epw_dir), gnn_config or GNNConfig())
        except Exception as e:
            print(f"⚠️ GNN failed ({e}). Continuing with thresholds and EVT only.")

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
            stats_df, events_df, _, _, _, _ = process_file(epw_path, gnn_daily=gnn_daily, evt=evt,
                                                           thresholds=thresholds)

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

        epw_stats = recompute_stats_from_events(ev_resolved, thresholds.tmax_abs)

        print("\n📈 Severity scores across all years:")
        for _, row in epw_stats[['year', 'severity']].iterrows():
            print(f"   - Year {int(row['year'])}: Severity = {row['severity']:.2f}")

        peak_year = int(epw_stats.loc[epw_stats['severity'].idxmax(), 'year'])
        report_runner_up(epw_stats['year'], epw_stats['severity'], 'Heat-wave')
        peak_events = ev_resolved[ev_resolved['year'] == peak_year].reset_index(drop=True)
        peak_events = resolve_strict_overlaps(peak_events)

        peak_stats = recompute_stats_from_events(peak_events, thresholds.tmax_abs)
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
