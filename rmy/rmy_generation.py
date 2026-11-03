"""Assemble an RMY: insert peak-year extreme events into a base TMY.

Ported from the canonical notebook, section "3.0 Generate RMYs"
(single-location RMY assembly with 8-hour smoothed insertion boundaries).

Steps:
1. Insert the peak heat-wave and cold-spell event days from the peak AMY years.
2. Insert compensating AMY days so monthly means stay close to the base TMY:
   summer and winter at Northern/Southern sites, each month on its own at
   equatorial sites (see rmy.seasons).
3. Blend the hours around every inserted block once, after all insertions.
   Inserted AMY hours are never modified.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .coldspells import run_full_pipeline_cold
from .heatwaves import run_full_pipeline
from .seasons import season_plan_for_epw
from .utils import (
    WEATHER_COLS, extract_original_header, find_epw_by_year, get_peak_year,
    list_epws, read_epw_file, safe_load_events,
)

TRANSITION_HOURS = 8
SMOOTH_COLS = [
    'temp_air', 'temp_dew', 'relative_humidity',
    'atmospheric_pressure', 'wind_speed'
]
HEAT_EVENT_COLS = ['begin_date', 'end_date', 'duration', 'avg_tmax', 'std_tmax', 'max_tmax']
COLD_EVENT_COLS = ['begin_date', 'end_date', 'duration', 'avg_tmin', 'std_tmin', 'min_tmin']


def match_events(base_df, peak_df):
    """Pair base-TMY events with peak-year events that start on the same day.

    Among same-day candidates the closest duration wins; each peak event is
    used at most once. Returns (matched pairs, unmatched peak events).
    """
    matched, unmatched = [], peak_df.copy()
    if base_df.empty:
        return matched, unmatched
    for _, base in base_df.iterrows():
        best, min_diff = None, float('inf')
        for idx, peak in unmatched.iterrows():
            if (base['begin_date'].month == peak['begin_date'].month and
                    base['begin_date'].day == peak['begin_date'].day):
                diff = abs(base['duration'] - peak['duration'])
                if diff < min_diff:
                    best, min_diff = idx, diff
        if best is not None:
            matched.append((base, unmatched.loc[best]))
            unmatched = unmatched.drop(best)
    return matched, unmatched


def replace_whole_day(df, source_day, month, day, inserted_hours):
    """Insert all source-hour weather values, preserving baseline EPW timestamps."""
    target_idx = df.index[(df['month'] == month) & (df['day'] == day)]
    if target_idx.empty:
        # e.g. February 29 in a 365-day base TMY.
        return False

    source_day = source_day.sort_values('hour')
    target_idx = df.loc[target_idx].sort_values('hour').index
    if (len(source_day) != len(target_idx) or
            not np.array_equal(source_day['hour'].to_numpy(),
                               df.loc[target_idx, 'hour'].to_numpy())):
        raise ValueError(f'Hours mismatch while inserting {month:02d}-{day:02d}')

    # Per-column assignment preserves numeric float and categorical string types.
    for col in WEATHER_COLS:
        df.loc[target_idx, col] = source_day[col].to_numpy()
    inserted_hours[np.asarray(target_idx, dtype=int)] = True
    return True


def integrate_events(df, matched, unmatched, peak_epw, label, inserted_hours):
    all_events = matched + [(None, e) for e in unmatched.itertuples()]
    replaced = set()
    for _, event in all_events:
        for d in pd.date_range(event.begin_date, event.end_date, freq='D'):
            month, day = int(d.month), int(d.day)
            source_day = peak_epw[(peak_epw['month'] == month) &
                                  (peak_epw['day'] == day)]
            if source_day.empty:
                print(f'WARNING: {label} source lacks {month:02d}-{day:02d}')
                continue
            if replace_whole_day(df, source_day, month, day, inserted_hours):
                replaced.add(f'{month:02d}-{day:02d}')
    print(f'{label} events integrated. Days replaced: {len(replaced)}')
    return df, replaced


def calculate_monthly_avg_conditions(df, months):
    return (df[df['month'].isin(months)]
            .groupby('month')[['temp_air', 'relative_humidity']].mean())


def find_days_to_adjust_avg(epw_files, targets, months, condition):
    """Collect up to 40 candidate compensation days from the AMYs.

    Files are visited in sorted order so the result does not depend on the
    order the filesystem lists them.
    """
    days = {m: [] for m in months}
    for path in sorted(Path(p) for p in epw_files):
        source = read_epw_file(path)
        for month in months:
            for day in range(1, 32):
                subset = source[(source['month'] == month) &
                                (source['day'] == day)]
                if not subset.empty and condition(subset, targets.loc[month]):
                    days[month].append(subset)
                if sum(map(len, days.values())) >= 40:
                    return days
    return days


def integrate_days(df, days, protected_days, months, inserted_hours):
    inserted = set()
    for month in months:
        for day_df in days[month]:
            if len(inserted) >= 30:
                break
            month_value = int(day_df.iloc[0]['month'])
            day_value = int(day_df.iloc[0]['day'])
            tag = f'{month_value:02d}-{day_value:02d}'
            if tag in protected_days or tag in inserted:
                continue
            if replace_whole_day(df, day_df, month_value, day_value, inserted_hours):
                inserted.add(tag)
    return df, inserted


def compensate_monthly(df, base_epw, epw_files, months, protected_days, inserted_hours,
                       temp_tol=0.25, rh_tol=2.0, max_days_per_month=10, max_days_total=60):
    """Pull each month's mean temperature and RH back to the base TMY.

    Used at equatorial sites, where there is no summer or winter. For each
    month whose means moved after event insertion, AMY days from the same
    calendar month are inserted one at a time, each time choosing the day
    that most reduces the remaining deviation, until the month is within
    `temp_tol` (degrees C) and `rh_tol` (percentage points), no candidate
    helps, or a day budget is reached. Event days and already-inserted days
    are never replaced. Months without deviation are left untouched.

    Returns (df, inserted day tags, per-month report DataFrame).
    """
    targets = calculate_monthly_avg_conditions(base_epw, months)
    sources = [read_epw_file(p) for p in sorted(Path(p) for p in epw_files)]
    inserted, report = set(), []

    def objective(dt, drh):
        return (dt / temp_tol) ** 2 + (drh / rh_tol) ** 2

    for m in months:
        in_month = df['month'] == m
        n_hours = int(in_month.sum())
        dt = df.loc[in_month, 'temp_air'].mean() - targets.loc[m, 'temp_air']
        drh = df.loc[in_month, 'relative_humidity'].mean() - targets.loc[m, 'relative_humidity']
        row = {'month': m, 'temp_dev_before': dt, 'rh_dev_before': drh, 'days_inserted': 0}

        # Current hourly sums of each replaceable target day in this month.
        target_sums = {}
        for d, day in df[in_month].groupby('day'):
            if f'{m:02d}-{d:02d}' not in protected_days:
                target_sums[d] = (len(day), day['temp_air'].sum(), day['relative_humidity'].sum())

        candidates = []
        for src in sources:
            for d, day in src[src['month'] == m].groupby('day'):
                t_sum, rh_sum = day['temp_air'].sum(), day['relative_humidity'].sum()
                if d in target_sums and len(day) == target_sums[d][0] \
                        and np.isfinite(t_sum) and np.isfinite(rh_sum):
                    candidates.append((d, day, t_sum, rh_sum))

        while (row['days_inserted'] < max_days_per_month and len(inserted) < max_days_total
               and not (abs(dt) <= temp_tol and abs(drh) <= rh_tol)):
            best, best_obj = None, objective(dt, drh)
            for d, day, t_sum, rh_sum in candidates:
                if d not in target_sums:
                    continue
                _, cur_t, cur_rh = target_sums[d]
                new_dt = dt + (t_sum - cur_t) / n_hours
                new_drh = drh + (rh_sum - cur_rh) / n_hours
                obj = objective(new_dt, new_drh)
                if obj < best_obj - 1e-12:
                    best, best_obj = (d, day, new_dt, new_drh), obj
            if best is None:
                break
            d, day, dt, drh = best
            replace_whole_day(df, day, m, d, inserted_hours)
            inserted.add(f'{m:02d}-{d:02d}')
            del target_sums[d]          # each date is replaced at most once
            row['days_inserted'] += 1

        row.update(temp_dev_after=dt, rh_dev_after=drh)
        report.append(row)

    report = pd.DataFrame(report)
    print('Monthly compensation (deviation from base TMY, °C / %RH):')
    for r in report.itertuples():
        print(f'   - Month {r.month:2d}: {r.temp_dev_before:+.2f} / {r.rh_dev_before:+.1f} -> '
              f'{r.temp_dev_after:+.2f} / {r.rh_dev_after:+.1f} ({r.days_inserted} days)')
    return df, inserted, report


def smooth_inserted_boundaries(df, protected, hours=TRANSITION_HOURS):
    """Linearly blend up to `hours` untouched hours around inserted AMY blocks.

    The inserted AMY hours themselves are never modified. Close blocks share a
    single blend across the intervening hours. Categorical fields, radiation,
    and calendar information are not interpolated. Blocks that touch the start
    or end of the year are not blended across the year boundary.
    """
    protected = np.asarray(protected, dtype=bool)
    if not protected.any():
        return df, 0

    near = protected.copy()
    for shift in range(1, hours + 1):
        near[shift:] |= protected[:-shift]
        near[:-shift] |= protected[shift:]
    blend = near & ~protected
    edges = np.flatnonzero(np.diff(np.r_[False, blend, False].astype(int)))
    smoothed_hours = 0

    for start, end_exclusive in zip(edges[::2], edges[1::2]):
        end = end_exclusive - 1
        left, right = start - 1, end + 1
        if left < 0 or right >= len(df):
            continue
        for col in SMOOTH_COLS:
            a, b = float(df.at[left, col]), float(df.at[right, col])
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            values = np.linspace(a, b, right - left + 1)[1:-1]
            if col == 'relative_humidity':
                values = np.clip(values, 0, 100)
            elif col == 'wind_speed':
                values = np.maximum(values, 0)
            df.loc[start:end, col] = values
        smoothed_hours += end - start + 1

    return df, smoothed_hours


def report_temperature_jumps(df, description, threshold_c=8):
    jumps = df['temp_air'].diff().abs()
    n = int((jumps > threshold_c).sum())
    print(f'{description}: {n} jumps >{threshold_c}°C/h; '
          f'largest {jumps.max():.2f}°C/h')
    return n


def construct_final_rmy(base_epw_path, epw_dir, hot_dir, cold_dir, output_path,
                        summer_months=None, winter_months=None,
                        transition_hours=TRANSITION_HOURS):
    """Build the RMY EPW from detection outputs and write it to `output_path`.

    Parameters
    ----------
    base_epw_path : the base TMY EPW.
    epw_dir : folder with the AMY EPWs (filenames must contain the year).
    hot_dir, cold_dir : folders written by `run_full_pipeline` and
        `run_full_pipeline_cold`.
    output_path : where to write the RMY EPW.
    summer_months, winter_months : months used for compensation. By default
        they come from the base EPW's latitude (rmy.seasons): JJA/DJF in the
        north, DJF/JJA in the south, and month-by-month compensation at
        equatorial sites. Passing both overrides this with a two-season plan.
    transition_hours : width of the blend around each inserted block.
    """
    base_epw_path, epw_dir = Path(base_epw_path), Path(epw_dir)
    hot_dir, cold_dir, output_path = Path(hot_dir), Path(cold_dir), Path(output_path)

    if summer_months is None and winter_months is None:
        plan = season_plan_for_epw(base_epw_path)
        print(f'Seasons: {plan.describe()}')
        equatorial = plan.is_equatorial
        summer_months, winter_months = plan.summer_months, plan.winter_months
    elif summer_months is None or winter_months is None:
        raise ValueError('Pass both summer_months and winter_months, or neither.')
    else:
        equatorial = False

    header = extract_original_header(base_epw_path)
    base_epw = read_epw_file(base_epw_path)
    if len(base_epw) not in (8760, 8784):
        raise ValueError(f'Unexpected baseline length: {len(base_epw)} hours')

    epw_list = list_epws(epw_dir)
    if not epw_list:
        raise FileNotFoundError(f'No AMY EPWs found in {epw_dir}')

    heat_path = find_epw_by_year(get_peak_year(hot_dir / 'heatwave_stats_peak.csv'), epw_dir)
    cold_path = find_epw_by_year(get_peak_year(cold_dir / 'coldspells_stats_peak.csv'), epw_dir)
    peak_heat = read_epw_file(heat_path)
    peak_cold = read_epw_file(cold_path)

    heat_base = safe_load_events(hot_dir / 'heatwave_events_base.csv', HEAT_EVENT_COLS)
    heat_events = safe_load_events(hot_dir / 'heatwave_events_peak.csv', HEAT_EVENT_COLS)
    cold_base = safe_load_events(cold_dir / 'coldspells_events_base.csv', COLD_EVENT_COLS)
    cold_events = safe_load_events(cold_dir / 'coldspells_events_peak.csv', COLD_EVENT_COLS)

    mh, uh = match_events(heat_base, heat_events)
    mc, uc = match_events(cold_base, cold_events)

    # Mark every inserted AMY hour. These will be untouched by smoothing.
    inserted_hours = np.zeros(len(base_epw), dtype=bool)
    final, rh = integrate_events(base_epw.copy(), mh, uh, peak_heat,
                                 'Heatwave', inserted_hours)
    final, rc = integrate_events(final, mc, uc, peak_cold,
                                 'Coldspell', inserted_hours)

    # Protect both types of extremes against monthly compensation.
    all_extreme_days = rh | rc
    if equatorial:
        final, _, _ = compensate_monthly(final, base_epw, epw_list, list(range(1, 13)),
                                         all_extreme_days, inserted_hours)
    else:
        summer, winter = list(summer_months), list(winter_months)
        base_summer = calculate_monthly_avg_conditions(base_epw, summer)
        base_winter = calculate_monthly_avg_conditions(base_epw, winter)
        # Summer: insert cooler, drier days to offset heat waves.
        # Winter: insert warmer, wetter days to offset cold spells.
        s_condition = lambda day, ref: (day['temp_air'].mean() < ref['temp_air'] and
                                        day['relative_humidity'].mean() < ref['relative_humidity'])
        w_condition = lambda day, ref: (day['temp_air'].mean() > ref['temp_air'] and
                                        day['relative_humidity'].mean() > ref['relative_humidity'])

        summer_days = find_days_to_adjust_avg(epw_list, base_summer, summer, s_condition)
        winter_days = find_days_to_adjust_avg(epw_list, base_winter, winter, w_condition)

        final, inserted_summer = integrate_days(
            final, summer_days, all_extreme_days, summer, inserted_hours)
        final, _ = integrate_days(
            final, winter_days, all_extreme_days | inserted_summer,
            winter, inserted_hours)

    # Blend the shoulders ONCE, after all insertions.
    report_temperature_jumps(final, 'Before smoothing')
    event_values = final.loc[inserted_hours, SMOOTH_COLS].copy()
    final, n_smoothed = smooth_inserted_boundaries(
        final, inserted_hours, hours=transition_hours)
    pd.testing.assert_frame_equal(final.loc[inserted_hours, SMOOTH_COLS],
                                  event_values, check_exact=True)
    report_temperature_jumps(final, 'After smoothing')
    print(f'Inserted AMY hours preserved: {int(inserted_hours.sum())}')
    print(f'Surrounding hours included in smoothing: {n_smoothed}')

    # Check (do not silently alter) possible psychrometric conflicts.
    issues = ((final['temp_dew'] > final['temp_air'] + 0.1) &
              final[['temp_dew', 'temp_air']].notna().all(axis=1)).sum()
    if issues:
        print(f'WARNING: {issues} hours have dew point > dry bulb by >0.1°C; '
              'review before using this EPW in EnergyPlus.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as out:
        for line in header:
            out.write(line if line.endswith('\n') else line + '\n')
        final.to_csv(out, index=False, header=False, float_format='%g', na_rep='999')

    print(f'✅ Final RMY saved to: {output_path}')
    print(f'📂 Heatwave peak AMY: {heat_path}')
    print(f'📂 Cold-spell peak AMY: {cold_path}')
    return output_path


def run_full_rmy_pipeline(epw_dir, base_dir, output_dir, **rmy_kwargs):
    """Run heat-wave and cold-spell detection, then assemble the RMY.

    `epw_dir` holds the AMY EPWs, `base_dir` holds exactly one base TMY EPW.
    Detection CSVs go to `output_dir/hotspells` and `output_dir/coldspells`;
    the RMY is written to `output_dir/RMY_<base file name>`. Extra keyword
    arguments are passed to `construct_final_rmy`. Returns the RMY path.
    """
    epw_dir, base_dir, output_dir = Path(epw_dir), Path(base_dir), Path(output_dir)
    print('📂 Running full RMY pipeline:')
    print(f'   AMY Folder: {epw_dir}')
    print(f'   Base Folder: {base_dir}')
    print(f'   Output Folder: {output_dir}')

    base_files = list_epws(base_dir)
    if len(base_files) != 1:
        raise ValueError(f'Expected exactly one base EPW in {base_dir}; '
                         f'found {len(base_files)}')
    base_epw_path = base_files[0]

    hot_dir = output_dir / 'hotspells'
    cold_dir = output_dir / 'coldspells'
    hot_dir.mkdir(parents=True, exist_ok=True)
    cold_dir.mkdir(parents=True, exist_ok=True)

    run_full_pipeline(str(epw_dir), str(base_dir), str(hot_dir))
    run_full_pipeline_cold(str(epw_dir), str(base_dir), str(cold_dir))

    return construct_final_rmy(base_epw_path, epw_dir, hot_dir, cold_dir,
                               output_dir / f'RMY_{base_epw_path.name}',
                               **rmy_kwargs)
