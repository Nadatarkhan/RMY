"""Assemble an RMY: insert peak-year extreme events into a base TMY.

Ported from the canonical notebook, section "3.0 Generate RMYs"
(single-location RMY assembly with 8-hour smoothed insertion boundaries).

Steps:
1. Insert the peak heat-wave and cold-spell event days from the peak AMY years.
2. Insert compensating AMY days so each month's mean temperature and
   humidity stay close to the base TMY (compensate_monthly).
3. Blend the hours around every inserted block once, after all insertions.
   Inserted AMY hours are never modified.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .coldspells import run_full_pipeline_cold
from .heatwaves import run_full_pipeline
from .seasons import season_plan_for_epw
from .utils import (
    WEATHER_COLS, extract_original_header, find_epw_by_year, get_peak_year,
    label_rmy_header, list_amys, list_epws, read_epw_file, safe_load_events, set_typical_extreme_periods, write_epw,
)
from .validation import audit_epws, print_report

TRANSITION_HOURS = 8
# Longest event inserted into the RMY, in days. Detection keeps events at
# their full length; a longer event is inserted as its most severe window
# (see clip_events_to_window). None inserts events whole.
MAX_INSERT_DAYS = 21
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


def clip_events_to_window(events, source_epw, kind, max_days):
    """Shorten events longer than `max_days` to their most severe window.

    For heat waves the window with the highest mean daily Tmax is kept, for
    cold spells the one with the lowest mean daily Tmin, computed from the
    AMY the events are inserted from. Returns (events, list of changes).
    """
    if max_days is None or events.empty:
        return events, []
    agg = 'max' if kind == 'heat' else 'min'
    daily = source_epw.groupby(['month', 'day'])['temp_air'].agg(agg)
    events = events.copy()
    changes = []
    for i, ev in events.iterrows():
        dates = pd.date_range(ev['begin_date'], ev['end_date'], freq='D')
        if len(dates) <= max_days:
            continue
        values = pd.Series([daily.get((d.month, d.day), np.nan) for d in dates])
        window = values.rolling(max_days).mean()
        end = int(window.idxmax() if kind == 'heat' else window.idxmin())
        start = end - max_days + 1
        changes.append((dates[0], dates[-1], dates[start], dates[end]))
        events.loc[i, ['begin_date', 'end_date', 'duration']] = [dates[start], dates[end], max_days]
    return events, changes


def new_source_map(base_epw):
    """Where each RMY hour comes from, starting as the base TMY (see write_source_map)."""
    return {'origin': np.full(len(base_epw), 'TMY', dtype=object),
            'source_year': base_epw['year'].to_numpy(dtype=int, copy=True),
            'wind_filled': np.zeros(len(base_epw), dtype=bool)}


def write_source_map(final, sources, path):
    """Write the source of every RMY hour next to the RMY.

    origin is TMY (the base TMY's own hour), heat_wave or cold_spell (an event
    day inserted from the peak AMY), compensation (an AMY day inserted to keep
    the monthly means), or blend (a TMY hour adjusted to join an inserted
    block). source_year is the year the weather was observed. The EPW's year
    column keeps the base TMY's years, so this file is what says that, for
    example, a heat-wave day came from 2015. wind_filled marks inserted hours
    whose missing wind was filled (repair_missing_wind).
    """
    table = pd.DataFrame({'month': final['month'].to_numpy(), 'day': final['day'].to_numpy(),
                          'hour': final['hour'].to_numpy(), 'origin': sources['origin'],
                          'source_year': sources['source_year'], 'wind_filled': sources['wind_filled']})
    table.to_csv(path, index=False)
    return table


def replace_whole_day(df, source_day, month, day, inserted_hours, sources=None, origin=None):
    """Insert all source-hour weather values, preserving baseline EPW timestamps.

    With `sources` (new_source_map), the replaced hours are recorded as
    `origin` from the source day's year.
    """
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
    if sources is not None:
        rows = np.asarray(target_idx, dtype=int)
        sources['origin'][rows] = origin
        sources['source_year'][rows] = source_day['year'].to_numpy(dtype=int)
    return True


def integrate_events(df, matched, unmatched, peak_epw, label, inserted_hours, sources=None):
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
            origin = 'heat_wave' if label == 'Heatwave' else 'cold_spell'
            if replace_whole_day(df, source_day, month, day, inserted_hours, sources, origin):
                replaced.add(f'{month:02d}-{day:02d}')
    print(f'{label} events integrated. Days replaced: {len(replaced)}')
    return df, replaced


def calculate_monthly_avg_conditions(df, months):
    return (df[df['month'].isin(months)]
            .groupby('month')[['temp_air', 'relative_humidity']].mean())


def compensate_monthly(df, base_epw, epw_files, months, protected_days, inserted_hours,
                       temp_tol=0.25, rh_tol=2.0, max_days_per_month=10, max_days_total=60,
                       max_midnight_step=None, source_map=None):
    """Pull each month's mean temperature and RH back to the base TMY.

    Paper Section 2.4: "the algorithm adjusts monthly and seasonal dry bulb
    temperature and relative humidity averages back to those of the original
    TMY ... Replacement days are added iteratively until the deviation ...
    is minimized". For each month whose means moved after event insertion, AMY days from the same
    calendar month are inserted one at a time, each time choosing the day
    that most reduces the remaining deviation, until the month is within
    `temp_tol` (degrees C) and `rh_tol` (percentage points), no candidate
    helps, or a day budget is reached. Event days and already-inserted days
    are never replaced. Months without deviation are left untouched.

    A compensation day placed directly next to another inserted day (an
    event day or an earlier compensation day) meets it at midnight with no
    TMY hours between them, so the boundary smoothing cannot blend that
    step. Such a day is only used if the temperature step at each shared
    midnight is at most `max_midnight_step` degrees C.

    Returns (df, inserted day tags, per-month report DataFrame).
    """
    targets = calculate_monthly_avg_conditions(base_epw, months)
    sources = [read_epw_file(p) for p in sorted(Path(p) for p in epw_files)]
    inserted, report = set(), []

    # Order of the calendar days in the file, to find each day's neighbors
    # (also across month ends, and across 28 Feb / 1 Mar in a 365-day TMY).
    keys = list(zip(df['month'].astype(int), df['day'].astype(int)))
    position, first_row, last_row = {}, [], []
    for row_i, key in enumerate(keys):
        if key not in position:
            position[key] = len(first_row)
            first_row.append(row_i)
            last_row.append(row_i)
        last_row[position[key]] = row_i
    occupied = {position[(int(t[:2]), int(t[3:]))] for t in protected_days
                if (int(t[:2]), int(t[3:])) in position}
    temp_col = df.columns.get_loc('temp_air')
    if max_midnight_step is None:
        # Steps the base TMY itself takes at midnight (99th percentile), at least 3 C.
        steps = base_epw['temp_air'].diff().abs()[base_epw['hour'] == 1]
        max_midnight_step = max(3.0, float(np.nanpercentile(steps, 99)))

    def joins_smoothly(m, d, day):
        """True unless the day would meet an inserted neighbor with a big step."""
        p, t = position[(m, d)], day['temp_air'].to_numpy(dtype=float)
        if p - 1 in occupied:
            before = df.iat[last_row[p - 1], temp_col]
            if not abs(t[0] - before) <= max_midnight_step:
                return False
        if p + 1 in occupied:
            after = df.iat[first_row[p + 1], temp_col]
            if not abs(after - t[-1]) <= max_midnight_step:
                return False
        return True

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
                if d not in target_sums or not joins_smoothly(m, d, day):
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
            replace_whole_day(df, day, m, d, inserted_hours, source_map, 'compensation')
            inserted.add(f'{m:02d}-{d:02d}')
            occupied.add(position[(m, d)])
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


def repair_missing_wind(df, base_epw, inserted_hours, max_gap=3):
    """Fill missing wind speed and direction in the inserted AMY hours.

    NOAA AMYs leave wind direction missing for "variable" wind and have
    occasional gaps in wind speed. Only hours that are missing (NaN) and
    were inserted from an AMY are filled; recorded values and the base-TMY
    hours are never changed:
    - gaps of up to `max_gap` hours are interpolated from the hours on either
      side (direction around the compass, through the shorter arc);
    - longer gaps take the base TMY's value for the same hour.

    Returns (df, {column: (hours interpolated, hours from the TMY)}).
    """
    inserted_hours = np.asarray(inserted_hours, dtype=bool)
    counts = {}
    for col in ('wind_speed', 'wind_direction'):
        values = df[col].to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(values)
        edges = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)))
        n_interp = n_tmy = 0
        for start, stop in zip(edges[::2], edges[1::2]):
            fill = np.arange(start, stop)
            fill = fill[inserted_hours[fill]]
            if fill.size == 0:
                continue
            left, right = start - 1, stop
            anchors_ok = (left >= 0 and right < len(values)
                          and np.isfinite(values[left]) and np.isfinite(values[right]))
            if stop - start <= max_gap and anchors_ok:
                w = (fill - left) / (right - left)
                if col == 'wind_direction':
                    a, b = np.radians(values[left]), np.radians(values[right])
                    u = (1 - w) * np.sin(a) + w * np.sin(b)
                    v = (1 - w) * np.cos(a) + w * np.cos(b)
                    new = np.round(np.degrees(np.arctan2(u, v)) % 360)
                    new[new == 360] = 0
                else:
                    new = (1 - w) * values[left] + w * values[right]
                values[fill] = new
                n_interp += fill.size
            else:
                tmy = base_epw[col].to_numpy(dtype=float)[fill]
                ok = np.isfinite(tmy)
                values[fill[ok]] = tmy[ok]
                n_tmy += int(ok.sum())
        df[col] = values
        counts[col] = (n_interp, n_tmy)
    return df, counts


def relative_humidity(temp_c, dew_c):
    """Relative humidity (%) from dry-bulb and dew-point temperature (Magnus, over water),
    the formula used for the month joins of the TMY."""
    es = lambda x: np.exp(17.625 * x / (x + 243.04))
    with np.errstate(invalid='ignore', over='ignore'):
        return np.clip(100 * es(np.asarray(dew_c, dtype=float)) / es(np.asarray(temp_c, dtype=float)), 0, 100)


def set_calm_direction(df):
    """Wind direction 0 wherever the wind speed is 0 (calm), as in the AMYs. Returns the hours changed."""
    calm = df['wind_speed'].eq(0) & df['wind_direction'].notna() & df['wind_direction'].ne(0)
    df.loc[calm, 'wind_direction'] = 0.0
    return int(calm.sum())


def smooth_inserted_boundaries(df, protected, hours=TRANSITION_HOURS, base=None,
                               max_shape_step=8.0):
    """Blend up to `hours` untouched hours around inserted AMY blocks.

    The inserted AMY hours themselves are never modified. Close blocks share a
    single blend across the intervening hours. Categorical fields, radiation,
    and calendar information are not interpolated. Blocks that touch the start
    or end of the year are not blended across the year boundary.

    With `base` (the base TMY, same rows as `df`), the blended hours keep the
    TMY's hourly shape (its day-night cycle): the TMY values are shifted by an
    offset that changes linearly from the offset at one end of the blend
    (inserted value minus TMY value) to the offset at the other end, so the
    blend meets both neighbors without a step. Without `base`, where the TMY
    has missing values, or where the TMY's own temperature jumps by more than
    `max_shape_step` degrees C within one hour, the blended hours are a
    straight line between the two neighboring hours. Dew point in the blended
    hours is capped at the dry-bulb temperature, relative humidity is
    recalculated from the blended temperature and dew point (blending it
    separately left it up to 8 points off), and an hour whose blended wind
    speed is 0 gets wind direction 0 (calm).
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
        # Keep the TMY's hourly shape unless the TMY itself has a step here.
        use_shape = base is not None and not (
            np.abs(np.diff(base['temp_air'].to_numpy(dtype=float)[left:right + 1]))
            > max_shape_step).any()
        for col in SMOOTH_COLS:
            a, b = float(df.at[left, col]), float(df.at[right, col])
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            values = np.linspace(a, b, right - left + 1)[1:-1]
            if use_shape:
                shape = base[col].to_numpy(dtype=float)[left:right + 1]
                if np.isfinite(shape).all():
                    offsets = np.linspace(a - shape[0], b - shape[-1], right - left + 1)
                    values = (shape + offsets)[1:-1]
            if col == 'relative_humidity':
                values = np.clip(values, 0, 100)
            elif col == 'wind_speed':
                values = np.maximum(values, 0)
            df.loc[start:end, col] = values
        # Blended temperature and dew point move separately; keep Td <= T.
        seg_t = df.loc[start:end, 'temp_air']
        df.loc[start:end, 'temp_dew'] = df.loc[start:end, 'temp_dew'].where(
            ~(df.loc[start:end, 'temp_dew'] > seg_t), seg_t)
        rh = relative_humidity(seg_t.to_numpy(dtype=float),
                               df.loc[start:end, 'temp_dew'].to_numpy(dtype=float))
        ok = np.isfinite(rh)
        df.loc[np.arange(start, end + 1)[ok], 'relative_humidity'] = rh[ok]
        if 'wind_direction' in df.columns:
            calm = df.loc[start:end, 'wind_speed'].eq(0)
            df.loc[calm[calm].index, 'wind_direction'] = 0.0
        smoothed_hours += end - start + 1

    return df, smoothed_hours


def report_temperature_jumps(df, description, threshold_c=8):
    jumps = df['temp_air'].diff().abs()
    n = int((jumps > threshold_c).sum())
    print(f'{description}: {n} jumps >{threshold_c}°C/h; '
          f'largest {jumps.max():.2f}°C/h')
    return n


def _report_missing_hours(base_epw, final):
    """Print fields that have more missing hours in the RMY than in the base TMY."""
    from .validation import FIELD_RULES
    rows = [(c, int(base_epw[c].isna().sum()), int(final[c].isna().sum()))
            for c in FIELD_RULES if c in final.columns]
    rows = [r for r in rows if r[2] > r[1]]
    if rows:
        print('Fields with more missing hours than the base TMY '
              '(written with the EPW missing code):')
        for c, b, f in rows:
            print(f'   - {c}: {b} -> {f} hours')


def construct_final_rmy(base_epw_path, epw_dir, hot_dir, cold_dir, output_path,
                        transition_hours=TRANSITION_HOURS, invalid_values='missing',
                        max_insert_days=MAX_INSERT_DAYS):
    """Build the RMY EPW from detection outputs and write it to `output_path`.

    Parameters
    ----------
    base_epw_path : the base TMY EPW.
    epw_dir : folder with the AMY EPWs (filenames must contain the year).
    hot_dir, cold_dir : folders written by `run_full_pipeline` and
        `run_full_pipeline_cold`.
    output_path : where to write the RMY EPW.
    transition_hours : width of the blend around each inserted block.
    max_insert_days : longest event inserted, in days (default 21). A longer
        event is inserted as its most severe window; None inserts it whole.
    invalid_values : what to do when an input EPW holds invalid values
        (netCDF fill values, out-of-range numbers; see rmy.validation).
        'missing' (default) reports them and treats them as missing data;
        'error' stops before anything is written.
    """
    if invalid_values not in ('missing', 'error'):
        raise ValueError("invalid_values must be 'missing' or 'error'")
    base_epw_path, epw_dir = Path(base_epw_path), Path(epw_dir)
    hot_dir, cold_dir, output_path = Path(hot_dir), Path(cold_dir), Path(output_path)

    print(f'Seasons: {season_plan_for_epw(base_epw_path).describe()}')

    header = extract_original_header(base_epw_path)
    base_epw = read_epw_file(base_epw_path)
    if len(base_epw) not in (8760, 8784):
        raise ValueError(f'Unexpected baseline length: {len(base_epw)} hours')

    epw_list = list_amys(epw_dir, verbose=False)
    if not epw_list:
        raise FileNotFoundError(f'No AMY EPWs found in {epw_dir}')

    input_report = audit_epws([base_epw_path, *epw_list])
    invalid = input_report[input_report['problem'] == 'invalid']
    print_report(invalid, 'Invalid values in the input EPWs')
    if not invalid.empty:
        if invalid_values == 'error':
            raise ValueError('Input EPWs contain invalid values (see the report above). '
                             "Fix or remove those files, or pass invalid_values='missing'.")
        print('   These values are treated as missing: never used for detection or '
              'compensation, and written with the EPW missing code if inserted.')

    heat_year = get_peak_year(hot_dir / 'heatwave_stats_peak.csv')
    cold_year = get_peak_year(cold_dir / 'coldspells_stats_peak.csv')
    heat_path = find_epw_by_year(heat_year, epw_dir)
    cold_path = find_epw_by_year(cold_year, epw_dir)
    peak_heat = read_epw_file(heat_path)
    peak_cold = read_epw_file(cold_path)

    heat_base = safe_load_events(hot_dir / 'heatwave_events_base.csv', HEAT_EVENT_COLS)
    heat_events = safe_load_events(hot_dir / 'heatwave_events_peak.csv', HEAT_EVENT_COLS)
    cold_base = safe_load_events(cold_dir / 'coldspells_events_base.csv', COLD_EVENT_COLS)
    cold_events = safe_load_events(cold_dir / 'coldspells_events_peak.csv', COLD_EVENT_COLS)

    for label, kind, src in (('Heatwave', 'heat', peak_heat), ('Coldspell', 'cold', peak_cold)):
        events = heat_events if kind == 'heat' else cold_events
        events, changes = clip_events_to_window(events, src, kind, max_insert_days)
        for b0, e0, b1, e1 in changes:
            print(f'{label} {b0:%d %b %Y}-{e0:%d %b} ({(e0 - b0).days + 1} days) inserted as its most '
                  f'severe {max_insert_days} days: {b1:%d %b}-{e1:%d %b}')
        if kind == 'heat':
            heat_events = events
        else:
            cold_events = events

    mh, uh = match_events(heat_base, heat_events)
    mc, uc = match_events(cold_base, cold_events)

    # Mark every inserted AMY hour. These will be untouched by smoothing.
    inserted_hours = np.zeros(len(base_epw), dtype=bool)
    sources = new_source_map(base_epw)
    final, rh = integrate_events(base_epw.copy(), mh, uh, peak_heat,
                                 'Heatwave', inserted_hours, sources)
    final, rc = integrate_events(final, mc, uc, peak_cold,
                                 'Coldspell', inserted_hours, sources)

    # Protect both types of extremes against monthly compensation.
    all_extreme_days = rh | rc
    final, _, _ = compensate_monthly(final, base_epw, epw_list, list(range(1, 13)),
                                     all_extreme_days, inserted_hours, source_map=sources)

    # Fill wind values missing in the inserted AMY hours (before smoothing,
    # so the boundary blend has wind anchors).
    wind_missing = final[['wind_speed', 'wind_direction']].isna().any(axis=1).to_numpy()
    final, wind = repair_missing_wind(final, base_epw, inserted_hours)
    sources['wind_filled'] = (wind_missing & inserted_hours
                              & final[['wind_speed', 'wind_direction']].notna().any(axis=1).to_numpy())
    print('Wind repair (inserted AMY hours): ' + '; '.join(
        f'{col.replace("_", " ")} {a} h interpolated, {b} h from the TMY'
        for col, (a, b) in wind.items()))

    # Blend the shoulders ONCE, after all insertions.
    report_temperature_jumps(final, 'Before smoothing')
    event_values = final.loc[inserted_hours, SMOOTH_COLS].copy()
    final, n_smoothed = smooth_inserted_boundaries(
        final, inserted_hours, hours=transition_hours, base=base_epw)
    pd.testing.assert_frame_equal(final.loc[inserted_hours, SMOOTH_COLS],
                                  event_values, check_exact=True)
    report_temperature_jumps(final, 'After smoothing')
    n_calm = set_calm_direction(final)
    if n_calm:
        print(f'Calm hours given wind direction 0: {n_calm}')
    changed = ~inserted_hours & ~np.isclose(final[SMOOTH_COLS].to_numpy(dtype=float),
                                            base_epw[SMOOTH_COLS].to_numpy(dtype=float),
                                            equal_nan=True).all(axis=1)
    sources['origin'][changed] = 'blend'
    print(f'Inserted AMY hours preserved: {int(inserted_hours.sum())}')
    print(f'Surrounding hours included in smoothing: {n_smoothed}')

    # Check (do not silently alter) possible psychrometric conflicts.
    issues = ((final['temp_dew'] > final['temp_air'] + 0.1) &
              final[['temp_dew', 'temp_air']].notna().all(axis=1)).sum()
    if issues:
        print(f'WARNING: {issues} hours have dew point > dry bulb by >0.1°C; '
              'review before using this EPW in EnergyPlus.')

    amy_years = [int(read_epw_file(p, mask=False)['year'].mode().iloc[0]) for p in epw_list]
    header = label_rmy_header(header, amy_years, heat_year, cold_year)
    header = set_typical_extreme_periods(header, final)   # design weeks of the RMY itself
    write_epw(final, header, output_path)
    _report_missing_hours(base_epw, final)
    map_path = source_map_path(output_path)
    table = write_source_map(final, sources, map_path)
    counts = table['origin'].value_counts()
    print('Source of the RMY hours: ' + ', '.join(
        f'{k} {int(counts.get(k, 0))} h' for k in ('TMY', 'heat_wave', 'cold_spell', 'compensation', 'blend')))
    print(f'📄 Source year of every hour: {map_path}')

    print(f'✅ Final RMY saved to: {output_path}')
    print(f'📂 Heatwave peak AMY: {heat_path}')
    print(f'📂 Cold-spell peak AMY: {cold_path}')
    return output_path


# "TMY", "TMYx", "TMY2" or "TMY3" as a word in a file name (any case), but not
# the "TMY" of "TMY2009", whose digits are a year.
_TMY_TOKEN = re.compile(r'(?<![A-Za-z])TMY(?:x|[23](?!\d))?(?![A-Za-z])', re.IGNORECASE)


def source_map_path(rmy_path):
    """'.../X_RMY_2009-2023.epw' -> '.../X_RMY_2009-2023.sources.csv'."""
    rmy_path = Path(rmy_path)
    return rmy_path.with_name(rmy_path.stem + '.sources.csv')


def rmy_file_name(tmy_file_name):
    """Name of the RMY made from a base TMY file: every TMY variant becomes RMY.

    'Boston_MA_725090_TMYx_2009-2023.epw' -> 'Boston_MA_725090_RMY_2009-2023.epw'
    'USA_MA_Boston.725090_TMY3.epw'       -> 'USA_MA_Boston.725090_RMY.epw'
    A name without any TMY variant gets '_RMY' before the extension.
    """
    path = Path(tmy_file_name)
    stem, n = _TMY_TOKEN.subn('RMY', path.stem)
    if n == 0:
        stem = f'{path.stem}_RMY'
    return stem + path.suffix


def run_full_rmy_pipeline(epw_dir, base_dir, output_dir, gnn_config=None, **rmy_kwargs):
    """Run heat-wave and cold-spell detection, then assemble the RMY.

    `epw_dir` holds the AMY EPWs, `base_dir` holds exactly one base TMY EPW.
    Detection CSVs go to `output_dir/hotspells` and `output_dir/coldspells`;
    the RMY is written to `output_dir/<base file name with TMY -> RMY>`
    (see rmy_file_name). `gnn_config`
    (rmy.gnn.GNNConfig) changes the GNN settings. Extra keyword arguments
    are passed to `construct_final_rmy`. Returns the RMY path.
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

    run_full_pipeline(str(epw_dir), str(base_dir), str(hot_dir), gnn_config=gnn_config)
    run_full_pipeline_cold(str(epw_dir), str(base_dir), str(cold_dir), gnn_config=gnn_config)

    return construct_final_rmy(base_epw_path, epw_dir, hot_dir, cold_dir,
                               output_dir / rmy_file_name(base_epw_path.name),
                               **rmy_kwargs)
