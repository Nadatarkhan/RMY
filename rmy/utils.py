"""Shared EPW helpers for the RMY package.

Ported from the canonical notebook (RMYFRMY_Generation_Oct2026, sections 1-3).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import fill_missing_codes, mask_invalid

EPW_HEADER_LINES = 8

EPW_COLUMNS = [
    'year', 'month', 'day', 'hour', 'minute', 'data_source_unct',
    'temp_air', 'temp_dew', 'relative_humidity', 'atmospheric_pressure',
    'etr', 'etrn', 'ghi_infrared', 'ghi', 'dni', 'dhi',
    'global_hor_illum', 'direct_normal_illum', 'diffuse_horizontal_illum',
    'zenith_luminance', 'wind_direction', 'wind_speed',
    'total_sky_cover', 'opaque_sky_cover', 'visibility', 'ceiling_height',
    'present_weather_observation', 'present_weather_codes',
    'precipitable_water', 'aerosol_optical_depth', 'snow_depth',
    'days_since_last_snowfall', 'albedo', 'liquid_precipitation_depth',
    'liquid_precipitation_quantity',
]
DATE_COLS = ['year', 'month', 'day', 'hour', 'minute']
CODE_COLS = ['data_source_unct', 'present_weather_observation', 'present_weather_codes']
WEATHER_COLS = [c for c in EPW_COLUMNS if c not in DATE_COLS]


def list_epws(folder) -> list[Path]:
    """All .epw files in `folder`, in sorted (filesystem-independent) order."""
    return sorted(Path(folder).glob('*.epw'))


def read_epw_raw(path) -> pd.DataFrame:
    """Read the hourly block of an EPW for event detection.

    Invalid values (fill values, out-of-range numbers) and EPW missing codes
    in the checked weather fields become NaN, so that, for example, a
    missing temperature coded 99.9 is not mistaken for a heat wave.
    """
    df = pd.read_csv(path, skiprows=EPW_HEADER_LINES, header=None,
                     names=EPW_COLUMNS, usecols=range(len(EPW_COLUMNS)),
                     low_memory=False)
    return mask_invalid(df)


def read_epw_file(path, mask=True) -> pd.DataFrame:
    """Read the hourly block of an EPW with explicit types, for RMY assembly.

    Calendar columns are int64, weather codes are strings, and every other
    column is float64 so that smoothing can write fractional values. Invalid
    values and EPW missing codes in the checked fields become NaN (see
    rmy.validation); `write_epw` turns them back into missing codes.
    Pass mask=False to get the values exactly as stored in the file.
    """
    df = pd.read_csv(path, header=None, skiprows=EPW_HEADER_LINES,
                     names=EPW_COLUMNS, dtype={c: 'string' for c in CODE_COLS})
    for c in DATE_COLS:
        df[c] = pd.to_numeric(df[c], errors='raise').astype('int64')
    for c in WEATHER_COLS:
        if c not in CODE_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    return mask_invalid(df) if mask else df


def write_epw(df, header_lines, path):
    """Write an EPW: the given 8 header lines, then the hourly rows.

    Line endings follow the header (CRLF or LF) so the file is not mixed.
    NaN in a checked field is written as that field's EPW missing code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    newline = '\r\n' if header_lines and header_lines[0].endswith('\r\n') else '\n'
    with open(path, 'w', newline='') as out:
        for line in header_lines:
            out.write(line.rstrip('\r\n') + newline)
        fill_missing_codes(df).to_csv(out, index=False, header=False, float_format='%g',
                                      na_rep='999', lineterminator=newline)
    return path


RMY_LABEL = 'RMY extreme weather file'


def label_rmy_header(header_lines, amy_years, peak_heat_year=None, peak_cold_year=None):
    """Return the header with a short RMY note added to the COMMENTS 2 line.

    Only COMMENTS 2 changes, and only by prepending text inside its existing
    text field. The note has no commas, so every header line keeps its field
    count; all other lines are returned unchanged. Example:

        COMMENTS 2,"RMY extreme weather file: heat waves and cold spells from
        AMYs 2009-2023 inserted (peak heat year 2022; peak cold year 2015);
        Downloaded from Climate.Onebuilding.org ..."
    """
    years = sorted({int(y) for y in amy_years})
    if not years:
        raise ValueError('No AMY years given for the header note')
    note = f'{RMY_LABEL}: heat waves and cold spells from AMYs {years[0]}-{years[-1]} inserted'
    gaps = sorted(set(range(years[0], years[-1] + 1)) - set(years))
    if gaps:
        note += ' (no AMY for ' + ' '.join(str(y) for y in gaps) + ')'
    peaks = [f'peak {k} year {y}' for k, y in (('heat', peak_heat_year), ('cold', peak_cold_year))
             if y is not None]
    if peaks:
        note += ' (' + '; '.join(peaks) + ')'
    assert ',' not in note and '"' not in note

    lines = list(header_lines)
    idx = next((i for i, l in enumerate(lines) if l.upper().startswith('COMMENTS 2,')), None)
    if idx is None:
        raise ValueError('EPW header has no COMMENTS 2 line')
    line = lines[idx]
    ending = line[len(line.rstrip('\r\n')):]
    keyword, rest = line.rstrip('\r\n').split(',', 1)
    if rest.startswith('"') and rest.endswith('"') and len(rest) >= 2:
        inner = rest[1:-1]
        rest = '"' + note + ('; ' + inner if inner else '') + '"'
    else:
        rest = note + ('; ' + rest if rest else '')
    lines[idx] = f'{keyword},{rest}{ending}'
    return lines


def extract_original_header(epw_path) -> list[str]:
    """Return the 8 raw EPW header lines (with their line endings, CRLF or LF)."""
    with open(epw_path, 'r', newline='') as f:
        return [f.readline() for _ in range(EPW_HEADER_LINES)]


def get_peak_year(stats_csv) -> int:
    stats = pd.read_csv(stats_csv)
    if stats.empty or 'year' not in stats:
        raise ValueError(f'Peak statistics missing/empty: {stats_csv}')
    return int(stats.iloc[0]['year'])


def find_epw_by_year(year, folder) -> Path:
    """Return the one EPW whose filename contains `year` as a whole number.

    '2010' matches 'Boston_2010.epw' but not 'Boston_20101.epw'. Raises if no
    file or more than one file matches, instead of picking one arbitrarily.
    """
    matches = [p for p in list_epws(folder)
               if re.search(rf'(?<!\d){int(year)}(?!\d)', p.stem)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected one EPW for {year} in {folder}; got {len(matches)}: {matches}')
    return matches[0]


def safe_load_events(path, columns) -> pd.DataFrame:
    """Load an events CSV, or an empty table with `columns` if it is missing.

    A missing or empty file is expected (no events). Any other problem, such
    as unparseable dates, raises instead of being silently ignored.
    """
    path = Path(path)
    if not path.exists():
        print(f'WARNING: missing events CSV: {path}')
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:     # written from an empty table: no events
        return pd.DataFrame(columns=columns)
    if df.empty:
        return pd.DataFrame(columns=columns)
    for c in ['begin_date', 'end_date']:
        df[c] = pd.to_datetime(df[c], dayfirst=True, errors='raise')
    return df


def calculate_heat_index(temp_air_c, rh):
    """Heat index in degrees Celsius.

    Uses the simple NOAA approximation below 80 F and the Rothfusz
    regression above it.
    """
    t_c = pd.Series(temp_air_c, copy=False).astype(float)
    rh = pd.Series(rh, copy=False).astype(float)

    t_f = (t_c * 9 / 5) + 32
    hi_simple_f = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))
    hi_full_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 6.83783e-3 * t_f**2
        - 5.481717e-2 * rh**2
        + 1.22874e-3 * t_f**2 * rh
        + 8.5282e-4 * t_f * rh**2
        - 1.99e-6 * t_f**2 * rh**2
    )
    hi_f = np.where(hi_simple_f < 80, hi_simple_f, hi_full_f)
    return (pd.Series(hi_f) - 32) * 5 / 9


def report_runner_up(years, scores, label, close_pct=5.0):
    """Print the peak year's lead over the runner-up; flag near-ties.

    When two years score within `close_pct` percent, small changes in the
    inputs or settings can swap them, so both are worth a look.
    """
    ranked = sorted(zip(scores, years), reverse=True)
    if len(ranked) < 2 or ranked[0][0] <= 0:
        return
    (s1, y1), (s2, y2) = ranked[0], ranked[1]
    margin = 100.0 * (s1 - s2) / s1
    note = '  <- close: both years are plausible peaks' if margin < close_pct else ''
    print(f"{label} peak year {int(y1)}; runner-up {int(y2)} scores {margin:.1f}% lower{note}")
