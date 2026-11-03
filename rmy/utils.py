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

    NaN in a checked field is written as that field's EPW missing code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as out:
        for line in header_lines:
            out.write(line if line.endswith('\n') else line + '\n')
        fill_missing_codes(df).to_csv(out, index=False, header=False,
                                      float_format='%g', na_rep='999')
    return path


def extract_original_header(epw_path) -> list[str]:
    """Return the 8 raw EPW header lines (with their line endings)."""
    with open(epw_path, 'r') as f:
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
    df = pd.read_csv(path)
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
