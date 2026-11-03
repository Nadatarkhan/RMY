"""EPW value checks: invalid values, EPW missing codes, and a report.

Each checked field has a valid range and a missing-value code from the EPW
data dictionary (EnergyPlus Auxiliary Programs, "Weather File Data Fields").

Two kinds of problem are reported:
- invalid: not finite, a netCDF fill value such as 9.96921e+36, or outside
  the field's valid range (for example a wind speed of 447 m/s). These break
  or distort simulations and must not reach an output file.
- missing: the field's official EPW missing code (for example 9999 for
  radiation). This is valid EPW, but the hour has no data.

Inside the package, both are held as NaN (`mask_invalid`). When an RMY is
written, NaN becomes the field's missing code (`fill_missing_codes`).

Run `python -m rmy.audit FILE.epw ...` to audit files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# field: (min, max, missing code). None = no bound.
FIELD_RULES = {
    'temp_air': (-70, 70, 99.9),
    'temp_dew': (-70, 70, 99.9),
    'relative_humidity': (0, 110, 999),
    'atmospheric_pressure': (31000, 120000, 999999),
    'etr': (0, None, 9999),
    'etrn': (0, None, 9999),
    'ghi_infrared': (0, None, 9999),
    'ghi': (0, None, 9999),
    'dni': (0, None, 9999),
    'dhi': (0, None, 9999),
    'global_hor_illum': (0, None, 999999),
    'direct_normal_illum': (0, None, 999999),
    'diffuse_horizontal_illum': (0, None, 999999),
    'zenith_luminance': (0, None, 9999),
    'wind_direction': (0, 360, 999),
    'wind_speed': (0, 40, 999),
    'total_sky_cover': (0, 10, 99),
    'opaque_sky_cover': (0, 10, 99),
}


def _masks(series, rule):
    lo, hi, code = rule
    v = pd.to_numeric(series, errors='coerce').astype('float64')
    missing = v == code
    bad = ~np.isfinite(v) & series.notna() | (v.abs() > 1e30)
    out = pd.Series(False, index=v.index)
    if lo is not None:
        out |= v < lo
    if hi is not None:
        out |= v > hi
    invalid = (bad | out) & ~missing
    return invalid, missing


def mask_invalid(df):
    """Return a copy with invalid values and EPW missing codes set to NaN."""
    df = df.copy()
    for col, rule in FIELD_RULES.items():
        if col in df.columns:
            invalid, missing = _masks(df[col], rule)
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            df.loc[invalid | missing, col] = np.nan
    return df


def fill_missing_codes(df):
    """Return a copy with NaN in checked fields replaced by the EPW missing code."""
    df = df.copy()
    for col, (_, _, code) in FIELD_RULES.items():
        if col in df.columns:
            df[col] = df[col].fillna(code)
    return df


def check_epw_values(df, name=''):
    """One row per (field, problem): hours affected and the years they come from."""
    rows = []
    for col, rule in FIELD_RULES.items():
        if col not in df.columns:
            continue
        for problem, mask in zip(('invalid', 'missing'), _masks(df[col], rule)):
            if mask.any():
                vals = pd.to_numeric(df.loc[mask, col], errors='coerce')
                rows.append({
                    'file': name, 'field': col, 'problem': problem,
                    'hours': int(mask.sum()),
                    'example_values': ', '.join(f'{x:g}' for x in vals.drop_duplicates().head(3)),
                    'source_years': ', '.join(str(int(y)) for y in sorted(df.loc[mask, 'year'].unique()))
                    if 'year' in df.columns else '',
                })
    return pd.DataFrame(rows, columns=['file', 'field', 'problem', 'hours',
                                       'example_values', 'source_years'])


def audit_epws(paths):
    """Check several EPW files and return one combined report."""
    from .utils import read_epw_file
    reports = [check_epw_values(read_epw_file(p, mask=False), Path(p).name) for p in paths]
    return pd.concat(reports, ignore_index=True) if reports else check_epw_values(pd.DataFrame())


def print_report(report, title='EPW value check', show_missing=True):
    rep = report if show_missing else report[report['problem'] == 'invalid']
    if rep.empty:
        print(f'{title}: no problems found.')
        return
    print(f'{title}:')
    with pd.option_context('display.width', 200, 'display.max_colwidth', 60):
        print(rep.to_string(index=False))


def main(argv=None):
    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        print('usage: python -m rmy.audit FILE.epw [FILE.epw ...]')
        return 2
    report = audit_epws(paths)
    print_report(report)
    return 1 if (report['problem'] == 'invalid').any() else 0

