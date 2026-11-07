#!/usr/bin/env python3
"""
Generate a typical meteorological year (TMY) EPW from a set of AMY EPWs, following
ISO 15927-4:2005 ("Hourly data for assessing the annual energy use for heating and
cooling"), the method also used for the TMYx files from climate.onebuilding.org.

Steps (clause numbers refer to ISO 15927-4:2005):
  Clause 4    Missing values (EPW missing codes and physically impossible values) are
              filled by linear interpolation for gaps of up to --max-gap hours.
              Night-time solar gaps are set to zero. Longer gaps are left missing; a
              month that still has missing hours in a primary parameter is not used.
  5.3.1 a)    At least 10 years of data (--min-years).
  5.3.1 a-d)  For each calendar month and each primary parameter (daily mean dry-bulb
              temperature, relative humidity and global horizontal radiation), the
              Finkelstein-Schafer statistic of every candidate month is computed with
              Equations (1) to (3) of the standard.
  5.3.1 e-g)  The candidate months are ranked per parameter, the ranks are summed, and
              of the three months with the lowest total the one whose mean wind speed is
              closest to the multi-year mean of that calendar month is selected (months
              with wind speed in fewer than --min-wind-completeness of their hours are
              left out of this wind comparison). Each year may appear only once, and a
              month is used only if its days and hours are complete and in order.
  5.3.2       The last 8 hours of each month and the first 8 hours of the next month
              (including December -> January) are adjusted by interpolation so the months
              join smoothly. Dry-bulb, dew point, pressure and wind speed are adjusted,
              and relative humidity is rescaled to stay consistent with the adjusted dry
              bulb and dew point. Solar radiation, illuminance and wind direction are not.
  6           The years each month was taken from, the parameters used and the period of
              record are written to COMMENTS 1 and to a month-source CSV.

Inputs:
  --amy-dir      Directory containing AMY EPW files (the year must be in each file name).
  --header-epw   EPW whose 8 header lines are used for the output (location metadata).
  --output       Output EPW path.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Column index constants (0-based, per EPW)
YEAR_COL = 0
MONTH_COL = 1
DAY_COL = 2
HOUR_COL = 3
TEMP_COL = 6          # Dry bulb temperature (C)
DEWPOINT_COL = 7      # Dew point temperature (C)
RH_COL = 8            # Relative humidity (%)
PRESSURE_COL = 9      # Atmospheric station pressure (Pa)
EXTRA_HOR_COL = 10    # Extraterrestrial horizontal radiation (Wh/m2), 0 at night
GHI_COL = 13          # Global horizontal radiation (Wh/m2)
DNI_COL = 14          # Direct normal radiation (Wh/m2)
DHI_COL = 15          # Diffuse horizontal radiation (Wh/m2)
WS_COL = 21           # Wind speed (m/s)

# EPW missing codes and plausible ranges of the fields this script reads or adjusts.
# Values equal to the missing code or outside the range are treated as missing.
FIELDS = {
    TEMP_COL: (99.9, -70.0, 70.0, '{:.1f}'),
    DEWPOINT_COL: (99.9, -70.0, 70.0, '{:.1f}'),
    RH_COL: (999.0, 0.0, 110.0, '{:.0f}'),
    PRESSURE_COL: (999999.0, 31000.0, 120000.0, '{:.0f}'),
    GHI_COL: (9999.0, 0.0, 1500.0, '{:.0f}'),
    DNI_COL: (9999.0, 0.0, 1500.0, '{:.0f}'),
    DHI_COL: (9999.0, 0.0, 1000.0, '{:.0f}'),
    WS_COL: (999.0, 0.0, 40.0, '{:.1f}'),
}
SOLAR_COLS = (GHI_COL, DNI_COL, DHI_COL)

# Primary parameters of ISO 15927-4 5.3.1, and the secondary one.
PRIMARY = {'temperature': TEMP_COL, 'humidity': RH_COL, 'solar': GHI_COL}
SECONDARY = WS_COL

# Parameters adjusted at the month joins (5.3.2).
DEFAULT_SMOOTH_COLS = [TEMP_COL, DEWPOINT_COL, RH_COL, PRESSURE_COL, WS_COL]

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTH_LENGTHS = (672, 696, 720, 744)


# --------------------------------------------------------------------------- reading

def read_epw_header(epw_path: Path) -> List[str]:
    """Return the first 8 EPW header lines, with their original line endings."""
    with Path(epw_path).open('r', encoding='utf-8', errors='ignore', newline='') as f:
        return [next(f) for _ in range(8)]


def read_epw_data(epw_path: Path) -> pd.DataFrame:
    """Read the EPW hourly data section as text, exactly as written in the file."""
    return pd.read_csv(epw_path, skiprows=8, header=None, dtype=str,
                       keep_default_na=False)


def numeric_fields(raw: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Numeric copies of the fields in FIELDS, with missing and impossible values as NaN."""
    out = {}
    for col, (code, lo, hi, _) in FIELDS.items():
        if col >= raw.shape[1]:
            continue
        v = pd.to_numeric(raw.iloc[:, col], errors='coerce').to_numpy(dtype=float, copy=True)
        v[np.isclose(v, code) | (v < lo) | (v > hi)] = np.nan
        out[col] = v
    return out


def fill_gaps(values: np.ndarray, max_gap: int, night: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate runs of NaN of at most `max_gap` hours (ISO 15927-4 clause 4).

    `night` (solar fields only): missing night hours are set to 0 regardless of gap length.
    Returns (filled values, boolean mask of the hours that were filled).
    """
    v = values.copy()
    filled = np.zeros(len(v), dtype=bool)
    if night is not None:
        m = np.isnan(v) & night
        v[m] = 0.0
        filled |= m
    missing = np.isnan(v)
    edges = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)))
    for start, stop in zip(edges[::2], edges[1::2]):
        left, right = start - 1, stop
        if stop - start > max_gap or left < 0 or right >= len(v):
            continue
        w = (np.arange(start, stop) - left) / (right - left)
        v[start:stop] = (1 - w) * v[left] + w * v[right]
        filled[start:stop] = True
    return v, filled


def find_epw_files(amy_dir: Path, pattern: str = '*.epw', recursive: bool = False) -> List[Path]:
    """Find EPW files in the AMY directory."""
    iterator = amy_dir.rglob(pattern) if recursive else amy_dir.glob(pattern)
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() == '.epw')


def year_from_filename(path: Path) -> Optional[int]:
    """The calendar year in an EPW file name (e.g. Boston_..._2015_AMY.epw -> 2015)."""
    match = re.findall(r'(?<!\d)((?:19|20|21)\d{2})(?!\d)', Path(path).name)
    return int(match[-1]) if match else None


# ------------------------------------------------------------------ selection (5.3.1)

def fs_statistic(candidate: np.ndarray, pooled_sorted: np.ndarray) -> float:
    """Finkelstein-Schafer statistic, ISO 15927-4 Equations (1) to (3).

    F(i)   = J(i) / (n + 1): J(i) is the rank of the i-th daily mean within the month.
    Phi(i) = K(i) / (N + 1): K(i) is the rank of that same daily mean among all N daily
             means of that calendar month in the whole data set.
    FS     = sum over the n days of |F(i) - Phi(i)|.
    Tied values (common for daily mean relative humidity) get the average of the ranks
    they span, in both J and K; the standard does not specify a tie rule.
    """
    x = np.sort(np.asarray(candidate, dtype=float))
    n, N = len(x), len(pooled_sorted)
    J = (np.searchsorted(x, x, side='left') + np.searchsorted(x, x, side='right') + 1) / 2
    K = (np.searchsorted(pooled_sorted, x, side='left')
         + np.searchsorted(pooled_sorted, x, side='right') + 1) / 2
    return float(np.sum(np.abs(J / (n + 1) - K / (N + 1))))


def _daily_means(month: Dict[int, np.ndarray], day: np.ndarray, col: int) -> np.ndarray:
    return pd.Series(month[col]).groupby(day).mean().to_numpy()


def _rank(scores: Sequence[float]) -> np.ndarray:
    """Rank 1 = lowest score; ties keep the candidate order (stable)."""
    order = np.argsort(np.asarray(scores, dtype=float), kind='mergesort')
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def select_months(candidates: Dict[int, List[dict]], min_wind_completeness: float = 0.95
                  ) -> Dict[int, dict]:
    """Apply ISO 15927-4 5.3.1 b) to g) to the candidate months of each calendar month.

    Step g) compares monthly mean wind speeds. A month whose wind speed is missing in more
    than 1 - `min_wind_completeness` of its hours (after gap filling) is not used for that
    comparison: its mean is not representative, and it cannot win the wind criterion. The
    multi-year mean is taken over the complete months. If none of the three months is
    complete enough, the one with the lowest rank sum is selected.

    Ties are not specified in the standard. Months with equal rank sums at the cut-off
    for the three lowest are ordered by the same wind-speed deviation, then by year.
    """
    selected = {}
    for month, cands in candidates.items():
        rows = [{'year': c['year'], 'file': c['file']} for c in cands]
        for name, col in PRIMARY.items():
            daily = [_daily_means(c['values'], c['day'], col) for c in cands]
            pooled = np.sort(np.concatenate(daily))
            scores = [fs_statistic(d, pooled) for d in daily]
            for r, s, k in zip(rows, scores, _rank(scores)):
                r[f'fs_{name}'], r[f'rank_{name}'] = s, int(k)
        wind = [c['values'][SECONDARY] for c in cands]
        complete = [float(np.isfinite(w).mean()) for w in wind]
        ws = [float(np.nanmean(w)) if f > 0 else float('nan') for w, f in zip(wind, complete)]
        usable = [w for w, f in zip(ws, complete) if f >= min_wind_completeness]
        ws_ref = float(np.mean(usable)) if usable else float('nan')
        for r, w, f in zip(rows, ws, complete):
            r['rank_sum'] = sum(r[f'rank_{n}'] for n in PRIMARY)
            r['ws_mean'], r['ws_ref_mean'], r['ws_completeness'] = w, ws_ref, f
            ok = f >= min_wind_completeness and np.isfinite(w) and np.isfinite(ws_ref)
            r['ws_abs_diff'] = abs(w - ws_ref) if ok else float('inf')
        ranked = sorted(rows, key=lambda r: (r['rank_sum'], r['ws_abs_diff'], r['year']))
        best = min(ranked[:3], key=lambda r: (r['ws_abs_diff'], r['rank_sum'], r['year']))
        best['top3_years'] = ' '.join(str(r['year']) for r in ranked[:3])
        best['n_candidates'] = len(rows)
        selected[month] = best
    return selected


# ------------------------------------------------------------------ month joins (5.3.2)

def smooth_month_joins(values: Dict[int, np.ndarray], starts: Sequence[int],
                       cols: Iterable[int], span: int = 8, method: str = 'offset'
                       ) -> np.ndarray:
    """Adjust the last `span` hours of each month and the first `span` of the next.

    Every join is adjusted, including December -> January (the year wraps around).

    method='offset' (default): each month keeps its own hourly profile. The step at the
        join (beyond the average hourly change on either side of it) is removed by an
        offset that grows linearly over the `span` hours before the join and shrinks
        linearly over the `span` hours after it.
    method='linear': the 2*span hours are replaced by a straight line between the hour
        before and the hour after the window.
    Relative humidity is not adjusted on its own where dry bulb and dew point are known:
    it is scaled by the change in the vapour-pressure ratio e(Td)/e(T) that the adjusted
    dry bulb and dew point imply, so the three stay consistent (and each source keeps its
    own humidity convention). Dew point is capped at the dry bulb.
    Returns a boolean mask of the adjusted hours.
    """
    n = len(next(iter(values.values())))
    touched = np.zeros(n, dtype=bool)
    if span <= 0:
        return touched
    original = {c: values[c].copy() for c in (TEMP_COL, DEWPOINT_COL, RH_COL) if c in values}
    for b_start in starts:
        a_end = (b_start - 1) % n
        before = (a_end - np.arange(span - 1, -1, -1)) % n        # last span hours of A
        after = (b_start + np.arange(span)) % n                   # first span hours of B
        left, right = (a_end - span) % n, (b_start + span) % n    # unchanged anchors
        for col in cols:
            if col not in values:
                continue
            v = values[col]
            if method == 'linear':
                if not (np.isfinite(v[left]) and np.isfinite(v[right])):
                    continue
                line = np.linspace(v[left], v[right], 2 * span + 2)[1:-1]
                v[np.r_[before, after]] = line
            else:
                pts = [a_end, (a_end - 1) % n, b_start, (b_start + 1) % n]
                if not np.isfinite(v[pts]).all():
                    continue
                natural = ((v[a_end] - v[(a_end - 1) % n]) + (v[(b_start + 1) % n] - v[b_start])) / 2
                excess = (v[b_start] - v[a_end]) - natural
                ramp = np.arange(1, span + 1) / span                # 1/span .. 1
                v[before] += excess / 2 * ramp
                v[after] -= excess / 2 * ramp[::-1]
            touched[before] = touched[after] = True
    if WS_COL in values:
        values[WS_COL] = np.maximum(values[WS_COL], 0)
    if TEMP_COL in values and DEWPOINT_COL in values:
        t, td = values[TEMP_COL], values[DEWPOINT_COL]
        cap = touched & np.isfinite(t) & (td > t)
        td[cap] = t[cap]
        if RH_COL in values:
            def ratio(temp, dew):          # e(Td)/e(T), Magnus formula over water
                es = lambda x: np.exp(17.625 * x / (x + 243.04))
                return es(dew) / es(temp)
            with np.errstate(invalid='ignore', divide='ignore'):
                scaled = original[RH_COL] * ratio(t, td) / ratio(original[TEMP_COL], original[DEWPOINT_COL])
            use = touched & np.isfinite(scaled)
            values[RH_COL][use] = scaled[use]
    if RH_COL in values:
        values[RH_COL] = np.clip(values[RH_COL], 0, 100)
    return touched


# ------------------------------------- header: design conditions and ground temperatures
# Used only when the header EPW has none ('DESIGN CONDITIONS,0', 'GROUND TEMPERATURES,0'),
# e.g. AMYs built from NOAA data; published values in a header EPW are kept.

WD_COL = 20                       # Wind direction (degrees)
SOIL_DIFFUSIVITY = 2.322576e-3 * 24  # m2/day (2.322576E-03 m2/h, the EnergyPlus standard soil)
MID_MONTH_DAY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)


def _saturation_pressure(t_c):
    """Saturation vapour pressure (Pa) over ice below 0 C and over water above
    (Hyland-Wexler, ASHRAE Handbook - Fundamentals 2017, ch. 1, Eq. 5 and 6)."""
    t = np.asarray(t_c, dtype=float) + 273.15
    ice = np.exp(-5.6745359e3 / t + 6.3925247 - 9.677843e-3 * t + 6.2215701e-7 * t ** 2
                 + 2.0747825e-9 * t ** 3 - 9.484024e-13 * t ** 4 + 4.1635019 * np.log(t))
    water = np.exp(-5.8002206e3 / t + 1.3914993 - 4.8640239e-2 * t + 4.1764768e-5 * t ** 2
                   - 1.4452093e-8 * t ** 3 + 6.5459673 * np.log(t))
    return np.where(t < 273.15, ice, water)


def _humidity_ratio(p_w, p):
    """Humidity ratio (kg/kg) from vapour pressure and air pressure (Pa)."""
    return 0.621945 * p_w / (p - p_w)


def _wet_bulb(db, dp, p, iterations=40):
    """Thermodynamic wet-bulb temperature (C), by bisection on ASHRAE Eq. 33 and 35."""
    w = _humidity_ratio(_saturation_pressure(dp), p)
    lo, hi = np.minimum(dp, db) - 1.0, np.asarray(db, dtype=float).copy()
    for _ in range(iterations):
        t = (lo + hi) / 2
        ws = _humidity_ratio(_saturation_pressure(t), p)
        w_t = np.where(t >= 0,
                       ((2501 - 2.326 * t) * ws - 1.006 * (db - t)) / (2501 + 1.86 * db - 4.186 * t),
                       ((2830 - 0.24 * t) * ws - 1.006 * (db - t)) / (2830 + 1.86 * db - 2.1 * t))
        too_wet = w_t > w            # the wet bulb is lower
        hi, lo = np.where(too_wet, t, hi), np.where(too_wet, lo, t)
    return (lo + hi) / 2


def _coincident(x, design, y, min_hours=20, tol=0.5):
    """Mean of y over the hours whose x is within tol of the design value (at least the
    min_hours nearest hours), as for the ASHRAE mean coincident values."""
    dist = np.abs(x - design)
    n = max(int((dist <= tol).sum()), min(min_hours, len(x)))
    return float(np.mean(y[np.argsort(dist, kind='stable')[:n]]))


def _prevailing_direction(wd, ws):
    """Most frequent wind direction (degrees, 10-degree sectors), calm hours left out."""
    d = wd[(ws > 0) & np.isfinite(wd)]
    if len(d) == 0:
        return 0
    sectors = (np.rint(d / 10) * 10) % 360
    values, counts = np.unique(sectors, return_counts=True)
    return int(values[np.argmax(counts)])


def design_conditions_line(hours: pd.DataFrame, eol: str = '\n') -> str:
    """EPW 'DESIGN CONDITIONS' line (ASHRAE 2009 layout, as written by the EnergyPlus
    Weather Converter) calculated from multi-year hourly data with the ASHRAE Handbook -
    Fundamentals ch. 14 definitions: annual percentiles of the pooled hours, mean
    coincident values, coldest/hottest month, and extreme annual dry-bulb statistics with
    Gumbel return periods. `hours` has columns year, month, hour, db, dp, p, ws, wd."""
    h = hours.dropna(subset=['db', 'dp', 'ws']).copy()
    p_std = float(np.nanmedian(h['p'])) if h['p'].notna().any() else 101325.0
    h['p'] = h['p'].fillna(p_std)
    h['w'] = _humidity_ratio(_saturation_pressure(h['dp']), h['p'])
    h['wb'] = _wet_bulb(h['db'].to_numpy(), h['dp'].to_numpy(), h['p'].to_numpy())
    h['en'] = 1.006 * h['db'] + h['w'] * (2501 + 1.86 * h['db'])
    db, dp, wb, en, ws, wd = (h[c].to_numpy(float) for c in ('db', 'dp', 'wb', 'en', 'ws', 'wd'))
    pct = lambda v, q: float(np.percentile(v, q))
    hr = lambda d: 1000 * float(_humidity_ratio(_saturation_pressure(d), p_std))   # g/kg
    f1 = lambda v: f'{v:.1f}'
    monthly = h.groupby('month')['db'].mean()
    cold, hot = int(monthly.idxmin()), int(monthly.idxmax())

    heating = [str(cold)]
    db996, db990 = pct(db, 0.4), pct(db, 1.0)
    heating += [f1(db996), f1(db990)]
    for q in (0.4, 1.0):                                  # humidification
        d = pct(dp, q)
        heating += [f1(d), f1(hr(d)), f1(_coincident(dp, d, db))]
    cm = (h['month'] == cold).to_numpy()
    for q in (99.6, 99.0):                                # coldest-month wind
        s = pct(ws[cm], q)
        heating += [f1(s), f1(_coincident(ws[cm], s, db[cm]))]
    tail = db <= db996
    heating += [f1(ws[tail].mean()), str(_prevailing_direction(wd[tail], ws[tail]))]

    hot_days = h[h['month'] == hot].groupby(['year', 'day'])['db']
    cooling = [str(hot), f1((hot_days.max() - hot_days.min()).mean())]
    db004 = pct(db, 99.6)
    for q in (99.6, 99.0, 98.0):                          # dry bulb / mean coincident wet bulb
        t = pct(db, q)
        cooling += [f1(t), f1(_coincident(db, t, wb))]
    for q in (99.6, 99.0, 98.0):                          # wet bulb / mean coincident dry bulb
        t = pct(wb, q)
        cooling += [f1(t), f1(_coincident(wb, t, db))]
    tail = db >= db004
    cooling += [f1(ws[tail].mean()), str(_prevailing_direction(wd[tail], ws[tail]))]
    for q in (99.6, 99.0, 98.0):                          # dehumidification
        d = pct(dp, q)
        cooling += [f1(d), f1(hr(d)), f1(_coincident(dp, d, db))]
    for q in (99.6, 99.0, 98.0):                          # enthalpy (kJ/kg)
        e = pct(en, q)
        cooling += [f1(e), f1(_coincident(en, e, db, tol=1.0))]
    n_years = h['year'].nunique()
    work = h['hour'].between(9, 16) & h['db'].between(12.8, 20.6)   # 8:00-16:00, 12.8-20.6 C
    cooling.append(str(int(round(work.sum() / n_years))))

    extremes = [f1(pct(ws, q)) for q in (99.0, 97.5, 95.0)] + [f1(wb.max())]
    yearly = h.groupby('year')['db'].agg(['min', 'max'])
    m_min, m_max = yearly['min'].mean(), yearly['max'].mean()
    s_min, s_max = (yearly['min'].std(ddof=1), yearly['max'].std(ddof=1)) if n_years > 1 else (0.0, 0.0)
    extremes += [f1(m_min), f1(m_max), f1(s_min), f1(s_max)]
    for n in (5, 10, 20, 50):                             # Gumbel return periods (ASHRAE)
        f = -np.sqrt(6) / np.pi * (0.5772 + np.log(np.log(n / (n - 1))))
        extremes += [f1(m_min - f * s_min), f1(m_max + f * s_max)]

    years = sorted(h['year'].unique())
    source = f'Calculated from AMYs {years[0]}-{years[-1]} (ASHRAE Handbook method)'
    return ','.join(['DESIGN CONDITIONS', '1', source, '', 'Heating', *heating,
                     'Cooling', *cooling, 'Extremes', *extremes]) + eol


def ground_temperatures_line(month: np.ndarray, db: np.ndarray, eol: str = '\n') -> str:
    """EPW 'GROUND TEMPERATURES' line: undisturbed ground temperatures at 0.5, 2 and 4 m
    from the monthly mean dry-bulb temperatures (Kusuda and Achenbach 1965, as used by
    EnergyPlus), with the EnergyPlus standard soil diffusivity:
        T(z, t) = Tm - A exp(-z sqrt(pi / (365 a))) cos(2 pi / 365 (t - t0 - z / 2 sqrt(365 / (pi a))))
    Tm, A and t0 (day of the coldest surface temperature) come from a one-harmonic fit
    of the 12 monthly means."""
    means = pd.Series(db).groupby(np.asarray(month)).mean().reindex(range(1, 13)).to_numpy()
    w = 2 * np.pi / 365
    basis = np.column_stack([np.ones(12), np.cos(w * MID_MONTH_DAY), np.sin(w * MID_MONTH_DAY)])
    tm, a, b = np.linalg.lstsq(basis, means, rcond=None)[0]
    amp, t0 = np.hypot(a, b), (np.arctan2(b, a) + np.pi) / w
    parts = ['GROUND TEMPERATURES', '3']
    for z in (0.5, 2.0, 4.0):
        damp = np.exp(-z * np.sqrt(np.pi / (365 * SOIL_DIFFUSIVITY)))
        lag = z / 2 * np.sqrt(365 / (np.pi * SOIL_DIFFUSIVITY))
        temps = tm - amp * damp * np.cos(w * (MID_MONTH_DAY - t0 - lag))
        parts += [f'{z:g}' if z != 0.5 else '.5', '', '', ''] + [f'{t:.2f}' for t in temps]
    return ','.join(parts) + eol


def _header_line_is_empty(line: str) -> bool:
    fields = line.strip().split(',')
    return len(fields) < 3 or fields[1].strip() in ('', '0')


# ------------------------------------------------------------------------- header (6)

def build_header(header: List[str], selected: Dict[int, dict], header_name: str,
                 update_comments_1: bool = True, update_comments_2: bool = True,
                 design_line: Optional[str] = None, ground_line: Optional[str] = None) -> List[str]:
    """Header for the generated file: the 8 lines of the header EPW, with COMMENTS 1
    reporting this file's period of record and source year of every month (ISO 15927-4
    clause 6), and COMMENTS 2 stating where the remaining metadata came from.
    design_line / ground_line replace DESIGN CONDITIONS / GROUND TEMPERATURES only where
    the header EPW has none."""
    out = list(header)
    eol = '\r\n' if header[0].endswith('\r\n') else '\n'
    n_years = max(r['n_candidates'] for r in selected.values())
    first, last = min(r['first_year'] for r in selected.values()), max(r['last_year'] for r in selected.values())
    months = '; '.join(f'{MONTH_NAMES[m - 1]}={selected[m]["year"]}' for m in range(1, 13))
    copied, calculated = ['Location'], []
    for i, line in enumerate(out):
        for key, new, what in (('DESIGN CONDITIONS', design_line, 'design conditions'),
                               ('GROUND TEMPERATURES', ground_line, 'ground temperatures')):
            if line.upper().startswith(key):
                if new is not None and _header_line_is_empty(line):
                    out[i] = new.rstrip('\r\n') + eol
                    calculated.append(what)
                else:
                    copied.append(what)
    for i, line in enumerate(out):
        if update_comments_1 and line.upper().startswith('COMMENTS 1'):
            out[i] = (f'COMMENTS 1,"ISO 15927-4 reference year from AMY EPWs - #years=[{n_years}] '
                      f'Period of Record={first}-{last}; {months}; selection parameters: daily mean '
                      f'dry-bulb temperature, relative humidity and global horizontal radiation, '
                      f'with wind speed as secondary parameter"{eol}')
        if update_comments_2 and line.upper().startswith('COMMENTS 2'):
            names = ', '.join(copied[:-1]) + ' and ' + copied[-1] if len(copied) > 1 else copied[0]
            note = f'{names} copied from {header_name}; not recalculated for this file'
            if 'design conditions' in calculated:
                note += ('; design conditions calculated from the AMYs with the ASHRAE Handbook '
                         'method (not the published ASHRAE values)')
            if 'ground temperatures' in calculated:
                note += ("; ground temperatures calculated from this TMY's monthly dry-bulb "
                         'temperatures (Kusuda-Achenbach; soil diffusivity 0.0557 m2/day)')
            out[i] = f'COMMENTS 2,"{note}"{eol}'
    return out


# ------------------------------------------------------------------------ main routine

def generate_tmy_from_amys(
    amy_dir: Path | str,
    output_epw: Path | str,
    header_epw: Path | str,
    sources_csv: Optional[Path | str] = None,
    pattern: str = '*.epw',
    recursive: bool = False,
    smooth_span: int = 8,
    smooth_method: str = 'offset',
    years: Optional[Tuple[int, int]] = None,
    min_years: int = 10,
    max_gap: int = 6,
    min_wind_completeness: float = 0.95,
    set_year: Optional[int] = None,
    update_header_years: bool = True,
    add_header_generation_note: bool = True,
    fill_design_conditions: bool = True,
    verbose: bool = True,
) -> Dict[int, str]:
    """
    Generate an ISO 15927-4 TMY EPW from AMYs.

    Parameters
    ----------
    amy_dir: Directory containing AMY EPW files (year in each file name).
    output_epw: Path where the generated EPW is written.
    header_epw: EPW whose 8 header lines are copied to the output.
    sources_csv: Month-source CSV (default: <output>.month_sources.csv).
    pattern, recursive: How AMY files are found.
    smooth_span: Hours adjusted on each side of every month join (8 in ISO 15927-4).
    smooth_method: 'offset' (keeps each month's hourly profile) or 'linear'.
    years: Optional (first, last) year range of AMY files to use, e.g. (2009, 2023).
    min_years: Minimum number of years per calendar month (10 in ISO 15927-4).
    max_gap: Longest gap in hours filled by linear interpolation.
    min_wind_completeness: Share of hours with wind speed a month needs to take part in the
        wind-speed comparison of 5.3.1 g).
    set_year: Optional year written into the year column of every row.
    update_header_years: Rewrite COMMENTS 1 with this file's period and month years.
    add_header_generation_note: Rewrite COMMENTS 2 with a note on the header source.
    fill_design_conditions: Where the header EPW has no design conditions or ground
        temperatures, calculate them (from all AMYs, and from this TMY's monthly
        temperatures). Published values in the header EPW are always kept.
    verbose: Print progress.

    Returns
    -------
    Dict[int, str]: month number -> selected source file name.
    """
    amy_dir, output_epw, header_epw = Path(amy_dir), Path(output_epw), Path(header_epw)
    sources_csv = Path(sources_csv) if sources_csv is not None else output_epw.with_suffix('.month_sources.csv')
    if not amy_dir.exists():
        raise FileNotFoundError(f'AMY directory does not exist: {amy_dir}')
    if not header_epw.exists():
        raise FileNotFoundError(f'Header EPW does not exist: {header_epw}')
    if smooth_method not in ('offset', 'linear'):
        raise ValueError("smooth_method must be 'offset' or 'linear'")

    files = find_epw_files(amy_dir, pattern=pattern, recursive=recursive)
    if years is not None:
        files = [f for f in files if (y := year_from_filename(f)) is not None and years[0] <= y <= years[1]]
    if not files:
        raise FileNotFoundError(f'No EPW files found in {amy_dir} (pattern={pattern!r}, years={years})')
    if verbose:
        print(f'Found {len(files)} AMY EPW files in {amy_dir}')

    candidates: Dict[int, List[dict]] = {m: [] for m in range(1, 13)}
    design_hours: List[pd.DataFrame] = []
    expected_cols = None
    fill_log = []
    seen_years: Dict[int, str] = {}
    for path in files:
        try:
            raw = read_epw_data(path)
        except Exception as exc:  # unreadable file
            print(f'Skipping {path.name}: could not read EPW data ({exc})')
            continue
        if expected_cols is None:
            expected_cols = raw.shape[1]
        if raw.shape[1] != expected_cols:
            print(f'Skipping {path.name}: column mismatch ({raw.shape[1]} vs {expected_cols})')
            continue
        year = year_from_filename(path)
        if year is None:
            year = int(pd.to_numeric(raw.iloc[:, YEAR_COL], errors='coerce').mode().iloc[0])
        if year in seen_years:
            raise ValueError(f'Two AMY files for {year}: {seen_years[year]} and {path.name}. '
                             f'Each year may appear once, or it is counted twice in the '
                             f'long-term distribution.')
        seen_years[year] = path.name

        # Clause 4: fill missing values by linear interpolation.
        values = numeric_fields(raw)
        night = None
        if EXTRA_HOR_COL < raw.shape[1]:
            night = pd.to_numeric(raw.iloc[:, EXTRA_HOR_COL], errors='coerce').to_numpy() == 0
        filled = {}
        for col in list(values):
            values[col], filled[col] = fill_gaps(values[col], max_gap,
                                                 night if col in SOLAR_COLS else None)
        fill_log.append({'file': path.name, **{f'filled_col{c}': int(f.sum()) for c, f in filled.items()},
                         **{f'missing_col{c}': int(np.isnan(v).sum()) for c, v in values.items()}})

        month_col = pd.to_numeric(raw.iloc[:, MONTH_COL], errors='coerce').to_numpy()
        day_col = pd.to_numeric(raw.iloc[:, DAY_COL], errors='coerce').to_numpy()
        hour_col = pd.to_numeric(raw.iloc[:, HOUR_COL], errors='coerce').to_numpy()
        if fill_design_conditions:
            wd = (pd.to_numeric(raw.iloc[:, WD_COL], errors='coerce').to_numpy(dtype=float, copy=True)
                  if WD_COL < raw.shape[1] else np.full(len(raw), np.nan))
            wd[(wd < 0) | (wd > 360)] = np.nan
            design_hours.append(pd.DataFrame({
                'year': year, 'month': month_col, 'day': day_col, 'hour': hour_col,
                **{k: values.get(c, np.full(len(raw), np.nan)) for k, c in
                   (('db', TEMP_COL), ('dp', DEWPOINT_COL), ('p', PRESSURE_COL), ('ws', WS_COL))},
                'wd': wd}))
        for m in range(1, 13):
            idx = np.flatnonzero(month_col == m)
            if len(idx) not in MONTH_LENGTHS:
                continue
            # Every day 1..n_days must have hours 1..24, in order and without repeats.
            n_days = len(idx) // 24
            in_order = (np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx)))
                        and np.array_equal(day_col[idx], np.repeat(np.arange(1, n_days + 1), 24))
                        and np.array_equal(hour_col[idx], np.tile(np.arange(1, 25), n_days)))
            if not in_order:
                print(f'   {path.name} {MONTH_NAMES[m - 1]}: not a candidate, '
                      f'days/hours are not complete and in order')
                continue
            month_vals = {c: v[idx] for c, v in values.items()}
            gaps = [name for name, c in PRIMARY.items() if np.isnan(month_vals[c]).any()]
            if gaps:
                if verbose:
                    print(f'   {path.name} {MONTH_NAMES[m - 1]}: not a candidate, '
                          f'missing hours remain in {", ".join(gaps)}')
                continue
            candidates[m].append({'year': year, 'file': path.name, 'raw': raw.iloc[idx],
                                  'values': month_vals, 'filled': {c: f[idx] for c, f in filled.items()},
                                  'day': day_col[idx]})

    if expected_cols is None:
        raise RuntimeError('No valid EPW files could be read.')
    short = {MONTH_NAMES[m - 1]: len({c['year'] for c in cands}) for m, cands in candidates.items()
             if len({c['year'] for c in cands}) < min_years}
    if short:
        raise ValueError(f'ISO 15927-4 needs at least {min_years} years per calendar month; '
                         f'available: {short}. Lower min_years to proceed anyway.')

    selected = select_months(candidates, min_wind_completeness=min_wind_completeness)
    for m, r in selected.items():
        ys = [c['year'] for c in candidates[m]]
        r['first_year'], r['last_year'] = min(ys), max(ys)

    # Assemble the year from the selected months (29 Feb dropped).
    blocks, block_vals, block_filled = [], [], []
    for m in range(1, 13):
        cand = next(c for c in candidates[m] if c['year'] == selected[m]['year'] and c['file'] == selected[m]['file'])
        keep = ~((m == 2) & (cand['day'] == 29))
        blocks.append(cand['raw'][keep])
        block_vals.append({c: v[keep] for c, v in cand['values'].items()})
        block_filled.append({c: f[keep] for c, f in cand['filled'].items()})
    out = pd.concat(blocks, axis=0, ignore_index=True)
    values = {c: np.concatenate([b[c] for b in block_vals]) for c in block_vals[0]}
    changed = {c: np.concatenate([b[c] for b in block_filled]) for c in block_filled[0]}
    if len(out) != 8760:
        raise AssertionError(f'Expected 8760 rows, got {len(out)}')

    # Clause 5.3.2: adjust the month joins, December -> January included.
    starts = np.cumsum([0] + [len(b) for b in blocks[:-1]])
    smooth_cols = [c for c in DEFAULT_SMOOTH_COLS if c in values]
    joins = smooth_month_joins(values, starts, smooth_cols, span=smooth_span, method=smooth_method)
    for c in smooth_cols:
        changed[c] = changed[c] | joins

    # Write: unchanged cells keep their original text; filled or adjusted cells are
    # formatted like the EPW field; hours still missing keep the EPW missing code.
    for c, mask in changed.items():
        if not mask.any():
            continue
        code, _, _, fmt = FIELDS[c]
        text = [fmt.format(v) if np.isfinite(v) else fmt.format(code) for v in values[c][mask]]
        out.iloc[np.flatnonzero(mask), c] = text
    if set_year is not None:
        out.iloc[:, YEAR_COL] = str(int(set_year))

    design_line = ground_line = None
    if fill_design_conditions:
        design_line = design_conditions_line(pd.concat(design_hours, ignore_index=True))
        ground_line = ground_temperatures_line(pd.to_numeric(out.iloc[:, MONTH_COL]).to_numpy(),
                                               values[TEMP_COL])
    header = build_header(read_epw_header(header_epw), selected, header_epw.name,
                          update_comments_1=update_header_years,
                          update_comments_2=add_header_generation_note,
                          design_line=design_line, ground_line=ground_line)
    eol = '\r\n' if header[0].endswith('\r\n') else '\n'
    output_epw.parent.mkdir(parents=True, exist_ok=True)
    with output_epw.open('w', encoding='utf-8', newline='') as f:
        f.writelines(header)
        f.write(eol.join(','.join(row) for row in out.itertuples(index=False, name=None)) + eol)

    log = pd.DataFrame([{'month': m, **{k: v for k, v in r.items()}} for m, r in selected.items()])
    log['hours_adjusted_at_joins'] = int(joins.sum())
    sources_csv.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(sources_csv, index=False)
    pd.DataFrame(fill_log).to_csv(sources_csv.with_name(sources_csv.stem + '_gap_fill.csv'), index=False)

    if verbose:
        print('Month sources (ISO 15927-4):')
        for m in range(1, 13):
            r = selected[m]
            print(f'  {MONTH_NAMES[m - 1]} -> {r["year"]}  (three lowest rank sums: {r["top3_years"]}; '
                  f'{r["n_candidates"]} candidates)')
        print(f'Hours filled by interpolation in the output: '
              + ', '.join(f'col {c}: {int((changed[c] & ~joins).sum())}' for c in changed))
        print(f'Hours adjusted at the 12 month joins: {int(joins.sum())} ({smooth_method})')
        print(f'TMY written to: {output_epw}')
        print(f'Month-source log written to: {sources_csv}')
    return {m: r['file'] for m, r in selected.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate an ISO 15927-4 TMY EPW from a folder of AMY EPWs, using a separate EPW for the output header.')
    parser.add_argument('--amy-dir', required=True, type=Path, help='Directory containing AMY EPW files.')
    parser.add_argument('--header-epw', required=True, type=Path, help='EPW file whose first 8 header lines are used.')
    parser.add_argument('--output', required=True, type=Path, help='Output TMY EPW path.')
    parser.add_argument('--sources-csv', type=Path, default=None, help='Optional month-source CSV output path.')
    parser.add_argument('--pattern', default='*.epw', help='Glob pattern for EPW files. Default: *.epw')
    parser.add_argument('--recursive', action='store_true', help='Search the AMY directory recursively.')
    parser.add_argument('--years', default=None, help='Year range of AMY files to use, e.g. 2009-2023.')
    parser.add_argument('--smooth-span', type=int, default=8, help='Hours adjusted on each side of month joins (0 disables).')
    parser.add_argument('--smooth-method', choices=('offset', 'linear'), default='offset',
                        help="'offset' keeps each month's hourly profile; 'linear' draws a straight line.")
    parser.add_argument('--min-years', type=int, default=10, help='Minimum years per month (ISO 15927-4: 10).')
    parser.add_argument('--max-gap', type=int, default=6, help='Longest gap (hours) filled by interpolation.')
    parser.add_argument('--min-wind-completeness', type=float, default=0.95,
                        help='Share of hours with wind speed a month needs for the wind comparison (5.3.1 g).')
    parser.add_argument('--set-year', type=int, default=None, help='Optional year written into every row.')
    parser.add_argument('--no-update-header-years', action='store_true',
                        help='Keep COMMENTS 1 of the header EPW unchanged.')
    parser.add_argument('--no-header-generation-note', action='store_true',
                        help='Keep COMMENTS 2 of the header EPW unchanged.')
    parser.add_argument('--no-design-conditions', action='store_true',
                        help='Do not calculate design conditions and ground temperatures '
                             'missing from the header EPW.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    years = None
    if args.years:
        first, _, last = args.years.partition('-')
        years = (int(first), int(last or first))
    generate_tmy_from_amys(
        amy_dir=args.amy_dir, output_epw=args.output, header_epw=args.header_epw,
        sources_csv=args.sources_csv, pattern=args.pattern, recursive=args.recursive,
        smooth_span=args.smooth_span, smooth_method=args.smooth_method, years=years,
        min_years=args.min_years, max_gap=args.max_gap,
        min_wind_completeness=args.min_wind_completeness, set_year=args.set_year,
        update_header_years=not args.no_update_header_years,
        add_header_generation_note=not args.no_header_generation_note,
        fill_design_conditions=not args.no_design_conditions, verbose=True)


if __name__ == '__main__':
    main()
