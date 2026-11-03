"""Checks of tmy/generate_tmy_from_amys.py against ISO 15927-4:2005."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tmy_gen', REPO / 'tmy' / 'generate_tmy_from_amys.py')
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

AMY_DIR = REPO / 'data' / 'cities' / 'Boston' / 'AMY'
HEADER_EPW = next((REPO / 'data' / 'cities' / 'Boston' / 'TMY').glob('*.epw'))


def test_fs_statistic_follows_iso_equations_1_to_3():
    # Pooled record of N = 6 daily means; candidate month of n = 2 days.
    pooled = np.sort(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    candidate = np.array([5.0, 2.0])
    # F = J/(n+1) = 1/3, 2/3 for the sorted values 2, 5; Phi = K/(N+1) = 2/7, 5/7.
    expected = abs(1 / 3 - 2 / 7) + abs(2 / 3 - 5 / 7)
    assert tm.fs_statistic(candidate, pooled) == pytest.approx(expected)


def test_gap_fill_interpolates_short_gaps_only_and_zeros_solar_at_night():
    v = np.array([1.0, np.nan, np.nan, 4.0, 5.0] + [np.nan] * 8 + [9.0])
    filled, mask = tm.fill_gaps(v, max_gap=6)
    np.testing.assert_allclose(filled[:4], [1, 2, 3, 4])
    assert np.isnan(filled[5:13]).all() and mask.sum() == 2
    night = np.zeros(len(v), dtype=bool)
    night[5:13] = True
    filled, _ = tm.fill_gaps(v, max_gap=6, night=night)
    assert (filled[5:13] == 0).all()


def test_month_joins_remove_the_step_and_leave_other_fields_alone():
    n = 48
    hours = np.arange(n)
    t = 5 * np.sin(2 * np.pi * hours / 24)
    t[24:] += 6.0                                         # 6 C step at the join (hour 24)
    values = {tm.TEMP_COL: t.copy(), tm.GHI_COL: np.full(n, 100.0)}
    touched = tm.smooth_month_joins(values, [0, 24], [tm.TEMP_COL], span=8)
    out = values[tm.TEMP_COL]
    natural = ((t[23] - t[22]) + (t[25] - t[24])) / 2
    assert out[24] - out[23] == pytest.approx(natural)    # step reduced to the natural change
    assert touched.sum() == 32                            # 8 + 8 hours at both joins (wrap included)
    np.testing.assert_array_equal(out[~touched], t[~touched])
    assert (values[tm.GHI_COL] == 100.0).all()            # radiation not adjusted
    # Each month keeps its own hourly profile: hour-to-hour changes differ only by the ramp.
    ramp_step = abs((t[24] - t[23]) - natural) / 2 / 8
    assert np.abs(np.diff(out[16:24]) - np.diff(t[16:24])).max() == pytest.approx(ramp_step)


@pytest.fixture(scope='module')
def boston(tmp_path_factory):
    out = tmp_path_factory.mktemp('tmy') / 'Boston_TMY_ISO.epw'
    sources = tm.generate_tmy_from_amys(AMY_DIR, out, HEADER_EPW, years=(2009, 2023), verbose=False)
    return out, sources


def test_generated_tmy_keeps_source_data_outside_the_joins(boston):
    out, sources = boston
    g = pd.read_csv(out, skiprows=8, header=None, dtype=str, keep_default_na=False)
    assert g.shape == (8760, 35)
    month = g[1].astype(int).to_numpy()
    starts = np.flatnonzero(np.r_[True, np.diff(month) != 0])
    near = np.zeros(8760, dtype=bool)
    for s in starts:
        near[(s + np.arange(-8, 8)) % 8760] = True
    for m, fname in sources.items():
        a = pd.read_csv(AMY_DIR / fname, skiprows=8, header=None, dtype=str, keep_default_na=False)
        a = a[(a[1].astype(int) == m) & ~((m == 2) & (a[2].astype(int) == 29))].reset_index(drop=True)
        gm = g[month == m].reset_index(drop=True)
        # Radiation, illuminance and all other unsmoothed fields are the source month's text.
        other = [c for c in range(35) if c not in tm.DEFAULT_SMOOTH_COLS]
        assert (gm[other].values == a[other].values).all(), m
        # Smoothed fields only change near the joins or where the source was missing.
        changed = (gm[tm.DEFAULT_SMOOTH_COLS].values != a[tm.DEFAULT_SMOOTH_COLS].values).any(axis=1)
        far = changed & ~near[month == m]
        src = a.loc[far, tm.DEFAULT_SMOOTH_COLS].astype(float).values
        assert np.isin(src, [99.9, 999, 999999]).any(axis=1).all() if far.any() else True


def test_generated_tmy_header_reports_the_month_years(boston):
    out, sources = boston
    new = open(out, newline='').read().splitlines(True)[:8]
    old = open(HEADER_EPW, newline='').read().splitlines(True)[:8]
    assert [i for i in range(8) if new[i] != old[i]] == [5, 6]          # COMMENTS 1 and 2
    assert all(line.endswith('\r\n') for line in new) == all(line.endswith('\r\n') for line in old)
    years = dict(re.findall(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)=(\d{4})', new[5]))
    assert {tm.MONTH_NAMES.index(k) + 1: int(v) for k, v in years.items()} == \
        {m: tm.year_from_filename(Path(f)) for m, f in sources.items()}
    assert 'Period of Record=2009-2023' in new[5] and '#years=[15]' in new[5]


def test_fewer_than_ten_years_is_refused(tmp_path):
    with pytest.raises(ValueError, match='at least 10 years'):
        tm.generate_tmy_from_amys(AMY_DIR, tmp_path / 'x.epw', HEADER_EPW,
                                  years=(2009, 2015), verbose=False)
