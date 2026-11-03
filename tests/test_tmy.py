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


def test_fs_statistic_gives_tied_values_their_average_rank():
    pooled = np.sort(np.array([1.0, 2.0, 2.0, 3.0]))           # the two 2.0s span ranks 2-3
    candidate = np.array([2.0, 2.0])                           # J = 1.5 for both
    expected = 2 * abs(1.5 / 3 - 2.5 / 5)
    assert tm.fs_statistic(candidate, pooled) == pytest.approx(expected)


def _candidate(year, temp, wind):
    days = np.repeat(np.arange(1, 4), 24)
    n = len(days)
    return {'year': year, 'file': f'{year}.epw', 'day': days,
            'values': {tm.TEMP_COL: np.full(n, temp), tm.RH_COL: np.full(n, 50.0),
                       tm.GHI_COL: np.full(n, 100.0), tm.WS_COL: np.asarray(wind, dtype=float)}}


def test_wind_criterion_ignores_months_with_incomplete_wind():
    n = 72
    gappy = np.full(n, 3.0)
    gappy[:10] = np.nan                                         # 86 % complete
    cands = [_candidate(2001, 10.0, gappy),                     # mean 3.0 = closest, but gappy
             _candidate(2002, 10.0, np.full(n, 3.4)),
             _candidate(2003, 10.0, np.full(n, 2.5))]
    best = tm.select_months({1: cands})[1]
    assert best['year'] == 2002                                 # closest of the complete months
    assert best['ws_ref_mean'] == pytest.approx((3.4 + 2.5) / 2)


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


def test_relative_humidity_stays_consistent_with_adjusted_temperatures():
    n = 48
    t = np.r_[np.full(24, 10.0), np.full(24, 20.0)]             # 10 C step at the join
    td = t - 5.0
    es = lambda x: np.exp(17.625 * x / (x + 243.04))
    rh = 100 * es(td) / es(t)
    values = {tm.TEMP_COL: t.copy(), tm.DEWPOINT_COL: td.copy(), tm.RH_COL: rh.copy()}
    touched = tm.smooth_month_joins(values, [0, 24], tm.DEFAULT_SMOOTH_COLS, span=8)
    implied = 100 * es(values[tm.DEWPOINT_COL]) / es(values[tm.TEMP_COL])
    np.testing.assert_allclose(values[tm.RH_COL][touched], implied[touched], atol=1e-9)
    assert (values[tm.TEMP_COL][touched] != t[touched]).any()


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


def _amy_copy(tmp_path, years=range(2009, 2024)):
    folder = tmp_path / 'amy'
    folder.mkdir()
    for p in AMY_DIR.glob('*.epw'):
        if tm.year_from_filename(p) in years:
            (folder / p.name).write_bytes(p.read_bytes())
    return folder


def test_duplicate_years_are_refused(tmp_path):
    folder = _amy_copy(tmp_path)
    src = next(folder.glob('*2015*.epw'))
    (folder / 'Boston_copy_2015_AMY.epw').write_bytes(src.read_bytes())
    with pytest.raises(ValueError, match='Two AMY files for 2015'):
        tm.generate_tmy_from_amys(folder, tmp_path / 'x.epw', HEADER_EPW, verbose=False)


def test_month_with_hours_out_of_order_is_not_a_candidate(tmp_path, capsys):
    folder = _amy_copy(tmp_path)
    path = next(folder.glob('*2015*.epw'))
    lines = path.read_bytes().split(b'\n')
    lines[8 + 100], lines[8 + 101] = lines[8 + 101], lines[8 + 100]     # swap two January hours
    path.write_bytes(b'\n'.join(lines))
    sources = tm.generate_tmy_from_amys(folder, tmp_path / 'x.epw', HEADER_EPW, verbose=False)
    assert 'Jan: not a candidate, days/hours are not complete and in order' in capsys.readouterr().out
    assert '2015' not in sources[1]


def test_fewer_than_ten_years_is_refused(tmp_path):
    with pytest.raises(ValueError, match='at least 10 years'):
        tm.generate_tmy_from_amys(AMY_DIR, tmp_path / 'x.epw', HEADER_EPW,
                                  years=(2009, 2015), verbose=False)
