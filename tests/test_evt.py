from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rmy.coldspells import calculate_daily_tmin_tmax_cold, identify_evt_cold_extremes, pooled_daily_cold
from rmy.evt import select_pot_threshold
from rmy.heatwaves import identify_evt_extremes, pooled_daily
from rmy.utils import list_epws

REPO = Path(__file__).resolve().parents[1]
AMYS = list_epws(REPO / 'data' / 'cities' / 'Boston' / 'AMY')


def _series(n_years=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2001-01-01', periods=365 * n_years)
    return dates, rng.normal(0, 1, len(dates))


def test_grid_search_picks_the_most_extreme_quantile_with_an_event():
    dates, v = _series()
    # One 3-day run at the very top; nothing else persists that high.
    v[100:103] = 10.0
    evt = select_pot_threshold(v, dates, tail='upper')
    assert evt.quantile == 0.99 and evt.threshold < 10
    assert evt.n_excesses >= 30 and evt.scale > 0


def test_grid_search_lowers_the_quantile_until_a_run_exists():
    dates, v = _series(seed=1)
    v = np.where(np.arange(len(v)) % 2 == 0, v, -5.0)    # no two high days in a row at the top
    v[200:203] = np.quantile(v, 0.93)                     # a 3-day run around the 93rd percentile
    evt = select_pot_threshold(v, dates, tail='upper')
    assert evt is not None and 0.90 <= evt.quantile < 0.95


def test_lower_tail_mirrors_upper_tail():
    dates, v = _series(seed=2)
    v[50:54] = -10.0
    evt = select_pot_threshold(v, dates, tail='lower')
    assert evt.quantile == 0.01
    assert evt.flags(v)[50:54].all()


def test_returns_none_when_no_threshold_is_valid():
    dates = pd.date_range('2001-01-01', periods=20)
    assert select_pot_threshold(np.arange(20.0), dates) is None


@pytest.fixture(scope='module')
def boston_thresholds():
    heat = pooled_daily(AMYS)
    cold = pooled_daily_cold(AMYS)
    return (heat, select_pot_threshold(heat['heat_index'], heat['date'], 'upper'),
            select_pot_threshold(cold['Tmin (oC)'], cold['date'], 'lower'))


def test_heat_evt_finds_the_paper_s_late_june_2021_heat_wave(boston_thresholds):
    heat, evt, _ = boston_thresholds
    d = heat[heat['YEAR'] == 2021].reset_index(drop=True)
    events = identify_evt_extremes(d, evt=evt)
    starts = [d['date'][s].strftime('%Y-%m-%d') for s in events['Start']]
    assert '2021-06-28' in starts


@pytest.mark.parametrize('year, day', [(2015, '2015-02-15'), (2016, '2016-02-14')])
def test_cold_evt_fires_on_record_cold(boston_thresholds, year, day):
    _, _, evt = boston_thresholds
    path = next(p for p in AMYS if f'_{year}_' in p.name)
    from rmy.utils import read_epw_raw
    d = calculate_daily_tmin_tmax_cold(read_epw_raw(path))
    events = identify_evt_cold_extremes(d, evt=evt)
    dates = pd.to_datetime(dict(year=d.YEAR, month=d.MONTH, day=d.DAY))
    t = pd.Timestamp(day)
    assert any(dates[s] <= t <= dates[e] for s, e in zip(events['Start'], events['End']))
