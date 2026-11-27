import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from rmy import event_probabilities as ep

YEARS = list(range(2010, 2022))


def _rmy_dir(tmp_path, n_cold=40, n_heat=30, log=True, seed=0):
    """A city folder with AMY file names, event CSVs, peak CSVs and the pipeline's threshold log."""
    rng = np.random.default_rng(seed)
    city = tmp_path / 'Test_City_ST'
    (city / 'AMY').mkdir(parents=True)
    for y in YEARS:
        (city / 'AMY' / f'Test_City_ST_123456_{y}_AMY.epw').write_text('')
    rmy = city / 'RMY'
    for sub, n, kind in (('coldspells', n_cold, 'cold'), ('hotspells', n_heat, 'heat')):
        (rmy / sub).mkdir(parents=True)
        rows = []
        for i in range(n):
            y = YEARS[i % len(YEARS)]
            start = pd.Timestamp(y, 1 if kind == 'cold' else 7, 1) + pd.Timedelta(days=int(rng.integers(0, 50)))
            d = int(rng.integers(3, 12))
            row = dict(begin_date=f'{start:%d/%m/%Y}', end_date=f'{start + pd.Timedelta(days=d - 1):%d/%m/%Y}',
                       duration=d, year=y, source_years=y)
            if kind == 'cold':
                tmin = -12 - rng.gamma(2, 2)
                row.update(avg_tmin=round(tmin + 2, 1), std_tmin=1.0, min_tmin=round(tmin, 1), avg_wind_speed=4.0)
            else:
                tmax = 31 + rng.gamma(2, 1.5)
                row.update(avg_tmax=round(tmax - 1.5, 1), std_tmax=1.0, max_tmax=round(tmax, 1), avg_heat_index=36.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        name = 'coldspells_events' if kind == 'cold' else 'heatwave_events'
        df.to_csv(rmy / sub / f'{name}.csv', index=False)
        df.iloc[:3].to_csv(rmy / sub / f'{name}_peak.csv', index=False)
    if log:
        (rmy / 'log.txt').write_text('Heat thresholds (all AMY years): percentile Tmin >= 18.0 and Tmax >= 31.0; '
                                     'temperature Tmax >= 31.0; months [6, 7, 8]\n'
                                     'Cold thresholds (all AMY years): percentile Tmin < -12.0; temperature Tmin < -12.0\n')
    return rmy


def test_pwm_recovers_gpd_parameters():
    x = genpareto.rvs(-0.2, scale=2.0, size=4000, random_state=np.random.default_rng(1))
    shape, scale = ep._gpd_pwm(x)
    assert shape == pytest.approx(-0.2, abs=0.05)
    assert scale == pytest.approx(2.0, rel=0.06)


def test_pwm_shape_is_kept_within_limits():
    # whole-degree values with many ties at the top give an implausibly short tail
    shape, scale = ep._gpd_pwm(np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], float))
    assert abs(shape) <= ep.SHAPE_LIMIT and scale > 0


def test_data_counts_years_and_observed_return_periods(tmp_path):
    d = ep.event_probability_data(_rmy_dir(tmp_path), n_boot=50)
    assert d['kinds'] == ['cold', 'heat'] and d['thresholds'] == {'cold': -12.0, 'heat': 31.0}
    rec = d['periods']['all']['cold']
    assert rec['Y'] == len(YEARS) and rec['n'] == 40 and sum(rec['per_year']) == 40
    m = rec['measures']['peak']
    coldest = min(rec['events'], key=lambda e: e['vals']['peak'])
    assert m['T_emp'][coldest['id']] == pytest.approx(len(YEARS))        # the most extreme: once in Y years
    assert m['sign'] == -1 and len(m['T']) == len(m['level']) == len(m['lo']) == len(m['hi']) > 0
    # colder levels at longer return periods, and the band (lo <= hi in degrees) brackets the curve
    assert np.all(np.diff(m['level']) <= 1e-9)
    lo, level, hi = (np.array(m[k]) for k in ('lo', 'level', 'hi'))
    assert np.all(lo <= level + 1e-6) and np.all(level <= hi + 1e-6)
    assert sum(e['rmy'] for e in rec['events']) == 3
    assert set(d['periods']) == {'all', 'early', 'late'}
    assert d['periods']['early']['cold']['Y'] + d['periods']['late']['cold']['Y'] == len(YEARS)


def test_severity_is_degree_days_beyond_the_threshold(tmp_path):
    d = ep.event_probability_data(_rmy_dir(tmp_path), n_boot=10)
    for e in d['periods']['all']['heat']['events']:
        assert e['vals']['severity'] == pytest.approx(max(0, e['duration'] * (e['vals']['mean'] - 31.0)), abs=0.06)


def test_kind_with_too_few_events_is_left_out(tmp_path):
    d = ep.event_probability_data(_rmy_dir(tmp_path, n_heat=2), n_boot=10)
    assert d['kinds'] == ['cold'] and 'heat' not in d['periods']['all']


def test_few_events_give_observed_return_periods_only(tmp_path):
    d = ep.event_probability_data(_rmy_dir(tmp_path, n_cold=5), n_boot=10)
    m = d['periods']['all']['cold']['measures']['peak']
    assert m['T'] == [] and m['T_fit'] == m['T_emp']


def test_explorer_writes_a_page_with_the_data(tmp_path):
    rmy = _rmy_dir(tmp_path)
    out = ep.event_probability_explorer(rmy, show=False, n_boot=10)
    assert out == rmy / 'event_probabilities.html'
    page = out.read_text(encoding='utf-8')
    start = page.index('const DATA = ') + len('const DATA = ')
    data = json.loads(page[start:page.index(';\nconst S =', start)].replace('<\\/', '</'))
    assert data['city'] == 'Test City ST' and '/*DATA*/' not in page and '<!--PLOTLY-->' not in page
