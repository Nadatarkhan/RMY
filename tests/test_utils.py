import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rmy.rmy_generation import smooth_inserted_boundaries
from rmy.utils import calculate_heat_index, find_epw_by_year, safe_load_events

REPO = Path(__file__).resolve().parents[1]


def test_import_has_no_side_effects(tmp_path):
    # Run from an empty folder so nothing like /content or data/ is found.
    result = subprocess.run(
        [sys.executable, '-c', 'import rmy, rmy.visualization'],
        cwd=tmp_path, capture_output=True, text=True,
        env={'PYTHONPATH': str(REPO), 'MPLBACKEND': 'Agg'},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''


def test_find_epw_by_year_matches_whole_number(tmp_path):
    (tmp_path / 'Boston_2010.epw').write_text('x')
    (tmp_path / 'Boston_20101.epw').write_text('x')
    assert find_epw_by_year(2010, tmp_path).name == 'Boston_2010.epw'


def test_find_epw_by_year_rejects_missing_and_ambiguous(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_epw_by_year(2010, tmp_path)
    (tmp_path / 'a_2010.epw').write_text('x')
    (tmp_path / 'b_2010.epw').write_text('x')
    with pytest.raises(FileNotFoundError):
        find_epw_by_year(2010, tmp_path)


def test_safe_load_events_missing_file_gives_empty_table(tmp_path):
    df = safe_load_events(tmp_path / 'nope.csv', ['begin_date', 'end_date'])
    assert df.empty and list(df.columns) == ['begin_date', 'end_date']


def test_safe_load_events_parses_day_first_dates(tmp_path):
    p = tmp_path / 'events.csv'
    p.write_text('begin_date,end_date,duration\n05/07/2010,08/07/2010,4\n')
    df = safe_load_events(p, [])
    assert df.loc[0, 'begin_date'] == pd.Timestamp(2010, 7, 5)


def test_heat_index_simple_branch():
    # 77 F, 50 % RH: 0.5 * (77 + 61 + 10.8 + 4.7) = 76.75 F = 24.86 C (< 80 F).
    assert calculate_heat_index(25.0, 50.0).iloc[0] == pytest.approx(24.86, abs=0.01)


def test_heat_index_rothfusz_branch_matches_nws_table():
    # NWS heat index table: 86 F at 60 % RH reads 91 F (32.8 C).
    assert calculate_heat_index(30.0, 60.0).iloc[0] == pytest.approx(32.8, abs=0.2)


def test_smoothing_blends_shoulders_and_keeps_inserted_hours():
    n = 48
    df = pd.DataFrame({c: np.zeros(n) for c in
                       ['temp_air', 'temp_dew', 'relative_humidity',
                        'atmospheric_pressure', 'wind_speed']})
    protected = np.zeros(n, dtype=bool)
    protected[20:28] = True
    df.loc[20:27, 'temp_air'] = 16.0
    out, n_smoothed = smooth_inserted_boundaries(df.copy(), protected, hours=4)

    assert n_smoothed == 8
    assert (out.loc[20:27, 'temp_air'] == 16.0).all()      # inserted hours untouched
    # Left shoulder is a straight line from hour 15 (0) to hour 20 (16).
    np.testing.assert_allclose(out.loc[16:19, 'temp_air'], [3.2, 6.4, 9.6, 12.8])
    np.testing.assert_allclose(out.loc[28:31, 'temp_air'], [12.8, 9.6, 6.4, 3.2])
    assert (out.loc[:15, 'temp_air'] == 0).all() and (out.loc[32:, 'temp_air'] == 0).all()


def test_safe_load_events_file_with_no_columns_gives_empty_table(tmp_path):
    # pandas writes an empty DataFrame as a file with no header at all.
    p = tmp_path / 'events.csv'
    pd.DataFrame().to_csv(p, index=False)
    df = safe_load_events(p, ['begin_date', 'end_date'])
    assert df.empty and list(df.columns) == ['begin_date', 'end_date']



def test_exceedance_severity_ranks_one_extreme_event_above_many_mild_ones():
    from rmy.heatwaves import exceedance_severity
    events = pd.DataFrame({
        'year': [2015] * 4 + [2021],
        'avg_tmax': [31.0, 31.0, 31.0, 31.0, 39.0],
        'duration': [5, 5, 5, 5, 4],
    })
    sev = exceedance_severity(events, reference=30.0)
    assert sev[2015] == pytest.approx(20.0)          # 4 events x 5 days x 1 C
    assert sev[2021] == pytest.approx(36.0)          # 4 days x 9 C
    cold = pd.DataFrame({'year': [2009, 2013], 'avg_tmin': [-10.0, -4.0], 'duration': [3, 10]})
    sev = exceedance_severity(cold, reference=-3.0, value_col='avg_tmin', sign=-1)
    assert sev[2009] == pytest.approx(21.0) and sev[2013] == pytest.approx(10.0)
