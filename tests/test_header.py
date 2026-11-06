import pytest

import numpy as np
import pandas as pd

from rmy.utils import label_rmy_header, set_typical_extreme_periods, typical_extreme_periods_line

HEADER = [
    'LOCATION,Boston,MA,USA,SRC-TMYx,725090,42.36,-71.01,-5.0,3.7\n',
    'DESIGN CONDITIONS,0\n',
    'TYPICAL/EXTREME PERIODS,0\n',
    'GROUND TEMPERATURES,0\n',
    'HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n',
    'COMMENTS 1,"Period of Record=2009-2023"\n',
    'COMMENTS 2,"Downloaded from Climate.Onebuilding.org"\n',
    'DATA PERIODS,1,1,Data,Sunday,1/ 1,12/31\n',
]


def test_only_comments_2_changes_and_field_counts_hold():
    new = label_rmy_header(HEADER, range(2009, 2024), 2022, 2015)
    assert new[:6] == HEADER[:6] and new[7] == HEADER[7]
    assert new[6] == ('COMMENTS 2,"RMY extreme weather file: heat waves and cold spells from '
                      'AMYs 2009-2023 inserted (peak heat year 2022; peak cold year 2015); '
                      'Downloaded from Climate.Onebuilding.org"\n')
    assert [l.count(',') for l in new] == [l.count(',') for l in HEADER]


def test_gaps_in_the_amy_years_are_listed():
    new = label_rmy_header(HEADER, [2010, 2011, 2013, 2014, 2016])
    assert 'AMYs 2010-2016 inserted (no AMY for 2012 2015)' in new[6]


@pytest.mark.parametrize('line, expected', [
    ('COMMENTS 2,\r\n', 'COMMENTS 2,RMY extreme weather file: heat waves and cold spells from AMYs 2020-2021 inserted\r\n'),
    ('COMMENTS 2,""\n', 'COMMENTS 2,"RMY extreme weather file: heat waves and cold spells from AMYs 2020-2021 inserted"\n'),
    ('COMMENTS 2,plain text\n', 'COMMENTS 2,RMY extreme weather file: heat waves and cold spells from AMYs 2020-2021 inserted; plain text\n'),
])
def test_empty_unquoted_and_crlf_comment_lines(line, expected):
    header = HEADER[:6] + [line] + HEADER[7:]
    assert label_rmy_header(header, [2020, 2021])[6] == expected


def test_missing_comments_2_raises():
    with pytest.raises(ValueError):
        label_rmy_header(HEADER[:6] + HEADER[7:], [2020])


def _year(phase_days=0, hot_week=None, cold_week=None):
    """Hourly year: a seasonal sine (warmest around day 200 + phase), optional anomaly weeks."""
    idx = pd.date_range('2001-01-01', periods=8760, freq='h')
    doy = idx.dayofyear.to_numpy()
    temp = 10 + 12 * np.sin(2 * np.pi * (doy - 110 - phase_days) / 365) + 4 * np.sin(2 * np.pi * idx.hour.to_numpy() / 24)
    for week, delta in ((hot_week, 8), (cold_week, -10)):
        if week is not None:
            temp[(doy >= week) & (doy < week + 7)] += delta
    return pd.DataFrame({'month': idx.month, 'day': idx.day, 'temp_air': temp})


def _periods(line):
    f = line.split(',')
    assert f[0] == 'TYPICAL/EXTREME PERIODS' and f[1] == '6' and len(f) == 26
    return {(f[2 + 4 * k].split(' - ')[0], f[3 + 4 * k]): (f[4 + 4 * k], f[5 + 4 * k]) for k in range(6)}


def test_design_weeks_find_the_inserted_extremes():
    # Hot week 22-28 July (day 203), cold week 5-11 February (day 36).
    p = _periods(typical_extreme_periods_line(_year(hot_week=203, cold_week=36)))
    assert p[('Summer', 'Extreme')] == ('7/22', '7/28')
    assert p[('Winter', 'Extreme')] == ('2/ 5', '2/11')
    assert set(k[0] for k in p) == {'Summer', 'Winter', 'Autumn', 'Spring'}


def test_design_week_seasons_follow_the_climate():
    # Southern-hemisphere-like year: warmest in January, so Summer is Dec-Feb.
    p = _periods(typical_extreme_periods_line(_year(phase_days=182)))
    start_month = int(p[('Summer', 'Extreme')][0].split('/')[0])
    assert start_month in (12, 1, 2)


def test_only_the_periods_line_changes():
    new = set_typical_extreme_periods(HEADER, _year(hot_week=203))
    assert [i for i in range(8) if new[i] != HEADER[i]] == [2]
    assert new[2].endswith('\n') and new[2].startswith('TYPICAL/EXTREME PERIODS,6,')
