from pathlib import Path

import pandas as pd
import pytest

from rmy import run_full_rmy_pipeline
from rmy.coldspells import identify_cold_spells
from rmy.seasons import read_epw_latitude, season_plan_for_epw, season_plan_for_latitude
from rmy.utils import read_epw_file

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'
BOSTON = {'EPWs': DATA / 'cities' / 'Boston' / 'AMY', 'base': DATA / 'cities' / 'Boston' / 'TMY'}


@pytest.mark.parametrize('lat, band', [
    ('42.361', 'northern'),       # Boston
    ('-33.97', 'southern'),       # Cape Town
    ('-37.67', 'southern'),       # Melbourne
    ('3.12', 'equatorial'),       # Kuala Lumpur
    ('-3.04', 'equatorial'),      # Manaus (was "Southern" in the notebook list)
    ('-6.87', 'equatorial'),      # Dar es Salaam (was "Southern" in the notebook list)
])
def test_band_from_epw_latitude(tmp_path, lat, band):
    _copy_with_latitude(tmp_path, lat)
    assert season_plan_for_epw(next((tmp_path / 'base').glob('*.epw'))).band == band


def test_band_of_the_published_files():
    # 40 cities from 35 S to 65 N: each file's season band follows its own latitude.
    bands = set()
    for f in sorted((DATA / 'RMY').glob('*.epw')):
        lat = float(f.read_text().split('\n', 1)[0].split(',')[6])
        band = season_plan_for_epw(f).band
        assert band == season_plan_for_latitude(lat).band, f.name
        bands.add(band)
    assert bands == {'northern', 'southern', 'equatorial'}


def test_season_months_by_band():
    north, south, eq = (season_plan_for_latitude(x) for x in (42.4, -34.0, 1.3))
    assert (north.summer_months, north.winter_months) == ((6, 7, 8), (12, 1, 2))
    assert (south.summer_months, south.winter_months) == ((12, 1, 2), (6, 7, 8))
    assert eq.cold_months == tuple(range(1, 13)) and eq.summer_months == ()
    # The band edge belongs to the hemisphere, not the equatorial band.
    assert season_plan_for_latitude(10.0).band == 'northern'
    assert season_plan_for_latitude(-10.0).band == 'southern'


def test_bad_location_line_raises(tmp_path):
    p = tmp_path / 'bad.epw'
    p.write_text('NOT A HEADER\n')
    with pytest.raises(ValueError):
        read_epw_latitude(p)


def test_cold_filter_uses_given_winter_months():
    # GNN flags a 5-day cold anomaly in July: winter in the south, summer in the north.
    dates = pd.date_range('2010-01-01', '2010-12-31')
    dip = (dates.month == 7) & (dates.day >= 10) & (dates.day <= 14)
    df = pd.DataFrame({'YEAR': dates.year, 'MONTH': dates.month, 'DAY': dates.day,
                       'Tmin (oC)': 10.0, 'Wind Speed (m/s)': 5.0})
    north = identify_cold_spells(df.copy(), method='gnn', winter_months=(12, 1, 2), gnn_flags=dip)
    south = identify_cold_spells(df.copy(), method='gnn', winter_months=(6, 7, 8), gnn_flags=dip)
    assert north.empty
    assert not south.empty and dates[south['Start'].iloc[0]].month == 7


def _copy_with_latitude(dst, lat):
    for sub in ('EPWs', 'base'):
        (dst / sub).mkdir(parents=True)
        for f in sorted(BOSTON[sub].glob('*.epw')):
            lines = f.read_text().splitlines(True)
            fields = lines[0].split(',')
            fields[6] = lat
            lines[0] = ','.join(fields)
            (dst / sub / f.name).write_text(''.join(lines))


def test_equatorial_run_brings_monthly_means_back(tmp_path):
    # Boston data relabelled to Singapore's latitude, to exercise the
    # month-by-month compensation path end to end.
    _copy_with_latitude(tmp_path, '1.350')
    rmy_path = run_full_rmy_pipeline(tmp_path / 'EPWs', tmp_path / 'base', tmp_path / 'out')
    base = read_epw_file(next((tmp_path / 'base').glob('*.epw')))
    rmy = read_epw_file(rmy_path)
    dev = (rmy.groupby('month')['temp_air'].mean() - base.groupby('month')['temp_air'].mean()).abs()
    # February is almost entirely cold-spell days in 2015 (21 inserted days
    # plus 23-28 Feb), leaving almost no days to compensate with, so what it
    # cannot take back goes to January and March, each moved by at most
    # SPILL_CAP. The other months are compensated to within 0.25 C; the
    # boundary smoothing applied afterwards can move a month by a few tenths more.
    from rmy.rmy_generation import SPILL_CAP
    assert dev.drop([1, 2, 3]).max() < 0.5, dev.round(2).to_dict()
    assert dev[[1, 3]].max() < SPILL_CAP + 0.5, dev.round(2).to_dict()
    # Without the adjacent-month step, the annual mean is further from the TMY.
    no_spill = read_epw_file(run_full_rmy_pipeline(tmp_path / 'EPWs', tmp_path / 'base',
                                                   tmp_path / 'out_no_spill', spill_cap=0))
    gap = abs(rmy['temp_air'].mean() - base['temp_air'].mean())
    gap_no_spill = abs(no_spill['temp_air'].mean() - base['temp_air'].mean())
    assert gap < gap_no_spill - 0.03, (gap, gap_no_spill)


def test_long_runs_are_kept_whole():
    # A 25-day cold run (the notebook dropped anything over 21 days).
    dates = pd.date_range('2015-01-01', '2015-03-31')
    tmin = pd.Series(0.0, index=range(len(dates)))
    tmin[20:45] = -15.0
    df = pd.DataFrame({'YEAR': dates.year, 'MONTH': dates.month, 'DAY': dates.day,
                       'Tmin (oC)': tmin, 'Wind Speed (m/s)': 3.0})
    events = identify_cold_spells(df, method='temperature')
    assert len(events) == 1 and events.End[0] - events.Start[0] + 1 == 25
