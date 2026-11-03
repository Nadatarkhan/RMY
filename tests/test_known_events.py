"""Detection must find well-documented extreme events in the test cities.

Detection only (no RMY assembly). Each city takes a few seconds.
"""
from pathlib import Path

import pandas as pd
import pytest

from rmy.coldspells import run_full_pipeline_cold
from rmy.heatwaves import run_full_pipeline

CITIES = Path(__file__).resolve().parents[1] / 'data' / 'cities'

KNOWN = [
    # city, kind, a day inside the event, description
    ('Seattle', 'heat', '2021-06-27', 'Pacific Northwest heat dome, record 108 F on 28 Jun 2021'),
    ('Phoenix', 'heat', '2023-07-15', '31 consecutive days of 110 F or more, 30 Jun - 30 Jul 2023'),
    ('Dulles_Washington', 'heat', '2012-06-29', 'June 2012 derecho heat wave'),
    ('Chicago', 'cold', '2019-01-30', 'polar vortex, -23 F on 30 Jan 2019'),
    ('Chicago', 'cold', '2014-01-06', 'polar vortex, Jan 2014'),
]


@pytest.fixture(scope='module')
def detected(tmp_path_factory):
    cache = {}

    def get(city, kind):
        if (city, kind) not in cache:
            out = tmp_path_factory.mktemp(f'{city}_{kind}')
            amy, tmy = CITIES / city / 'AMY', CITIES / city / 'TMY'
            if kind == 'heat':
                run_full_pipeline(str(amy), str(tmy), str(out))
                ev = pd.read_csv(out / 'heatwave_events.csv')
            else:
                run_full_pipeline_cold(str(amy), str(tmy), str(out))
                ev = pd.read_csv(out / 'coldspells_events.csv')
            ev['b'] = pd.to_datetime(ev.begin_date, dayfirst=True)
            ev['e'] = pd.to_datetime(ev.end_date, dayfirst=True)
            cache[(city, kind)] = ev
        return cache[(city, kind)]
    return get


@pytest.mark.parametrize('city, kind, day, what', KNOWN, ids=[k[0] + '-' + k[2] for k in KNOWN])
def test_known_event_is_detected(detected, city, kind, day, what):
    ev = detected(city, kind)
    t = pd.Timestamp(day)
    assert ((ev.b <= t) & (ev.e >= t)).any(), f'{city}: missed {what}'


def test_phoenix_events_are_not_whole_calendar_months(detected):
    # With a fixed 30 C threshold and a one-month warm season, every Phoenix
    # "event" used to start on the 1st and end on the last day of a month.
    ev = detected('Phoenix', 'heat')
    whole_months = ((ev.b.dt.day == 1) & (ev.e == ev.e + pd.offsets.MonthEnd(0))).sum()
    assert whole_months <= 2 and ev.duration.median() <= 10
