"""End-to-end checks on the Boston sample data in data/ (about 10 seconds)."""
from pathlib import Path

import pandas as pd
import pytest

from rmy import run_full_rmy_pipeline
import numpy as np

from rmy.rmy_generation import compensate_monthly
from rmy.utils import extract_original_header, list_epws, read_epw_file
from rmy.visualization import visualize_extreme_events

REPO = Path(__file__).resolve().parents[1]
EPW_DIR = REPO / 'data' / 'cities' / 'Boston' / 'AMY'
BASE_DIR = REPO / 'data' / 'cities' / 'Boston' / 'TMY'


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp('rmy')
    rmy_path = run_full_rmy_pipeline(EPW_DIR, BASE_DIR, out)
    return out, rmy_path


def test_rmy_header_changes_only_comments_2(run):
    _, rmy_path = run
    base = list_epws(BASE_DIR)[0]
    old, new = extract_original_header(base), extract_original_header(rmy_path)
    assert len(new) == 8
    changed = [i for i in range(8) if old[i] != new[i]]
    assert changed == [6]                                   # COMMENTS 2 only
    assert [l.count(',') for l in new] == [l.count(',') for l in old]
    assert new[6].startswith('COMMENTS 2,"RMY extreme weather file: heat waves and cold spells '
                             'from AMYs 2009-2024 inserted (peak heat year 2022; peak cold year 2015); ')
    assert new[6].endswith(old[6][len('COMMENTS 2,"'):])     # original text kept
    # Same line endings as the base TMY throughout (this TMYx uses CRLF).
    raw, base_raw = Path(rmy_path).read_bytes(), Path(base).read_bytes()
    assert raw.isascii()
    crlf = base_raw.count(b'\r\n') == base_raw.count(b'\n')
    assert (raw.count(b'\r\n') == raw.count(b'\n')) if crlf else (b'\r' not in raw)
    assert len(read_epw_file(rmy_path)) == len(read_epw_file(base)) == 8760


def test_rmy_keeps_base_timestamps(run):
    _, rmy_path = run
    base = read_epw_file(list_epws(BASE_DIR)[0])
    rmy = read_epw_file(rmy_path)
    cols = ['year', 'month', 'day', 'hour', 'minute']
    pd.testing.assert_frame_equal(rmy[cols], base[cols])


def test_peak_events_are_inserted_hour_for_hour(run):
    """Every peak event day is in the RMY exactly; long events as their 21 most severe days."""
    out, rmy_path = run
    rmy = read_epw_file(rmy_path)
    for folder, prefix in (('hotspells', 'heatwave'), ('coldspells', 'coldspells')):
        stats = pd.read_csv(out / folder / f'{prefix}_stats_peak.csv')
        year = int(stats.loc[0, 'year'])
        source = read_epw_file(next(p for p in list_epws(EPW_DIR) if str(year) in p.name))
        events = pd.read_csv(out / folder / f'{prefix}_events_peak.csv')
        assert not events.empty
        for _, ev in events.iterrows():
            days = pd.date_range(pd.to_datetime(ev.begin_date, dayfirst=True),
                                 pd.to_datetime(ev.end_date, dayfirst=True))
            pick = lambda df, d: df[(df.month == d.month) & (df.day == d.day)]['temp_air'].tolist()
            same = [pick(rmy, d) == pick(source, d) for d in days]
            if len(days) <= 21:
                assert all(same), (ev.begin_date, ev.end_date)
            else:
                # Longer events are inserted as one run of their 21 most severe days.
                assert sum(same) >= 21
                run_len = max(len(r) for r in ''.join('1' if x else '0' for x in same).split('0'))
                assert run_len >= 21


def test_output_does_not_depend_on_file_order():
    base = read_epw_file(list_epws(BASE_DIR)[0])
    # Shift January to create a deviation for compensation to remove.
    shifted = base.copy()
    shifted.loc[shifted.month == 1, 'temp_air'] -= 2.0
    files = list_epws(EPW_DIR)
    out = []
    for order in (files, files[::-1]):
        df, inserted, report = compensate_monthly(shifted.copy(), base, order, [1], set(),
                                                  np.zeros(len(base), dtype=bool))
        out.append((sorted(inserted), df['temp_air'].round(6).tolist()))
    assert out[0] == out[1]
    assert report.loc[0, 'days_inserted'] > 0
    assert abs(report.loc[0, 'temp_dev_after']) < abs(report.loc[0, 'temp_dev_before'])


def test_visualization_writes_png(run, tmp_path):
    out, _ = run
    png = tmp_path / 'timeline.png'
    fig = visualize_extreme_events(out / 'hotspells' / 'heatwave_events.csv',
                                   out / 'coldspells' / 'coldspells_events.csv',
                                   output_path=png, show=False)
    assert fig is not None and png.stat().st_size > 10_000


def test_visualization_handles_missing_csvs(tmp_path):
    assert visualize_extreme_events(tmp_path / 'a.csv', tmp_path / 'b.csv', show=False) is None


def test_rmy_is_named_after_the_tmy_with_tmy_replaced_by_rmy(run):
    _, rmy_path = run
    assert Path(rmy_path).name == 'Boston_MA_725090_RMY_2009-2023.epw'


@pytest.mark.parametrize('tmy, rmy', [
    ('Boston_MA_725090_TMYx_2009-2023.epw', 'Boston_MA_725090_RMY_2009-2023.epw'),
    ('USA_MA_Boston-Logan.Intl.AP.725090_TMYx.2009-2023.epw', 'USA_MA_Boston-Logan.Intl.AP.725090_RMY.2009-2023.epw'),
    ('USA_MA_Boston.725090_TMY3.epw', 'USA_MA_Boston.725090_RMY.epw'),
    ('City_TMY2.epw', 'City_RMY.epw'),
    ('city_tmy.epw', 'city_RMY.epw'),
    ('MyCity_TMY2009-2023.epw', 'MyCity_RMY2009-2023.epw'),     # 2009 is a year, not TMY2
    ('Toronto_base.epw', 'Toronto_base_RMY.epw'),              # no TMY in the name
])
def test_rmy_file_name(tmy, rmy):
    from rmy.rmy_generation import rmy_file_name
    assert rmy_file_name(tmy) == rmy
