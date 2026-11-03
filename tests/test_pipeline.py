"""End-to-end checks on the Boston sample data in data/ (about 10 seconds)."""
from pathlib import Path

import pandas as pd
import pytest

from rmy import run_full_rmy_pipeline
from rmy.rmy_generation import calculate_monthly_avg_conditions, find_days_to_adjust_avg
from rmy.utils import extract_original_header, list_epws, read_epw_file
from rmy.visualization import visualize_extreme_events

REPO = Path(__file__).resolve().parents[1]
EPW_DIR = REPO / 'data' / 'EPWs'
BASE_DIR = REPO / 'data' / 'base'


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp('rmy')
    rmy_path = run_full_rmy_pipeline(EPW_DIR, BASE_DIR, out)
    return out, rmy_path


def test_rmy_has_base_header_and_full_year(run):
    _, rmy_path = run
    base = list_epws(BASE_DIR)[0]
    assert extract_original_header(rmy_path) == extract_original_header(base)
    assert len(read_epw_file(rmy_path)) == len(read_epw_file(base)) == 8760


def test_rmy_keeps_base_timestamps(run):
    _, rmy_path = run
    base = read_epw_file(list_epws(BASE_DIR)[0])
    rmy = read_epw_file(rmy_path)
    cols = ['year', 'month', 'day', 'hour', 'minute']
    pd.testing.assert_frame_equal(rmy[cols], base[cols])


def test_peak_events_are_inserted_hour_for_hour(run):
    out, rmy_path = run
    rmy = read_epw_file(rmy_path)
    for folder, prefix in (('hotspells', 'heatwave'), ('coldspells', 'coldspells')):
        stats = pd.read_csv(out / folder / f'{prefix}_stats_peak.csv')
        year = int(stats.loc[0, 'year'])
        source = read_epw_file(next(p for p in list_epws(EPW_DIR) if str(year) in p.name))
        events = pd.read_csv(out / folder / f'{prefix}_events_peak.csv')
        assert not events.empty
        for _, ev in events.iterrows():
            for d in pd.date_range(pd.to_datetime(ev.begin_date, dayfirst=True),
                                   pd.to_datetime(ev.end_date, dayfirst=True)):
                pick = lambda df: df[(df.month == d.month) & (df.day == d.day)]
                assert pick(rmy)['temp_air'].tolist() == pick(source)['temp_air'].tolist()


def test_output_does_not_depend_on_file_order():
    base = read_epw_file(list_epws(BASE_DIR)[0])
    months = [6, 7, 8]
    targets = calculate_monthly_avg_conditions(base, months)
    cond = lambda day, ref: (day['temp_air'].mean() < ref['temp_air'] and
                             day['relative_humidity'].mean() < ref['relative_humidity'])
    files = list_epws(EPW_DIR)
    forward = find_days_to_adjust_avg(files, targets, months, cond)
    backward = find_days_to_adjust_avg(files[::-1], targets, months, cond)
    for m in months:
        assert [d.index.tolist() for d in forward[m]] == [d.index.tolist() for d in backward[m]]


def test_visualization_writes_png(run, tmp_path):
    out, _ = run
    png = tmp_path / 'timeline.png'
    fig = visualize_extreme_events(out / 'hotspells' / 'heatwave_events.csv',
                                   out / 'coldspells' / 'coldspells_events.csv',
                                   output_path=png, show=False)
    assert fig is not None and png.stat().st_size > 10_000


def test_visualization_handles_missing_csvs(tmp_path):
    assert visualize_extreme_events(tmp_path / 'a.csv', tmp_path / 'b.csv', show=False) is None
