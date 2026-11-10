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


def test_rmy_header_changes_only_design_weeks_and_comments_2(run):
    _, rmy_path = run
    base = list_epws(BASE_DIR)[0]
    old, new = extract_original_header(base), extract_original_header(rmy_path)
    assert len(new) == 8
    changed = [i for i in range(8) if old[i] != new[i]]
    assert changed in ([2, 6], [6])                         # TYPICAL/EXTREME PERIODS and COMMENTS 2 only
    assert [l.count(',') for l in new] == [l.count(',') for l in old]   # same six periods as the TMYx
    assert new[2].startswith('TYPICAL/EXTREME PERIODS,6,Summer - Week Nearest Max Temperature For Period,Extreme,')
    assert new[6].startswith('COMMENTS 2,"RMY extreme weather file: heat waves and cold spells '
                             'from AMYs 2009-2024 inserted (peak heat year 2022; peak cold year 2015); ')
    assert new[6].endswith(old[6][len('COMMENTS 2,"'):])     # original text kept
    # Same line endings as the base TMY throughout (this TMYx uses CRLF).
    raw, base_raw = Path(rmy_path).read_bytes(), Path(base).read_bytes()
    assert raw.isascii()
    crlf = base_raw.count(b'\r\n') == base_raw.count(b'\n')
    assert (raw.count(b'\r\n') == raw.count(b'\n')) if crlf else (b'\r' not in raw)
    assert len(read_epw_file(rmy_path)) == len(read_epw_file(base)) == 8760


def test_write_epw_never_uses_scientific_notation(run, tmp_path):
    from rmy.utils import write_epw
    _, rmy_path = run
    assert b'e-' not in Path(rmy_path).read_bytes().split(b'\n', 8)[8]
    df = read_epw_file(rmy_path)
    header = extract_original_header(rmy_path)
    df.loc[0:3, 'temp_air'] = [8.88178e-16, -1e-12, 14.77777777, -3.25]
    out = write_epw(df, header, tmp_path / 'x.epw')
    first = [l.split(',')[6] for l in out.read_text().splitlines()[8:12]]
    assert first == ['0', '0', '14.7778', '-3.25']
    pd.testing.assert_frame_equal(read_epw_file(out).iloc[4:], df.iloc[4:], atol=5e-5, check_exact=False)


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


def test_compensation_days_join_inserted_days_smoothly():
    """A compensation day next to an inserted day must not create a big midnight step."""
    base = read_epw_file(list_epws(BASE_DIR)[0])
    shifted = base.copy()
    shifted.loc[shifted.month == 1, 'temp_air'] -= 2.0
    # Protect every other January day, so most candidates have inserted neighbors.
    protected = {f'01-{d:02d}' for d in range(1, 32, 2)}
    df, inserted, _ = compensate_monthly(shifted.copy(), base, list_epws(EPW_DIR), [1], protected,
                                         np.zeros(len(base), dtype=bool), max_midnight_step=2.0)
    assert inserted
    occupied = protected | inserted
    for tag in inserted:
        d = int(tag[3:])
        for a, b in ((d - 1, d), (d, d + 1)):
            if f'01-{a:02d}' in occupied and f'01-{b:02d}' in occupied:
                last = df[(df.month == 1) & (df.day == a)].temp_air.iloc[-1]
                first = df[(df.month == 1) & (df.day == b)].temp_air.iloc[0]
                assert abs(first - last) <= 2.0, (a, b)


def test_wind_repair_fills_only_missing_inserted_hours():
    from rmy.rmy_generation import repair_missing_wind
    n = 48
    base = pd.DataFrame({'wind_speed': np.full(n, 9.0), 'wind_direction': np.full(n, 90.0)})
    df = pd.DataFrame({'wind_speed': np.full(n, 2.0), 'wind_direction': np.full(n, 180.0)})
    df.loc[0:3, 'wind_direction'] = [350.0, np.nan, np.nan, 30.0]    # short gap across north
    df.loc[10:15, 'wind_speed'] = np.nan                               # long gap -> TMY
    df.loc[20, 'wind_speed'] = np.nan                                  # not inserted -> untouched
    inserted = np.ones(n, dtype=bool)
    inserted[20] = False
    before = df.copy()
    out, counts = repair_missing_wind(df, base, inserted)
    assert out.loc[1:2, 'wind_direction'].tolist() == [3.0, 17.0]      # 350 -> 30 the short way
    assert (out.loc[10:15, 'wind_speed'] == 9.0).all()
    assert np.isnan(out.loc[20, 'wind_speed'])
    recorded = before.notna()
    pd.testing.assert_frame_equal(out[recorded], before[recorded])     # recorded values unchanged
    assert counts == {'wind_speed': (0, 6), 'wind_direction': (2, 0)}


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


def test_source_map_accounts_for_every_hour(run):
    out, rmy_path = run
    from rmy.rmy_generation import source_map_path
    m = pd.read_csv(source_map_path(rmy_path))
    tmy = read_epw_file(next(BASE_DIR.glob('*.epw')))
    rmy = read_epw_file(rmy_path)
    assert len(m) == len(rmy)
    assert set(m.origin) <= {'TMY', 'heat_wave', 'cold_spell', 'compensation', 'blend'}
    # TMY hours are the base TMY's, with its source years.
    kept = (m.origin == 'TMY').to_numpy()
    assert np.allclose(rmy.temp_air[kept], tmy.temp_air[kept], equal_nan=True)
    assert (m.source_year[kept].to_numpy() == tmy.year[kept].to_numpy()).all()
    # Inserted heat-wave hours are the peak year's own hours.
    peak = int(pd.read_csv(out / 'hotspells' / 'heatwave_stats_peak.csv').year.iloc[0])
    heat = m[m.origin == 'heat_wave']
    assert len(heat) and (heat.source_year == peak).all()
    amy = read_epw_file(next(p for p in EPW_DIR.glob('*.epw') if str(peak) in p.name)).set_index(['month', 'day', 'hour'])
    src = amy.loc[list(zip(heat.month, heat.day, heat.hour)), 'temp_air'].to_numpy()
    assert np.allclose(rmy.temp_air[heat.index].to_numpy(), src, equal_nan=True)


def test_event_csvs_have_no_internal_columns(run):
    out, _ = run
    for f in (out / 'hotspells').glob('*events*.csv'):
        cols = pd.read_csv(f).columns
        assert '_day0' not in cols and 'source_years' in cols
