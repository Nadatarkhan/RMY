"""Offline checks of amy_sourcing/generate_amys.py (no network: downloads are mocked)."""
import contextlib
import importlib.util
import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('requests')
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_amys', REPO / 'amy_sourcing' / 'generate_amys.py')
ga = importlib.util.module_from_spec(spec)
_out = io.StringIO()
with contextlib.redirect_stdout(_out):
    spec.loader.exec_module(ga)

BOSTON = {'USAF': '725090', 'WBAN': '14739', 'STATION NAME': 'GEN E L LOGAN INTERNATIONAL AIRPORT',
          'CTRY': 'US', 'STATE': 'MA', 'ICAO': 'KBOS', 'LAT': '+42.361', 'LON': '-071.010',
          'ELEV(M)': '+0003.3', 'BEGIN': '19431121', 'END': '20250827'}


def test_import_is_silent_and_self_tests_pass():
    assert _out.getvalue() == ''
    with contextlib.redirect_stdout(io.StringIO()):
        assert ga.run_self_tests()


def test_last_complete_year_reads_the_station_record():
    today = date(2026, 11, 1)
    assert ga.last_complete_year({'end': '20250827'}, 'NSRDB', today) == 2024   # record stops mid-2025
    assert ga.last_complete_year({'end': '20251231'}, 'NSRDB', today) == 2025   # 2025 complete
    assert ga.last_complete_year({'end': '20260920'}, 'NSRDB', today) == 2025   # capped at last year
    assert ga.last_complete_year({'end': '20261231', 'tz_offset': -5}, 'PVGIS', today) == \
        ga.PVGIS5_3_BOUNDS[1] - 1                                                 # PVGIS coverage


def test_keys_and_emails_are_removed_from_saved_text():
    report = {'error': 'Download failed: ...?api_key=SECRET123&email=me@x.org', 'year': 2020}
    assert ga._redact(report, ['SECRET123', 'me@x.org']) == {'error': 'Download failed: ...?api_key=***&email=***',
                                                             'year': 2020}
    # Very short values (placeholders) never mangle ordinary words.
    assert ga._redact('RuntimeError: failed', ['e', 'k']) == 'RuntimeError: failed'


def test_sky_fill_keeps_recorded_values_and_rounds_only_filled_hours():
    total = np.array([2.5, np.nan, np.nan, 9.5, 4.0])            # 2.5 and 9.5 come from oktas
    t, o, flags, report = ga.infill_sky_cover(total, total.copy(), np.full(5, np.nan))
    np.testing.assert_array_equal(t[[0, 3, 4]], [2.5, 9.5, 4.0])  # recorded values unchanged
    assert list(flags['total']) == ['', 'B', 'B', '', '']
    assert t[1] == np.rint(2.5 + (9.5 - 2.5) / 3) and t[2] == np.rint(2.5 + 2 * (9.5 - 2.5) / 3)
    assert report['total_sky_interpolated_hours'] == 2


def _fake_station_catalog(*_):
    return pd.DataFrame([BOSTON])


def _write_fake_amy(path, year):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'fake AMY {year}\n')
    return path


def test_failed_and_flagged_years_are_replaced_by_earlier_years(tmp_path, monkeypatch):
    monkeypatch.setattr(ga, 'station_catalog', _fake_station_catalog)

    def fake_year(session, st, year, cache_dir, out_dir, key, email, **kw):
        if year == 2024:
            raise RuntimeError(f'NOAA_MET stage: gap, url?api_key={key}')
        folder = 'flagged' if year == 2023 else 'epws'
        p = _write_fake_amy(Path(out_dir) / folder / f"Boston_{st['station_id']}_{year}_AMY.epw", year)
        return {'year': year, 'status': 'written', 'qc_grade': 'GOOD', 'output_epw': str(p)}

    monkeypatch.setattr(ga, 'create_station_year', fake_year)
    with contextlib.redirect_stdout(io.StringIO()):
        run = ga.generate_amys('72509014739', tmp_path, 'SECRET123', 'me@x.org', n_years=10, tz_offset=-5)
    assert run['city'] == 'Boston'
    assert run['years'] == list(range(2013, 2023))                # 2024 failed, 2023 flagged
    assert sorted(p.name for p in run['amy_dir'].glob('*.epw'))[0] == 'Boston_72509014739_2013_AMY.epw'
    manifest = pd.read_csv(tmp_path / 'Boston' / 'AMY_reports' / 'AMY_manifest.csv')
    assert set(manifest.loc[manifest.year.isin([2023, 2024]), 'status']) == {'flagged_not_used', 'failed'}
    saved = ''.join(p.read_text() for p in (tmp_path / 'Boston' / 'AMY_reports').rglob('*') if p.is_file())
    assert 'SECRET123' not in saved and 'me@x.org' not in saved


def test_setup_errors_stop_after_three_years(tmp_path, monkeypatch):
    monkeypatch.setattr(ga, 'station_catalog', _fake_station_catalog)
    calls = []

    def always_fails(session, st, year, *a, **kw):
        calls.append(year)
        raise RuntimeError('SOLAR stage: HTTP 403 (request details withheld)')

    monkeypatch.setattr(ga, 'create_station_year', always_fails)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match='first three years'):
        ga.generate_amys('72509014739', tmp_path, 'k', 'e', tz_offset=-5)
    assert len(calls) == 3


def test_americas_station_without_key_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ga, 'station_catalog', _fake_station_catalog)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match='NREL API key'):
        ga.generate_amys('72509014739', tmp_path, None, None, tz_offset=-5)
