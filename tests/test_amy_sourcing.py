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


def test_calm_and_variable_hours_are_not_wind_outages():
    # 8 calm hours, 8 variable (VRB) hours, then a real 3-hour outage.
    n = 24
    ws = [3.0] * 2 + [0.0] * 8 + [2.0] * 8 + [4.0, np.nan, np.nan, np.nan, 5.0, 6.0]
    wd = [200.0] * 2 + [np.nan] * 16 + [210.0, np.nan, np.nan, np.nan, 220.0, 230.0]
    wt = ['N'] * 2 + ['C'] * 8 + ['V'] * 8 + ['N', '', '', '', 'N', 'N']
    s = pd.DataFrame({'WS': ws, 'WD': wd, 'WIND_TYPE': wt},
                     index=pd.date_range('2020-01-01', periods=n, freq='h'))
    _, rep = ga.repair_wind_direction(s, mode='CIRCULAR_LINEAR')
    assert rep['wind_direction_max_original_missing_run_hours'] == 16   # unchanged report field
    assert rep['wind_direction_max_outage_run_hours'] == 3               # real outage only
    assert rep['wind_direction_variable_imputed_hours'] == 8             # still reported


def test_nsrdb_aerosol_and_precipitable_water_are_read_under_their_csv_names(tmp_path):
    # Column names exactly as the NSRDB GOES v4 CSV gives them (Las Vegas run, 2010-2024).
    st = {'station_id': '72386023169', 'lat': 36.072, 'lon': -115.163}
    idx = pd.date_range('2023-01-01 00:30', periods=8760, freq='h')
    body = pd.DataFrame({'Year': idx.year, 'Month': idx.month, 'Day': idx.day, 'Hour': idx.hour,
                         'Minute': 30, 'GHI': 0, 'DNI': 0, 'DHI': 0,
                         'Aerosol Optical Depth': 0.08, 'Precipitable Water': 1.25,
                         'Surface Albedo': 0.2, 'Solar Zenith Angle': 95.0,
                         'Cloud Type': 0, 'Fill Flag': 0})
    cache = tmp_path / 'radiation' / 'nsrdb' / st['station_id'] / '2023_local_extended_v4.csv'
    cache.parent.mkdir(parents=True)
    cache.write_text('Source,Time Zone,Precipitable Water Units\nNSRDB,-8,cm\n'
                     + body.to_csv(index=False))
    with contextlib.redirect_stdout(io.StringIO()):
        rad, _ = ga.nsrdb_radiation(None, 2023, st, tmp_path, 'key', 'a@b.c', -8)
    assert (rad['AOD'] == 0.08).all() and (rad['PW_CM'] == 1.25).all()
    assert (rad['ALBEDO'] == 0.2).all() and rad['CLOUD_TYPE'].notna().all()
    epw_pw = ga.optional_epw_numeric(rad['PW_CM'], 10, 0, 999, 999, 1)     # cm -> mm, as in build_epw
    epw_aod = ga.optional_epw_numeric(rad['AOD'], 1, 0, .999, .999, 3)
    assert (epw_pw == 12.5).all() and (epw_aod == 0.08).all()


def test_city_name_comes_from_the_station_coordinates():
    pytest.importorskip('geonamescache')
    us = {'country': 'US'}
    assert ga.city_from_coordinates({**us, 'lat': 36.072, 'lon': -115.163}) == 'Las_Vegas_NV'
    assert ga.city_from_coordinates({**us, 'lat': 39.217, 'lon': -82.233}) == 'Athens_OH'
    # Border airports keep their own country's city (not Ciudad Juarez or Mexicali).
    assert ga.city_from_coordinates({**us, 'lat': 31.812, 'lon': -106.377}) == 'El_Paso_TX'
    assert ga.city_from_coordinates({**us, 'lat': 18.433, 'lon': -66.011}) == 'San_Juan_PR'
    assert ga.city_from_coordinates({'country': 'AE', 'lat': 25.255, 'lon': 55.364}) == 'Dubai_AE'
    assert ga.city_from_coordinates({**us, 'lat': 0.0, 'lon': -140.0}) is None   # open ocean


def test_pvgis_snapshots_are_moved_to_the_hour_midpoints():
    # PVGIS SARAH snapshots at hh:10 UTC of a clear sky (beam 70%, diffuse 10% of the
    # horizontal extraterrestrial irradiance); Cairo, local standard time UTC+2.
    lat, lon, tz = 30.122, 31.406, 2
    t = pd.date_range('2022-06-17 00:10', '2022-06-24 00:10', freq='h')
    cz, e0 = ga.solar_geometry_utc(t, lat, lon)
    eh = e0 * np.maximum(cz, 0)
    samples = pd.DataFrame({'BEAM_H': 0.7 * eh, 'DHI': 0.1 * eh}, index=t)
    local = pd.date_range('2022-06-18', '2022-06-23', freq='h', inclusive='left')
    mid = local + pd.Timedelta(minutes=30) - pd.Timedelta(hours=tz)
    r = ga.pvgis_to_hour_midpoints(samples, mid, lat, lon)
    mcz, me0 = ga.solar_geometry_utc(mid, lat, lon)
    assert np.allclose(r['GHI'], 0.8 * me0 * np.maximum(mcz, 0), atol=0.01)   # exact at midpoints
    assert (r.loc[mcz <= 0, ['GHI', 'DNI', 'DHI']] == 0).all().all()        # night is 0
    # The daily radiation centroid falls on solar noon (it was ~20 min late before).
    hours = local.hour.to_numpy() + 0.5                 # EPW hour midpoints
    g = r['GHI'].to_numpy()
    centroid = (hours * g).sum() / g.sum() * 60        # minutes after local midnight (5-day mean)
    noon = (12 - (lon - 15 * tz) / 15) * 60 + 1.5      # equation of time in mid-June: about -1.5 min
    assert abs(centroid - noon) < 2
    etr, etrn = ga.extraterrestrial_from_nsrdb(2022, local, r)
    assert (etrn[mcz < -0.01] == 0).all()                # no extraterrestrial normal at night


def test_cavok_and_other_clear_sky_metars_mean_zero_cover():
    cover = lambda rem, **groups: ga.parse_noaa_cloud({'REM': rem, **groups}, 'TC_GA_PROXY')[0]
    assert cover('MET051METAR HECA 021030Z 03002KT CAVOK 19/10 Q1020 NOSIG=') == 0.0
    assert cover('MET045METAR XXXX 021000Z 03003KT 9999 NSC 18/09 Q1020=') == 0.0
    # Clear sky only in the trend part (BECMG/TEMPO) is not clear now.
    assert np.isnan(cover('MET060METAR XXXX 021000Z 00000KT 0800 FG VV002 12/12 Q1020 BECMG NSC='))
    assert np.isnan(cover(''))                                                  # no report text
    assert cover('METAR XXXX CAVOK', GA1='04,1,+01500,1,99,9') == 5.0           # a real layer wins


def test_precipitation_totals_over_six_hours_are_not_used():
    p = lambda **groups: ga.parse_noaa_precip(groups)[:2]
    assert np.isnan(p(AA1='24,1000,3,1')[0])                         # 24 h SYNOP total: ignored
    assert np.isnan(p(AA1='12,0500,9,1')[0])
    assert p(AA1='24,1000,3,1', AA2='06,0010,9,1') == (1.0, 6.0)     # the 6 h total is kept
    assert p(AA1='01,0025,9,1', AA2='03,0050,9,1') == (2.5, 1.0)     # shortest period preferred


def test_sky_fill_thresholds():
    # Linear up to 12 h (B), blended cloud-type estimate for 13-36 h (C), estimate beyond (E).
    def flags_for_gap(n):
        cover = np.r_[np.full(30, 5.0), np.full(n, np.nan), np.full(30, 5.0)]
        ctype = np.r_[np.zeros(30), np.full(n, 3.0), np.full(30, 3.0)]   # type 3 seen with 5 tenths
        _, _, flags, _ = ga.infill_sky_cover(cover, cover.copy(), ctype)
        return set(flags['total'][30:30 + n])
    assert flags_for_gap(12) == {'B'}
    assert flags_for_gap(13) == {'C'} and flags_for_gap(36) == {'C'}
    assert flags_for_gap(37) == {'E'}


def test_longwave_default_is_energyplus_clark_allen_with_dilley_obrien_option():
    db, dp = pd.Series([20.0]), pd.Series([10.0])
    clear, rep = ga.energyplus_infrared(db, dp, pd.Series([0.0]))
    assert abs(clear[0] - 5.6697e-8 * 293.15 ** 4 * (0.787 + 0.764 * np.log(283.15 / 273))) < 1e-6
    assert 'Clark-Allen' in rep['infrared_method']
    overcast, _ = ga.energyplus_infrared(db, dp, pd.Series([10.0]))
    assert abs(overcast[0] / clear[0] - (1 + 0.224 - 0.35 + 0.28)) < 1e-9     # Walton cloud correction
    do, rep = ga.energyplus_infrared(db, dp, pd.Series([0.0]), clear_sky='DILLEY_OBRIEN')
    # Dilley and O'Brien (1998) by hand: e = 1.2271 kPa, w = 4650 e / T = 19.47 kg/m2
    assert abs(do[0] - (59.38 + 113.7 * (293.15 / 273.16) ** 6 + 96.96 * np.sqrt(19.465 / 25))) < 0.1
    assert "Dilley-O'Brien" in rep['infrared_method']


def _era5_surface(gap_slices):
    idx = pd.date_range('2022-01-01', '2022-03-01', freq='h', inclusive='left')
    hod = idx.hour.to_numpy()
    s = pd.DataFrame({'DB': 10 + 5 * np.sin(2 * np.pi * (hod - 9) / 24), 'DP': np.full(len(idx), 2.0),
                      'PR': np.full(len(idx), 1013.0), 'WS': np.full(len(idx), 3.0), 'WD': np.full(len(idx), 90.0)},
                     index=idx)
    for sl in gap_slices:
        s.iloc[sl] = np.nan
    return s


def _fake_era5(offset_c=-2.0):
    """ERA5 = the true station series shifted by offset_c (C) in temperature; UTC = local + 5 h."""
    def month(fs, ym, field, lat, lon, cache_dir):
        t = pd.date_range(f'{ym[:4]}-{ym[4:]}-01', periods=24 * 31, freq='h')
        local = t - pd.Timedelta(hours=5)
        hod = local.hour.to_numpy()
        val = {'2t': 10 + 5 * np.sin(2 * np.pi * (hod - 9) / 24) + offset_c, '2d': np.full(len(t), 1.0),
               'msl': np.full(len(t), 1012.0), '10u': np.full(len(t), -2.0), '10v': np.zeros(len(t))}[field]
        return pd.Series(val, index=t)
    return month


def test_era5_infill_fills_long_gaps_with_bias_corrected_values(monkeypatch):
    monkeypatch.setattr(ga, 'era5_point_month', _fake_era5())
    st = {'lat': 42.0, 'lon': -71.0, 'tz_offset': -5}
    surface = _era5_surface([slice(100, 130), slice(500, 506)])            # 30 h and 6 h gaps
    with contextlib.redirect_stdout(io.StringIO()):
        out, rep = ga.era5_infill(surface, st, '/tmp', fs=object())
    truth = _era5_surface([])
    np.testing.assert_allclose(out.DB.iloc[100:130], truth.DB.iloc[100:130], atol=1e-6)   # bias removed
    assert out.DB.iloc[500:506].isna().all()                                 # short gap left for linear filling
    assert (out.ERA5_FILLED.iloc[100:130].str.split(',').apply(set) == {'DB', 'DP', 'PR', 'WS', 'WD'}).all()
    np.testing.assert_allclose(out.WD.iloc[100:130], 90.0)                  # wind from the east (u < 0)
    assert rep['era5_infill_DB_hours'] == 30 and rep['era5_infill_months'] == '202201'


def test_era5_infill_respects_the_cap_and_failures():
    st = {'lat': 42.0, 'lon': -71.0, 'tz_offset': -5}
    big = _era5_surface([slice(0, 200)])                                     # 200 of 1416 h > 5%
    out, rep = ga.era5_infill(big, st, '/tmp', fs=object())
    assert out.DB.iloc[:200].isna().all() and '> 5%' in rep['era5_infill_DB_skipped']
    class Broken:
        def ls(self, *_):
            raise OSError('no network')
    with contextlib.redirect_stdout(io.StringIO()):
        out, rep = ga.era5_infill(_era5_surface([slice(100, 130)]), st, '/tmp/era5_test_cache_none', fs=Broken())
    assert out.DB.iloc[100:130].isna().all() and rep['era5_infill_status'].startswith('ERA5 unavailable')


def test_era5_infill_leaves_calm_and_variable_hours_without_direction(monkeypatch):
    monkeypatch.setattr(ga, 'era5_point_month', _fake_era5())
    st = {'lat': 42.0, 'lon': -71.0, 'tz_offset': -5}
    s = _era5_surface([])
    s['WIND_TYPE'] = 'N'
    s.iloc[200:230, s.columns.get_loc('WS')] = 0.0                          # 30 h calm night
    s.iloc[200:230, s.columns.get_loc('WD')] = np.nan
    s.iloc[200:230, s.columns.get_loc('WIND_TYPE')] = 'C'
    s.iloc[300:320, s.columns.get_loc('WD')] = np.nan                        # 20 h variable wind
    s.iloc[300:320, s.columns.get_loc('WIND_TYPE')] = 'V'
    s.iloc[500:530, s.columns.get_loc('WD')] = np.nan                        # 30 h real outage
    s.iloc[500:530, s.columns.get_loc('WS')] = np.nan
    with contextlib.redirect_stdout(io.StringIO()):
        out, rep = ga.era5_infill(s, st, '/tmp', fs=object())
    assert out.WD.iloc[200:230].isna().all() and out.WD.iloc[300:320].isna().all()
    assert out.WD.iloc[500:530].notna().all() and rep['era5_infill_WD_hours'] == 30


def test_era5_prefetch_skips_cached_months_and_never_raises(tmp_path, monkeypatch):
    jobs = [('202101', f, 5.605, -0.167, str(tmp_path)) for f in ('2t', '2d')]
    for ym, f, lat, lon, cd in jobs:                                         # already cached: nothing to do
        p = tmp_path / 'era5_point' / f'{lat:.3f}_{lon:.3f}' / f'{ym}_{f}.csv'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('utc,v\n')
    called = []
    monkeypatch.setattr('subprocess.run', lambda *a, **k: called.append(1))
    ga._era5_prefetch(jobs)
    assert not called
    monkeypatch.undo()
    monkeypatch.setattr('sys.executable', str(tmp_path / 'no_python_here'))  # helper cannot start
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ga._era5_prefetch([('202102', f, 5.605, -0.167, str(tmp_path)) for f in ('2t', '2d')])
    assert 'one at a time' in out.getvalue()


def test_calm_hours_get_direction_zero_even_when_a_direction_was_reported():
    idx = pd.date_range('2022-01-01', periods=6, freq='h')
    s = pd.DataFrame({'WS': [3.0, 0.0, 0.0, 2.0, 0.0, 4.0], 'WD': [90.0, 140.0, np.nan, 180.0, 0.0, 200.0],
                      'WIND_TYPE': ['N', 'N', 'C', 'N', 'C', 'N']}, index=idx)
    with contextlib.redirect_stdout(io.StringIO()):
        out, rep = ga.repair_wind_direction(s, 'CIRCULAR_LINEAR', 2)
    assert (out.WD[out.WS == 0] == 0).all()                                   # all calm hours 0 deg
    assert list(out.WD[out.WS > 0]) == [90.0, 180.0, 200.0]                   # real directions untouched
    assert rep['wind_direction_calm_reported_set_to_zero_hours'] == 1


def test_sparse_records_are_flagged_for_review():
    assert ga.sparse_grade('GOOD', 0.33) == 'REVIEW_SPARSE'
    assert ga.sparse_grade('REVIEW_WIND', 0.33) == 'REVIEW_WIND+SPARSE'
    assert ga.sparse_grade('GOOD', 0.95) == 'GOOD'
    assert ga.sparse_grade('REVIEW_WIND', 0.5) == 'REVIEW_WIND'               # at the threshold: not sparse


def test_pvgis_timebase_says_era5_values_are_not_shifted():
    assert 'no time shift' in ga.pvgis_timebase('PVGIS-ERA5', [30, 30])
    sarah = ga.pvgis_timebase('PVGIS-SARAH3', [10])
    assert 'snapshots' in sarah and 'minute(s) 10' in sarah


def test_observed_share_note_marks_sparse_records():
    assert ga.observed_share_comment(0.95) == '; observed dry-bulb share 0.950'
    assert ga.observed_share_comment(0.33).endswith('(sparse record)')
