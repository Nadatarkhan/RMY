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


def test_sky_fill_without_cloud_type_uses_the_years_own_cover():
    # PVGIS years have no NSRDB cloud type: long gaps used to stay missing (EPW 99).
    hours = pd.date_range('2021-01-01', periods=24 * 60, freq='h')
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 11, len(hours)).astype(float)
    cover[500:600] = np.nan                                         # 100-hour gap
    recorded = np.isfinite(cover)
    old_t, _, _, old_rep = ga.infill_sky_cover(cover, cover.copy(), np.full(len(hours), np.nan))
    assert old_rep['total_sky_still_missing_hours'] == 100          # without a fallback: unchanged behaviour
    t, o, flags, rep = ga.infill_sky_cover(cover, cover.copy(), np.full(len(hours), np.nan),
                                           month=hours.month.to_numpy(), hour=hours.hour.to_numpy())
    assert rep['total_sky_still_missing_hours'] == 0 and np.isfinite(t).all()
    np.testing.assert_array_equal(t[recorded], cover[recorded])    # recorded values unchanged
    assert set(flags['total'][500:600]) == {'E'} and rep['total_sky_estimate_from_month_hour_hours'] == 100
    assert (o <= t).all() and t.min() >= 0 and t.max() <= 10


def test_sky_fill_fallback_follows_clearness_index_by_day():
    n = 24 * 90
    kt = np.tile(np.r_[np.full(6, np.nan), np.linspace(.1, .8, 12), np.full(6, np.nan)], 90)
    cover = np.where(np.isfinite(kt), np.rint(10 * (1 - kt)), 5.0)   # cloudier when the sky is darker
    cover[1000:1100] = np.nan
    hours = pd.date_range('2021-01-01', periods=n, freq='h')
    t, _, _, rep = ga.infill_sky_cover(cover, cover.copy(), np.full(n, np.nan), kt=kt,
                                       month=hours.month.to_numpy(), hour=hours.hour.to_numpy())
    day = np.isfinite(kt[1000:1100])
    assert rep['total_sky_estimate_from_clearness_hours'] > 0
    np.testing.assert_allclose(t[1000:1100][day], np.rint(10 * (1 - kt[1000:1100][day])), atol=1)


def test_sky_fill_uses_era5_when_the_year_has_no_cloud_reports():
    # Automatic stations (e.g. Bern-Zollikofen) can go whole years without cloud reports: the year's
    # own cover gives no fallback, so ERA5 cover is the estimate of last resort.
    n = 24 * 60
    hours = pd.date_range('2021-01-01', periods=n, freq='h')
    none = np.full(n, np.nan)
    era_t = np.tile(np.r_[np.full(12, 8.0), np.full(12, 2.0)], 60)
    era_o = era_t / 2
    _, _, _, rep0 = ga.infill_sky_cover(none, none, none, month=hours.month.to_numpy(), hour=hours.hour.to_numpy())
    assert rep0['total_sky_still_missing_hours'] == n
    t, o, flags, rep = ga.infill_sky_cover(none, none, none, month=hours.month.to_numpy(),
                                           hour=hours.hour.to_numpy(), era5_total=era_t, era5_opaque=era_o)
    assert rep['total_sky_still_missing_hours'] == 0 and rep['opaque_sky_still_missing_hours'] == 0
    np.testing.assert_array_equal(t, era_t); np.testing.assert_array_equal(o, era_o)
    assert set(flags['total']) == {'E'} and rep['total_sky_estimate_from_era5_hours'] == n


def test_era5_sky_estimate_is_shifted_to_the_recorded_cover_and_never_replaces_it():
    n = 24 * 60
    rng = np.random.default_rng(1)
    era_t = rng.integers(0, 9, n).astype(float)
    cover = np.clip(era_t + 2, 0, 10)                               # the observer reports 2 tenths more
    cover[200:] = np.nan                                            # 200 recorded hours, then a long gap
    t, _, flags, rep = ga.infill_sky_cover(cover, cover.copy(), np.full(n, np.nan),
                                           era5_total=era_t, era5_opaque=era_t)
    np.testing.assert_array_equal(t[:200], cover[:200])
    gap = slice(300, n)
    assert abs(np.mean(t[gap] - era_t[gap]) - 2) < 0.3 and rep['total_sky_estimate_from_era5_hours'] > 0


def test_era5_sky_cover_reads_only_needed_months_and_keeps_opaque_below_total(monkeypatch):
    st = {'tz_offset': 1.0, 'lat': 46.99, 'lon': 7.46}
    index = pd.date_range('2021-01-01', '2021-02-28 23:00', freq='h')
    need = np.zeros(len(index), dtype=bool); need[(index.month == 2) & (index.day >= 10)] = True
    asked = []
    def fake(fs, ym, field, lat, lon, cache_dir):
        asked.append((ym, field))
        utc = pd.date_range(f'{ym[:4]}-{ym[4:]}-01', periods=24 * 28, freq='h')
        return pd.Series({'tcc': 6.0, 'lcc': 5.0, 'mcc': 5.0}[field], index=utc)
    monkeypatch.setattr(ga, 'era5_point_month', fake)
    total, opaque = ga.era5_sky_cover(st, index, need, '/nonexistent', fs=object())
    assert {ym for ym, _ in asked} == {'202102'}
    feb = index.month == 2
    assert np.isnan(total[~feb]).all()
    ok = feb & np.isfinite(total)
    np.testing.assert_allclose(total[ok], 6.0)
    np.testing.assert_allclose(opaque[ok], 6.0)                     # 1-(1-.5)^2 = 7.5 tenths, capped at total


def _fake_noaa(reports_per_year):
    """noaa_csv stand-in: `reports_per_year[year]` hourly temperature reports (None = no file)."""
    def fake(session, station_id, year, cache_dir):
        n = reports_per_year.get(year)
        if n is None:
            raise RuntimeError(f'HTTP 404 from https://example/{year}/{station_id}.csv (request details withheld)')
        dates = pd.date_range(f'{year}-01-01', periods=n, freq='h', tz='UTC')
        return pd.DataFrame({'DATE': dates, 'TMP': ['+0215,1'] * n})
    return fake


def test_last_reporting_year_skips_years_a_silent_station_left_in_its_record(monkeypatch):
    # Lahore: hourly to 2018, then a few reports a year while the NOAA record stays open.
    years = {y: 8760 for y in range(2005, 2019)}
    years.update({2019: 40, 2020: 40, 2021: 40, 2022: None, 2023: 30})
    monkeypatch.setattr(ga, 'noaa_csv', _fake_noaa(years))
    end, silent = ga.last_reporting_year(None, {'station_id': '41641099999'}, 2023, '/nonexistent')
    assert end == 2018 and silent == [2023, 2022, 2021, 2020, 2019]


def test_last_reporting_year_keeps_a_station_that_reports(monkeypatch):
    years = {y: 8760 for y in range(2009, 2025)}
    years[2024] = 5000                                               # 57% of hours: still regular
    monkeypatch.setattr(ga, 'noaa_csv', _fake_noaa(years))
    assert ga.last_reporting_year(None, {'station_id': 'x'}, 2024, '/nonexistent') == (2024, [])


def test_last_reporting_year_does_not_skip_on_network_errors_or_give_up_silently(monkeypatch):
    def broken(session, station_id, year, cache_dir):
        raise RuntimeError('Download failed from https://example: timeout')
    monkeypatch.setattr(ga, 'noaa_csv', broken)
    assert ga.last_reporting_year(None, {'station_id': 'x'}, 2024, '/nonexistent') == (2024, [])
    monkeypatch.setattr(ga, 'noaa_csv', _fake_noaa({}))                # no files at all
    assert ga.last_reporting_year(None, {'station_id': 'x'}, 2024, '/nonexistent') == (2024, [])


def test_isolated_temperature_spikes_are_removed_but_fronts_and_storms_kept():
    db = np.r_[np.full(10, 15.0), 2.0, np.full(10, 15.0),             # corrupt report: T and Td both 2 C
               np.full(5, 15.0), 30.0, np.full(5, 15.5),                # one-hour warm spike
               np.full(5, 35.0), 24.0, 26.0, np.full(5, 35.0),          # storm outflow: Td holds -> kept
               np.full(5, 15.0), np.full(10, 4.0)]                      # cold front: a step that stays
    dp = np.full(len(db), 8.0); dp[10] = 2.0; dp[32:44] = 22.0; dp[-10:] = 1.0   # humid air mass (Td holds in the storm); dry air behind the front
    out, rep = ga.remove_temperature_spikes(pd.DataFrame({'DB': db, 'DP': dp}))
    assert np.isnan(out.DB[10]) and np.isnan(out.DP[10])             # T = Td pair blanked together
    assert np.isnan(out.DB[26])
    assert out.DB.iloc[37:39].notna().all()                          # storm kept
    assert out.DB.iloc[-10:].notna().all() and out.DB.iloc[-11] == 15.0
    assert rep['DB_spikes_removed_hours'] == 2


def test_temperature_below_dew_point_is_treated_as_a_corrupt_report():
    # Orly 2020: a garbled METAR temperature (2 C) in an hour whose SYNOP dew point is 9.3 C.
    surface = pd.DataFrame({'DB': [15.0, 14.5, 2.0, 12.0, 12.4], 'DP': [4.0, 5.0, 9.3, 10.0, 11.0]})
    out, rep = ga.remove_temperature_spikes(surface)
    assert np.isnan(out.DB[2]) and out.DP[2] == 9.3
    assert rep['DB_below_DP_removed_hours'] == 1 and rep['DP_above_DB_removed_hours'] == 0


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


def test_dew_point_above_dry_bulb_is_capped():
    # Fairbanks-like ice-fog hours: dew point reported above the air temperature.
    surface = pd.DataFrame({'DB': [-40.0, -36.1, 5.0, np.nan], 'DP': [-30.4, -35.8, 2.0, 1.0]})
    report = ga.cap_dew_point(surface)
    assert report == {'dew_point_capped_hours': 2, 'dew_point_capped_max_excess_C': 9.6}
    assert surface['DP'].tolist()[:3] == [-40.0, -36.1, 2.0]
    assert np.isnan(surface['DB'].iloc[3]) and surface['DP'].iloc[3] == 1.0   # missing values left alone


def _noaa_frame(times_utc, temps):
    return pd.DataFrame({'DATE': pd.to_datetime(times_utc, utc=True), 'SOURCE': '4', 'REPORT_TYPE': 'FM-15',
                         'TMP': [f'{int(round(t * 10)):+05d},1' for t in temps]})


def test_reports_are_grouped_by_local_standard_hour_in_half_hour_zones():
    # India (+5:30): METARs at :00 and :30 UTC fall at :30 and :00 local time.
    d = _noaa_frame(['2021-06-01 00:00', '2021-06-01 00:30', '2021-06-01 01:00'], [30.0, 31.0, 32.0])
    h = ga.extract_hourly(d, 5.5)
    assert list(h.index) == [pd.Timestamp('2021-06-01 05:00'), pd.Timestamp('2021-06-01 06:00')]
    assert h.DB.iloc[0] == 30.0                         # 05:30 local is in the 05:00-06:00 hour
    # Nepal (+5:45)
    h = ga.extract_hourly(_noaa_frame(['2021-06-01 00:20'], [25.0]), 5.75)
    assert list(h.index) == [pd.Timestamp('2021-06-01 06:00')]


def test_whole_hour_zones_give_the_same_hours_as_before():
    # floor(UTC) + offset (old) equals floor(UTC + offset) (new) for whole-hour offsets
    times = pd.date_range('2021-03-01 00:10', periods=200, freq='37min', tz='UTC')
    d = _noaa_frame(times, np.linspace(-5, 20, len(times)))
    for tz in (-8, -5, 0, 1, 9):
        new = ga.extract_hourly(d, tz)
        old = ga.extract_hourly(d, 0)
        old.index = old.index + pd.Timedelta(hours=tz)
        pd.testing.assert_series_equal(new.DB, old.DB, check_names=False, check_index_type=False)


def test_hourly_reanalysis_is_interpolated_to_half_hour_zones():
    s = pd.Series([0.0, 10.0, np.nan, 30.0], index=pd.date_range('2021-01-01', periods=4, freq='h'))
    q = pd.DatetimeIndex(['2021-01-01 00:00', '2021-01-01 00:30', '2021-01-01 00:45', '2021-01-01 01:30', '2021-01-01 05:30'])
    out = ga._reindex_time(s, q).to_numpy()
    np.testing.assert_allclose(out[:3], [0.0, 5.0, 7.5])
    assert np.isnan(out[3]) and np.isnan(out[4])        # never across a missing value or past the end


def _noaa_fields_frame(tmp, dew, slp, wnd, n=48):
    dates = pd.date_range('2020-06-01', periods=n, freq='h')
    return pd.DataFrame({'DATE': dates, 'TMP': [tmp] * n, 'DEW': [dew] * n, 'SLP': [slp] * n, 'WND': [wnd] * n})


def test_required_field_coverage_finds_fields_a_station_never_reports():
    uscrn = _noaa_fields_frame('+0251,1', '+9999,9', '99999,9', '999,9,9,9999,9')
    share = ga.required_field_coverage(uscrn, 2020)
    assert share['dry-bulb temperature'] > 0
    assert share['dew point'] == 0 and share['pressure'] == 0 and share['wind speed'] == 0
    airport = _noaa_fields_frame('+0251,1', '+0150,1', '10132,1', '270,1,N,0046,1')
    assert all(v > 0 for v in ga.required_field_coverage(airport, 2020).values())


def test_station_without_dew_point_or_pressure_stops_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(ga, 'noaa_csv', lambda *a, **k: _noaa_fields_frame('+0251,1', '+9999,9', '99999,9', '999,9,9,0030,1', n=8760))
    with pytest.raises(ValueError, match='reports no dew point, no pressure'):
        ga.check_required_fields(None, {'station_id': '99999923802', 'name': 'THOMASVILLE 2 S'}, 2020, '.')
    monkeypatch.setattr(ga, 'noaa_csv', lambda *a, **k: _noaa_fields_frame('+0251,1', '+0150,1', '10132,1', '270,1,N,0046,1', n=8760))
    ga.check_required_fields(None, {'station_id': '72509014739'}, 2020, '.')       # no error
