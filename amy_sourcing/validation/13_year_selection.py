"""Section 4.1 and Table 3: the method's year selection on the NOAA observations of each station
(fifteen most recent complete years, with ERA5 filling; years with a remaining temperature or
dew-point gap > 24 h, pressure gap > 72 h or wind outage >= 48 h replaced), and the ASHRAE
design conditions of the selected years (input to 14_design_conditions.py). Gaps are measured
after the ERA5 filling and before interpolation, as in the method."""
import io, contextlib, functools
import pandas as pd, requests
from common import work_dir, stations, catalog, station_row
import generate_amys as g
import generate_tmy_from_amys as tmy


def main():
    work = work_dir(); cat = catalog(work); out = work / 'years'; out.mkdir(exist_ok=True)
    sess = requests.Session(); cache = str(work / 'cache')
    extract = functools.lru_cache(maxsize=4)(lambda sid, y: g.extract_hourly(g.noaa_csv(sess, sid, y, cache)))

    def surface(sid, year, tz):
        target = pd.date_range(f'{year}-01-01', f'{year + 1}-01-01', freq='h', inclusive='left'); parts = []
        for y in [year] + ([year + 1] if tz < 0 else [year - 1] if tz > 0 else []):
            try:
                parts.append(extract(sid, y))
            except Exception:
                if y == year:
                    raise
        u = pd.concat(parts).sort_index(); u = u[~u.index.duplicated(keep='first')]
        u.index = u.index + pd.to_timedelta(tz, unit='h'); return u.reindex(target)

    for x in stations():
        sid, tz = x['id'], x['tz']; r = station_row(cat, sid); elev = float(r['ELEV(M)'])
        end = pd.to_datetime(r.END, format='%Y%m%d'); last = end.year if (end.month, end.day) == (12, 31) else end.year - 1
        last = min(last, 2024 if x['region'] == 'US' else 2023)          # last year of NSRDB / PVGIS used in the paper
        rows = []; hours = []
        for year in range(last, last - 20, -1):
            if sum(q['used'] for q in rows) == 15:
                break
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    raw = surface(sid, year, tz)
            except Exception:
                rows.append({'year': year, 'used': False, 'reason': 'no data'}); continue
            obs_gaps = {v: g.max_missing_run(raw[v]) for v in ('DB', 'DP', 'PR')}
            with contextlib.redirect_stdout(io.StringIO()):
                raw, er = g.era5_infill(raw, {'lat': float(r.LAT), 'lon': float(r.LON), 'tz_offset': tz}, cache)
            gaps = {v: g.max_missing_run(raw[v]) for v in ('DB', 'DP', 'PR')}
            if raw.DB.notna().sum() < 100:
                rows.append({'year': year, 'used': False, 'reason': 'no data'}); continue
            for v in ('DB', 'DP', 'PR'):
                raw[v] = g.caltech_linear_fill(raw[v])
            with contextlib.redirect_stdout(io.StringIO()):
                raw, wp = g.repair_wind_direction(raw, 'CIRCULAR_LINEAR', 2)
            wind_long = max(wp['wind_speed_max_original_missing_run_hours'], wp['wind_direction_max_outage_run_hours'])
            wind_missing = wp['wind_speed_remaining_missing_hours'] + wp['wind_direction_remaining_missing_hours']
            reason = []
            if gaps['DB'] > 24 or gaps['DP'] > 24:
                reason.append(f"T/Td gap {max(gaps['DB'], gaps['DP'])} h")
            if gaps['PR'] > 72:
                reason.append(f"pressure gap {gaps['PR']} h")
            if wind_long >= 48 or wind_missing > 0:
                reason.append(f'wind outage {wind_long} h')
            rows.append({'year': year, 'used': not reason, 'reason': '; '.join(reason), 'db_gap': gaps['DB'], 'dp_gap': gaps['DP'],
                         'pr_gap': gaps['PR'], 'wind_outage': wind_long, 'db_gap_observed': obs_gaps['DB'],
                         'era5_status': er.get('era5_infill_status', ''), 'era5_db_hours': er.get('era5_infill_DB_hours', 0),
                         'era5_skipped': ';'.join(f'{k}:{v}' for k, v in er.items() if k.endswith('_skipped'))})
            if reason:
                continue
            p, _ = g.rh_and_station_pressure(raw.DB, raw.DP, raw.PR, elev)
            hours.append(pd.DataFrame({'year': year, 'month': raw.index.month, 'day': raw.index.day, 'hour': raw.index.hour + 1,
                                       'db': raw.DB.to_numpy(), 'dp': raw.DP.to_numpy(), 'p': p, 'ws': raw.WS.to_numpy(), 'wd': raw.WD.to_numpy()}))
        pd.DataFrame(rows).to_csv(out / f'{sid}_years.csv', index=False)
        if hours:
            (out / f'{sid}_design.txt').write_text(tmy.design_conditions_line(pd.concat(hours, ignore_index=True)))
        print(sid, x['city'], 'used', sum(q['used'] for q in rows), 'of', len(rows), flush=True)


if __name__ == '__main__':
    main()
