"""Section 4.2, Fig. 2 and Table 4: timing and accuracy of hourly global horizontal irradiance
against USCRN ground measurements (2022-2023) for three alignments of the NSRDB data:
A, the method (HH:30 values at the hour midpoint); B, HH:00 snapshots moved to the midpoint
(the PVGIS procedure); C, direct assignment of the HH:00 snapshot to the hour that contains it."""
import io, json, time
import numpy as np, pandas as pd, requests
from common import HERE, work_dir, nsrdb_file, nsrdb_meta
import generate_amys as g

YEARS = (2022, 2023)
USCRN = 'https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02'


def fetch(work):
    """USCRN hourly files and the GHI of the nearest NSRDB cell, cached."""
    import fsspec
    cache = work / 'cache' / 'uscrn'; cache.mkdir(parents=True, exist_ok=True)
    cols = requests.get(f'{USCRN}/HEADERS.txt', timeout=60).text.splitlines()[1].split()
    fs = fsspec.filesystem('s3', anon=True)
    for y in YEARS:
        f = None
        for site in json.load(open(HERE / 'uscrn_sites.json')).values():
            if (cache / f'nsrdb_ghi_{site}_{y}.npz').exists():
                continue
            f = f or nsrdb_file(fs, y); mlat, mlon = nsrdb_meta(work, f)
            ti = pd.to_datetime(f['time_index'][:].astype(str)).tz_localize(None)
            txt = requests.get(f'{USCRN}/{y}/CRNH0203-{y}-{site}.txt', timeout=120).text
            u = pd.read_csv(io.StringIO(txt), sep=r'\s+', header=None, names=cols, dtype={'UTC_DATE': str, 'UTC_TIME': str})
            lat, lon = float(u.LATITUDE.iloc[0]), float(u.LONGITUDE.iloc[0])
            i = int(np.argmin((mlat - lat) ** 2 + ((mlon - lon) * np.cos(np.radians(lat))) ** 2))
            np.savez(cache / f'nsrdb_ghi_{site}_{y}.npz', ghi=f['ghi'][:, i].astype(float), time=ti.asi8, lat=mlat[i], lon=mlon[i])
            u.to_csv(cache / f'uscrn_{site}_{y}.csv', index=False)
            print(y, site, flush=True)
    return cache


def main():
    work = work_dir(); cache = fetch(work); rows = []; hourly = []
    for city, site in json.load(open(HERE / 'uscrn_sites.json')).items():
        for year in YEARS:
            z = np.load(cache / f'nsrdb_ghi_{site}_{year}.npz')
            ghi = pd.Series(z['ghi'], index=pd.to_datetime(z['time'])); lat, lon = float(z['lat']), float(z['lon'])
            u = pd.read_csv(cache / f'uscrn_{site}_{year}.csv', dtype={'UTC_DATE': str, 'UTC_TIME': str})
            end = pd.to_datetime(u.UTC_DATE + u.UTC_TIME.str.zfill(4), format='%Y%m%d%H%M')
            meas = pd.Series(np.where((u.SOLARAD > -99) & (u.SOLARAD_FLAG == 0), u.SOLARAD.clip(lower=0), np.nan),
                             index=end - pd.Timedelta(hours=1))                       # labelled by hour start (UTC)
            hours = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')
            A = ghi.reindex(hours + pd.Timedelta(minutes=30)).to_numpy()
            snap = ghi[ghi.index.minute == 0]
            C = snap.reindex(hours).to_numpy()
            B = g.pvgis_to_hour_midpoints(pd.DataFrame({'BEAM_H': snap.to_numpy(), 'DHI': 0.0}, index=snap.index),
                                          hours + pd.Timedelta(minutes=30), lat, lon)['GHI'].to_numpy()
            d = pd.DataFrame({'m': meas.reindex(hours).to_numpy(), 'A': A, 'B': B, 'C': C}, index=hours)
            d['site'] = site; d['year'] = year
            day = (d.m > 5) | (d.A > 5); ok = day & d[['m', 'A', 'B', 'C']].notna().all(axis=1)
            rec = {'site': site, 'city': city, 'year': year, 'hours': int(ok.sum()), 'mean_meas': d.m[ok].mean()}
            for k in 'ABC':
                e = d[k][ok] - d.m[ok]
                rec[f'{k}_mbe'] = e.mean(); rec[f'{k}_rmse'] = np.sqrt((e ** 2).mean()); rec[f'{k}_r'] = np.corrcoef(d[k][ok], d.m[ok])[0, 1]
            # timing: GHI centroid of clear days (top 30% of daily totals per month) minus the measured centroid
            loc = d.index + pd.Timedelta(hours=round(lon / 15))
            d['tt'] = (loc.hour + 0.5).to_numpy(); d['ld'] = loc.date
            good = [(dt, x.m.sum()) for dt, x in d.groupby('ld') if not x[['m', 'A', 'B', 'C']].isna().any().any() and x.m.sum() >= 1000]
            gd = pd.DataFrame(good, columns=['date', 'tot']); gd['mon'] = pd.to_datetime(gd.date).dt.month
            clear = set(gd[gd.tot >= gd.groupby('mon').tot.transform(lambda s: s.quantile(0.7))].date)
            for k in 'ABC':
                c = lambda x, v: (x.tt * v).sum() / v.sum()
                rec[f'{k}_timing_min'] = np.mean([(c(x, x[k]) - c(x, x.m)) * 60 for dt, x in d.groupby('ld') if dt in clear])
            rec['clear_days'] = len(clear); rows.append(rec)
            if site == 'VA_Charlottesville_2_SSE' and year == 2022:
                hourly.append(d[['m', 'A', 'C']])
    r = pd.DataFrame(rows)
    r.to_csv(work / 'results' / 'radiation_uscrn.csv', index=False)
    pd.concat(hourly).to_csv(work / 'results' / 'radiation_charlottesville_2022_hourly.csv')
    print(r[['city', 'year', 'A_timing_min', 'B_timing_min', 'C_timing_min']].round(1).to_string(index=False))
    print('mean |offset| of the station-years, method: %.1f min' % r.A_timing_min.abs().mean())


if __name__ == '__main__':
    main()
