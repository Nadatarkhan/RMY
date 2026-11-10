"""Section 4.4 and Table 8: horizontal infrared radiation of the method (sky cover filled, then the
Clark-Allen/Walton correlation) for 2022 at the twenty stations, against the ERA5 mean surface
downward longwave flux (hour-ending, UTC) of the nearest grid cell. Needs the NSRDB cloud type
of 01_nsrdb_cache.py for the U.S. stations. The ERA5 download reads whole monthly fields (slow)."""
import io, os, contextlib
import numpy as np, pandas as pd, requests
from common import work_dir, stations, catalog, station_row
import generate_amys as g


def era5_longwave(work, cat):
    """ERA5 mean surface downward longwave (W/m2) at each station, cached per station."""
    import fsspec, h5py
    todo = [x for x in stations() if not (work / 'cache' / f'era5_lw_{x["id"]}.csv').exists()]
    if not todo:
        return
    st = {x['id']: (float(station_row(cat, x['id']).LAT), float(station_row(cat, x['id']).LON)) for x in todo}
    fs = fsspec.filesystem('s3', anon=True); keys = []
    for ym in ['202112'] + [f'2022{m:02d}' for m in range(1, 13)] + ['202301']:
        keys += [k for k in fs.ls(f'nsf-ncar-era5/e5.oper.fc.sfc.meanflux/{ym}/') if '_msdwlwrf.' in k]
    keys = [k for k in keys if not (k.split('/')[-2] == '202112' and '2021120106' in k) and not (k.split('/')[-2] == '202301' and '2023011606' in k)]
    out = {i: [] for i in st}; tmp = work / 'cache' / '_era5_tmp.nc'
    for n, k in enumerate(keys):
        fs.get(k, str(tmp))
        with h5py.File(tmp, 'r') as f:
            lat, lon = f['latitude'][:], f['longitude'][:]
            ij = {i: (int(np.argmin(abs(lat - a))), int(np.argmin(abs(((lon - (b % 360)) + 180) % 360 - 180)))) for i, (a, b) in st.items()}
            init, fh, d = f['forecast_initial_time'][:], f['forecast_hour'][:], f['MSDWLWRF']
            for a in range(len(init)):
                fld = d[a]
                for i, (y, xx) in ij.items():
                    out[i] += [(int(init[a]) + int(fh[h]), float(fld[h, y, xx])) for h in range(len(fh))]
        os.remove(tmp); print(f'ERA5 longwave {n + 1}/{len(keys)}', flush=True)
    for i in st:
        s = pd.DataFrame(out[i], columns=['hours1900', 'lw']).drop_duplicates('hours1900').sort_values('hours1900')
        s['utc'] = pd.Timestamp('1900-01-01') + pd.to_timedelta(s.hours1900, unit='h')
        s[['utc', 'lw']].to_csv(work / 'cache' / f'era5_lw_{i}.csv', index=False)


def main():
    work = work_dir(); cat = catalog(work); era5_longwave(work, cat); sess = requests.Session(); rows = []
    for x in stations():
        sid, tz = x['id'], x['tz']
        with contextlib.redirect_stdout(io.StringIO()):
            raw = g.required_surface(sess, sid, 2022, str(work / 'cache'), tz)
            for v in ('DB', 'DP', 'PR'):
                raw[v] = g.caltech_linear_fill(raw[v])
            raw, _ = g.repair_wind_direction(raw, 'CIRCULAR_LINEAR', 2); raw, _ = g.repair_opaque_sky(raw, 'TOTAL_COVER_PROXY')
        f = work / 'cache' / 'radiation' / 'nsrdb' / sid / '2022_local_extended_v4.csv'
        ct = pd.read_csv(f, skiprows=2)['Cloud Type'].to_numpy(float) if (x['region'] == 'US' and f.exists()) else np.full(len(raw), np.nan)
        t_, o_, _, _ = g.infill_sky_cover(raw.TC.to_numpy(float), raw.OC.to_numpy(float), ct)
        ir, _ = g.energyplus_infrared(raw.DB, raw.DP, pd.Series(o_, index=raw.index), 'DEFAULT_5')
        era = pd.read_csv(work / 'cache' / f'era5_lw_{sid}.csv', parse_dates=['utc']).set_index('utc').lw
        end_utc = raw.index + pd.Timedelta(hours=1) - pd.Timedelta(hours=tz)
        d = pd.DataFrame({'amy': np.asarray(ir, float), 'era5': era.reindex(end_utc).to_numpy(), 'cover': t_}).dropna(subset=['amy', 'era5'])
        e = d.amy - d.era5
        rows.append({'city': x['city'], 'region': x['region'], 'n': len(d), 'mean_era5': d.era5.mean(), 'mbe': e.mean(),
                     'rmse': np.sqrt((e ** 2).mean()), 'r': np.corrcoef(d.amy, d.era5)[0, 1],
                     'mbe_clear': e[d.cover <= 2].mean(), 'mbe_overcast': e[d.cover >= 8].mean()})
        print(x['city'], round(e.mean(), 1), flush=True)
    pd.DataFrame(rows).to_csv(work / 'results' / 'longwave_era5.csv', index=False)


if __name__ == '__main__':
    main()
