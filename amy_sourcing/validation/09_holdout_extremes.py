"""Section 4.3 (Extremes): gaps of 24, 48 and 72 h centred on the hottest and on the coldest hour
of January, April, July and October 2022 at the twenty stations; does the filled series keep the
peak? ERA5 diurnal-bias filling against linear interpolation."""
import io, contextlib
import numpy as np, pandas as pd, requests
from common import work_dir, stations, catalog, station_row
import generate_amys as g

MONTHS = ['202201', '202204', '202207', '202210']


def main():
    import fsspec
    work = work_dir(); cat = catalog(work); sess = requests.Session(); rows = []
    fs = fsspec.filesystem('s3', anon=True)
    for x in stations():
        sid, tz = x['id'], x['tz']; r = station_row(cat, sid); lat, lon = float(r.LAT), float(r.LON)
        g._era5_prefetch([(ym, '2t', lat, lon, str(work / 'cache')) for ym in MONTHS])
        with contextlib.redirect_stdout(io.StringIO()):
            raw = g.required_surface(sess, sid, 2022, str(work / 'cache'), tz)
        obs = raw.DB.astype(float).to_numpy()
        er = pd.concat([g.era5_point_month(fs, ym, '2t', lat, lon, work / 'cache') for ym in MONTHS]); er = er[~er.index.duplicated()]
        cands = {lag: er.reindex(raw.index - pd.Timedelta(hours=tz) + pd.Timedelta(hours=lag)).to_numpy() for lag in (0, 1)}
        lag = min(cands, key=lambda k: np.nanmean(np.abs(cands[k] - obs))); e = cands[lag]
        hod, mon = raw.index.hour.to_numpy(), raw.index.month.to_numpy()
        for m in (1, 4, 7, 10):
            idx_m = np.flatnonzero((mon == m) & np.isfinite(obs) & np.isfinite(e))
            for kind in ('hot', 'cold'):
                c = idx_m[np.nanargmax(obs[idx_m])] if kind == 'hot' else idx_m[np.nanargmin(obs[idx_m])]
                for L in (24, 48, 72):
                    s0 = c - L // 2; s1 = s0 + L
                    if s0 < 1 or s1 + 1 > len(obs) or not np.isfinite(obs[s0 - 1:s1 + 1]).all() or not np.isfinite(e[s0:s1]).all():
                        continue
                    y = obs.copy(); y[s0:s1] = np.nan
                    lin = g.caltech_linear_fill(pd.Series(y)).to_numpy()
                    fill = e + pd.Series(y - e).groupby([mon, hod]).transform('mean').to_numpy()
                    pk = (lambda a: np.max(a[s0:s1])) if kind == 'hot' else (lambda a: np.min(a[s0:s1]))
                    for name, v in (('linear', lin), ('era5', fill)):
                        rows.append({'city': x['city'], 'region': x['region'], 'kind': kind, 'gap_h': L, 'method': name,
                                     'peak_err': pk(v) - pk(obs), 'mae': np.mean(np.abs(v[s0:s1] - obs[s0:s1]))})
        print(x['city'], flush=True)
    d = pd.DataFrame(rows); d.to_csv(work / 'results' / 'holdout_extremes.csv', index=False)
    print(d.groupby(['kind', 'gap_h', 'method']).agg(n=('mae', 'size'), mae=('mae', 'mean'), peak_bias=('peak_err', 'mean')).round(2).to_string())


if __name__ == '__main__':
    main()
