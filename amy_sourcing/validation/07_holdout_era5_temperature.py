"""Section 4.3 and Table 6 (temperature, dew point): gaps of 6-72 h hidden in January, April,
July and October 2022 at the twenty stations, filled by linear interpolation and by ERA5
corrected for the station's mean difference by month and hour of day."""
import io, contextlib
import numpy as np, pandas as pd, requests
from common import work_dir, stations, catalog, station_row
import generate_amys as g

MONTHS = ['202201', '202204', '202207', '202210']
rng = np.random.default_rng(5)


def era5_series(work, field, lat, lon):
    import fsspec
    fs = fsspec.filesystem('s3', anon=True)
    s = pd.concat([g.era5_point_month(fs, ym, field, lat, lon, work / 'cache') for ym in MONTHS])
    return s[~s.index.duplicated()]


def windows(ok, L, k=60):
    good = np.convolve(ok.astype(int), np.ones(L + 2, int), 'valid') == L + 2
    st = np.flatnonzero(good) + 1; rng.shuffle(st); ch = []; used = np.zeros(len(ok), bool)
    for s in st:
        if not used[s - 1:s + L + 1].any():
            ch.append(s); used[s - 1:s + L + 1] = True
        if len(ch) == k:
            break
    return ch


def main():
    work = work_dir(); cat = catalog(work); sess = requests.Session(); rows = []
    for x in stations():
        sid, tz = x['id'], x['tz']; r = station_row(cat, sid); lat, lon = float(r.LAT), float(r.LON)
        g._era5_prefetch([(ym, f, lat, lon, str(work / 'cache')) for f in ('2t', '2d') for ym in MONTHS])
        with contextlib.redirect_stdout(io.StringIO()):
            raw = g.required_surface(sess, sid, 2022, str(work / 'cache'), tz)
        hod, mon = raw.index.hour.to_numpy(), raw.index.month.to_numpy()
        for field, col in (('2t', 'DB'), ('2d', 'DP')):
            obs = raw[col].astype(float).to_numpy(); er = era5_series(work, field, lat, lon)
            cands = {lag: er.reindex(raw.index - pd.Timedelta(hours=tz) + pd.Timedelta(hours=lag)).to_numpy() for lag in (0, 1)}
            lag = min(cands, key=lambda k: np.nanmean(np.abs(cands[k] - obs))); e = cands[lag]
            ok = np.isfinite(obs) & np.isfinite(e)
            for L in (6, 12, 24, 48, 72):
                st = windows(ok, L)
                if not st:
                    continue
                idx = np.concatenate([np.arange(s, s + L) for s in st]); y = obs.copy(); y[idx] = np.nan
                lin = g.caltech_linear_fill(pd.Series(y)).to_numpy()
                bias = pd.Series(y - e).groupby([mon, hod]).transform('mean').to_numpy()   # hidden hours excluded
                for name, v in (('linear', lin), ('era5_diurnal_bias', e + bias)):
                    err = v[idx] - obs[idx]
                    rows.append({'city': x['city'], 'region': x['region'], 'var': col, 'gap_h': L, 'method': name, 'n': len(idx),
                                 'mae': np.nanmean(np.abs(err)), 'bias': np.nanmean(err), 'lag': lag})
        print(x['city'], flush=True)
    pd.DataFrame(rows).to_csv(work / 'results' / 'holdout_era5_temperature.csv', index=False)


if __name__ == '__main__':
    main()
