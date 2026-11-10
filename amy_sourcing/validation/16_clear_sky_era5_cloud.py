"""Section 4.6 (sensitivity): ERA5 total, high and low/middle cloud cover at the hours the method
records as clear from CAVOK/NSC or SKC/CLR/NCD reports, at Cairo, Dubai and Beijing (2022), and
the largest effect on the computed longwave radiation (all ERA5 cloud counted as opaque)."""
import io, contextlib, multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, requests
from common import work_dir, catalog, station_row
import generate_amys as g

Y = 2022
g.ERA5_FIELDS.update({'tcc': ('128_164_tcc', 'TCC', 0.0, 10.0), 'hcc': ('128_188_hcc', 'HCC', 0.0, 10.0),
                      'lcc': ('128_186_lcc', 'LCC', 0.0, 10.0), 'mcc': ('128_187_mcc', 'MCC', 0.0, 10.0)})   # in tenths
STATIONS = {'Cairo': ('62366099999', 2), 'Dubai': ('41194099999', 4), 'Beijing': ('54511099999', 8)}
FIELDS = ('tcc', 'hcc', 'lcc', 'mcc')


def walton(n):
    return 1 + 0.0224 * n - 0.0035 * n ** 2 + 0.00028 * n ** 3


def main():
    import fsspec
    work = work_dir(); cat = catalog(work); sess = requests.Session(); rows = []
    months = [f'{Y}{m:02d}' for m in range(1, 13)]; cache = str(work / 'cache')
    for city, (sid, tz) in STATIONS.items():
        r = station_row(cat, sid); lat, lon = float(r.LAT), float(r.LON)
        with contextlib.redirect_stdout(io.StringIO()):
            raw = g.required_surface(sess, sid, Y, cache, tz)
        csv = pd.read_csv(work / 'cache' / 'noaa_s3' / sid / f'{Y}.csv', dtype=str, keep_default_na=False,
                          usecols=lambda c: c in ('DATE', 'REPORT_TYPE', 'REM'))
        csv = csv[csv.REPORT_TYPE.str.strip() == 'FM-15']
        body = csv.REM.str.split(r'\b(?:BECMG|TEMPO|NOSIG)\b', n=1, regex=True).str[0]   # main body, without trend forecasts
        kind = np.where(body.str.contains(r'\b(?:CAVOK|NSC)\b', regex=True), 'CAVOK/NSC',
                        np.where(body.str.contains(r'\b(?:SKC|CLR|NCD)\b', regex=True), 'SKC/CLR/NCD', ''))
        k = pd.Series(kind, index=pd.to_datetime(csv.DATE).dt.floor('h') + pd.Timedelta(hours=tz))
        k = k[k != ''].groupby(level=0).first()
        clear = raw.index[raw.TC == 0]
        jobs = [(ym, f, lat, lon, cache) for f in FIELDS for ym in months]
        with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context('fork')) as ex:   # fork keeps the added fields
            list(ex.map(g._era5_fetch_one, jobs))
        fs = fsspec.filesystem('s3', anon=True)
        era = {}
        for f in FIELDS:
            s = pd.concat([g.era5_point_month(fs, ym, f, lat, lon, cache) for ym in months]); era[f] = s[~s.index.duplicated()]
        at = lambda f: era[f].reindex((clear - pd.Timedelta(hours=tz) + pd.Timedelta(minutes=30)).floor('h')).to_numpy()
        d = pd.DataFrame({'kind': k.reindex(clear).fillna('cloud groups = 0').to_numpy(), 'tcc': at('tcc'), 'hcc': at('hcc'),
                          'low_mid': np.maximum(at('lcc'), at('mcc'))})
        for name, q in list(d.groupby('kind')) + [('all hours recorded as clear', d)]:
            rows.append({'city': city, 'kind': name, 'hours': len(q), 'share_of_year_%': 100 * len(q) / len(raw),
                         'era5_total_cover_mean': q.tcc.mean(), 'era5_low_mid_cover_mean': q.low_mid.mean(),
                         'era5_high_cover_mean': q.hcc.mean(), 'share_total_cover_over_5_%': 100 * (q.tcc > 5).mean(),
                         'longwave_increase_if_era5_cover_opaque_%': 100 * (walton(q.tcc.clip(0, 10)).mean() - 1)})
        print(city, flush=True)
    out = pd.DataFrame(rows); out.to_csv(work / 'results' / 'clear_sky_era5_cloud.csv', index=False)
    print(out.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
