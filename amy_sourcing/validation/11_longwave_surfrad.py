"""Section 4.4 and Table 7: horizontal infrared radiation computed as in the method from the
nearest airport, against SURFRAD pyrgeometer measurements (2023, every third day). Also
reports the Dilley-O'Brien clear-sky alternative."""
import io, contextlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from common import work_dir
import generate_amys as g

BASE = 'https://gml.noaa.gov/aftp/data/radiation/surfrad'
SITES = {  # SURFRAD folder, file code, nearest NOAA airport, UTC offset
    'Bondville_IL': ('bon', '72531594870', -6), 'Boulder_CO': ('tbl', '72053300160', -7),
    'Desert_Rock_NV': ('dra', '72387003160', -8), 'Fort_Peck_MT': ('fpk', '72768694017', -7),
    'Goodwin_Creek_MS': ('gwn', '72054153806', -6), 'Penn_State_PA': ('psu', '72512854739', -5),
    'Sioux_Falls_SD': ('sxf', '72651014944', -6)}
sess = requests.Session()


def day(site, code, j):
    """Hourly means of downwelling longwave (QC 0, >= 45 good minutes) for one daily file."""
    for _ in range(3):
        try:
            r = sess.get(f'{BASE}/{site}/2023/{code}23{j:03d}.dat', timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status(); break
        except Exception:
            continue
    else:
        return None
    rows = [l.split() for l in r.text.splitlines()[2:]]
    a = np.array([[float(x) for x in (q[0], q[2], q[3], q[4], q[5], q[16], q[17])] for q in rows if len(q) >= 18])
    d = pd.DataFrame(a, columns=['y', 'm', 'd', 'H', 'M', 'lw', 'qc'])
    d = d[(d.qc == 0) & (d.lw > 50) & (d.lw < 700)]
    t = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.H))
    q = d.groupby(t).lw.agg(['mean', 'count'])
    return q[q['count'] >= 45]['mean']


def main():
    work = work_dir(); out = []; allh = []
    for site, (code, sid, tz) in SITES.items():
        f = work / 'cache' / 'surfrad' / f'{site}_2023_hourly.csv'; f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            with ThreadPoolExecutor(12) as ex:
                parts = [p for p in ex.map(lambda j: day(site, code, j), range(1, 366, 3)) if p is not None]
            s = pd.concat(parts).sort_index(); s.index.name = 'utc_start'; s.rename('lw').to_csv(f)
        with contextlib.redirect_stdout(io.StringIO()):
            raw = g.required_surface(sess, sid, 2023, str(work / 'cache'), tz)
            for v in ('DB', 'DP', 'PR'):
                raw[v] = g.caltech_linear_fill(raw[v])
            raw, _ = g.repair_opaque_sky(raw, 'TOTAL_COVER_PROXY')
        _, oc, _, _ = g.infill_sky_cover(raw.TC.to_numpy(float), raw.OC.to_numpy(float), np.full(len(raw), np.nan))
        db, dp, cv = raw.DB.reset_index(drop=True), raw.DP.reset_index(drop=True), pd.Series(oc)
        model = {k: np.asarray(g.energyplus_infrared(db, dp, cv, clear_sky=k)[0], float) for k in ('CLARK_ALLEN', 'DILLEY_OBRIEN')}
        meas = pd.read_csv(f, parse_dates=['utc_start']).set_index('utc_start').lw.reindex(raw.index - pd.Timedelta(hours=tz)).to_numpy()
        allh.append(pd.DataFrame({'site': site, 'meas': meas, 'cover': oc, **model}).dropna())
        print(site, flush=True)
    for d in allh + [pd.concat(allh).assign(site='ALL 7 SITES')]:
        r = {'site': d.site.iloc[0], 'hours': len(d), 'measured mean': d.meas.mean()}
        for k, lab in (('CLARK_ALLEN', 'method'), ('DILLEY_OBRIEN', 'dilley_obrien')):
            e = d[k] - d.meas
            r.update({f'{lab} bias': e.mean(), f'{lab} RMSE': np.sqrt((e ** 2).mean()),
                      f'{lab} bias clear': e[d.cover <= 2].mean(), f'{lab} bias overcast': e[d.cover >= 8].mean()})
        out.append(r)
    r = pd.DataFrame(out); r.to_csv(work / 'results' / 'longwave_surfrad.csv', index=False); print(r.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
