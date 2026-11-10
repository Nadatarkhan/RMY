"""Section 4.3 and Table 6 (pressure, wind): gaps of 24, 48 and 72 h hidden in January and July
2022 at six stations, filled by the method's own era5_infill and by interpolation."""
import io, contextlib
import numpy as np, pandas as pd, requests
from common import work_dir, stations, catalog, station_row
import generate_amys as g

STATIONS = ['72509014739', '72278023183', '72793024233', '03772099999', '08221099999', '47671099999']
VARS = ['PR', 'WS', 'WD']
rng = np.random.default_rng(11)


def circ_interp(wd):
    s = pd.Series(wd); ok = s.notna(); r = np.radians(s)
    x = pd.Series(np.cos(r)).interpolate(limit_area='inside'); y = pd.Series(np.sin(r)).interpolate(limit_area='inside')
    return np.where(ok, s, np.mod(np.degrees(np.arctan2(y, x)), 360))


def main():
    work = work_dir(); cat = catalog(work); sess = requests.Session(); rows = []
    st_all = {x['id']: x for x in stations()}
    for sid in STATIONS:
        x = st_all[sid]; tz = x['tz']; r = station_row(cat, sid)
        st = {'lat': float(r.LAT), 'lon': float(r.LON), 'tz_offset': tz}
        with contextlib.redirect_stdout(io.StringIO()):
            raw = g.required_surface(sess, sid, 2022, str(work / 'cache'), tz)
        base = raw[['DB', 'DP'] + VARS].astype(float).copy()
        base.loc[base.WS.eq(0) | base.WS.isna(), 'WD'] = np.nan                 # calm: no direction
        ok = base[['PR', 'WS']].notna().all(axis=1).to_numpy(); mon = base.index.month.to_numpy()
        for L in (24, 48, 72):
            good = np.convolve(ok.astype(int), np.ones(L + 2, int), 'valid') == L + 2
            cand = [s + 1 for s in np.flatnonzero(good) if mon[s + 1] in (1, 7) and mon[s + L] in (1, 7)]
            rng.shuffle(cand); used = np.zeros(len(ok), bool); win = []
            for s in cand:
                if not used[max(s - L, 0):s + 2 * L].any():
                    win.append(s); used[s - 1:s + L + 1] = True
                if len(win) == 6:
                    break
            if not win:
                continue
            idx = np.concatenate([np.arange(s, s + L) for s in win]); test = base.copy()
            test.iloc[idx, [test.columns.get_loc(v) for v in VARS]] = np.nan
            with contextlib.redirect_stdout(io.StringIO()):
                filled, rep = g.era5_infill(test, st, str(work / 'cache'), max_share=0.5)
            for v in VARS:
                o = base[v].to_numpy()[idx]
                if v == 'WD':
                    lin = circ_interp(test.WD.to_numpy())[idx]; era = filled.WD.to_numpy()[idx]
                    err = lambda e: np.abs((e - o + 180) % 360 - 180)
                else:
                    lin = test[v].interpolate(limit_area='inside').to_numpy()[idx]; era = filled[v].to_numpy()[idx]
                    err = lambda e: np.abs(e - o)
                m = np.isfinite(o)
                rows.append({'city': x['city'], 'region': x['region'], 'var': v, 'gap_h': L, 'n': int(m.sum()),
                             'mae_era5': np.nanmean(err(era)[m]), 'mae_linear': np.nanmean(err(lin)[m]),
                             'bias_era5': np.nanmean((era - o)[m]) if v != 'WD' else np.nan})
        print(x['city'], flush=True)
    pd.DataFrame(rows).to_csv(work / 'results' / 'holdout_era5_pressure_wind.csv', index=False)


if __name__ == '__main__':
    main()
