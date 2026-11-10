"""Section 4.3, Table 5 and Fig. 3a-c: interpolation errors for gaps of 1-24 h, from real 2022
observations hidden at up to 150 non-overlapping positions per gap length and station."""
import io, contextlib
import numpy as np, pandas as pd, requests
from common import work_dir, stations
import generate_amys as g

rng = np.random.default_rng(42)


def windows(ok, L, k=150):
    """Up to k non-overlapping starts where L hours and one neighbour on each side are observed."""
    good = np.convolve(ok.astype(int), np.ones(L + 2, int), 'valid') == L + 2
    starts = np.flatnonzero(good) + 1; rng.shuffle(starts); chosen = []; used = np.zeros(len(ok), bool)
    for s in starts:
        if not used[s - 1:s + L + 1].any():
            chosen.append(s); used[s - 1:s + L + 1] = True
        if len(chosen) == k:
            break
    return chosen


def main():
    work = work_dir(); sess = requests.Session(); rows = []
    for x in stations():
        sid, tz = x['id'], x['tz']
        with contextlib.redirect_stdout(io.StringIO()):
            raw = g.required_surface(sess, sid, 2022, str(work / 'cache'), tz)
        ws = pd.to_numeric(raw.WS, errors='coerce'); wd = pd.to_numeric(raw.WD, errors='coerce')
        wd_p = wd.copy(); wd_p[wd.isna() & ws.eq(0)] = 0.0                       # calm: 0 deg, as in the method
        for var in ('DB', 'DP', 'PR', 'WS', 'WD'):
            s = {'DB': raw.DB, 'DP': raw.DP, 'PR': raw.PR, 'WS': ws, 'WD': wd_p}[var].astype(float)
            ok = s.notna().to_numpy().copy()
            if var == 'WD':
                ok &= (ws > 0).to_numpy()
            for L in (1, 2, 3, 4, 6, 9, 12, 18, 24):
                st = windows(ok, L)
                if not st:
                    continue
                y = s.copy(); idx = np.concatenate([np.arange(a, a + L) for a in st]); y.iloc[idx] = np.nan
                f = g.caltech_linear_fill(y) if var in ('DB', 'DP', 'PR') else g._wind_fill_all_hours(y, angular=(var == 'WD'))[0]
                e = f.to_numpy()[idx] - s.to_numpy()[idx]
                if var == 'WD':
                    e = (e + 180) % 360 - 180
                rows.append({'station': sid, 'region': x['region'], 'var': var, 'gap_h': L, 'n': len(idx),
                             'mae': np.mean(np.abs(e)), 'rmse': np.sqrt(np.mean(e ** 2)), 'bias': np.mean(e),
                             'p95': np.percentile(np.abs(e), 95)})               # PR in hPa
        print(x['city'], flush=True)
    pd.DataFrame(rows).to_csv(work / 'results' / 'holdout_interpolation.csv', index=False)


if __name__ == '__main__':
    main()
