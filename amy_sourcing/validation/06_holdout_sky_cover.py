"""Section 4.3 and Fig. 3d: total sky cover hidden in gaps of 2-47 h at the ten U.S. stations
(2022), filled by the method, by linear interpolation only and by the cloud-type estimate only.
Needs the NSRDB cloud type cached by 01_nsrdb_cache.py."""
import io, contextlib
import numpy as np, pandas as pd, requests
from common import work_dir, stations
import generate_amys as g

rng = np.random.default_rng(7)


def windows(ok, L, k=120):
    good = np.convolve(ok.astype(int), np.ones(L + 2, int), 'valid') == L + 2
    st = np.flatnonzero(good) + 1; rng.shuffle(st); ch = []; used = np.zeros(len(ok), bool)
    for s in st:
        if not used[s - 1:s + L + 1].any():
            ch.append(s); used[s - 1:s + L + 1] = True
        if len(ch) == k:
            break
    return ch


def main():
    work = work_dir(); sess = requests.Session(); rows = []
    for x in stations('US'):
        sid, tz = x['id'], x['tz']
        f = work / 'cache' / 'radiation' / 'nsrdb' / sid / '2022_local_extended_v4.csv'
        ct = pd.read_csv(f, skiprows=2)['Cloud Type'].to_numpy(dtype=float)
        with contextlib.redirect_stdout(io.StringIO()):
            raw = g.required_surface(sess, sid, 2022, str(work / 'cache'), tz)
        tc = raw.TC.to_numpy(dtype=float); ok = np.isfinite(tc) & np.isfinite(ct)
        for L in (2, 4, 6, 8, 12, 18, 24, 36, 47):
            st = windows(ok, L)
            if not st:
                continue
            idx = np.concatenate([np.arange(s, s + L) for s in st]); y = tc.copy(); y[idx] = np.nan
            method, _, _ = g._fill_cover(y, ct)
            lin = pd.Series(y).interpolate(limit_area='inside').to_numpy()
            mp = g._cloud_type_map(y, ct); est = np.clip(np.rint(np.array([mp.get(int(c), np.nan) for c in ct])), 0, 10)
            for name, v in (('method', method), ('linear', np.clip(np.rint(lin), 0, 10)), ('estimate', est)):
                e = v[idx] - tc[idx]; e = e[np.isfinite(e)]
                rows.append({'station': sid, 'gap_h': L, 'method': name, 'n': len(e), 'mae': np.mean(np.abs(e)), 'bias': np.mean(e)})
        print(x['city'], flush=True)
    pd.DataFrame(rows).to_csv(work / 'results' / 'holdout_sky_cover.csv', index=False)


if __name__ == '__main__':
    main()
