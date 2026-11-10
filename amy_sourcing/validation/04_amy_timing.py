"""Section 4.2 (end-to-end check): clear-day irradiance centroid of the 2022 AMYs built by
02_build_amys_2022.py, relative to solar noon."""
import os, glob
import numpy as np, pandas as pd
from common import work_dir
from rmy.utils import read_epw_file


def main():
    work = work_dir(); rows = []
    for f in sorted(glob.glob(str(work / 'amy_2022' / '**' / '*_2022_AMY.epw'), recursive=True)):
        loc = open(f).readline().split(','); lon, tz = float(loc[7]), float(loc[8])
        e = read_epw_file(f)
        doy = pd.to_datetime(dict(year=e.year, month=e.month, day=e.day)).dt.dayofyear.to_numpy()
        B = 2 * np.pi * (doy - 81) / 364; eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
        e['noon'] = 12 - (lon - 15 * tz) / 15 - eot / 60; e['t'] = e.hour - 0.5; e['doy'] = doy
        tot = e.groupby('doy').dni.sum()
        clear = tot[tot >= tot.groupby((tot.index - 1) // 30).transform(lambda s: s.quantile(0.7))].index
        c = [((q.t * q.ghi).sum() / q.ghi.sum() - q.noon.iloc[0]) * 60 for _, q in e[e.doy.isin(clear)].groupby('doy')]
        ce = [((q.t * q.etr).sum() / q.etr.sum() - q.noon.iloc[0]) * 60 for _, q in e.groupby('doy')]
        rows.append({'file': os.path.basename(f), 'ghi_centroid_min': np.mean(c), 'etr_centroid_min': np.mean(ce)})
    r = pd.DataFrame(rows); r.to_csv(work / 'results' / 'amy_timing.csv', index=False); print(r.round(2).to_string())


if __name__ == '__main__':
    main()
