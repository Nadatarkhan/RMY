"""Sections 4.1 and 4.5, Tables 3 and 9, Fig. 4: years selected per station, ERA5 use per
station-year, and the design conditions of the selected years against ASHRAE 2025 (from the TMYx
files in data/cities or, for the other stations, ashrae_design_lines.json). Run after
13_year_selection.py."""
import glob, json
import numpy as np, pandas as pd
from common import HERE, REPO, work_dir, stations

REPO_TMY = {'72509014739': 'Boston', '72530094846': 'Chicago', '72403093738': 'Dulles_Washington', '72367723054': 'Las_Vegas',
            '72295023174': 'Los_Angeles', '72278023183': 'Phoenix', '72494023234': 'San_Francisco', '72793024233': 'Seattle'}
ITEMS = {  # name: (section, index) in the EPW DESIGN CONDITIONS line
    'Heating dry bulb, 99.6%': ('h', 1), 'Heating dry bulb, 99%': ('h', 2), 'Humidification dew point, 99.6%': ('h', 3),
    'Cooling dry bulb, 0.4%': ('c', 2), 'Mean coincident wet bulb, 0.4%': ('c', 3), 'Cooling dry bulb, 1%': ('c', 4),
    'Evaporation wet bulb, 0.4%': ('c', 8), 'Dehumidification dew point, 0.4%': ('c', 16), 'Enthalpy, 0.4% (kJ/kg)': ('c', 25),
    'Daily range, hottest month': ('c', 1), 'Mean annual maximum dry bulb': ('e', 5), 'Mean annual minimum dry bulb': ('e', 4),
    '50-year maximum dry bulb': ('e', 15), '50-year minimum dry bulb': ('e', 14), 'Extreme wind speed, 1% (m/s)': ('e', 0)}


def parse(line):
    f = line.strip().split(','); i, j, k = f.index('Heating'), f.index('Cooling'), f.index('Extremes')
    return f[i + 1:j], f[j + 1:k], f[k + 1:]


def main():
    work = work_dir(); res = work / 'results'; ref_lines = json.load(open(HERE / 'ashrae_design_lines.json'))
    rows, years, e5 = [], [], []
    for x in stations():
        sid = x['id']; y = pd.read_csv(work / 'years' / f'{sid}_years.csv'); used = y[y.used]
        years.append({'city': x['city'], 'region': x['region'], 'tried': len(y), 'used': len(used),
                      'first': int(used.year.min()) if len(used) else None, 'last': int(used.year.max()) if len(used) else None,
                      'replaced': ', '.join(str(v) for v in y[~y.used].year)})
        for _, q in y.iterrows():
            e5.append({'city': x['city'], 'region': x['region'], 'year': q.year, 'used': q.used, 'reason': q.reason,
                       'db_gap_observed': q.get('db_gap_observed'), 'db_gap_after': q.get('db_gap'),
                       'era5_db_hours': q.get('era5_db_hours'), 'era5_status': q.get('era5_status'), 'era5_skipped': q.get('era5_skipped')})
        if len(used) < 10:
            continue
        mine = (work / 'years' / f'{sid}_design.txt').read_text()
        if sid in REPO_TMY:
            ref = open(glob.glob(str(REPO / 'data' / 'cities' / REPO_TMY[sid] / 'TMY' / '*.epw'))[0]).read().splitlines()[1]
        else:
            ref = ref_lines[sid[:6]]['design_conditions']
        mh, mc, me = parse(mine); rh, rc, re_ = parse(ref)
        rh = rh[:15]; wbmax = rc[31]; rc = rc[:31]; re_ = re_[:3] + [wbmax] + re_[3:15]   # ASHRAE 2025 -> 2009 layout
        for name, (sec, ix) in ITEMS.items():
            a = {'h': mh, 'c': mc, 'e': me}[sec][ix]; b = {'h': rh, 'c': rc, 'e': re_}[sec][ix]
            rows.append({'city': x['city'], 'region': x['region'], 'item': name, 'calc': float(a), 'ashrae': float(b), 'diff': float(a) - float(b)})
    d = pd.DataFrame(rows); d.to_csv(res / 'design_vs_ashrae.csv', index=False)
    pd.DataFrame(years).to_csv(res / 'years_selected.csv', index=False); pd.DataFrame(e5).to_csv(res / 'era5_use_by_year.csv', index=False)
    s = d.groupby('item', sort=False)['diff'].agg(mean_diff='mean', mean_abs=lambda v: np.abs(v).mean(), max_abs=lambda v: np.abs(v).max())
    s = s.join(d.groupby(['item', 'region'], sort=False)['diff'].agg(lambda v: np.abs(v).mean()).unstack().rename(columns={'US': 'us_abs', 'INT': 'int_abs'}))
    s.round(2).to_csv(res / 'design_summary.csv'); print(s.round(2).to_string())


if __name__ == '__main__':
    main()
