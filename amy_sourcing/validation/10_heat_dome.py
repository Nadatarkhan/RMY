"""Section 4.3 (Preserving an observed extreme): the June 2021 heat dome at Seattle-Tacoma in the
raw NOAA observations and in the AMY temperature series produced by the method."""
import io, contextlib
import pandas as pd, requests
from common import work_dir, catalog, station_row
import generate_amys as g

SID, TZ = '72793024233', -8


def main():
    work = work_dir(); r = station_row(catalog(work), SID)
    with contextlib.redirect_stdout(io.StringIO()):
        raw = g.required_surface(requests.Session(), SID, 2021, str(work / 'cache'), TZ)
        obs = raw.DB.astype(float).copy()
        amy, rep = g.era5_infill(raw, {'lat': float(r.LAT), 'lon': float(r.LON), 'tz_offset': TZ}, str(work / 'cache'))
        amy_db = g.caltech_linear_fill(amy.DB)
    w = slice('2021-06-24', '2021-07-01'); o, a = obs[w], amy_db[w]

    def event(s, thr=30.0):
        return {'peak_C': s.max(), 'peak_time': str(s.idxmax()), 'hours_at_or_above_30C': int((s >= thr).sum()),
                'days_Tmax_32C_or_more': int((s.resample('D').max() >= 32).sum()),
                'degree_hours_above_30C': float((s - thr).clip(lower=0).sum())}
    out = pd.DataFrame({'observed': event(o.dropna()), 'AMY': event(a)}).T
    out['missing_hours_in_event'] = int(o.isna().sum()); out['era5_filled_hours_in_event'] = int((amy.ERA5_FILLED[w] != '').sum())
    out.to_csv(work / 'results' / 'heat_dome_seattle_2021.csv'); print(out.to_string())


if __name__ == '__main__':
    main()
