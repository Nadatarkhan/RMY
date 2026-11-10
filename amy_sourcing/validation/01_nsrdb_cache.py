"""Section 4.2 inputs: NSRDB GOES v4 for 2022 at the ten U.S. stations, read from NREL's public
archive on AWS (no API key) and written into the method's cache in the format of the NREL
download service, so that 02_build_amys_2022.py runs without a key."""
import time
import numpy as np, pandas as pd
from common import work_dir, stations, catalog, station_row, nsrdb_file, nsrdb_meta

YEAR = 2022
FIELDS = [('ghi', 'GHI', 1), ('dni', 'DNI', 1), ('dhi', 'DHI', 1), ('aod', 'Aerosol Optical Depth', 1000),
          ('total_precipitable_water', 'Precipitable Water', 10), ('surface_albedo', 'Surface Albedo', 100),
          ('solar_zenith_angle', 'Solar Zenith Angle', 100), ('cloud_type', 'Cloud Type', 1), ('fill_flag', 'Fill Flag', 1)]


def main():
    import fsspec
    work = work_dir(); cat = catalog(work)
    fs = fsspec.filesystem('s3', anon=True)
    files = {y: nsrdb_file(fs, y) for y in (YEAR, YEAR + 1)}
    mlat, mlon = nsrdb_meta(work, files[YEAR])
    tidx = {y: pd.to_datetime(f['time_index'][:].astype(str)).tz_localize(None) for y, f in files.items()}
    for x in stations('US'):
        sid, tz = x['id'], x['tz']
        out = work / 'cache' / 'radiation' / 'nsrdb' / sid / f'{YEAR}_local_extended_v4.csv'
        if out.exists():
            continue
        t0 = time.time(); r = station_row(cat, sid); lat, lon = float(r.LAT), float(r.LON)
        i = int(np.argmin((mlat - lat) ** 2 + ((mlon - lon) * np.cos(np.radians(lat))) ** 2))
        local = pd.date_range(f'{YEAR}-01-01 00:30', f'{YEAR}-12-31 23:30', freq='h')
        utc = local - pd.Timedelta(hours=tz)
        cols = {}
        for k, name, scale in FIELDS:
            parts = []
            for y, f in files.items():
                sel = np.flatnonzero(tidx[y].isin(utc))
                if len(sel):
                    lo, hi = sel.min(), sel.max() + 1
                    parts.append(pd.Series(f[k][lo:hi, i].astype(float) / scale, index=tidx[y][lo:hi]))
            cols[name] = pd.concat(parts).reindex(utc).to_numpy()
        df = pd.DataFrame({'Year': local.year, 'Month': local.month, 'Day': local.day, 'Hour': local.hour, 'Minute': 30, **cols})
        out.parent.mkdir(parents=True, exist_ok=True)
        head = ('Source,Location ID,Latitude,Longitude,Time Zone,Elevation,Local Time Zone,Precipitable Water Units\n'
                f'NSRDB,{i},{mlat[i]:.2f},{mlon[i]:.2f},{tz},0,{tz},cm\n')
        out.write_text(head + df.to_csv(index=False))
        print(sid, x['city'], 'cell %.3f,%.3f' % (mlat[i], mlon[i]), '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
