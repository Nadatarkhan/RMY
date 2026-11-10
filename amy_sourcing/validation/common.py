"""Shared set-up for the validation scripts.

Every script takes one argument, a working folder (default ./amy_validation_work).
Downloads are cached in <work>/cache and results are written to <work>/results, so a
second run reuses the data. The method itself is imported from this repository.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / 'amy_sourcing', REPO / 'tmy', REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def work_dir():
    work = Path(sys.argv[1] if len(sys.argv) > 1 else 'amy_validation_work').resolve()
    for sub in ('cache', 'results'):
        (work / sub).mkdir(parents=True, exist_ok=True)
    return work


def stations(region=None):
    """The twenty validation stations (id, city, standard UTC offset, region US or INT)."""
    st = json.load(open(HERE / 'stations.json'))
    return [x for x in st if region is None or x['region'] == region]


def catalog(work):
    """NOAA station catalogue (isd-history), downloaded once into the cache."""
    import requests
    import generate_amys as g
    return g.station_catalog(requests.Session(), work / 'cache')


def station_row(cat, sid):
    return cat[(cat.USAF == sid[:6]) & (cat.WBAN == sid[6:])].iloc[0]


def nsrdb_file(fs, year):
    """One year of NSRDB GOES v4 (aggregated) on NREL's public AWS archive."""
    import h5py
    return h5py.File(fs.open(f'nrel-pds-nsrdb/GOES/aggregated/v4.0.0/nsrdb_{year}.h5', 'rb', block_size=4 * 2 ** 20), 'r')


def nsrdb_meta(work, f):
    """Latitude and longitude of every NSRDB cell (read once, cached)."""
    import numpy as np
    p = work / 'cache' / 'nsrdb_v4_meta.npz'
    if not p.exists():
        m = f['meta'].fields(['latitude', 'longitude'])[:]
        np.savez(p, lat=m['latitude'].astype(float), lon=m['longitude'].astype(float))
    z = np.load(p)
    return z['lat'], z['lon']
