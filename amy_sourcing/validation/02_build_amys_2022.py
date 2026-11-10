"""Section 4.2 (end-to-end check), 4.3 (sky cover) and 4.4 (longwave) inputs: the 2022 AMYs of
the ten U.S. stations, built with the complete method from the NSRDB cache of 01_nsrdb_cache.py."""
import io, contextlib
import pandas as pd, requests
from common import work_dir, stations
import generate_amys as g


def main():
    work = work_dir(); cache = work / 'cache'; out = work / 'amy_2022'
    sess = requests.Session(); cat = g.station_catalog(sess, cache); rows = []
    for x in stations('US'):
        st = g.identify_station(cat, x['id'], period_start=2010, period_end=2024)
        st['tz_offset'], st['timezone_resolution'] = g.standard_tz_offset(st['lat'], st['lon'], None)
        city = g.clean_city_name(st, g.STATION_CITY_OVERRIDES.get(x['id']) or g.city_from_coordinates(st))
        with contextlib.redirect_stdout(io.StringIO()):
            # The key is not used: the NSRDB data come from the cache written by 01_nsrdb_cache.py.
            rep = g.create_station_year(sess, st, 2022, cache, out, 'cached', 'cached@example.org', preferred='NSRDB',
                                        city_name=city, **g.GENERATION_SETTINGS)
        rep['city'] = city; rows.append(rep)
        print(x['id'], city, rep.get('qc_grade'), flush=True)
    pd.DataFrame(rows).to_csv(work / 'results' / 'amy_2022_reports.csv', index=False)


if __name__ == '__main__':
    main()
