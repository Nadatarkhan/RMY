#!/usr/bin/env python3
"""
Station-driven AMY (actual meteorological year) EPW generator.

For one NOAA station number, builds one EPW per calendar year for the most recent
complete years (15 by default), from:
  - NOAA Global Hourly (ISD) station observations (temperature, dew point, pressure,
    wind, cloud, precipitation), via the public NOAA archive on AWS S3;
  - NSRDB GOES v4 satellite solar radiation in the Americas (needs a free NREL API key
    and email), or PVGIS-SARAH3 / PVGIS-ERA5 elsewhere (no key).
Gaps are filled with documented, flagged methods; illuminance follows Perez et al.
(1990); sky cover is filled before the longwave (horizontal infrared) calculation.

The functions below are the AMY Generator V10.1 notebook code (Nada Tarkhan),
unchanged except for the module header, the computed last complete year and the
self-tests moved into run_self_tests(). generate_amys() at the end runs a whole
station. Command line:

    NSRDB_API_KEY=... NSRDB_EMAIL=... python amy_sourcing/generate_amys.py --station 72509014739 --out output
"""
from __future__ import annotations
import calendar
import csv
import io
import json
import math
import re
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

NOAA_STATIONS = 'https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv'
NOAA_BASE = 'https://noaa-global-hourly-pds.s3.amazonaws.com'
# NOAA retired ISD CSV direct access in July 2026; public archive is on AWS S3.
NOAA_ISD_LAST_COMPLETE_YEAR = date.today().year - 1  # upper bound; generate_amys() also checks the station record
NSRDB_URL = 'https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv'
PVGIS_URL = 'https://re.jrc.ec.europa.eu/api/v5_3/seriescalc'
NSRDB_YEARS = (1998, 2025)  # provider-supported advertised range, recheck in future
PVGIS_YEARS = (2005, 2023)  # provider historically published range; errors flagged
NSRDB_EXTENDED_ATTRIBUTES = [
    'ghi', 'dni', 'dhi', 'aod', 'total_precipitable_water',
    'surface_albedo', 'solar_zenith_angle', 'cloud_type', 'fill_flag'
]
# CSV column names of the optional NSRDB fields, lower case without spaces or symbols.
# The GOES v4 CSV names some fields differently from the API attribute
# ('Aerosol Optical Depth' for aod, 'Precipitable Water' for total_precipitable_water).
NSRDB_FIELD_COLUMNS = {
    'AOD': ('aod', 'aerosolopticaldepth'),
    'PW_CM': ('totalprecipitablewater', 'precipitablewater'),
    'ALBEDO': ('surfacealbedo',), 'ZENITH_DEG': ('solarzenithangle',),
    'CLOUD_TYPE': ('cloudtype',), 'FILL_FLAG': ('fillflag',),
}


def _nsrdb_column(lookup, field):
    """The CSV column holding an optional NSRDB field, or None."""
    return next((lookup[k] for k in NSRDB_FIELD_COLUMNS[field] if k in lookup), None)
# NOAA sky cover codes 00..08 represent 0..8 oktas (not 0..10 tenths).
# Approximate octas -> tenths using centers of documented NCEI ranges.
NOAA_OKTAS_TO_TENTHS = {0:0., 1:1., 2:2.5, 3:4., 4:5.,
                        5:6., 6:7.5, 7:9.5, 8:10.}


EPW_COLUMNS = [
    'year','month','day','hour','minute','data_source_uncertainty',
    'dry_bulb_temperature','dew_point_temperature','relative_humidity','atmospheric_station_pressure',
    'extraterrestrial_horizontal_radiation','extraterrestrial_direct_normal_radiation',
    'horizontal_infrared_radiation_intensity','global_horizontal_radiation','direct_normal_radiation',
    'diffuse_horizontal_radiation','global_horizontal_illuminance','direct_normal_illuminance',
    'diffuse_horizontal_illuminance','zenith_luminance','wind_direction','wind_speed',
    'total_sky_cover','opaque_sky_cover','visibility','ceiling_height',
    'present_weather_observation','present_weather_codes','precipitable_water',
    'aerosol_optical_depth','snow_depth','days_since_last_snowfall','albedo',
    'liquid_precipitation_depth','liquid_precipitation_quantity'
]
REJECT_QUALITY = {'2','3','6','7','','A'}
FAIL_SOURCE = {'2','9','A','B'}
QUALITY_GROUPS = ({'5'}, {'4','C','R'}, {'1','M'}, {'0','9','U'}, {'P','I'}, {'8'})
RANK = {c: i for i, group in enumerate(QUALITY_GROUPS) for c in group}


def get_bytes(session, url, params=None, timeout=180, retries=3):
    last = None
    for trial in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in [404, 400, 401, 403, 422]:
                raise RuntimeError(f'HTTP {resp.status_code} from {url} (request details withheld)')
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f'HTTP {resp.status_code}', response=resp)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as e:
            last = e
            if trial < retries - 1:
                time.sleep(2**trial)
    raise RuntimeError(f'Download failed from {url}: {last}')


def cached_download(session, url, cache_path, params=None):
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.stat().st_size == 0:
        data = get_bytes(session, url, params)
        p.with_suffix(p.suffix + '.tmp').write_bytes(data)
        p.with_suffix(p.suffix + '.tmp').replace(p)
    return p


def station_catalog(session, cache_dir):
    file = cached_download(session, NOAA_STATIONS, Path(cache_dir) / 'isd-history.csv')
    df = pd.read_csv(file, dtype=str, keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    for c in ['USAF','WBAN','LAT','LON','ELEV(M)','BEGIN','END']:
        if c not in df:
            raise ValueError(f'NOAA station catalog missing {c}')
    return df


def identify_station(df, station_number, wban_override=None, period_start=2009, period_end=2024):
    ident = str(station_number).strip().replace('-', '')
    if not re.fullmatch(r'[A-Za-z0-9]{6}([0-9]{5})?', ident):
        raise ValueError('Station ID must be six-character USAF/WMO or 11-character USAF+WBAN')
    usaf = ident[:6]
    candidates = df[df.USAF.str.zfill(6) == usaf].copy()
    if len(ident) == 11:
        candidates = candidates[candidates.WBAN.str.zfill(5) == ident[6:]]
    if wban_override is not None:
        candidates = candidates[candidates.WBAN.str.zfill(5) == str(wban_override).zfill(5)]
    if candidates.empty:
        raise ValueError(f'No NOAA Global Hourly station matches {station_number}; check USAF/WMO or provide 11-digit ID')
    candidates['start_num'] = pd.to_numeric(candidates.BEGIN.str[:4], errors='coerce').fillna(9999)
    candidates['end_num'] = pd.to_numeric(candidates.END.str[:4], errors='coerce').fillna(0)
    candidates['overlap'] = (np.minimum(candidates.end_num, period_end) - np.maximum(candidates.start_num, period_start) + 1).clip(lower=0)
    candidates['is_real_wban'] = (candidates.WBAN != '99999').astype(int)
    candidates = candidates.sort_values(['overlap','end_num','is_real_wban'], ascending=False)
    first = candidates.iloc[0]
    record = {'usaf': str(first.USAF).zfill(6), 'wban': str(first.WBAN).zfill(5),
              'station_id': str(first.USAF).zfill(6) + str(first.WBAN).zfill(5),
              'name': first.get('STATION NAME','Unnamed station'), 'country': first.get('CTRY',''),
              'state': first.get('STATE',''), 'icao': first.get('ICAO',''),
              'lat': float(first.LAT), 'lon': float(first.LON),
              'elevation_m': float(first['ELEV(M)']),
              'begin': str(first.BEGIN), 'end': str(first.END),
              'other_station_id_candidates': candidates.station_id.tolist() if 'station_id' in candidates else
                  (candidates.USAF + candidates.WBAN).tolist()}
    if not (-90 <= record['lat'] <= 90 and -180 <= record['lon'] <= 180):
        raise ValueError('NOAA station coordinate missing/invalid')
    if not (-500 <= record['elevation_m'] <= 9000):
        raise ValueError('NOAA station elevation missing/invalid')
    return record


def standard_tz_offset(lat, lon, override=None):
    if override is not None:
        return float(override), 'manual override'
    try:
        from timezonefinder import TimezoneFinder
        tzname = TimezoneFinder().timezone_at(lat=lat,lng=lon)
        if tzname:
            tz = ZoneInfo(tzname)
            jan = datetime(2020,1,15,12, tzinfo=tz).utcoffset().total_seconds()/3600
            jul = datetime(2020,7,15,12, tzinfo=tz).utcoffset().total_seconds()/3600
            return min(jan, jul), tzname + ' (non-DST standard offset)'
    except Exception as exc:
        print('[WARNING] Timezone geolocation unavailable:', exc)
    return int(lon/15), 'longitude / 15 truncated (Caltech legacy fallback; verify manually!)'


def provider_order(lat, lon):
    # Geographical heuristic; API coverage is ultimately determined by service response.
    america = (-175 <= lon <= -30) and (-60 <= lat <= 65)
    return ['NSRDB','PVGIS'] if america else ['PVGIS']


def quality_rank(code):
    q = str(code).strip().upper()
    if q in REJECT_QUALITY:
        return None
    return RANK.get(q)  # discard undocumented codes rather than assume valid


def _num(text, invalid_threshold, multiplier=1.):
    try:
        raw = float(text)
        if not np.isfinite(raw) or abs(raw) >= invalid_threshold:
            return np.nan
        return raw * multiplier
    except (ValueError, TypeError):
        return np.nan


def parse_noaa_value(row, variable):
    """Parse Global Hourly groups and documented per-variable QC priorities.
    Returns (numeric_value, quality_tier) or (nan, None).
    """
    def parts(group):
        v = str(row.get(group, '')).strip()
        return v.split(',') if v else []
    if variable in ('DB','DP','PR'):
        group = {'DB':'TMP','DP':'DEW','PR':'SLP'}[variable]
        tokens = parts(group)
        if len(tokens) < 2 and variable == 'PR':
            tokens = parts('MA1')
        if len(tokens) < 2:
            return np.nan, None
        r = quality_rank(tokens[1]); v = _num(tokens[0], 99999 if variable == 'PR' else 9999, 0.1)
        if (r is None or not np.isfinite(v)) and variable == 'PR':
            alt = parts('MA1')
            if len(alt) >= 2:
                r = quality_rank(alt[1]); v = _num(alt[0], 99999, 0.1)
        return (v, r) if r is not None and np.isfinite(v) else (np.nan, None)
    if variable in ('WS','WD'):
        tokens = parts('WND')
        idx, qidx, missing, scale = (3,4,9999,0.1) if variable == 'WS' else (0,1,999,1.)
        if len(tokens) <= qidx:
            return np.nan,None
        r = quality_rank(tokens[qidx]); v = _num(tokens[idx], missing, scale)
        return (v,r) if r is not None and np.isfinite(v) else (np.nan,None)
    if variable in ('CEIL','VIS'):
        tokens = parts('CIG' if variable == 'CEIL' else 'VIS')
        if len(tokens) < 2: return np.nan,None
        r = quality_rank(tokens[1]); v = _num(tokens[0], 99999, 1.)
        return (v,r) if r is not None and np.isfinite(v) else (np.nan,None)
    if variable == 'PC':
        tokens = parts('AA1')
        if len(tokens) < 4: return np.nan,None
        r = quality_rank(tokens[3])
        if r is None: return np.nan,None
        if tokens[1].strip() == '':
            return 0.,r
        v = _num(tokens[1], 9999, 0.1)
        return (v,r) if np.isfinite(v) else (np.nan,None)
    raise ValueError(variable)


def parse_noaa_cloud(row, variable):
    """NOAA GF1 aggregated octas or GA layer-maximum approximation.

    GF1 coverage and opaque coverage are separately recorded (0..8 oktas);
    GA records only individual cloud layers; their maximum is not a measured total.
    """
    if variable in ('TC_GF1', 'OC_GF1'):
        tok = str(row.get('GF1','') or '').split(',')
        if len(tok)<3: return np.nan,None
        rank=quality_rank(tok[2])
        try: oktas=int(tok[0 if variable=='TC_GF1' else 1])
        except (TypeError,ValueError): return np.nan,None
        return (NOAA_OKTAS_TO_TENTHS[oktas],rank) if rank is not None and oktas in NOAA_OKTAS_TO_TENTHS else (np.nan,None)
    if variable=='TC_GA_PROXY':
        layers=[]
        for label in ('GA1','GA2','GA3','GA4','GA5','GA6'):
            tok=str(row.get(label,'') or '').split(',')
            if len(tok)<2: continue
            rank=quality_rank(tok[1])
            try: oktas=int(tok[0])
            except (TypeError,ValueError): continue
            if rank is not None and oktas in NOAA_OKTAS_TO_TENTHS:
                layers.append((rank,NOAA_OKTAS_TO_TENTHS[oktas]))
        if not layers:
            # A METAR saying CAVOK, NSC, SKC, NCD or CLR (outside its trend part) has no
            # cloud groups in NOAA's data: it reports clear sky, not missing cover. It
            # gets the lowest quality rank, so any real cloud group in the hour wins.
            body=re.split(r'\b(?:BECMG|TEMPO|NOSIG|RMK|PROB\d\d|FM\d{4,6})\b',str(row.get('REM','') or ''))[0]
            if re.search(r'\b(?:CAVOK|NSC|SKC|NCD|CLR)\b',body):
                return 0.0,len(QUALITY_GROUPS)
            return np.nan,None
        return max(v for _,v in layers),min(r for r,_ in layers)
    raise ValueError(variable)


def parse_noaa_precip(row):
    """Valid NOAA AA1..AA4 interval-accumulation reports (mm and hours).

    Prefer a short valid reporting period, particularly one-hour accumulations;
    do NOT reinterpret a 6-hour amount as one-hour rainfall. Periods of 1 to 6
    hours only (longer totals overlap and are error-prone). One report per
    observation is retained. Repeated overlapping accumulation reports require
    additional reconciliation before computing climatological annual totals.
    """
    candidates=[]
    for grp in ('AA1','AA2','AA3','AA4'):
        t=str(row.get(grp,'') or '').split(',')
        if len(t)<4: continue
        try: duration=int(t[0]); depth=float(t[1])/10 if t[1].strip() else 0.0
        except (TypeError,ValueError): continue
        rank=quality_rank(t[3])
        # Totals over more than 6 hours are not used: they overlap the shorter reports
        # (the same rain counted again), and 12/24-hour SYNOP totals carry decoding
        # artefacts (e.g. 50, 99 or 100 mm on dry days at desert stations).
        if rank is None or duration not in (1,2,3,4,5,6) or not 0<=depth<999: continue
        candidates.append((duration,rank,depth,grp))
    if not candidates: return np.nan,np.nan,None,None
    period,rank,depth,source=min(candidates,key=lambda x:(x[0],x[1]))
    return depth,float(period),rank,source


def parse_noaa_snow_depth(row):
    """Read documented NOAA AJ1 snow-depth dimension, when supplied.

    AJ1 units mm (source); EPW uses cm. This is an observation, not interpolation.
    """
    t=str(row.get('AJ1','') or '').split(',')
    if len(t)<3: return np.nan,None
    # AJ1 depth in millimeters; condition code t[1]; quality t[2].
    r=quality_rank(t[2]); val=_num(t[0],9999,0.1)  # mm -> cm
    return (val,r) if r is not None and np.isfinite(val) else (np.nan,None)


def wind_type(row):
    t=str(row.get('WND','') or '').split(',')
    return t[2].strip().upper() if len(t)>2 else ''

def noaa_csv(session, station_id, year, cache_dir):
    # Separate cache folder prevents accidentally reusing error pages previously
    # downloaded from NOAA's discontinued NCEI-hosted CSV endpoint.
    file = cached_download(session, f'{NOAA_BASE}/{int(year)}/{station_id}.csv',
                           Path(cache_dir) / 'noaa_s3' / station_id / f'{year}.csv')
    try:
        with file.open(encoding='utf-8-sig', errors='replace', newline='') as fh:
            columns = [c.strip().upper() for c in next(csv.reader(fh), [])]
        if not {'STATION', 'DATE'}.issubset(columns):
            raise ValueError('NOAA download was not an ISD CSV (missing STATION/DATE header)')
        d = pd.read_csv(file, dtype=str, keep_default_na=False, low_memory=False)
    except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
        file.unlink(missing_ok=True)  # Never retain an error page or corrupted CSV.
        raise ValueError(f'Invalid NOAA CSV source for station {station_id}, year {year} ({NOAA_BASE}/{int(year)}/{station_id}.csv): {exc}') from exc
    d['DATE'] = pd.to_datetime(d['DATE'], utc=True, errors='coerce')
    d = d[d.DATE.notna()].copy()
    if 'SOURCE' in d:
        d = d[~d.SOURCE.astype(str).isin(FAIL_SOURCE)].copy()
    return d


def extract_hourly(d):
    """Caltech-style per-hour per-variable NOAA best-quality selection.

    Includes wind type and AA precipitation period provenance, with no
    undocumentable substitutions of wind directions for variable winds.
    """
    variables=['DB','DP','WS','WD','PR','CEIL','VIS','TC_GF1','OC_GF1','TC_GA_PROXY','SNOW_CM']
    best={}; columns=list(d.columns)
    for values in d.itertuples(index=False,name=None):
        row=dict(zip(columns,values))
        ts=row['DATE'].floor('h').tz_localize(None)
        entry=best.setdefault(ts,{})
        wtype=wind_type(row)
        if wtype in ('C','V'):
            # Capture the code even when the direction angle itself is missing.
            types=entry.setdefault('_wind_type_candidates',set()); types.add(wtype)
        for variable in variables:
            if variable=='SNOW_CM': val,rank=parse_noaa_snow_depth(row)
            elif variable in ('TC_GF1','OC_GF1','TC_GA_PROXY'):
                val,rank=parse_noaa_cloud(row,variable)
            else: val,rank=parse_noaa_value(row,variable)
            if rank is None: continue
            if variable not in entry or rank<entry[variable][0]:
                entry[variable]=(rank,val)
                if variable=='WD': entry['_selected_wind_type']=wtype
        pc,period,rank,pcsrc=parse_noaa_precip(row)
        if rank is not None:
            old=entry.get('_precip')
            candidate=(period,rank,pc,pcsrc)
            if old is None or (period,rank)<old[:2]: entry['_precip']=candidate
    if not best: raise ValueError('Station-year contains no valid NOAA hours')
    result=pd.DataFrame(index=pd.DatetimeIndex(sorted(best),name='utc_hour'))
    for var in variables:
        result[var]=[best[t].get(var,(None,np.nan))[1] for t in result.index]
    precip=[best[t].get('_precip',(np.nan,None,np.nan,'')) for t in result.index]
    result['PC']=[x[2] for x in precip]
    result['PC_PERIOD']=[x[0] for x in precip]
    result['PC_SOURCE']=[x[3] for x in precip]
    result['WIND_TYPE']=[best[t].get('_selected_wind_type','') or
                         ('V' if 'V' in best[t].get('_wind_type_candidates',set()) else
                          'C' if 'C' in best[t].get('_wind_type_candidates',set()) else '')
                         for t in result.index]
    result['TC']=result['TC_GF1'].combine_first(result['TC_GA_PROXY'])
    result['OC']=result['OC_GF1'].copy()
    result['TC_method']=np.where(result['TC_GF1'].notna(),'GF1',
                                 np.where(result['TC_GA_PROXY'].notna(),'GA_LAYER_MAX_PROXY','MISSING'))
    conflicts=result['TC_GF1'].notna() & result['OC_GF1'].notna() & (result['OC_GF1']>result['TC_GF1'])
    result.loc[conflicts,'OC']=np.nan
    return result

def max_missing_run(series):
    a = pd.isna(series).to_numpy() if hasattr(series, 'to_numpy') else np.asarray(pd.isna(series))
    if len(a) == 0 or not a.any(): return 0
    edges = np.diff(np.r_[False,a,False].astype(int))
    return int(max((edges == -1).nonzero()[0] - (edges == 1).nonzero()[0]))


def caltech_linear_fill(s):
    """Documented linear interior interpolation + genuine linear edge EXTRAPOLATION."""
    x = pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
    valid = np.flatnonzero(np.isfinite(x))
    if len(valid) < 2:
        raise ValueError('Need >=2 valid points to extrapolate a required variable; no fabrication')
    ix = np.arange(len(x))
    interp = np.interp(ix, valid, x[valid])
    lead = ix < valid[0]
    trail = ix > valid[-1]
    v0,v1 = valid[:2]; z0,z1 = valid[-2:]
    interp[lead] = x[v0] + (ix[lead]-v0) * (x[v1]-x[v0])/(v1-v0)
    interp[trail] = x[z1] + (ix[trail]-z1) * (x[z1]-x[z0])/(z1-z0)
    return pd.Series(interp,index=s.index)


def required_surface(session, station_id, year, cache_dir, tz_offset):
    target_idx = pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='h',inclusive='left')
    if abs(tz_offset - round(tz_offset)) > 1e-8:
        raise ValueError('Half-hour / fractional standard timezones require subhourly resampling; not yet supported')
    needed = [year]
    if tz_offset < 0: needed.append(year+1)
    if tz_offset > 0: needed.append(year-1)
    chunks = []
    for y in needed:
        try:
            chunks.append(extract_hourly(noaa_csv(session, station_id, y, cache_dir)))
        except Exception as exc:
            if y == year: raise
            print(f'[WARN] NOAA adjacent year {y} unavailable: {exc}')
    if not chunks: raise RuntimeError('No NOAA observation years downloaded')
    utc = pd.concat(chunks).sort_index()
    utc = utc[~utc.index.duplicated(keep='first')]
    utc.index = utc.index + pd.to_timedelta(tz_offset,unit='h')
    local = utc.reindex(target_idx)
    local.index.name='local_standard_time'
    return local


def rh_and_station_pressure(db, dp, slp_mb, elevation_m):
    """Caltech appendix pressure and Rogers-Yau-style humidity relationship."""
    db = np.asarray(db,dtype=float)
    dp = np.asarray(dp,dtype=float)
    p0 = np.asarray(slp_mb,dtype=float)
    t = db+273.15
    if elevation_m == 0:
        p = p0.copy()
    else:
        gamma = (288.0-t)/elevation_m
        small = np.abs(gamma) < 1e-7
        power = np.where(small, -9.80665*elevation_m/(287.055*288.),
                         (9.80665/(287.055*np.where(small,1.,gamma)))*np.log(np.maximum(t,1.)/288.))
        p = p0*np.exp(power)
    w = (0.622*2.53e9*np.exp(-5420.0/(dp+273.15)))/p
    e = p*w/(0.622+w)
    es = 6.112*np.exp(17.67*db/(db+243.5))
    rh = e/es*100
    return p*100,rh


def nsrdb_radiation(session, year, station, cache_dir, key, email, tz_offset):
    """Fetch and correctly align NSRDB hourly midpoint timestamps to EPW hours.

    NSRDB 60-minute files conventionally label irradiance at HH:30, the
    midpoint of the corresponding hour. EnergyPlus EPW hour 1 represents
    00:00--01:00. Thus 00:30 maps to EPW hour 1 without shifting to another
    hour. 00:00 is also supported if an NSRDB product uses start-hour labels.
    No daylight-saving adjustment is made: the API request uses utc=false.
    """
    if not NSRDB_YEARS[0] <= year <= NSRDB_YEARS[1]:
        raise ValueError('Year outside configured NSRDB supported range')
    if not key or not email:
        raise ValueError('NSRDB_API_KEY and NSRDB_EMAIL needed')
    cache = Path(cache_dir)/'radiation'/'nsrdb'/station['station_id']/f'{year}_local_extended_v4.csv'
    params = {'wkt':f"POINT({station['lon']} {station['lat']})",'names':str(year),
              'interval':60,'leap_day':'true','utc':'false',
              'attributes':','.join(NSRDB_EXTENDED_ATTRIBUTES),'email':email,'api_key':key,
              'full_name':'AMY research user','affiliation':'Research',
              'reason':'Actual meteorological year building simulations','mailing_list':'false'}
    f = cached_download(session, NSRDB_URL, cache, params)
    lines = f.read_text(encoding='utf-8-sig',errors='replace').splitlines()
    hi = next((i for i,line in enumerate(lines) if line.strip().lower().startswith('year,month,day,hour,minute')),None)
    if hi is None:
        f.unlink(missing_ok=True)
        raise ValueError('Invalid NSRDB CSV response; cache removed')
    import csv as _csv
    metadata = dict(zip(next(_csv.reader([lines[0]])),next(_csv.reader([lines[1]])))) if len(lines)>2 else {}
    data_tz = metadata.get('Data Timezone',metadata.get('Time Zone'))
    if data_tz is not None and str(data_tz).strip():
        if abs(float(data_tz)-tz_offset)>0.01:
            raise ValueError(f'NSRDB site timezone ({data_tz}) != station standard timezone ({tz_offset}); refuse merge')
    d = pd.read_csv(io.StringIO('\n'.join(lines[hi:])),low_memory=False)
    print('[AMY NSRDB] CSV fields:', list(d.columns),flush=True)
    lower = {re.sub(r'[^a-z0-9]','',str(c).strip().lower()):c for c in d.columns}
    need = ['year','month','day','hour','minute','ghi','dni','dhi']
    if any(c not in lower for c in need):
        raise ValueError(f'NSRDB CSV missing required date/radiation columns: {sorted(set(need)-set(lower))}')
    parts = {k:pd.to_numeric(d[lower[k]],errors='coerce') for k in ['year','month','day','hour','minute']}
    minutes = parts['minute'].dropna().unique()
    if len(minutes)!=1 or minutes[0] not in (0,30) or parts['minute'].isna().any():
        raise ValueError(f'NSRDB unexpected hourly time-minute convention: {sorted(set(parts["minute"].dropna()))[:12]}; expected 0 or 30')
    minute = int(minutes[0])
    # Assemble each timestamp directly from the YEAR, MONTH, DAY, HOUR, MINUTE CSV fields.
    raw_ts = pd.to_datetime(pd.DataFrame(parts),errors='coerce')
    if raw_ts.isna().any():
        raise ValueError(f'NSRDB has {int(raw_ts.isna().sum())} unparseable time records')
    # Hourly irradiance at HH:30 belongs to the hour HH:00--HH+1:00.
    # This is a *midpoint label mapping*, NOT a timezone conversion.
    hourly_ts = raw_ts.dt.floor('h') if minute==30 else raw_ts
    rad_values = {v.upper():pd.to_numeric(d[lower[v]],errors='coerce').to_numpy()
                  for v in ['ghi','dni','dhi']}
    # Optional NSRDB ancillary fields are returned with irradiance and retain
    # their original time alignment; missing attrs must NOT masquerade as zero.
    for outcol in NSRDB_FIELD_COLUMNS:
        column = _nsrdb_column(lower, outcol)
        rad_values[outcol] = (pd.to_numeric(d[column],errors='coerce').to_numpy()
                              if column is not None else np.full(len(d),np.nan))
    # NLR metadata specifies precipitable water in centimeters.
    pwunit = next((str(v).strip().lower() for k,v in metadata.items()
                   if 'precipitable water units' in k.lower()), 'cm')
    if pwunit not in ('cm','centimeter','centimeters','millimeter','millimeters','mm'):
        raise ValueError(f'Unexpected NSRDB precipitable-water units: {pwunit!r}; refusing implicit conversion')
    if pwunit in ('mm','millimeter','millimeters'):
        rad_values['PW_CM'] = rad_values['PW_CM'] / 10.0
    # These missing values MUST remain missing, not become physical zeroes.
    # NSRDB optional fields can be absent or marked with negative placeholders.
    bounds={'AOD':(0,5),'PW_CM':(0,20),'ALBEDO':(0,1),
            'ZENITH_DEG':(0,180),'CLOUD_TYPE':(0,100),'FILL_FLAG':(0,10000)}
    for field,(low,high) in bounds.items():
        v=np.array(rad_values[field],dtype=float)  # copy: pandas 3 arrays are read-only
        v[~np.isfinite(v) | (v<low) | (v>high)] = np.nan
        rad_values[field]=v
    frame = pd.DataFrame(rad_values,index=pd.DatetimeIndex(hourly_ts))
    if frame.index.has_duplicates:
        raise ValueError('NSRDB duplicate hourly timestamps after midpoint alignment')
    frame = frame.sort_index()
    target = expected_local_index(year)
    overlap = len(frame.index.intersection(target))
    nan_counts = frame[['GHI','DNI','DHI']].isna().sum().to_dict()
    print(f'[AMY NSRDB] input rows={len(frame)}; raw timestamp minute={minute}; '
          f'first={raw_ts.iloc[0]}; last={raw_ts.iloc[-1]}; '
          f'aligned overlap={overlap}/{len(target)}; NaNs={nan_counts}',flush=True)
    if overlap != len(target):
        missing = target.difference(frame.index)
        extras = frame.index.difference(target)
        raise ValueError(f'NSRDB timestamp mismatch: overlap={overlap}/{len(target)}, '
                         f'missing={len(missing)}, outside-year={len(extras)}, '
                         f'first missing={list(missing[:3])}, first extra={list(extras[:3])}')
    return frame, {'radiation_source':'NSRDB GOES Aggregated v4',
                   'radiation_year':year,
                   'radiation_timebase':'NSRDB local standard time; hourly midpoint 30->hour start' if minute==30
                                         else 'NSRDB local standard time; timestamps at hour start',
                   'nsrdb_timestamp_minute':minute,
                   'nsrdb_midpoint_alignment':bool(minute==30),
                   'nsrdb_raw_csv':str(f),
                   'radiation_url':NSRDB_URL,
                   'nsrdb_extended_attributes':','.join(NSRDB_EXTENDED_ATTRIBUTES),
                   'nsrdb_ancillary_provenance':'NLR NSRDB PSM v4; ancillary atmospheric inputs can use MERRA-2',
                   'nsrdb_precipitable_water_units':'cm (raw aligned PW_CM)'}


def pvgis_radiation(session, year, station, cache_dir):
    if not PVGIS_YEARS[0] <= year <= PVGIS_YEARS[1]:
        raise ValueError('Year outside configured PVGIS supported range')
    err=[]
    for db in ['PVGIS-SARAH3','PVGIS-ERA5']:
        by_comp={}
        try:
            for comp in [0,1]:
                cached=Path(cache_dir)/'radiation'/'pvgis'/station['station_id']/f'{year}_{db}_{comp}.json'
                params={'lat':station['lat'],'lon':station['lon'],'startyear':year,'endyear':year,
                        'raddatabase':db,'components':comp,'angle':0,'aspect':0,
                        'outputformat':'json'}
                f=cached_download(session,PVGIS_URL,cached,params)
                try:
                    obj=json.loads(f.read_text())
                    rows=obj['outputs']['hourly']
                except (ValueError, KeyError, TypeError):
                    f.unlink(missing_ok=True)
                    raise
                d=pd.DataFrame(rows)
                if 'time' not in d: raise ValueError('PVGIS JSON missing time')
                idx=pd.to_datetime(d['time'],format='%Y%m%d:%H%M',errors='coerce').dt.floor('h')
                d.index=pd.DatetimeIndex(idx)
                if d.index.has_duplicates: raise ValueError('PVGIS duplicate hourly timestamps')
                by_comp[comp]=d
            base=by_comp[0]; secondary=by_comp[1]
            if 'G(i)' not in base or 'Gd(i)' not in secondary or 'H_sun' not in base:
                raise ValueError(f'PVGIS radiation columns not found: {list(base)} / {list(secondary)}')
            d=pd.DataFrame(index=base.index)
            d['GHI']=pd.to_numeric(base['G(i)'],errors='coerce')
            d['DHI']=pd.to_numeric(secondary.reindex(base.index)['Gd(i)'],errors='coerce')
            el=pd.to_numeric(base['H_sun'],errors='coerce')
            sinel=np.sin(np.deg2rad(el))
            d['DNI']=np.where(sinel>0.035,np.maximum(0,(d.GHI-d.DHI)/sinel),0.)
            # PVGIS returns timestamps in UTC, not local standard time.
            return d, {'radiation_source':db,'radiation_year':year,
                       'radiation_timebase':'PVGIS UTC, floored subhour sample','radiation_url':PVGIS_URL}
        except Exception as e:
            err.append(f'{db}: {type(e).__name__}: {str(e)[:190]}')
    raise RuntimeError('PVGIS radiation unavailable: '+' | '.join(err))


def expected_local_index(year):
    return pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='h',inclusive='left')


def validate_radiation(frame, year):
    target=expected_local_index(year)
    r=frame.reindex(target).copy()
    if len(r)!=len(target) or r[['GHI','DNI','DHI']].isna().any().any():
        missing_required = int(r[['GHI','DNI','DHI']].isna().any(axis=1).sum())
        raise ValueError(f'Radiation incomplete ({missing_required} missing local hours)')
    if (r[['GHI','DNI','DHI']]<0).any().any() or (r[['GHI','DNI','DHI']]>1600).any().any():
        raise ValueError('Radiation outside physical screening range [0,1600] W/m2')
    if r['GHI'].sum()<=1e3 or (r['GHI']>10).sum()<500:
        raise ValueError('Radiation appears zero-filled or unusable')
    return r


def _fetch_radiation_for_year(session, y, st, cache, key, email, tz, provider):
    if provider=='NSRDB': return nsrdb_radiation(session,y,st,cache,key,email,tz)
    return pvgis_radiation(session,y,st,cache)


def radiation_for_year(session, year, st, cache, key, email, tz, preferred='AUTO',
                       allow_borrowed=False, borrow_radius=2, allow_edge_wrap=False):
    providers=provider_order(st['lat'],st['lon']) if preferred.upper()=='AUTO' else [preferred.upper()]
    errors=[]
    for provider in providers:
        # NSRDB supplies already-local annual data; PVGIS supplies UTC requiring neighboring year edges.
        if provider=='NSRDB' and not (NSRDB_YEARS[0]<=year<=NSRDB_YEARS[1]): continue
        if provider=='PVGIS' and not (PVGIS_YEARS[0]<=year<=PVGIS_YEARS[1]): continue
        try:
            base,prov=_fetch_radiation_for_year(session,year,st,cache,key,email,tz,provider)
            if provider=='NSRDB':
                return validate_radiation(base,year),{**prov,'borrowed_radiation_hours':0,'edge_wrapped_hours':0}
            # PVGIS UTC -> local standard time, include adjacent UTC year before validation.
            frames=[base]
            edgeyear=year+1 if tz<0 else year-1 if tz>0 else None
            edge_wrapped=0
            if edgeyear is not None:
                try:
                    adjacent,_=_fetch_radiation_for_year(session,edgeyear,st,cache,key,email,tz,provider)
                    frames.append(adjacent)
                except Exception as e:
                    if not allow_edge_wrap:
                        raise RuntimeError(f'Missing PVGIS year-boundary data {edgeyear}: {e}')
                    # For just the missing 1..12 boundary hours, wrap same-year radiation.
                    edge_wrapped=int(abs(tz))
                    shifted=base.copy()
                    shifted.index=shifted.index+pd.DateOffset(years=1 if tz<0 else -1)
                    frames.append(shifted)
            joined=pd.concat(frames).sort_index()
            joined=joined[~joined.index.duplicated(keep='first')]
            joined.index=joined.index+pd.to_timedelta(tz,unit='h')
            return validate_radiation(joined,year),{**prov,'borrowed_radiation_hours':0,'edge_wrapped_hours':edge_wrapped}
        except Exception as e:
            errors.append(f'{provider}: {str(e)[:210]}')
    if allow_borrowed:
        for offset in range(1,borrow_radius+1):
            for y in [year-offset,year+offset]:
                for provider in providers:
                    if provider=='NSRDB' and not (NSRDB_YEARS[0]<=y<=NSRDB_YEARS[1]): continue
                    if provider=='PVGIS' and not (PVGIS_YEARS[0]<=y<=PVGIS_YEARS[1]): continue
                    try:
                        source,prov=radiation_for_year(session,y,st,cache,key,email,tz,provider,
                                           allow_borrowed=False,allow_edge_wrap=allow_edge_wrap)
                        # Calendar substitution by month/day/hour; Feb29 is Feb28 replication.
                        t=expected_local_index(year)
                        dates=[]
                        for dt in t:
                            mo,day=dt.month,dt.day
                            if mo==2 and day==29 and not calendar.isleap(y): day=28
                            dates.append(pd.Timestamp(y,mo,day,dt.hour))
                        borrowed=source.reindex(pd.DatetimeIndex(dates))
                        borrowed.index=t
                        out=validate_radiation(borrowed,year)
                        return out,{**prov,'radiation_year':y,'borrowed_radiation_hours':len(out),
                                    'edge_wrapped_hours':prov.get('edge_wrapped_hours',0)}
                    except Exception as e:
                        errors.append(f'borrow {provider} {y}: {str(e)[:150]}')
    raise RuntimeError('No valid complete radiation: '+' | '.join(errors[:8]))


def optional_nsrdb_ancillary_retry(session,year,st,cache_dir,key,email,tz,rad):
    """Optional additional NSRDB fetch for absent columns (no ERA5 fallback).

    Retries once with the subset of missing attributes, in a v5 cache. This
    avoids redownloading the original large GHI/DNI/DHI file. If the API still
    does not expose them, leave missing and report an explicit reason.
    """
    desired={'PW_CM':'total_precipitable_water','AOD':'aod',
             'ALBEDO':'surface_albedo','ZENITH_DEG':'solar_zenith_angle'}
    missing=[api for col,api in desired.items() if col not in rad or rad[col].notna().sum()==0]
    provenance={'nsrdb_ancillary_retry_attempted':bool(missing),
                'nsrdb_ancillary_retry_attributes':','.join(missing)}
    if not missing: return rad,provenance
    params={'wkt':f"POINT({st['lon']} {st['lat']})",'names':str(year),'interval':60,
            'leap_day':'true','utc':'false','attributes':','.join(missing),
            'email':email,'api_key':key,'full_name':'AMY research user',
            'affiliation':'Research','reason':'AMY QC supplemental fields',
            'mailing_list':'false'}
    cached=Path(cache_dir)/'radiation'/'nsrdb_v5_ancillary'/st['station_id']/f'{year}_ancillary.csv'
    try:
        f=cached_download(session,NSRDB_URL,cached,params)
        lines=f.read_text(encoding='utf-8-sig',errors='replace').splitlines()
        head=next((i for i,line in enumerate(lines) if line.lower().startswith('year,month,day,hour,minute')),None)
        if head is None: raise ValueError('Ancillary response not an NSRDB table')
        dd=pd.read_csv(io.StringIO('\n'.join(lines[head:])))
        lookup={re.sub('[^a-z0-9]','',c.lower()):c for c in dd.columns}
        cols=[lookup[k] for k in ['year','month','day','hour','minute']]
        stamp=pd.to_datetime(dd[cols].set_axis(['year','month','day','hour','minute'],axis=1),errors='coerce').dt.floor('h')
        if stamp.isna().any() or stamp.duplicated().any(): raise ValueError('Ancillary dates invalid')
        supplement=pd.DataFrame(index=pd.DatetimeIndex(stamp))
        for col,attr in desired.items():
            column=_nsrdb_column(lookup,col)
            if attr not in missing or column is None: continue
            values=pd.to_numeric(dd[column],errors='coerce').to_numpy(dtype=float,copy=True)
            if col=='PW_CM':
                # API standard NSRDB unit for total precipitable water: cm.
                values[(values<0)|(values>20)]=np.nan
            elif col=='AOD': values[(values<0)|(values>5)]=np.nan
            elif col=='ALBEDO': values[(values<0)|(values>1)]=np.nan
            elif col=='ZENITH_DEG': values[(values<0)|(values>180)]=np.nan
            supplement[col]=values
        if len(supplement.index.intersection(rad.index))!=len(rad):
            raise ValueError('Ancillary timestamps do not align to main NSRDB year')
        for col in supplement:
            if col not in rad: rad[col]=np.nan
            rad[col]=rad[col].combine_first(supplement[col].reindex(rad.index))
        provenance['nsrdb_ancillary_retry_error']=''
        provenance['nsrdb_ancillary_retry_columns']=','.join(supplement.columns)
    except Exception as exc:
        provenance['nsrdb_ancillary_retry_error']=f'{type(exc).__name__}: {exc}'[:220]
        print('[AMY NSRDB ANCILLARY] missing fields retained:',provenance['nsrdb_ancillary_retry_error'])
    return rad,provenance


def _wind_fill_all_hours(series, angular=False):
    """Full interior interpolation + genuine linear edge extrapolation.

    Angular mode unwraps successive observed angles before filling and then
    wraps the answer into [0,360); scalar mode treats angles as ordinary numbers.
    The same first/last TWO valid anchors are used to extend boundary slopes.
    All outputs are accompanied by per-hour method and original-gap-length tags.
    """
    original=pd.to_numeric(series,errors='coerce').to_numpy(dtype=float)
    n=len(original)
    valid=np.flatnonzero(np.isfinite(original))
    tags=np.full(n,'MISSING_UNRESOLVED',dtype=object)
    tags[valid]='NOAA_OBSERVED'
    gaps=np.zeros(n,dtype=int)
    missing=~np.isfinite(original)
    boundaries=np.diff(np.r_[False,missing,False].astype(int))
    for start,end in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
        gaps[start:end]=end-start
    if len(valid)<2:
        return pd.Series(original,index=series.index),tags,gaps,0,0
    anchors=original[valid]
    if angular:
        anchors=np.rad2deg(np.unwrap(np.deg2rad(anchors)))
    xx=np.arange(n)
    filled=np.interp(xx,valid,anchors)
    lead=xx<valid[0];trail=xx>valid[-1]
    i0,i1=valid[:2];j0,j1=valid[-2:]
    filled[lead]=anchors[0]+(xx[lead]-i0)*(anchors[1]-anchors[0])/(i1-i0)
    filled[trail]=anchors[-1]+(xx[trail]-j1)*(anchors[-1]-anchors[-2])/(j1-j0)
    tags[missing & ~(lead|trail)]='LINEAR_INTERPOLATION'
    tags[missing & lead]='LINEAR_EXTRAPOLATION_START'
    tags[missing & trail]='LINEAR_EXTRAPOLATION_END'
    if angular:
        filled=np.mod(filled,360.0)
    clipped_low=clipped_high=0
    if not angular and np.isfinite(original[valid]).all():
        badlow=missing & (filled<0)
        badhigh=missing & (filled>60)
        clipped_low=int(badlow.sum());clipped_high=int(badhigh.sum())
        filled[missing]=np.clip(filled[missing],0,60)
        tags[badlow]=np.array([v+'+CLAMPED_LOW' for v in tags[badlow]],dtype=object)
        tags[badhigh]=np.array([v+'+CLAMPED_HIGH' for v in tags[badhigh]],dtype=object)
    # Preserve source anchors *exactly* (no numeric noise from unwrap/modulo).
    filled[valid]=original[valid]
    return pd.Series(filled,index=series.index),tags,gaps,clipped_low,clipped_high


def repair_wind_direction(surface,mode='CIRCULAR_LINEAR',max_gap=2):
    """Caltech-inspired full linear interpolation/extrapolation of wind fields.

    CliMA documentation states linear interpolation/extrapolation for missing
    observations, not its wind-specific internals. CIRCULAR_LINEAR improves
    the directional topology; CALTECH_LINEAR is literal scalar-angle handling.
    Every hourly change is tagged. No undocumented meteorological data grafts.
    """
    s=surface.copy()
    mode=str(mode).upper()
    if mode not in ('PRESERVE','CIRCULAR_SHORT','CIRCULAR_LINEAR','CALTECH_LINEAR'):
        raise ValueError('WIND_FILL_MODE must be PRESERVE, CIRCULAR_SHORT, CIRCULAR_LINEAR or CALTECH_LINEAR')
    if 'WIND_TYPE' not in s:s['WIND_TYPE']=''
    types=s.WIND_TYPE.fillna('').astype(str).str.upper()
    s['WS_original']=pd.to_numeric(s.WS,errors='coerce')
    s['WD_original']=pd.to_numeric(s.WD,errors='coerce')
    initial_ws_missing=s.WS.isna();initial_wd_missing=s.WD.isna()
    # Preserve ORIGINAL missing-run lengths even when confirmed calm codes resolve some hours.
    _,_,original_ws_gaps,_,_=_wind_fill_all_hours(s.WS,angular=False)
    _,_,original_wd_gaps,_,_=_wind_fill_all_hours(s.WD,angular=False)
    s['WS_METHOD']=np.where(s.WS.notna(),'NOAA_OBSERVED','MISSING_UNRESOLVED')
    s['WD_METHOD']=np.where(s.WD.notna(),'NOAA_OBSERVED','MISSING_UNRESOLVED')
    recovered_speed=types.eq('C') & s.WS.isna()
    s.loc[recovered_speed,'WS']=0.0
    s.loc[recovered_speed,'WS_METHOD']='CALM_ZERO_FROM_NOAA_TYPE'
    # Gaps are computed after explicitly reported calm-speed recovery.
    if mode in ('CIRCULAR_LINEAR','CALTECH_LINEAR'):
        s.WS,ws_tags,ws_gaps,ws_lo,ws_hi=_wind_fill_all_hours(s.WS,angular=False)
        changed=s.WS_METHOD.eq('MISSING_UNRESOLVED')
        s.loc[changed,'WS_METHOD']=ws_tags[changed.to_numpy()]
    else:
        _,_,ws_gaps,_,_=_wind_fill_all_hours(s.WS,angular=False)
        ws_lo=ws_hi=0
    # Fill truly calm directions with 0°, including calm speeds inferred by NOAA.
    calm=s.WD.isna() & s.WS.eq(0)
    s.loc[calm,'WD']=0.0
    s.loc[calm,'WD_METHOD']='CALM_ZERO'
    variable=s.WD.isna() & types.eq('V') & s.WS.gt(0)
    if mode in ('CIRCULAR_LINEAR','CALTECH_LINEAR'):
        s.WD,wd_tags,wd_gaps,_,_=_wind_fill_all_hours(s.WD,angular=mode=='CIRCULAR_LINEAR')
        changed=s.WD_METHOD.eq('MISSING_UNRESOLVED')
        s.loc[changed,'WD_METHOD']=wd_tags[changed.to_numpy()]
        variable_filled=variable & s.WD.notna()
        s.loc[variable_filled,'WD_METHOD']=s.loc[variable_filled,'WD_METHOD'].astype(str)+'+VARIABLE_WIND_DIRECTION_IMPUTED'
    else:
        _,_,wd_gaps,_,_=_wind_fill_all_hours(s.WD,angular=False)
        if mode=='CIRCULAR_SHORT':
            orig=pd.to_numeric(s.WD,errors='coerce').to_numpy(dtype=float)
            obs=np.flatnonzero(np.isfinite(orig))
            for ia,ib in zip(obs[:-1],obs[1:]):
                ng=ib-ia-1
                if not 0<ng<=max_gap or variable.iloc[ia+1:ib].any():continue
                delta=((orig[ib]-orig[ia]+180)%360)-180
                for j in range(ia+1,ib):
                    s.iat[j,s.columns.get_loc('WD')]=(orig[ia]+delta*(j-ia)/(ib-ia))%360
                    s.iat[j,s.columns.get_loc('WD_METHOD')]='CIRCULAR_SHORT_INTERPOLATION'
        variable_filled=pd.Series(False,index=s.index)
        s.loc[variable & s.WD.isna(),'WD_METHOD']='VARIABLE_UNDEFINED'
    s['WS_GAP_H']=original_ws_gaps
    s['WD_GAP_H']=original_wd_gaps
    ws_fill=s.WS_METHOD.astype(str).str.contains('INTERPOLATION|EXTRAPOLATION')
    wd_fill=s.WD_METHOD.astype(str).str.contains('INTERPOLATION|EXTRAPOLATION')
    result={
        'wind_fill_algorithm_version':'V10.1_full_linear_extrapolation',
        'wind_direction_mode':mode,
        'wind_direction_max_gap_hours':int(max_gap),
        'wind_speed_missing_before_hours':int(initial_ws_missing.sum()),
        'wind_direction_missing_before_hours':int(initial_wd_missing.sum()),
        'wind_speed_calm_type_recovered_hours':int(recovered_speed.sum()),
        'wind_direction_calm_zero_hours':int(calm.sum()),
        'wind_direction_variable_undefined_hours':int((variable & s.WD.isna()).sum()),
        'wind_direction_variable_imputed_hours':int(variable_filled.sum()),
        'wind_speed_interpolated_hours':int(s.WS_METHOD.astype(str).str.contains('INTERPOLATION').sum()),
        'wind_speed_extrapolated_start_hours':int(s.WS_METHOD.astype(str).str.contains('EXTRAPOLATION_START').sum()),
        'wind_speed_extrapolated_end_hours':int(s.WS_METHOD.astype(str).str.contains('EXTRAPOLATION_END').sum()),
        'wind_direction_interpolated_hours':int(s.WD_METHOD.astype(str).str.contains('INTERPOLATION').sum()),
        'wind_direction_extrapolated_start_hours':int(s.WD_METHOD.astype(str).str.contains('EXTRAPOLATION_START').sum()),
        'wind_direction_extrapolated_end_hours':int(s.WD_METHOD.astype(str).str.contains('EXTRAPOLATION_END').sum()),
        'wind_speed_clipped_low_hours':int(ws_lo),
        'wind_speed_clipped_high_hours':int(ws_hi),
        'wind_speed_max_original_missing_run_hours':int(max_missing_run(pd.Series(initial_ws_missing.map({True:np.nan,False:0}),index=s.index))),
        'wind_direction_max_original_missing_run_hours':int(max_missing_run(pd.Series(initial_wd_missing.map({True:np.nan,False:0}),index=s.index))),
        # Same, without hours reported as calm or variable (no direction by definition, not outages).
        'wind_direction_max_outage_run_hours':int(max_missing_run(pd.Series(
            (initial_wd_missing & ~types.isin(['C','V']) & ~s.WS_original.eq(0)).map({True:np.nan,False:0}),index=s.index))),
        'wind_speed_max_filled_run_hours':int(np.max(s.WS_GAP_H.to_numpy()[ws_fill.to_numpy()],initial=0)),
        'wind_direction_max_filled_run_hours':int(np.max(s.WD_GAP_H.to_numpy()[wd_fill.to_numpy()],initial=0)),
        'wind_speed_remaining_missing_hours':int(s.WS.isna().sum()),
        'wind_direction_remaining_missing_hours':int(s.WD.isna().sum()),
    }
    return s,result

def repair_opaque_sky(surface,mode='TOTAL_COVER_PROXY'):
    """Use observed NOAA opaque first, optional *explicitly modeled* proxy.

    TOTAL_COVER_PROXY assumes all visible cover is opaque; it is an upper
    bound, not an observation and may overestimate infrared radiation. The GA
    layer maximum is a still weaker proxy. The method is logged per hour.
    """
    s=surface.copy()
    s['OC_METHOD']=np.where(s.OC.notna(),'NOAA_GF1_OPAQUE','MISSING')
    mode=mode.upper()
    if mode not in ('PRESERVE','TOTAL_COVER_PROXY'):
        raise ValueError('opaque_mode must be PRESERVE or TOTAL_COVER_PROXY')
    if mode=='TOTAL_COVER_PROXY':
        source=s.OC.isna() & s.TC.notna()
        s.loc[source,'OC']=s.loc[source,'TC']
        s.loc[source,'OC_METHOD']=np.where(
            s.loc[source,'TC_method'].eq('GF1'),'MODELED_TOTAL_GF1_UPPER_BOUND',
            'MODELED_GA_LAYER_PROXY')
    # data QA: every reported opaque <= total
    over=s.OC.notna() & s.TC.notna() & (s.OC>s.TC)
    s.loc[over,'OC']=s.loc[over,'TC']
    return s,{'opaque_sky_model_mode':mode,
              'opaque_sky_GF1_hours':int(s.OC_METHOD.eq('NOAA_GF1_OPAQUE').sum()),
              'opaque_sky_total_proxy_hours':int(s.OC_METHOD.eq('MODELED_TOTAL_GF1_UPPER_BOUND').sum()),
              'opaque_sky_GA_proxy_hours':int(s.OC_METHOD.eq('MODELED_GA_LAYER_PROXY').sum()),
              'opaque_sky_remaining_missing_hours':int(s.OC.isna().sum())}


def precip_diagnostics(surface):
    duration=pd.to_numeric(surface.PC_PERIOD,errors='coerce')
    observed=pd.to_numeric(surface.PC,errors='coerce')
    return {
        'precip_1hour_reports':int((duration==1).sum()),
        'precip_multi_hour_reports':int((duration>1).sum()),
        'precip_missing_reports':int(observed.isna().sum()),
        'precip_raw_nonoverlap_annual_total_not_estimated':True,
        'precip_event_time_resolution_caveat':'NOAA AA observed period totals may overlap across neighboring reports; DO NOT sum as independent hourly precipitation.'}

def energyplus_infrared(db_c, dp_c, opaque_tenths, cloud_missing_policy='DEFAULT_5'):
    """EnergyPlus default Clark-Allen/Walton longwave correlation (W/m2).

    For absent GF1 opaque cloud cover: DEFAULT_5 explicitly uses 5 tenths,
    while LEAVE_MISSING retains EPW 9999. No ERA5/CDS data are fetched.
    Returns (estimated_ir, counts_flags).
    """
    d = pd.to_numeric(db_c,errors='coerce').to_numpy(dtype=float)
    p = pd.to_numeric(dp_c,errors='coerce').to_numpy(dtype=float)
    cloud = pd.to_numeric(opaque_tenths,errors='coerce').to_numpy(dtype=float).copy()
    missing=~np.isfinite(cloud)
    mode=str(cloud_missing_policy).upper()
    if mode=='DEFAULT_5': cloud[missing]=5.0
    elif mode=='LEAVE_MISSING': pass
    else: raise ValueError('cloud_missing_policy must be DEFAULT_5 or LEAVE_MISSING')
    eps_clear=0.787 + 0.764*np.log((p+273.15)/273.0)
    multiplier=1 + .0224*cloud - .0035*cloud**2 + .00028*cloud**3
    ir=5.6697e-8 * (d+273.15)**4 * eps_clear * multiplier
    # Respect the missing sentinel instead of writing NaN into EPW.
    ir[~np.isfinite(ir)] = np.nan
    return ir, {'infrared_opaque_cloud_missing_hours':int(missing.sum()),
                'infrared_5_tenths_assumed_hours':int(missing.sum()) if mode=='DEFAULT_5' else 0,
                'infrared_missing_hours':int(np.isnan(ir).sum()),
                'infrared_method':'EnergyPlus Clark-Allen/Walton modeled from NOAA DB/DP/GF1 opaque sky',
                'infrared_cloud_missing_policy':mode}


def optional_epw_numeric(series, multiplier, low, high, missing, decimals=0):
    """Map valid source values; keep true unknowns as EPW field-specific sentinel."""
    s=pd.to_numeric(series,errors='coerce') * multiplier
    good=s.notna() & (s>=low) & (s<high)
    s=s.where(good,missing)
    return s.round(decimals)


def extraterrestrial_from_nsrdb(year, idx, rad):
    """Approximate extraterrestrial solar fields using day-of-year + NSRDB zenith.
    These fields are ancillary and not used in EnergyPlus building load equations.
    Zenith is the existing NSRDB hour midpoint value; no resampling performed.
    """
    day=np.asarray(idx.dayofyear,dtype=float)
    e0=1367.0 * (1.0 + 0.033*np.cos(2*np.pi*day/365.25))
    z=pd.to_numeric(rad.get('ZENITH_DEG',pd.Series(np.nan,index=idx)), errors='coerce').to_numpy(dtype=float)
    cosz=np.cos(np.deg2rad(z))
    valid=np.isfinite(z)&(z>=0)&(z<=180)
    eh=np.where(valid,np.maximum(0.0,e0*cosz),9999.0)
    edn=np.where(valid,np.where(cosz>0,e0,0),9999.0)
    return np.rint(eh).astype(int),np.rint(edn).astype(int)

# Perez et al. (1990) luminous efficacy model: daylight illuminance from irradiance.
# Perez R, Ineichen P, Seals R, Michalsky J, Stewart R. "Modeling daylight availability and
# irradiance components from direct and global irradiance". Solar Energy 44(5):271-289, 1990.
# Coefficient table (sky-clearness bins 1..8): a, b, c, d for global, direct and diffuse
# efficacy; a, c, c', d for zenith luminance. (Some published copies give -5.52 instead of
# -5.53 for diffuse bin 3, c; the difference is below 5 lux.)
PEREZ_EPS_BINS = np.array([1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200])
PEREZ_GLOBAL = np.array([[96.63, -0.47, 11.50, -9.16], [107.54, 0.79, 1.79, -1.19],
                         [98.73, 0.70, 4.40, -6.95], [92.72, 0.56, 8.36, -8.31],
                         [86.73, 0.98, 7.10, -10.94], [88.34, 1.39, 6.06, -7.60],
                         [78.63, 1.47, 4.93, -11.37], [99.65, 1.86, -4.46, -3.15]])
PEREZ_DIRECT = np.array([[57.20, -4.55, -2.98, 117.12], [98.99, -3.46, -1.21, 12.38],
                         [109.83, -4.90, -1.71, -8.81], [110.34, -5.84, -1.99, -4.56],
                         [106.36, -3.97, -1.75, -6.16], [107.19, -1.25, -1.51, -26.73],
                         [105.75, 0.77, -1.26, -34.44], [101.18, 1.58, -1.10, -8.29]])
PEREZ_DIFFUSE = np.array([[97.24, -0.46, 12.00, -8.91], [107.22, 1.15, 0.59, -3.95],
                          [104.97, 2.96, -5.53, -8.77], [102.39, 5.59, -13.95, -13.90],
                          [100.71, 5.94, -22.75, -23.74], [106.42, 3.83, -36.15, -28.83],
                          [141.88, 1.90, -53.24, -14.03], [152.23, 0.35, -45.27, -7.98]])
PEREZ_ZENITH = np.array([[40.86, 26.77, -29.59, -45.75], [26.58, 14.73, 58.46, -21.25],
                         [19.34, 2.28, 100.00, 0.25], [13.25, -1.39, 124.79, 15.66],
                         [14.47, -5.09, 160.09, 9.13], [19.76, -3.88, 154.61, -19.21],
                         [28.39, -9.67, 151.58, -69.39], [42.91, -19.62, 130.80, -164.08]])


def perez_illuminance(ghi, dni, dhi, zenith_deg, dew_point_c, day_of_year,
                      fallback_efficacy=(110.0, 105.0, 119.0)):
    """Global, direct-normal and diffuse illuminance (lux) and zenith luminance (cd/m2).

    Uses the same hourly zenith angle as the radiation (NSRDB hour midpoint), so the
    illuminance stays on the radiation's clock. Hours with the sun down and no radiation
    are 0. Hours with radiation but no valid Perez inputs (zenith missing, sun at or below
    the horizon, or zero diffuse) use the fixed efficacies, and zenith luminance 9999.
    Returns a dict of integer arrays plus 'fallback_hours'.
    """
    ghi, dni, dhi = (np.clip(np.asarray(x, dtype=float), 0, None) for x in (ghi, dni, dhi))
    zdeg = np.asarray(zenith_deg, dtype=float)
    td = np.asarray(dew_point_c, dtype=float)
    doy = np.asarray(day_of_year, dtype=float)

    ok = np.isfinite(zdeg) & (zdeg < 90) & (dhi > 0) & (ghi > 0)
    zd = np.where(ok, zdeg, 0.0)
    z = np.radians(zd)
    cosz = np.cos(z)
    e0 = 1367.0 * (1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.25))
    air_mass = 1.0 / (cosz + 0.50572 * (96.07995 - zd) ** -1.6364)        # Kasten & Young 1989
    safe_dhi = np.where(ok, dhi, 1.0)
    eps = ((safe_dhi + dni) / safe_dhi + 1.041 * z ** 3) / (1 + 1.041 * z ** 3)  # sky clearness
    delta = np.where(ok, safe_dhi * air_mass / e0, 1.0)                    # sky brightness
    w = np.exp(0.07 * np.where(np.isfinite(td), td, 10.0) - 0.075)         # precipitable water, cm
    i = np.digitize(eps, PEREZ_EPS_BINS)
    ln_delta = np.log(np.maximum(delta, 1e-6))

    a, b, c, d = PEREZ_GLOBAL[i].T
    glob = ghi * (a + b * w + c * cosz + d * ln_delta)
    a, b, c, d = PEREZ_DIRECT[i].T
    direct = dni * np.maximum(a + b * w + c * np.exp(5.73 * z - 5) + d * delta, 0)
    a, b, c, d = PEREZ_DIFFUSE[i].T
    diffuse = dhi * (a + b * w + c * cosz + d * ln_delta)
    a, c, cp, d = PEREZ_ZENITH[i].T
    zenith_lum = dhi * (a + c * cosz + cp * np.exp(-3 * z) + d * delta)

    fallback = ~ok & ((ghi > 0) | (dni > 0) | (dhi > 0))
    ge, de, fe = fallback_efficacy
    glob = np.where(ok, glob, np.where(fallback, ghi * ge, 0.0))
    direct = np.where(ok, direct, np.where(fallback, dni * de, 0.0))
    diffuse = np.where(ok, diffuse, np.where(fallback, dhi * fe, 0.0))
    clean = lambda x: np.rint(np.clip(x, 0, None)).astype(int)
    # EPW reserves 9999 cd/m2 as the zenith-luminance missing code, so valid values are
    # capped at 9998; 9999 marks only the fallback hours.
    zenith = np.where(fallback, 9999, np.minimum(clean(np.where(ok, zenith_lum, 0.0)), 9998))
    return {'global': clean(glob), 'direct': clean(direct), 'diffuse': clean(diffuse),
            'zenith': zenith, 'fallback_hours': int(fallback.sum())}

# Sky-cover gap filling (total and opaque), following ISO 15927-4 clause 4 ("missing values
# shall be generated by linear interpolation or by estimate") and the TMY2/EPW data-source
# flags: B = linearly interpolated, C = non-linearly filled gap of 13-36 h, E = estimated.
# Thresholds from hold-out tests at ten U.S. stations (2022): linear interpolation is the most
# accurate up to 12 h, the blended cloud-type estimate for 13-36 h, the estimate alone beyond.
# Estimates come from NSRDB cloud_type, mapped to cover with this station-year's own data.
NSRDB_UNUSABLE_CLOUD_TYPES = (10, -15)       # 'unknown' and 'N/A' in NSRDB cloud_type
SKY_FLAG_POSITIONS = {'total': 28, 'opaque': 30}   # character index of the source flag in the EPW flag string


def _cloud_type_map(cover, ctype, min_hours=20):
    """Median observed cover (tenths) for each NSRDB cloud type, from hours with both."""
    both = np.isfinite(cover) & np.isfinite(ctype) & ~np.isin(ctype, NSRDB_UNUSABLE_CLOUD_TYPES)
    table = pd.Series(cover[both]).groupby(ctype[both].astype(int)).agg(['median', 'size'])
    table = table[table['size'] >= min_hours]['median']
    return {int(k): float(v) for k, v in table.items()}


def _fill_cover(cover, ctype, max_linear=12, max_shaped=36):
    """Fill one cover series. Returns (filled values, flag letters, estimate check)."""
    v = np.asarray(cover, dtype=float).copy()
    ct = np.asarray(ctype, dtype=float)
    n = len(v)
    flags = np.full(n, '', dtype=object)
    mapping = _cloud_type_map(v, ct)
    est = np.array([mapping.get(int(c), np.nan) if np.isfinite(c) else np.nan for c in ct])

    # How well the cloud-type estimate matches observed cover at this station (reported).
    both = np.isfinite(v) & np.isfinite(est)
    check = {'estimate_mae_tenths': round(float(np.mean(np.abs(v[both] - est[both]))), 2) if both.any() else None,
             'estimate_compared_hours': int(both.sum()), 'cloud_types_mapped': len(mapping)}

    missing = ~np.isfinite(v)
    edges = np.flatnonzero(np.diff(np.r_[False, missing, False].astype(int)))
    for start, stop in zip(edges[::2], edges[1::2]):
        idx = np.arange(start, stop)
        left, right = start - 1, stop
        has_left, has_right = left >= 0, right < n
        length = stop - start
        if length <= max_linear and has_left and has_right:
            w = (idx - left) / (right - left)
            v[idx] = (1 - w) * v[left] + w * v[right]
            flags[idx] = 'B'
            continue
        e = est[idx]
        ok = np.isfinite(e)
        if not ok.any():
            continue                                   # no usable cloud type: leave missing
        if length <= max_shaped and has_left and has_right and np.isfinite(est[left]) and np.isfinite(est[right]):
            # Estimate, shifted so it meets the recorded values at both ends of the gap.
            w = (idx - left) / (right - left)
            shift = (1 - w) * (v[left] - est[left]) + w * (v[right] - est[right])
            v[idx[ok]] = e[ok] + shift[ok]
            flags[idx[ok]] = 'C'
        else:
            v[idx[ok]] = e[ok]
            flags[idx[ok]] = 'E'
    # Round only the filled hours to whole tenths; recorded values (e.g. 2.5 from oktas) stay
    # as observed, so the longwave calculation uses them unchanged. The EPW writer rounds later.
    filled = flags != ''
    v[filled] = np.clip(np.rint(v[filled]), 0, 10)
    return v, flags, check


def infill_sky_cover(total, opaque, cloud_type, max_linear=12, max_shaped=36):
    """Fill missing total and opaque sky cover (tenths); opaque is kept <= total.

    total, opaque: hourly cover in tenths with NaN for missing (surface.TC, surface.OC).
    cloud_type: NSRDB cloud_type for the same hours (NaN if unavailable, e.g. PVGIS).
    Recorded values are never changed. Returns (total, opaque, flags, report), where
    flags is a dict of per-hour source letters ('' where unchanged) for each field.
    """
    t, ft, ct_t = _fill_cover(total, cloud_type, max_linear, max_shaped)
    o, fo, ct_o = _fill_cover(opaque, cloud_type, max_linear, max_shaped)
    over = np.isfinite(t) & np.isfinite(o) & (o > t) & (fo != '')
    o[over] = t[over]                                  # only filled opaque values are lowered
    report = {f'{name}_sky_{label}_hours': int((f == letter).sum())
              for name, f in (('total', ft), ('opaque', fo))
              for letter, label in (('B', 'interpolated'), ('C', 'shaped_fill'), ('E', 'estimated'))}
    report.update({'total_sky_still_missing_hours': int((~np.isfinite(t)).sum()),
                   'opaque_sky_still_missing_hours': int((~np.isfinite(o)).sum()),
                   'sky_cover_fill_method': 'linear <=12 h (B); NSRDB cloud-type estimate blended to '
                                            'recorded ends for 13-36 h (C); cloud-type estimate >=37 h (E)'})
    for name, chk in (('total', ct_t), ('opaque', ct_o)):        # flat keys for the QA manifest CSV
        for key, val in chk.items():
            report[f'{name}_sky_{key}'] = val
    return t, o, {'total': ft, 'opaque': fo}, report


def source_flags_with_sky_fill(base_flags, flags):
    """Per-hour EPW data-source strings with B/C/E at the total/opaque sky-cover positions."""
    out = []
    for ft, fo in zip(flags['total'], flags['opaque']):
        s = list(base_flags)
        if ft: s[SKY_FLAG_POSITIONS['total']] = ft
        if fo: s[SKY_FLAG_POSITIONS['opaque']] = fo
        out.append(''.join(s))
    return out

def build_epw(st,year,surface,rad,source_description,cloud_missing_policy='DEFAULT_5',era5_ir=None,era5_mode='FALLBACK'):
    expected=expected_local_index(year)
    if not surface.index.equals(expected) or not rad.index.equals(expected):
        raise ValueError('Weather/radiation time index mismatch')
    n=len(expected)
    p_pa,rh=rh_and_station_pressure(surface.DB,surface.DP,surface.PR,st['elevation_m'])
    if (~np.isfinite(p_pa)).any() or (p_pa<50000).any() or (p_pa>115000).any():
        raise ValueError('Derived atmospheric station pressure physically invalid')
    if (~np.isfinite(rh)).any(): raise ValueError('Derived RH invalid')
    # Fill sky cover first, so longwave uses the filled opaque cover instead of the
    # cloud_missing_policy default in hours where NOAA reported no cloud.
    sky_t,sky_o,sky_flags,sky_report=infill_sky_cover(
        pd.to_numeric(surface.TC,errors='coerce').to_numpy(dtype=float),
        pd.to_numeric(surface.OC,errors='coerce').to_numpy(dtype=float),
        pd.to_numeric(rad.get('CLOUD_TYPE',pd.Series(np.nan,index=expected)),errors='coerce').to_numpy(dtype=float))
    infrared, ir_report=energyplus_infrared(surface.DB,surface.DP,pd.Series(sky_o,index=expected),cloud_missing_policy)
    ir_report.update(sky_report)
    # Fast primary is the existing V5 physical model. ERA5 point time-series
    # can replace it when PREFER is selected, or cover actual missing values.
    model_ir=np.asarray(infrared,dtype=float).copy()
    valid_era5=np.zeros(n,dtype=bool)
    mode=str(era5_mode).upper()
    if era5_ir is not None:
        candidate=pd.to_numeric(era5_ir.reindex(expected),errors='coerce').to_numpy(dtype=float)
        valid_era5=np.isfinite(candidate)&(candidate>=100)&(candidate<=700)
        use=valid_era5 if mode=='PREFER' else valid_era5&~np.isfinite(model_ir)
        infrared=np.where(use,candidate,model_ir)
    else:
        use=np.zeros(n,dtype=bool)
    ir_report.update({'infrared_ERA5_hours':int(use.sum()),
        'infrared_V5_modeled_hours':int(np.isfinite(infrared).sum()-use.sum()),
        'infrared_missing_hours':int((~np.isfinite(infrared)).sum()),
        'infrared_source_policy':mode,
        'infrared_method':'V5 EnergyPlus modeled primary; optional ERA5 point time-series backup'})
    ext_h,ext_dn=extraterrestrial_from_nsrdb(year,expected,rad)
    ill=perez_illuminance(rad.GHI.to_numpy(),rad.DNI.to_numpy(),rad.DHI.to_numpy(),
                          pd.to_numeric(rad.get('ZENITH_DEG',pd.Series(np.nan,index=expected)),errors='coerce').to_numpy(),
                          surface.DP.to_numpy(),expected.dayofyear)
    ir_report.update({'illuminance_method':'Perez et al. (1990) luminous efficacy; fixed 110/105/119 lm/W only where the sun is at or below the horizon',
                      'illuminance_fallback_hours':ill['fallback_hours']})
    out=pd.DataFrame(index=expected,columns=EPW_COLUMNS)
    values={
        'year':year,'month':expected.month,'day':expected.day,'hour':expected.hour+1,'minute':60,
        'data_source_uncertainty':source_flags_with_sky_fill('?9'*21,sky_flags),
        'dry_bulb_temperature':surface.DB.round(2),'dew_point_temperature':surface.DP.round(2),
        'relative_humidity':np.clip(np.rint(rh),0,100).astype(int),
        'atmospheric_station_pressure':np.rint(p_pa).astype(int),
        'extraterrestrial_horizontal_radiation':ext_h,'extraterrestrial_direct_normal_radiation':ext_dn,
        'horizontal_infrared_radiation_intensity':np.where(np.isfinite(infrared),np.rint(infrared),9999).astype(int),
        'global_horizontal_radiation':np.rint(rad.GHI).astype(int),
        'direct_normal_radiation':np.rint(rad.DNI).astype(int),
        'diffuse_horizontal_radiation':np.rint(rad.DHI).astype(int),
        'global_horizontal_illuminance':ill['global'],
        'direct_normal_illuminance':ill['direct'],
        'diffuse_horizontal_illuminance':ill['diffuse'],
        'zenith_luminance':ill['zenith'],
        'wind_direction':np.where(surface.WS.eq(0)&surface.WD.isna(),0,surface.WD.fillna(999)).astype(int),
        'wind_speed':surface.WS.fillna(999).round(2),
        'total_sky_cover':optional_epw_numeric(pd.Series(sky_t,index=expected),1,0,10.01,99),
        'opaque_sky_cover':optional_epw_numeric(pd.Series(sky_o,index=expected),1,0,10.01,99),
        'visibility':(surface.VIS/1000).fillna(9999).round(2),
        'ceiling_height':surface.CEIL.fillna(99999).round(0),
        'present_weather_observation':9,'present_weather_codes':'999999999',
        'precipitable_water':optional_epw_numeric(rad.get('PW_CM',pd.Series(np.nan,index=expected)),10,0,999,999,1),
        'aerosol_optical_depth':optional_epw_numeric(rad.get('AOD',pd.Series(np.nan,index=expected)),1,0,.999,.999,3),
        'snow_depth':optional_epw_numeric(surface.SNOW_CM,1,0,999,999,1),
        'days_since_last_snowfall':99,
        'albedo':optional_epw_numeric(rad.get('ALBEDO',pd.Series(np.nan,index=expected)),1,0,1.001,999,3),
        'liquid_precipitation_depth':surface.PC.fillna(999).round(2),
        'liquid_precipitation_quantity':surface.PC_PERIOD.fillna(99).round(0).astype(int)
    }
    for key,val in values.items(): out[key]=val
    assert out.shape==(n,35)
    clean=lambda s: str(s).replace(',',' ').replace('\n',' ').strip()
    country=clean(st['country']); region=clean(st['state']); name=clean(st['name'])
    tz=st['tz_offset']
    headers=[
        f"LOCATION,{name},{region},{country},NOAA-GlobalHourly,{st['station_id']},{st['lat']:.4f},{st['lon']:.4f},{tz:g},{st['elevation_m']:.1f}",
        'DESIGN CONDITIONS,0','TYPICAL/EXTREME PERIODS,0','GROUND TEMPERATURES,0',
        f'HOLIDAYS/DAYLIGHT SAVINGS,{"Yes" if calendar.isleap(year) else "No"},0,0,0',
        f'COMMENTS 1,Chronological AMY {year}; NOAA hourly met plus {clean(source_description)} radiation',
        f'COMMENTS 2,NOAA station weather; {clean(source_description)} shortwave; V5 modeled longwave with optional ERA5 point backup; see hourly provenance',
        f'DATA PERIODS,1,1,Data,{date(year,1,1).strftime("%A")},1/1,12/31'
    ]
    return headers,out,ir_report


def write_epw(headers,df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        f.write('\n'.join(headers)+'\n')
        df.to_csv(f,index=False,header=False,lineterminator='\n',na_rep='')
    try:
        check=pd.read_csv(path,skiprows=8,header=None)
    except pd.errors.ParserError as exc:
        raise ValueError(f'EPW postwrite parsing failed for {path}: {exc}') from exc
    if check.shape!=(len(df),35) or check.iloc[0,3]!=1 or check.iloc[-1,3]!=24:
        path.unlink(missing_ok=True)
        raise ValueError('EPW postwrite structural validation failed')
    return path



# ---------- V7: optional single-request ERA5 longwave backup ----------------
import zipfile
import tempfile
import hashlib

ERA5_TIMESERIES_DATASET='reanalysis-era5-single-levels-timeseries'
ERA5_TIMESERIES_VARIABLES=['surface_thermal_radiation_downwards','total_cloud_cover']


US_TERRITORIES = ('US', 'PR', 'VI', 'GU', 'AS', 'MP')


def city_from_coordinates(st, radius_km=30.0, fallback_km=75.0):
    """City name for a station from its coordinates, e.g. 'Las_Vegas_NV' or 'Dubai_AE'.

    Uses the offline GeoNames city list (towns of 5000+ people) of the geonamescache
    package: the largest city within radius_km of the station, otherwise the nearest
    city within fallback_km. Only cities in the station's country are considered (US
    stations: the US and its territories), so a border airport is not named after the
    city across the border. US names end in the state, others in the ISO country code.
    Returns None if the package is missing or no city is near, so the caller can fall
    back to the NOAA station name.
    """
    try:
        import geonamescache
        cities = pd.DataFrame(geonamescache.GeonamesCache(min_city_population=5000).get_cities().values())
    except Exception:
        return None
    import unicodedata
    la, lo = np.radians(float(st['lat'])), np.radians(float(st['lon']))
    lat, lon = np.radians(cities.latitude.to_numpy(float)), np.radians(cities.longitude.to_numpy(float))
    cities['km'] = 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((lat - la) / 2) ** 2
                                                  + np.cos(la) * np.cos(lat) * np.sin((lon - lo) / 2) ** 2))
    if str(st.get('country', '')).strip().upper() == 'US':
        cities = cities[cities.countrycode.isin(US_TERRITORIES)]
    elif not cities.empty:
        cities = cities[cities.countrycode == cities.loc[cities.km.idxmin(), 'countrycode']]
    near = cities[cities.km <= radius_km]
    if not near.empty:
        best = near.sort_values(['population', 'km'], ascending=[False, True]).iloc[0]
    else:
        near = cities[cities.km <= fallback_km]
        if near.empty:
            return None
        best = near.loc[near.km.idxmin()]
    name = unicodedata.normalize('NFKD', str(best['name'])).encode('ascii', 'ignore').decode()
    region = best.admin1code if best.countrycode == 'US' else best.countrycode
    label = re.sub(r'[^A-Za-z0-9]+', '_', f'{name} {region}').strip('_')
    return label or None


def clean_city_name(st, city_override=None):
    """File-safe city name; explicit override recommended for ambiguous stations."""
    label=str(city_override or st.get('name') or 'UnknownCity').strip()
    if not city_override:
        # NOAA name includes airport suffixes, so use the first location token
        # unless user supplies city_override (e.g. New_York).
        label=re.split(r'\s+(?:INTL|INTERNATIONAL|MUNICIPAL|REGIONAL|AIRPORT|AP|A/P)\b',label,flags=re.I)[0]
        label=label.split(' - ')[0]
        label=label.strip().split()[0] if len(label.split())>1 else label
    label=re.sub(r'[^A-Za-z0-9]+','_',label).strip('_')
    return label or 'UnknownCity'


def era5_timeseries_request(year,st):
    """The fast ECMWF ARCO-backed point product, not the monthly gridded dataset."""
    return {'variable':ERA5_TIMESERIES_VARIABLES,
            'location':{'longitude':float(st['lon']),'latitude':float(st['lat'])},
            'date':[f'{int(year)-1}-12-30/{int(year)+1}-01-02'],
            'data_format':'netcdf'}


def _era5_open_dataset_from_archive(download_path):
    """Read CDS response which can be single NetCDF or zipped NetCDFs."""
    import xarray as xr
    path=Path(download_path)
    if not path.is_file() or path.stat().st_size<100:
        raise RuntimeError('ERA5 time-series response missing or too small')
    datasets=[]
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            candidates=[name for name in z.namelist() if name.lower().endswith(('.nc','.nc4','.netcdf'))]
            if not candidates: raise ValueError('ERA5 ZIP lacks a NetCDF file')
            for name in candidates:
                with z.open(name) as member:
                    raw=member.read()
                import io
                try:
                    ds=xr.open_dataset(io.BytesIO(raw),engine='h5netcdf').load()
                except Exception:
                    with tempfile.NamedTemporaryFile(suffix='.nc') as fp:
                        fp.write(raw);fp.flush()
                        ds=xr.open_dataset(fp.name).load()
                datasets.append(ds)
    else:
        ds=xr.open_dataset(path).load()
        datasets.append(ds)
    if len(datasets)==1: return datasets[0]
    return xr.merge(datasets,compat='override',join='outer')


def parse_era5_timeseries_file(download_path,year,st):
    """Surface longwave J/m2 over PRIOR hour, indexed at hour end (UTC).

    EPW hour one covers local 00:00-01:00. So UTC valid_hour 01:00
    maps to local standard start 00:00 plus tz offset. NOT a solar
    timestamp; never modify NSRDB GHI/DNI/DHI through this function.
    """
    ds=_era5_open_dataset_from_archive(download_path)
    names=set(ds.variables)
    time_dim=next((x for x in ('valid_time','time','datetime','date') if x in names or x in ds.coords),None)
    if not time_dim: raise ValueError(f'ERA5 NetCDF contains no time coordinate: {sorted(names)[:30]}')
    t=pd.DatetimeIndex(pd.to_datetime(ds[time_dim].values,utc=True)).tz_localize(None)
    # Fast time series uses the nearest 0.25-degree grid cell; no interpolation.
    longwave_key=next((x for x in ('strd','surface_thermal_radiation_downwards','surface_thermal_radiation_downward') if x in ds.data_vars),None)
    if not longwave_key: raise ValueError('ERA5 NetCDF lacks surface thermal radiation downwards; variables: '+str(sorted(ds.data_vars)))
    arr=ds[longwave_key]
    for coord in list(arr.dims):
        if coord!=time_dim:
            arr=arr.isel({coord:0},drop=True)
    v=np.asarray(arr.values,dtype=float).reshape(-1)
    if len(v)!=len(t): raise ValueError('ERA5 time and longwave dimensions differ')
    units=str(ds[longwave_key].attrs.get('units','J m**-2')).lower().replace(' ','')
    if 'w/m' in units or 'wm-' in units:
        watts=v
    elif 'j' in units or not units:
        watts=v/3600.0
    else:
        raise ValueError(f'Unexpected ERA5 longwave units: {units}')
    offset=pd.Timedelta(hours=float(st['tz_offset']))
    ir=pd.Series(watts,index=t-pd.Timedelta(hours=1)+offset)
    ir=ir[~ir.index.duplicated(keep='last')].sort_index()
    idx=expected_local_index(year)
    ir=ir.reindex(idx).where(lambda x:x.between(100,700))
    return ir,{'era5_response_fields':sorted(ds.data_vars),
               'era5_valid_longwave_hours':int(ir.notna().sum()),
               'era5_raw_longwave_units':str(ds[longwave_key].attrs.get('units','J m-2')),
               'era5_time_alignment':'UTC validity time minus 1h plus local standard offset',
               'era5_grid_selection':'nearest 0.25-degree ERA5 grid cell'}


def era5_timeseries_longwave(year,st,cache_dir,cds_api_key=None,
                              cds_url='https://cds.climate.copernicus.eu/api',client=None):
    """One station-year request to CDS point time-series, plus disk caching."""
    idx=expected_local_index(year)
    empty=pd.Series(np.nan,index=idx,dtype=float)
    # normalized coordinate key permits shared grid-cell cache across stations
    lat=round(float(st['lat'])*4)/4
    lon=round(float(st['lon'])*4)/4
    cache=Path(cache_dir)/'era5_points_v7'/f'{lat:+08.2f}_{lon:+09.2f}_{year}_longwave_download'
    try:
        if not cache.is_file() or cache.stat().st_size<100:
            if client is None:
                if not cds_api_key: raise RuntimeError('CDS_API_KEY missing')
                import cdsapi
                client=cdsapi.Client(url=cds_url,key=cds_api_key,quiet=True,progress=False)
            req=era5_timeseries_request(year,st)
            req['location']={'latitude':lat,'longitude':lon}
            cache.parent.mkdir(parents=True,exist_ok=True)
            tmp=cache.with_name(cache.name+'.part')
            tmp.unlink(missing_ok=True)
            print(f'[AMY V7 ERA5] one point request: {year} nearest grid {lat:.2f},{lon:.2f}',flush=True)
            client.retrieve(ERA5_TIMESERIES_DATASET,req,str(tmp))
            if not tmp.exists(): raise RuntimeError('CDS did not create download')
            tmp.replace(cache)
        ir,meta=parse_era5_timeseries_file(cache,year,st)
        if not ir.notna().any(): raise RuntimeError('ERA5 time-series contains no aligned valid longwave')
        meta.update({'era5_status':'ok','era5_dataset':ERA5_TIMESERIES_DATASET})
        return ir,meta
    except Exception as exc:
        # Do not reuse corrupt downloads; leave cache to help diagnose only when valid.
        return empty,{'era5_status':f'{type(exc).__name__}: {str(exc)[:200]}',
                      'era5_valid_longwave_hours':0,'era5_dataset':ERA5_TIMESERIES_DATASET}




# ---------------------------------------------------------------------------
# V8 geographically adaptive shortwave selection and strict annual alignment
# ---------------------------------------------------------------------------
# PVGIS 5.3 coverage published by the European Commission is 2005–2023.
# Do not claim 2024+ radiation via this endpoint. Future datasets require a
# separately validated integration and should not silently borrow earlier years.
PVGIS5_3_BOUNDS=(2005, 2023)


def is_americas_location(st):
    """Broad station-screening heuristic, not an NSRDB service footprint polygon."""
    return -175.0 <= float(st['lon']) <= -30.0 and -60.0 <= float(st['lat']) <= 72.0


def radiation_source_priority(st, preferred='AUTO'):
    pref=str(preferred).upper().strip().replace('PVGIS-', '')
    if pref=='AUTO':
        # NSRDB first throughout the Americas. If it fails, try any SARAH3
        # coverage (including parts of South America) before ERA5 reanalysis.
        return (['NSRDB','PVGIS-SARAH3','PVGIS-ERA5']
                if is_americas_location(st) else ['PVGIS-SARAH3','PVGIS-ERA5'])
    aliases={'SARAH3':'PVGIS-SARAH3','ERA5':'PVGIS-ERA5',
             'PVGIS':'PVGIS-SARAH3'}
    normalized=aliases.get(pref,pref)
    if normalized not in ('NSRDB','PVGIS-SARAH3','PVGIS-ERA5'):
        raise ValueError('RADIATION_SOURCE must be AUTO, NSRDB, PVGIS-SARAH3, or PVGIS-ERA5')
    return [normalized]


def pvgis_hourly_utc(session, year, st, cache_dir, database):
    """Fetch PVGIS 5.3 hourly horizontal radiation samples, at their own UTC times.

    A single components=1 response supplies direct and diffuse radiation on a
    horizontal 0-degree plane (Gb(i), Gd(i), Gr(i)); reflected is ~0 at 0 tilt.
    Each value is a snapshot at its time stamp (SARAH: about hh:10 UTC, the
    satellite scan time), not an hourly mean, so the samples are returned with
    their exact times; merge_pvgis_local_year moves them to the EPW hour midpoints.
    """
    if not PVGIS5_3_BOUNDS[0] <= int(year) <= PVGIS5_3_BOUNDS[1]:
        raise ValueError(f'PVGIS 5.3 {database}: {year} outside 2005–2023 availability')
    if database not in ('PVGIS-SARAH3','PVGIS-ERA5'):
        raise ValueError('Unexpected PVGIS database')
    stid=str(st['station_id'])
    cache=Path(cache_dir)/'radiation'/'pvgis_v8'/stid/f'{year}_{database}_components1.json'
    params={'lat':float(st['lat']),'lon':float(st['lon']),
            'startyear':int(year),'endyear':int(year),
            'raddatabase':database,'components':1,'pvcalculation':0,
            'angle':0,'aspect':0,'usehorizon':0,'outputformat':'json'}
    try:
        f=cached_download(session,PVGIS_URL,cache,params)
        payload=json.loads(f.read_text(encoding='utf-8'))
        hourly=payload['outputs']['hourly']
        if not isinstance(hourly,list) or len(hourly)<8000:
            raise ValueError(f'PVGIS unexpectedly returned only {len(hourly)} hours')
        # The API's selected source must match our requested provenance.
        meta=payload.get('inputs',{}).get('meteo_data',{})
        actual=str(meta.get('radiation_db',''))
        if actual and actual.upper()!=database.upper():
            raise ValueError(f'PVGIS database mismatch: requested {database} but reported {actual}')
        df=pd.DataFrame(hourly)
        columns={'time','Gb(i)','Gd(i)','H_sun'}
        if not columns.issubset(df.columns):
            raise ValueError(f'PVGIS components response missing {columns-set(df.columns)}; fields={list(df.columns)}')
        raw=pd.to_datetime(df['time'],format='%Y%m%d:%H%M',errors='coerce')
        if raw.isna().any(): raise ValueError('PVGIS timestamps are invalid')
        minutes=raw.dt.minute
        times=pd.DatetimeIndex(raw)
        if times.has_duplicates: raise ValueError('Duplicate PVGIS UTC time stamps')
        beam=pd.to_numeric(df['Gb(i)'],errors='coerce').to_numpy(dtype=float,copy=True)
        diffuse=pd.to_numeric(df['Gd(i)'],errors='coerce').to_numpy(dtype=float,copy=True)
        refl=pd.to_numeric(df['Gr(i)'],errors='coerce').to_numpy(dtype=float) if 'Gr(i)' in df else np.zeros(len(df))
        # On a level plane the reflected component is 0; it is counted for QA only.
        invalid=~(np.isfinite(beam)&np.isfinite(diffuse))
        beam[invalid]=np.nan; diffuse[invalid]=np.nan
        out=pd.DataFrame({'BEAM_H':beam,'DHI':diffuse},index=times).sort_index()
        info={'pvgis_database':database,'pvgis_url':PVGIS_URL,
              'pvgis_utc_year':int(year),'pvgis_sample_minute_modes':','.join(str(int(m)) for m in sorted(minutes.unique())),
              'pvgis_nonzero_reflection_hours':int((np.abs(refl)>0.01).sum()),
              'pvgis_raw_json':str(cache),
              'pvgis_timestamp_processing':'PVGIS snapshots moved to EPW hour midpoints by '
                                            'interpolating the clearness index in time'}
        return out,info
    except Exception:
        # JSON that reached the cache but failed structural validation must not
        # poison future runs; retry only after re-downloading next time.
        # Valid JSON with incomplete annual coverage remains cached for review.
        if cache.exists():
            try:
                test=json.loads(cache.read_text(encoding='utf-8'))
                if 'outputs' not in test: cache.unlink(missing_ok=True)
            except Exception: cache.unlink(missing_ok=True)
        raise


def solar_geometry_utc(times, lat, lon):
    """cos(zenith) and extraterrestrial normal irradiance (W/m2) at UTC times.

    Spencer (1971) declination and equation of time (NOAA general solar position
    formulas, about 0.1 degree), sufficient for hourly weather files.
    """
    t=pd.DatetimeIndex(times)
    doy=t.dayofyear.to_numpy(dtype=float)
    hours=(t.hour+t.minute/60+t.second/3600).to_numpy(dtype=float)
    g=2*np.pi/365*(doy-1+(hours-12)/24)
    decl=(0.006918-0.399912*np.cos(g)+0.070257*np.sin(g)-0.006758*np.cos(2*g)
          +0.000907*np.sin(2*g)-0.002697*np.cos(3*g)+0.00148*np.sin(3*g))
    eot=229.18*(0.000075+0.001868*np.cos(g)-0.032077*np.sin(g)
                -0.014615*np.cos(2*g)-0.040849*np.sin(2*g))          # minutes
    hour_angle=np.deg2rad((hours*60+eot+4*lon)/4-180)
    phi=np.deg2rad(lat)
    cosz=np.sin(phi)*np.sin(decl)+np.cos(phi)*np.cos(decl)*np.cos(hour_angle)
    e0=1367.0*(1.0+0.033*np.cos(2*np.pi*doy/365.25))
    return np.clip(cosz,-1,1),e0


def pvgis_to_hour_midpoints(samples, mid_utc, lat, lon, min_cosz=0.05):
    """Move PVGIS snapshots (BEAM_H, DHI at their own UTC times) to hour midpoints.

    The beam and diffuse clearness indices (irradiance / horizontal extraterrestrial)
    are interpolated linearly in time between the two snapshots around each midpoint
    and multiplied by the extraterrestrial irradiance at the midpoint, so the solar
    geometry is exact at the midpoint. A snapshot with the sun lower than min_cosz
    gives no reliable clearness; then the other snapshot's is used, or the irradiance
    itself is interpolated if neither is usable. Midpoints with the sun below the
    horizon are 0. DNI = beam / cos(zenith) for a sun at least 2 degrees high.
    """
    st_times=pd.DatetimeIndex(samples.index)
    s_ns=st_times.asi8.astype(float); m_ns=pd.DatetimeIndex(mid_utc).asi8.astype(float)
    s_cz,s_e0=solar_geometry_utc(st_times,lat,lon)
    m_cz,m_e0=solar_geometry_utc(mid_utc,lat,lon)
    s_eh=s_e0*np.maximum(s_cz,0); m_eh=m_e0*np.maximum(m_cz,0)
    i1=np.clip(np.searchsorted(s_ns,m_ns),1,len(s_ns)-1); i0=i1-1
    span=s_ns[i1]-s_ns[i0]
    w=np.clip((m_ns-s_ns[i0])/np.where(span>0,span,1),0,1)
    too_far=(np.minimum(np.abs(m_ns-s_ns[i0]),np.abs(m_ns-s_ns[i1]))>2*3600e9)
    usable=s_cz>min_cosz
    out={}
    for col,cap in (('BEAM_H',1.2),('DHI',1.0)):
        v=samples[col].to_numpy(dtype=float)
        k=np.where(usable,np.clip(v/np.where(usable,s_eh,1),0,cap),np.nan)
        k0,k1,u0,u1=k[i0],k[i1],usable[i0],usable[i1]
        kt=np.where(u0&u1,(1-w)*k0+w*k1,np.where(u0,k0,np.where(u1,k1,np.nan)))
        value=kt*m_eh
        direct=(1-w)*v[i0]+w*v[i1]                     # low sun on both sides
        value=np.where(np.isfinite(kt),value,direct)
        value=np.where(m_cz>0,np.maximum(value,0),0.0)
        value[too_far|~np.isfinite(v[i0])|~np.isfinite(v[i1])]=np.nan
        out[col]=value
    beam,dhi=out['BEAM_H'],out['DHI']
    dni=np.where(m_cz>0.035,beam/np.where(m_cz>0.035,m_cz,1),0.0)
    dni=np.where(np.isfinite(beam),dni,np.nan)
    return pd.DataFrame({'GHI':beam+dhi,'DNI':dni,'DHI':dhi,
                         'ZENITH_DEG':np.rad2deg(np.arccos(m_cz)),
                         'PW_CM':np.nan,'AOD':np.nan,'ALBEDO':np.nan,
                         'CLOUD_TYPE':np.nan,'FILL_FLAG':np.nan},index=pd.DatetimeIndex(mid_utc))


def merge_pvgis_local_year(session, year, st, cache_dir, db, tz_offset):
    """PVGIS radiation for the EPW hours of one local-standard-time year.

    Uses adjacent UTC-year data when necessary. Every EPW hour gets the value at its
    midpoint (pvgis_to_hour_midpoints), which also handles half-hour and 45-minute
    time zones; no DST.
    """
    requested=int(year)
    tz=float(tz_offset)
    frames=[]; info=[]
    for yy in [requested]+(([requested+1] if tz<0 else [requested-1]) if tz!=0 else []):
        part,metadata=pvgis_hourly_utc(session,yy,st,cache_dir,db)
        frames.append(part);info.append(metadata)
    utc=pd.concat(frames).sort_index()
    if utc.index.has_duplicates: raise ValueError('PVGIS years contain overlapping UTC hours')
    target=expected_local_index(requested)
    # Each EPW hour (local standard time, no DST) is represented by its midpoint.
    mid_utc=target+pd.Timedelta(minutes=30)-pd.Timedelta(hours=tz)
    result=pvgis_to_hour_midpoints(utc,mid_utc,float(st['lat']),float(st['lon']))
    result.index=target
    method='PVGIS snapshots -> local standard hour midpoints (clearness interpolated in time); no DST'
    missing=int(result[['GHI','DNI','DHI']].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f'PVGIS {db} incomplete after time-zone merge: {missing}/{len(target)} local hours missing; '
                         'no artificial substitution permitted')
    result=validate_radiation(result,requested)
    meta={**info[0], 'radiation_source':db,
          'radiation_year':requested,'radiation_timebase':method,
          'pvgis_adjacent_utc_years':','.join(str(i['pvgis_utc_year']) for i in info[1:]),
          'radiation_url':PVGIS_URL,'radiation_is_reanalysis':db=='PVGIS-ERA5',
          'radiation_is_satellite_derived':db=='PVGIS-SARAH3',
          'borrowed_radiation_hours':0,'edge_wrapped_hours':0}
    return result,meta


def radiation_for_year_v8(session,year,st,cache,key,email,tz,preferred='AUTO'):
    """Try geographic sources in priority order; no synthetic radiation fills."""
    order=radiation_source_priority(st,preferred)
    errs=[]
    for source in order:
        try:
            if source=='NSRDB':
                # NLR GOES satellite coverage is validated by the API response.
                rad,prov=nsrdb_radiation(session,year,st,cache,key,email,tz)
                rad=validate_radiation(rad,year)
                prov.update({'radiation_is_reanalysis':False,'radiation_is_satellite_derived':True,
                             'borrowed_radiation_hours':0,'edge_wrapped_hours':0})
            else:
                rad,prov=merge_pvgis_local_year(session,year,st,cache,source,tz)
            prov['radiation_source_priority']=' > '.join(order)
            prov['radiation_fallback_count']=len(errs)
            prov['radiation_attempts_failed']=' | '.join(errs)
            prov['radiation_provider_selected']=source
            return rad,prov
        except Exception as exc:
            err=f'{source}: {type(exc).__name__}: {str(exc)[:250]}'
            errs.append(err)
            print(f'[AMY V8 SOLAR] {err}',flush=True)
    raise RuntimeError('No complete same-year radiation source. '+' | '.join(errs))


def create_station_year(session,st,year,cache_dir,out_dir,key,email,preferred='AUTO',
                        allow_borrowed=False,allow_edge_wrap=False,max_db_gap=24,max_dp_gap=24,
                        max_pr_gap=72,write_flagged=False,cloud_missing_policy='DEFAULT_5',
                        wind_fill_mode='CIRCULAR_LINEAR',max_wind_gap=2,
                        wind_review_gap=6,wind_high_risk_gap=48,
                        opaque_mode='TOTAL_COVER_PROXY',ancillary_retry=True,
                        era5_mode='FALLBACK',cds_api_key=None,era5_client=None,
                        city_name=None):
    # Continued V5/V7 atmospheric methods. New in V8: globally adaptive shortwave.
    if allow_borrowed or allow_edge_wrap:
        raise ValueError('V8 does not permit other-year radiation substitution or calendar-edge wrapping')
    mode=str(era5_mode).upper()
    if mode not in {'OFF','FALLBACK','PREFER'}:
        raise ValueError('ERA5_MODE must be OFF, FALLBACK or PREFER')
    city=clean_city_name(st,city_name)
    stem=f"{city}_{st['station_id']}_{year}_AMY"
    print(f"[AMY V10.1 MET] NOAA S3 {city}, station={st['station_id']} year={year}",flush=True)
    try: raw=required_surface(session,st['station_id'],year,cache_dir,st['tz_offset'])
    except Exception as exc:
        raise RuntimeError(f'NOAA_MET stage: {type(exc).__name__}: {exc}') from exc
    n=len(raw)
    report={'city':city,'station_id':st['station_id'],'year':int(year),'n_hours':n,
            'tz_offset':st['tz_offset'],'timezone_resolution':st['timezone_resolution'],
            'raw_noaa_hours':int(raw.DB.notna().sum())}
    originals={v:raw[v].isna() for v in ['DB','DP','PR']}
    for v in originals:
        report[f'{v}_missing_hours']=int(originals[v].sum())
        report[f'{v}_max_gap_hours']=max_missing_run(raw[v])
    report['WD_before_recovery_missing_hours']=int(raw.WD.isna().sum())
    report['WS_missing_hours']=int(raw.WS.isna().sum())
    report['GF1_total_cloud_observed_hours']=int(raw.TC_GF1.notna().sum())
    report['GF1_opaque_cloud_observed_hours']=int(raw.OC_GF1.notna().sum())
    report['NOAA_AJ1_snow_valid_hours']=int(raw.SNOW_CM.notna().sum())
    for v in ['DB','DP','PR']: raw[v]=caltech_linear_fill(raw[v])
    report['DB_filled_hot_tail_hours']=int(originals['DB'][raw.DB>=raw.DB.quantile(.99)].sum())
    report['DB_filled_cold_tail_hours']=int(originals['DB'][raw.DB<=raw.DB.quantile(.01)].sum())
    raw,windprov=repair_wind_direction(raw,wind_fill_mode,max_wind_gap)
    report.update(windprov)
    report['wind_review_gap_hours']=int(wind_review_gap)
    report['wind_high_risk_gap_hours']=int(wind_high_risk_gap)
    raw,cloudprov=repair_opaque_sky(raw,opaque_mode)
    report.update(cloudprov)
    report.update(precip_diagnostics(raw))
    print(f"[AMY V10.1 MET OK] {n} hours; NOAA DB={report['raw_noaa_hours']}; wind missing={report['wind_direction_remaining_missing_hours']}",flush=True)
    try:
        rad,prov=radiation_for_year_v8(session,year,st,cache_dir,key,email,st['tz_offset'],preferred=preferred)
        if ancillary_retry and prov['radiation_provider_selected']=='NSRDB':
            rad,attempt=optional_nsrdb_ancillary_retry(session,year,st,cache_dir,key,email,st['tz_offset'],rad)
            prov.update(attempt)
    except Exception as exc:
        raise RuntimeError(f'SOLAR stage: {type(exc).__name__}: {exc}') from exc
    report.update(prov)
    report['annual_ghi_Wh_m2']=float(rad.GHI.sum())
    report['annual_dni_Wh_m2']=float(rad.DNI.sum())
    report['annual_dhi_Wh_m2']=float(rad.DHI.sum())
    for col in ['PW_CM','AOD','ALBEDO','ZENITH_DEG']:
        report[f'solar_{col}_valid_hours']=int(rad[col].notna().sum()) if col in rad else 0
    print(f'[AMY V10.1 SOLAR OK] source={prov["radiation_source"]}, '
          f'GHI={rad.GHI.sum()/1000:.1f} kWh/m²; fallback_count={prov["radiation_fallback_count"]}',flush=True)
    model_ir,_=energyplus_infrared(raw.DB,raw.DP,raw.OC,cloud_missing_policy)
    model_bad=int((~np.isfinite(model_ir)).sum())
    era5_requested=mode=='PREFER' or (mode=='FALLBACK' and model_bad>0)
    era5_ir=None
    era5meta={'era5_status':'not needed: complete V5 infrared model',
              'era5_valid_longwave_hours':0, 'era5_dataset':ERA5_TIMESERIES_DATASET}
    if era5_requested:
        era5_ir,era5meta=era5_timeseries_longwave(year,st,cache_dir,cds_api_key,client=era5_client)
    elif mode=='OFF': era5meta['era5_status']='disabled'
    report.update(era5meta)
    report['era5_request_triggered']=bool(era5_requested)
    report['era5_mode']=mode
    max_exceed=(report['DB_max_gap_hours']>max_db_gap or
                report['DP_max_gap_hours']>max_dp_gap or report['PR_max_gap_hours']>max_pr_gap)
    if max_exceed and not write_flagged:
        raise ValueError('Required meteorological gap exceeds configured extremes threshold; '
                         +f"DB/DP/PR={report['DB_max_gap_hours']}/{report['DP_max_gap_hours']}/{report['PR_max_gap_hours']} h")
    # Wind reconstruction may create complete EPWs but carries its own risks.
    wind_longest=max(report['wind_speed_max_original_missing_run_hours'],
                     report['wind_direction_max_outage_run_hours'])
    wind_missing=report['wind_speed_remaining_missing_hours']+report['wind_direction_remaining_missing_hours']
    wind_high_risk=(wind_missing>0 or wind_longest>=wind_high_risk_gap)
    # Hours reported as calm or variable (VRB) have no direction by definition and occur at
    # almost every US airport in every year: they are still counted in the report, but only
    # real outages (and extrapolation or clipping) trigger the wind review grades.
    wind_review=(wind_longest>=wind_review_gap or
                 report['wind_speed_extrapolated_start_hours']+report['wind_speed_extrapolated_end_hours']>0 or
                 report['wind_direction_extrapolated_start_hours']+report['wind_direction_extrapolated_end_hours']>0 or
                 report['wind_speed_clipped_low_hours']+report['wind_speed_clipped_high_hours']>0)
    report['wind_high_risk_flag']=bool(wind_high_risk)
    report['wind_review_flag']=bool(wind_review)
    report['qc_grade']='HIGH_RISK' if max_exceed else 'GOOD'
    if wind_high_risk:
        report['qc_grade']='HIGH_RISK_WIND' if report['qc_grade']=='GOOD' else report['qc_grade']+'+WIND'
    elif wind_review and report['qc_grade']=='GOOD':
        report['qc_grade']='REVIEW_WIND'
    # No REVIEW_ATMOSPHERE grade for opaque sky cover taken from total cover
    # (TOTAL_COVER_PROXY): US stations report no opaque cover, so this documented method
    # applies to every hour of every year and the grade said nothing about the year.
    # The proxy hours are still counted in the report (opaque_sky_*_proxy_hours).
    # No REVIEW_PRECIP grade for multi-hour precipitation reports: their totals are written
    # with the accumulation period in the EPW liquid precipitation quantity field, and most
    # US stations have a few every year. They are still counted (precip_multi_hour_reports).
    if prov['radiation_is_reanalysis']:
        report['qc_grade']='REVIEW_REANALYSIS_SOLAR' if report['qc_grade']=='GOOD' else report['qc_grade']+'+ERA5_SOLAR'
    try:
        headers,epw,irprov=build_epw(st,year,raw,rad,prov['radiation_source'],cloud_missing_policy,
                                      era5_ir=era5_ir,era5_mode=mode)
        report.update(irprov)
        if report['infrared_missing_hours']>0 and report['qc_grade']=='GOOD': report['qc_grade']='REVIEW_LONGWAVE'
        dest=Path(out_dir)/('flagged' if (max_exceed or wind_high_risk) else 'epws')/f'{stem}.epw'
        report['output_epw']=str(write_epw(headers,epw,dest)); report['status']='written'
        # Avoid stale duplicates when a changed wind mode changes QA risk category.
        other_folder='epws' if dest.parent.name=='flagged' else 'flagged'
        prior=Path(out_dir)/other_folder/f'{stem}.epw'
        if prior.exists() and prior!=dest: prior.unlink()
        provenance=pd.DataFrame(index=raw.index)
        provenance.index.name='local_standard_time'
        for v in ['DB','DP','PR']: provenance[f'{v}_was_imputed']=originals[v].astype(int)
        provenance['WS_original']=raw.WS_original
        provenance['WD_original']=raw.WD_original
        provenance['WS_method']=raw.WS_METHOD
        provenance['WD_method']=raw.WD_METHOD
        provenance['WS_original_gap_hours']=raw.WS_GAP_H
        provenance['WD_original_gap_hours']=raw.WD_GAP_H
        provenance['WD_type']=raw.WIND_TYPE
        provenance['opaque_method']=raw.OC_METHOD
        provenance['total_cloud_method']=raw.TC_method
        provenance['precip_observed_duration_hours']=raw.PC_PERIOD
        provenance['precip_source_group']=raw.PC_SOURCE
        provenance['solar_source']=prov['radiation_provider_selected']
        provenance['solar_is_reanalysis']=int(prov['radiation_is_reanalysis'])
        provenance['solar_year']=prov['radiation_year']
        provenance['solar_AOD_available']=rad.AOD.notna().astype(int)
        provenance['solar_PW_available']=rad.PW_CM.notna().astype(int)
        modeled=pd.Series(model_ir,index=raw.index)
        if era5_ir is not None:
            valid=era5_ir.reindex(raw.index).between(100,700)
            replaced=valid if mode=='PREFER' else valid & modeled.isna()
        else: replaced=pd.Series(False,index=raw.index)
        provenance['longwave_source']=np.where(replaced,'CDS_ERA5_TIMESERIES',
                        np.where(modeled.notna(),'V5_CLARK_ALLEN_WALTON','MISSING'))
        qap=Path(out_dir)/'hourly_provenance'/f'{stem}_provenance.csv'
        qap.parent.mkdir(parents=True,exist_ok=True)
        provenance.to_csv(qap)
        report['hourly_provenance']=str(qap)
        print(f'[AMY V10.1 COMPLETE] {dest.name}; shortwave={prov["radiation_provider_selected"]}; '
              f'QC={report["qc_grade"]}; V5 longwave={report["infrared_V5_modeled_hours"]}; '
              f'CDS ERA5 longwave={report["infrared_ERA5_hours"]}',flush=True)
    except Exception as exc:
        raise RuntimeError(f'EPW_OUTPUT stage: {type(exc).__name__}: {exc}') from exc
    return report


def run_self_tests():
    """Offline checks of the calendar, source selection and wind reconstruction."""

    # Local calendar/source-selection smoke checks (no network needed).
    assert len(expected_local_index(2024)) == 8784, 'Leap-year hourly index broken'
    assert len(expected_local_index(2023)) == 8760, 'Ordinary-year hourly index broken'
    assert radiation_source_priority({'lat':47.4,'lon':-122.3},'NSRDB') == ['NSRDB']
    assert radiation_source_priority({'lat':30,'lon':31},'PVGIS-SARAH3') == ['PVGIS-SARAH3']
    print('[SELF-TEST] Calendar length and explicit radiation-provider selection: PASS')

    # V10.1 wind reconstruction smoke tests (no external API needed).
    _widx=pd.date_range('2020-01-01',periods=8,freq='h')
    _s0=pd.DataFrame({'WS':[np.nan,3.,2.,np.nan,np.nan,5.,7.,np.nan],
                      'WD':[np.nan,350.,10.,np.nan,np.nan,20.,40.,np.nan],
                      'WIND_TYPE':['']*8},index=_widx)
    _sf,_sr=repair_wind_direction(_s0,mode='CIRCULAR_LINEAR')
    assert len(_sf)==8 and _sf.WS.notna().all() and _sf.WD.notna().all()
    assert abs(_sf.WS.iloc[0]-4)<1e-9 and abs(_sf.WS.iloc[-1]-9)<1e-9
    assert abs(_sf.WD.iloc[0]-330)<1e-9 and abs(_sf.WD.iloc[3]-13.3333333333)<1e-5
    assert _sr['wind_speed_extrapolated_start_hours']==1 and _sr['wind_direction_extrapolated_end_hours']==1
    _sl,_=repair_wind_direction(_s0,mode='CALTECH_LINEAR')
    assert abs(_sl.WD.iloc[3]-13.3333333333)<1e-5
    print('[SELF-TEST] Wind interior interpolation, linear year-edge extrapolation and direction wrap: PASS')
    return True


# =====================================================================================
# Entry point for the RMY repository: one station number -> the AMYs of the most recent
# complete years, ready for tmy/generate_tmy_from_amys.py and the rmy package.
# =====================================================================================
import argparse
import os
import shutil

N_YEARS = 15            # Tarkhan et al. (2025) use the most recent 15 years of AMYs.
MIN_YEARS = 10          # ISO 15927-4 needs at least 10 years for the TMY step.
MAX_EXTRA_YEARS = 5     # how far back to look for replacements of failed or flagged years

# Readable city names for stations whose NOAA name does not start with the city.
STATION_CITY_OVERRIDES = {
    '72509014739': 'Boston', '72530094846': 'Chicago', '72403093738': 'Dulles_Washington',
    '72367723054': 'Las_Vegas', '72295023174': 'Los_Angeles', '72278023183': 'Phoenix',
    '72494023234': 'San_Francisco', '72793024233': 'Seattle',
}

# Settings of the AMY Generator V10.1 notebook. ERA5 longwave is off: the modeled
# longwave (Clark-Allen/Walton, with filled sky cover) is complete, and ERA5 would need
# a Copernicus key.
GENERATION_SETTINGS = dict(
    max_db_gap=24, max_dp_gap=24, max_pr_gap=72, write_flagged=True,
    cloud_missing_policy='DEFAULT_5', wind_fill_mode='CIRCULAR_LINEAR', max_wind_gap=2,
    wind_review_gap=6, wind_high_risk_gap=48, opaque_mode='TOTAL_COVER_PROXY',
    ancillary_retry=True, era5_mode='OFF')


def last_complete_year(st, radiation_source='NSRDB', today=None):
    """Most recent calendar year with complete NOAA data and radiation coverage.

    The NOAA station record ends on its last observation date (isd-history END): a year
    is complete only if the record reaches 31 December. Capped at last calendar year
    and at the radiation provider's coverage.
    """
    year = (today or date.today()).year - 1
    end = pd.to_datetime(str(st.get('end', '')), format='%Y%m%d', errors='coerce')
    if pd.notna(end):
        year = min(year, end.year if (end.month, end.day) == (12, 31) else end.year - 1)
    if radiation_source == 'NSRDB':
        year = min(year, NSRDB_YEARS[1])
    else:
        # PVGIS UTC data need the following year for stations west of Greenwich.
        year = min(year, PVGIS5_3_BOUNDS[1] - (1 if float(st.get('tz_offset', 0)) < 0 else 0))
    return year


def epw_is_complete(path, year):
    """Row count, 35 fields, the right year and no missing shortwave (as in the notebook)."""
    try:
        path = Path(path)
        if not path.is_file() or path.stat().st_size < 100_000:
            return False
        n = 0
        with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as fh:
            r = csv.reader(fh)
            for _ in range(8):
                next(r)
            for row in r:
                if len(row) != 35:
                    return False
                n += 1
                if n == 1 and int(float(row[0])) != year:
                    return False
                if any(float(row[k]) >= 9999 or float(row[k]) < 0 for k in (13, 14, 15)):
                    return False
        return n == (8784 if calendar.isleap(year) else 8760)
    except Exception:
        return False


def _redact(value, secrets):
    """Remove API keys and emails from text that could be saved or shown."""
    if isinstance(value, dict):
        return {k: _redact(v, secrets) for k, v in value.items()}
    if isinstance(value, str):
        for s in secrets:
            if s and len(str(s)) >= 6:        # never mangle text with very short values
                value = value.replace(str(s), '***')
    return value


def generate_amys(station_number, output_dir, nsrdb_api_key=None, nsrdb_email=None,
                  n_years=N_YEARS, city=None, radiation_source=None, cache_dir=None,
                  tz_offset=None, verbose=True):
    """Build the AMYs of the n most recent complete years for one NOAA station.

    station_number: 6-character USAF/WMO or 11-character USAF+WBAN (e.g. '72509014739').
    output_dir: AMYs go to <output_dir>/<City>/AMY/, reports to <City>/AMY_reports/.
    nsrdb_api_key, nsrdb_email: free NREL developer key (https://developer.nlr.gov/docs/climate/guide/),
        needed for stations in the Americas (NSRDB). Not needed elsewhere (PVGIS).
    city: optional file-safe city name; otherwise from the override list or NOAA.
    radiation_source: None picks NSRDB in the Americas and PVGIS elsewhere.
    tz_offset: optional standard UTC offset in hours; otherwise found with timezonefinder.
    Years that fail, or are flagged for long gaps (written to AMY_reports/flagged), are
    replaced by earlier years. Returns a dict with city, station, amy_dir, years, manifest.
    """
    secrets = [nsrdb_api_key, nsrdb_email]
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / '_cache'
    session = requests.Session()
    log = print if verbose else (lambda *a, **k: None)

    this_year = date.today().year
    st = identify_station(station_catalog(session, cache_dir), station_number,
                          period_start=this_year - n_years, period_end=this_year - 1)
    st['tz_offset'], st['timezone_resolution'] = standard_tz_offset(st['lat'], st['lon'], tz_offset)
    if str(st['timezone_resolution']).startswith('longitude'):
        raise ImportError('Could not determine the station time zone: install timezonefinder '
                          '(pip install timezonefinder) or pass tz_offset, e.g. -5 for US Eastern.')
    city = clean_city_name(st, city or STATION_CITY_OVERRIDES.get(st['station_id'])
                           or city_from_coordinates(st))
    source = (radiation_source or ('NSRDB' if is_americas_location(st) else 'AUTO')).upper()
    if source in ('NSRDB', 'AUTO') and is_americas_location(st) and not (nsrdb_api_key and nsrdb_email):
        raise ValueError('This station is in the Americas, so solar radiation comes from NSRDB, which '
                         'needs your own free NREL API key and email (https://developer.nlr.gov/docs/climate/guide/).')

    end = last_complete_year(st, 'NSRDB' if source == 'NSRDB' else 'PVGIS')
    first_allowed = max(int(str(st['begin'])[:4]),
                        NSRDB_YEARS[0] if source == 'NSRDB' else PVGIS5_3_BOUNDS[0] + 1)
    candidates = [y for y in range(end, end - n_years - MAX_EXTRA_YEARS, -1) if y >= first_allowed]
    log(f"Station {st['station_id']} ({st['name']}), city '{city}', UTC{st['tz_offset']:+g}, "
        f"NOAA record {st['begin']}-{st['end']}")
    log(f'Solar radiation: {source}. Most recent complete year: {end}. '
        f'Target: {n_years} years ({end - n_years + 1}-{end}).')

    amy_dir = output_dir / city / 'AMY'
    reports_dir = output_dir / city / 'AMY_reports'
    amy_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / 'per_year_reports').mkdir(parents=True, exist_ok=True)
    good, rows, failures = [], [], 0
    for year in candidates:
        if len(good) == n_years:
            break
        dest = amy_dir / f"{city}_{st['station_id']}_{year}_AMY.epw"
        if epw_is_complete(dest, year):
            log(f'{year}: existing AMY reused')
            good.append(year)
            rows.append({'year': year, 'status': 'reused', 'output_epw': str(dest)})
            continue
        try:
            report = create_station_year(session, st, year, cache_dir, reports_dir, nsrdb_api_key,
                                         nsrdb_email, preferred=source, city_name=city,
                                         **GENERATION_SETTINGS)
            written = Path(report['output_epw'])
            if written.parent.name == 'flagged':
                report['status'] = 'flagged_not_used'
                log(f"{year}: flagged ({report.get('qc_grade')}), kept in AMY_reports/flagged and replaced")
            else:
                shutil.move(str(written), dest)
                report['output_epw'] = str(dest)
                good.append(year)
                log(f"{year}: AMY written ({report.get('qc_grade')})")
        except Exception as exc:
            report = {'year': year, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'[:1200]}
            log(f"{year}: failed, replaced by an earlier year ({_redact(report['error'], secrets)[:200]})")
            failures += 1
            if failures == 3 and not good and len(rows) == 2:
                # The first three years all failed: almost certainly a setup problem
                # (key, email, network or time zone), not a data gap. Stop here.
                rows.append(_redact(report, secrets))
                pd.DataFrame(rows).to_csv(reports_dir / 'AMY_manifest.csv', index=False)
                raise RuntimeError('The first three years all failed, so the run was stopped. Last error: '
                                   + _redact(report['error'], secrets)[:600])
        report = _redact(report, secrets)
        rows.append(report)
        (reports_dir / 'per_year_reports' / f"{city}_{st['station_id']}_{year}.json").write_text(
            json.dumps(report, indent=2, default=str))

    manifest = pd.DataFrame(rows).sort_values('year')
    manifest.to_csv(reports_dir / 'AMY_manifest.csv', index=False)
    good = sorted(good)
    if len(good) < MIN_YEARS:
        raise RuntimeError(f'Only {len(good)} complete AMY years could be built (need at least '
                           f'{MIN_YEARS}); see {reports_dir / "AMY_manifest.csv"}.')
    if len(good) < n_years:
        log(f'WARNING: {len(good)} of {n_years} years built: {good[0]}-{good[-1]}')
    log(f'AMYs: {len(good)} years, {good[0]}-{good[-1]}, in {amy_dir}')
    return {'city': city, 'station': st, 'amy_dir': amy_dir, 'reports_dir': reports_dir,
            'years': good, 'radiation_source': source, 'manifest': manifest}


def parse_args():
    p = argparse.ArgumentParser(description='Build the AMYs of the most recent complete years for one NOAA station.')
    p.add_argument('--station', required=True, help='NOAA station number, e.g. 72509014739 (Boston Logan).')
    p.add_argument('--out', required=True, type=Path, help='Output folder; AMYs go to <out>/<City>/AMY/.')
    p.add_argument('--years', type=int, default=N_YEARS, help=f'Number of years (default {N_YEARS}).')
    p.add_argument('--city', default=None, help='Optional city name for the files.')
    p.add_argument('--source', default=None, help='Radiation source: NSRDB, PVGIS-SARAH3, PVGIS-ERA5 or AUTO.')
    p.add_argument('--cache', type=Path, default=None, help='Download cache (default <out>/_cache).')
    return p.parse_args()


def main():
    args = parse_args()
    generate_amys(args.station, args.out, os.environ.get('NSRDB_API_KEY'), os.environ.get('NSRDB_EMAIL'),
                  n_years=args.years, city=args.city, radiation_source=args.source, cache_dir=args.cache)


if __name__ == '__main__':
    main()
