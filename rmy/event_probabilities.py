"""Extreme Event Probability Explorer: how often heat waves and cold spells of a given length and
intensity occur at a station, from the events detected in its AMYs, and how rare the RMY's events are.

One interactive page (Plotly) with four linked panels:
  A. every event on a month-by-year timeline (circle size: duration; colour: heat index or wind
     speed); clicking one pins it, shows its statistics and marks it in the other panels;
  B. return period (or chance in a year / in N years) of an event at least this extreme;
  C. number of events in a year, with a Poisson fit;
  D. when in the year events start (half-month bins).
Toggles switch heat waves / cold spells, the intensity measure, the probability shown, the
horizon N and the period (all years, first half, second half).

Statistics:
  - Arrivals: Poisson with rate = events per AMY year.
  - Observed return period of the k-th most extreme event: years / k.
  - Fitted: generalised Pareto tail of the more extreme half of the event intensities (peaks over
    threshold), with a 90 % band from bootstrap resamples of the events. With 10-15 years of AMYs,
    values beyond about twice the record length are extrapolations and are drawn as such.

Usage (after the RMY pipeline):
    from rmy.event_probabilities import event_probability_explorer
    event_probability_explorer(RMY_DIR)          # shows it in the notebook, writes event_probabilities.html
"""
import html
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto

PLOTLY_JS = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'
TEMPLATE = Path(__file__).with_name('event_probabilities.html')
RETURN_PERIODS = np.unique(np.round(np.geomspace(0.1, 100, 120), 4))
SHAPE_LIMIT = 0.5          # |GPD shape| allowed: short records cannot support more
MIN_EVENTS_FIT = 8          # fewer events than this: observed return periods only, no fitted curve

KINDS = {
    'cold': dict(csv='coldspells/coldspells_events.csv', peak='coldspells/coldspells_events_peak.csv',
                 mean='avg_tmin', measures={
                     'peak': ('min_tmin', -1, 'Coldest night (Tmin, °C)'),
                     'mean': ('avg_tmin', -1, 'Mean night-time low (°C)'),
                     'severity': ('severity', 1, 'Severity (°C·days below {thr:g} °C)'),
                     'extra': ('avg_wind_speed', 1, 'Mean wind speed (m/s)')}),
    'heat': dict(csv='hotspells/heatwave_events.csv', peak='hotspells/heatwave_events_peak.csv',
                 mean='avg_tmax', measures={
                     'peak': ('max_tmax', 1, 'Hottest day (Tmax, °C)'),
                     'mean': ('avg_tmax', 1, 'Mean daytime high (°C)'),
                     'severity': ('severity', 1, 'Severity (°C·days above {thr:g} °C)'),
                     'extra': ('avg_heat_index', 1, 'Mean heat index (°C)')}),
}


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def detection_thresholds(rmy_dir, amy_paths):
    """Percentile thresholds the pipeline used: from its log, else recomputed from the AMYs."""
    log = Path(rmy_dir) / 'log.txt'
    text = log.read_text(errors='ignore') if log.exists() else ''
    heat = re.search(r'Heat thresholds .*?Tmax >= ([-\d.]+);', text)
    cold = re.search(r'Cold thresholds .*?percentile Tmin < ([-\d.]+);', text)
    if heat and cold:
        return float(cold.group(1)), float(heat.group(1))
    from .coldspells import cold_thresholds, pooled_daily_cold
    from .heatwaves import heat_thresholds, pooled_daily
    return (round(cold_thresholds(pooled_daily_cold(amy_paths)).tmin_pct, 1),
            round(heat_thresholds(pooled_daily(amy_paths)).tmax_pct, 1))


def amy_years(amy_dir):
    years = []
    for p in Path(amy_dir).glob('*.epw'):
        found = re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)', p.stem)
        if found:
            years.append(int(found[-1]))
    return sorted(set(years))


def load_events(rmy_dir, kind, threshold):
    """Events of one kind with dates, severity (degree-days beyond the threshold) and RMY flag."""
    k = KINDS[kind]
    e = _read_csv(Path(rmy_dir) / k['csv'])
    if e.empty:
        return e
    e['start'] = pd.to_datetime(e.begin_date, dayfirst=True)
    e['end'] = pd.to_datetime(e.end_date, dayfirst=True)
    beyond = (threshold - e[k['mean']]) if kind == 'cold' else (e[k['mean']] - threshold)
    e['severity'] = (e.duration * beyond).clip(lower=0).round(1)
    peak = _read_csv(Path(rmy_dir) / k['peak'])
    e['rmy'] = e.begin_date.isin(set(peak.begin_date)) if not peak.empty else False
    e['year'] = e['year'].astype(int) if 'year' in e else e.start.dt.year
    return e.reset_index(drop=True)


def _gpd_pwm(x):
    """Generalised Pareto (location 0) by probability-weighted moments (Hosking & Wallis 1987).
    Returns (shape, scale) in scipy's convention (shape > 0: heavy tail). Closed form, so the
    bootstrap is fast; for 10-50 events it is also steadier than maximum likelihood."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    a0 = x.mean()
    a1 = np.mean(x * (n - np.arange(1, n + 1)) / (n - 1))
    shape = -(a0 / (a0 - 2 * a1) - 2)
    if not np.isfinite(shape) or abs(shape) > SHAPE_LIMIT:
        # short records (and values rounded to whole degrees) can give an implausible tail;
        # keep the shape within +-0.5 and refit the scale to the mean excess, a0 = scale / (1 - shape)
        shape = float(np.clip(np.nan_to_num(shape), -SHAPE_LIMIT, SHAPE_LIMIT))
        return shape, a0 * (1 - shape)
    return shape, 2 * a0 * a1 / (a0 - 2 * a1)


def _fit(z, years):
    """Poisson-GPD over the more extreme half of z ('more extreme' = larger). Returns (lam, u, shape, scale)."""
    u = float(np.quantile(z, 0.5))
    exc = z[z > u] - u
    if len(exc) < 4 or np.ptp(exc) == 0:
        raise ValueError('too few distinct values above the threshold')
    c, s = _gpd_pwm(exc)
    if not (np.isfinite(c) and np.isfinite(s) and s > 0):
        raise ValueError('no valid fit')
    return len(exc) / years, u, c, s


def _levels(prm, T):
    lam, u, c, s = prm
    p = 1 - 1 / (lam * T)
    out = np.full(len(T), np.nan)
    ok = p > 0
    out[ok] = u + genpareto.ppf(p[ok], c, 0, s)
    return out


def _return_period(zv, z, years, prm):
    observed = years / max(int((z >= zv).sum()), 1)
    if prm is None or zv <= prm[1]:
        return observed
    sf = genpareto.sf(zv - prm[1], prm[2], 0, prm[3])
    return float(1 / (prm[0] * sf)) if sf > 0 else observed      # beyond a bounded tail's end


def _measure(e, col, sign, years, rng, n_boot):
    z = sign * e[col].to_numpy(float)
    order = np.argsort(-z, kind='stable')
    t_emp = np.empty(len(z))
    t_emp[order] = years / np.arange(1, len(z) + 1)
    out = dict(sign=sign, T=[], level=[], lo=[], hi=[], shape=None)
    prm = None
    if len(z) >= MIN_EVENTS_FIT and np.ptp(z) > 0:
        try:
            prm = _fit(z, years)
            curve = _levels(prm, RETURN_PERIODS)
            boots = []
            for _ in range(n_boot):
                try:
                    boots.append(_levels(_fit(rng.choice(z, len(z)), years), RETURN_PERIODS))
                except Exception:
                    pass
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)          # T shorter than a resample's reach
                lo, hi = np.nanpercentile(np.array(boots), [5, 95], axis=0)
            keep = np.isfinite(curve) & np.isfinite(lo) & np.isfinite(hi)
            lo_s, hi_s = (lo, hi) if sign > 0 else (hi, lo)
            out.update(T=RETURN_PERIODS[keep].tolist(), level=np.round(sign * curve[keep], 2).tolist(),
                       lo=np.round(sign * lo_s[keep], 2).tolist(), hi=np.round(sign * hi_s[keep], 2).tolist(),
                       shape=round(float(prm[2]), 3))
        except Exception:
            prm = None
    ids = e.index.tolist()
    out['T_emp'] = {int(i): round(float(t), 2) for i, t in zip(ids, t_emp)}
    out['T_fit'] = {int(i): round(min(_return_period(zv, z, years, prm), 999.0), 1) for i, zv in zip(ids, z)}
    return out


def event_probability_data(rmy_dir, amy_dir=None, n_boot=500, seed=1):
    """Everything the page draws, as a JSON-serialisable dict."""
    rmy_dir = Path(rmy_dir)
    amy_dir = Path(amy_dir) if amy_dir else rmy_dir.parent / 'AMY'
    amy_paths = sorted(amy_dir.glob('*.epw'))
    cold_thr, heat_thr = detection_thresholds(rmy_dir, amy_paths)
    thr = dict(cold=cold_thr, heat=heat_thr)
    events = {k: load_events(rmy_dir, k, thr[k]) for k in KINDS}
    years = amy_years(amy_dir) or sorted({int(y) for e in events.values() if len(e) for y in e.year})
    if not years:
        raise ValueError(f'No AMY years found in {amy_dir} and no events in {rmy_dir}')
    kinds = [k for k in KINDS if len(events[k]) >= 3]
    if not kinds:
        raise ValueError('Fewer than 3 heat waves and 3 cold spells were detected: nothing to analyse.')
    half = len(years) // 2
    periods = {'all': years}
    if half >= 5:
        periods.update(early=years[:half], late=years[half:])
    rng = np.random.default_rng(seed)
    data = dict(city=rmy_dir.parent.name.replace('_', ' '), kinds=kinds, thresholds=thr, n_boot=n_boot,
                labels={k: {m: lab.format(thr=thr[k]) for m, (col, _, lab) in KINDS[k]['measures'].items()
                            if col in events[k] or col == 'severity'} for k in kinds},
                periods={})
    for pk, ys in periods.items():
        Y = len(ys)
        for kind in kinds:
            e = events[kind][events[kind].year.isin(ys)]
            rec = dict(n=len(e), Y=Y, lam=len(e) / Y, years=ys,
                       per_year=e.groupby('year').size().reindex(ys, fill_value=0).astype(int).tolist(),
                       events=[dict(id=int(i), start=f'{r.start:%d %b %Y}', end=f'{r.end:%d %b %Y}', year=int(r.year),
                                    month=int(r.start.month), day=int(r.start.day),
                                    half=int((r.start.month - 1) * 2 + (r.start.day > 15)), duration=int(r.duration),
                                    rmy=bool(r.rmy), vals={m: float(r[col]) for m, (col, _, _) in KINDS[kind]['measures'].items()
                                                          if m in data['labels'][kind]})
                               for i, r in e.iterrows()],
                       measures={})
            if len(e):
                for m in data['labels'][kind]:
                    col, sign, _ = KINDS[kind]['measures'][m]
                    rec['measures'][m] = _measure(e, col, sign, Y, rng, n_boot)
            data['periods'].setdefault(pk, {})[kind] = rec
    return data


def event_probability_html(data, plotly_js=PLOTLY_JS):
    """The standalone page. `plotly_js` is a URL or the library's source text."""
    lib = (f'<script src="{plotly_js}"></script>' if plotly_js.startswith('http')
           else f'<script>{plotly_js}</script>')
    return (TEMPLATE.read_text(encoding='utf-8').replace('<!--PLOTLY-->', lib)
            .replace('/*DATA*/', json.dumps(data).replace('</', '<\\/')))


def local_plotly_js():
    """plotly.js shipped with the plotly Python package (preinstalled on Colab), or None."""
    try:
        import plotly
    except ImportError:
        return None
    path = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
    return path.read_text(encoding='utf-8') if path.exists() else None


def event_probability_explorer(rmy_dir, output_path=None, show=True, amy_dir=None, n_boot=500, height=1400):
    """Build the explorer for one RMY folder, save it as HTML and (in a notebook) show it.

    rmy_dir: the RMY output folder (with hotspells/, coldspells/ and log.txt).
    output_path: where to save the page; default <rmy_dir>/event_probabilities.html. The saved
        page carries plotly.js when the plotly package is installed, so it also works offline.
    amy_dir: the AMY folder; default <rmy_dir>/../AMY.
    The notebook display loads plotly.js from its CDN, which keeps the notebook small.
    """
    rmy_dir = Path(rmy_dir)
    out = Path(output_path) if output_path else rmy_dir / 'event_probabilities.html'
    data = event_probability_data(rmy_dir, amy_dir, n_boot=n_boot)
    out.write_text(event_probability_html(data, local_plotly_js() or PLOTLY_JS), encoding='utf-8')
    page = event_probability_html(data)
    if show:
        try:
            from IPython.display import HTML, display
            # an iframe keeps the page's styles and scripts out of the notebook's own page
            display(HTML(f'<iframe srcdoc="{html.escape(page, quote=True)}" '
                         f'style="width:100%;height:{height}px;border:0"></iframe>'))
        except ImportError:
            pass
    return out
