"""Section 4.3 (Sparse records): what a station reporting only every 3 hours loses. Karachi
reports at 02, 05, 08, 11, 14, 17, 20 and 23 h local time from 2019 on. Its well-observed 2018
record is thinned to those hours, filled again by the method's interpolation, and compared
with the full record: daily maxima and minima, hours at or above 40 C, and heat waves found
with the percentile criterion of the RMY method for Karachi (daily Tmax >= 36.5 C and
Tmin >= 22.0 C, the 90th and 40th percentiles of all its AMY days, for at least 3 days)."""
import io, contextlib
import pandas as pd, requests
from common import work_dir
import generate_amys as g

SID, TZ, YEAR, SPARSE_YEAR = '41780099999', 5.0, 2018, 2020
TMAX, TMIN, MIN_DAYS = 36.5, 22.0, 3


def heat_waves(tx, tn):
    hot = ((tx >= TMAX) & (tn >= TMIN)).to_numpy(); events, i = [], 0
    while i < len(hot):
        j = i
        while j < len(hot) and hot[j]:
            j += 1
        if j - i >= MIN_DAYS:
            events.append((tx.index[i].date(), j - i))
        i = max(j, i + 1)
    return events


def main():
    work = work_dir(); sess = requests.Session(); cache = str(work / 'cache')
    with contextlib.redirect_stdout(io.StringIO()):
        sparse = g.required_surface(sess, SID, SPARSE_YEAR, cache, TZ)
        raw = g.required_surface(sess, SID, YEAR, cache, TZ)
    share = sparse.DB.notna().groupby(sparse.index.hour).mean()
    hours = sorted(share[share > 0.5].index)                          # the hours reported in SPARSE_YEAR
    obs = raw.DB.astype(float)
    full = g.caltech_linear_fill(obs)
    thin = obs.where(obs.index.hour.isin(hours))
    rec = g.caltech_linear_fill(thin)
    d = pd.DataFrame({'full': full, 'thinned': rec})
    dmax, dmin = d.resample('D').max(), d.resample('D').min()
    emax, emin = dmax.thinned - dmax.full, dmin.thinned - dmin.full
    a, b = heat_waves(dmax.full, dmin.full), heat_waves(dmax.thinned, dmin.thinned)
    rows = {
        'reporting_hours_local': ' '.join(f'{h:02d}' for h in hours),
        'observed_share_full_%': 100 * obs.notna().mean(), 'observed_share_thinned_%': 100 * thin.notna().mean(),
        'hourly_mae_C': (d.thinned - d.full).abs().mean(),
        'daily_max_mean_error_C': emax.mean(), 'daily_max_mae_C': emax.abs().mean(), 'daily_max_worst_C': emax.loc[emax.abs().idxmax()],
        'daily_min_mean_error_C': emin.mean(), 'daily_min_mae_C': emin.abs().mean(), 'daily_min_worst_C': emin.loc[emin.abs().idxmax()],
        'annual_max_full_C': d.full.max(), 'annual_max_thinned_C': d.thinned.max(),
        'hours_40C_full': int((d.full >= 40).sum()), 'hours_40C_thinned': int((d.thinned >= 40).sum()),
        'heat_waves_full': len(a), 'heat_wave_days_full': sum(n for _, n in a),
        'heat_waves_thinned': len(b), 'heat_wave_days_thinned': sum(n for _, n in b),
        'heat_waves_full_list': '; '.join(f'{s} ({n} d)' for s, n in a),
        'heat_waves_thinned_list': '; '.join(f'{s} ({n} d)' for s, n in b)}
    out = pd.Series(rows, name='value').to_frame()
    out.to_csv(work / 'results' / 'sparse_records_karachi_2018.csv'); print(out.to_string())


if __name__ == '__main__':
    main()
