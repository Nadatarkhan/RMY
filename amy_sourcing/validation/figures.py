"""Figures 2-4 of the paper, drawn from the CSV files in <work>/results (written by the numbered
scripts) or, with --bundled, from the results bundled in this folder."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE

BLUE, RED, GREEN = '#0086b3', '#CF3952', '#1E7B3A'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})


def panel(ax, letter, x=-0.02):
    ax.text(x, 1.03, f'({letter})', transform=ax.transAxes, fontweight='bold', color=INK, ha='right')


def site_name(site):
    """'AZ_Tucson_11_W' -> 'Tucson, AZ'."""
    parts = site.split('_'); words = []
    for p in parts[1:]:
        if p.isdigit():
            break
        words.append(p)
    return f"{' '.join(words)}, {parts[0]}"


def fig2(res, out):
    r = pd.read_csv(res / 'radiation_uscrn.csv')
    m = r.groupby(['site', 'city'])[['A_timing_min', 'B_timing_min', 'C_timing_min']].mean().reset_index().sort_values('C_timing_min')
    lab = [f"{site_name(s).split(',')[0]} ({c})" for s, c in zip(m.site, m.city)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=300, gridspec_kw={'width_ratios': [1.15, 1]})
    y = np.arange(len(m)); a.axvline(0, color=MUTED, lw=0.8)
    a.scatter(m.C_timing_min, y, s=24, color=RED, marker='s', edgecolor='white', linewidth=0.7, zorder=3, label='Direct assignment: hh:00 snapshot filed under its hour')
    a.scatter(m.A_timing_min, y + 0.14, s=26, color=BLUE, marker='o', edgecolor='white', linewidth=0.7, zorder=3, label='This method: NSRDB hh:30 at the hour midpoint')
    a.scatter(m.B_timing_min, y - 0.14, s=20, color=GREEN, marker='D', edgecolor='white', linewidth=0.7, zorder=3, label='This method: hh:00 snapshot moved to midpoint')
    a.set_yticks(y); a.set_yticklabels(lab, fontsize=7.6); a.set_xlabel('Timing offset vs. ground (min)')
    a.grid(axis='x', color=GRID, lw=0.6); a.set_axisbelow(True); a.set_xlim(-10, 45); panel(a, 'a')
    a.legend(loc='upper center', bbox_to_anchor=(0.35, -0.17), frameon=False, fontsize=7.4, ncol=1)
    h = pd.read_csv(res / 'radiation_charlottesville_2022_hourly.csv', index_col=0, parse_dates=True).dropna()
    h['lh'] = (h.index.hour - 5) % 24 + 0.5; p = h.groupby('lh')[['m', 'A', 'C']].mean(); p = p[(p.m > 2) | (p.A > 2)]
    b.plot(p.index, p.m, color=INK, lw=2, label='Ground (USCRN)')
    b.plot(p.index, p.A, color=BLUE, lw=2, marker='o', ms=3.5, label='This method')
    b.plot(p.index, p.C, color=RED, lw=2, ls='--', marker='s', ms=3.2, label='hh:00 snapshot, direct assignment')
    b.set_xlabel('Local standard time (h)'); b.set_ylabel('Mean global horizontal (W/m²)'); b.set_xticks(range(6, 21, 3))
    b.grid(color=GRID, lw=0.6); b.set_axisbelow(True); panel(b, 'b')
    b.set_title('Charlottesville, VA (2022)', fontsize=8.5, color=INK2, loc='left', x=0.06, pad=4)
    b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False, fontsize=7.4)
    fig.tight_layout(); fig.savefig(out / 'fig2_radiation.png', bbox_inches='tight', facecolor='white'); plt.close(fig)


def fig3(res, out):
    h = pd.read_csv(res / 'holdout_interpolation.csv'); sky = pd.read_csv(res / 'holdout_sky_cover.csv')
    pooled = lambda d, keys: d.groupby(keys).apply(lambda q: (q.mae * q.n).sum() / q.n.sum(), include_groups=False)
    H = pooled(h, ['var', 'gap_h']); K = pooled(sky, ['method', 'gap_h'])
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), dpi=300); (a, b), (c, d) = axs
    line = lambda ax, s, col, mk, ls, lab: ax.plot(s.index, s.values, color=col, marker=mk, ls=ls, lw=2, ms=4, label=lab)
    line(a, H['DB'], BLUE, 'o', '-', 'Dry-bulb temperature'); line(a, H['DP'], RED, 's', '--', 'Dew-point temperature')
    a.set_ylabel('Mean absolute error (°C)'); a.legend(frameon=False, fontsize=7.4, loc='upper left'); a.set_title('Temperature', fontsize=8.5, color=INK, loc='left', x=0.04)
    line(b, H['WS'], BLUE, 'o', '-', 'Wind speed'); b.set_ylabel('Mean absolute error (m/s)'); b.set_title('Wind speed', fontsize=8.5, color=INK, loc='left', x=0.04)
    line(c, H['WD'], BLUE, 'o', '-', 'Wind direction'); c.set_ylabel('Mean absolute error (°)'); c.set_title('Wind direction', fontsize=8.5, color=INK, loc='left', x=0.04)
    for m, col, mk, ls, lab in (('linear', RED, 's', '--', 'Linear interpolation only'), ('estimate', GREEN, 'D', ':', 'Cloud-type estimate only'), ('method', BLUE, 'o', '-', 'This method')):
        line(d, K[m], col, mk, ls, lab)
    d.set_ylabel('Mean absolute error (tenths)'); d.set_title('Total sky cover (US stations)', fontsize=8.5, color=INK, loc='left', x=0.04)
    d.legend(frameon=False, fontsize=7.4, loc='lower right')
    for ax, l, xt in ((a, 'a', [1, 6, 12, 18, 24]), (b, 'b', [1, 6, 12, 18, 24]), (c, 'c', [1, 6, 12, 18, 24]), (d, 'd', [2, 6, 12, 24, 47])):
        ax.set_xlabel('Gap length (h)'); ax.set_xticks(xt); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(bottom=0); panel(ax, l, -0.12)
    fig.tight_layout(); fig.savefig(out / 'fig3_holdout.png', bbox_inches='tight', facecolor='white'); plt.close(fig)


def fig4(res, out):
    d = pd.read_csv(res / 'design_vs_ashrae.csv'); items = list(dict.fromkeys(d['item']))
    fig, ax = plt.subplots(figsize=(6.6, 4.9), dpi=300); ax.axvline(0, color=MUTED, lw=0.8); rng = np.random.default_rng(3)
    n = {r: d[d.region == r].city.nunique() for r in ('US', 'INT')}
    for i, it in enumerate(items[::-1]):
        for reg, col, mk, dy, lab in (('US', BLUE, 'o', 0.13, f"U.S. stations ({n['US']})"), ('INT', GREEN, 's', -0.13, f"International stations ({n['INT']})")):
            x = d[(d['item'] == it) & (d.region == reg)]['diff'].to_numpy()
            ax.scatter(x, np.full(len(x), i + dy) + rng.uniform(-0.05, 0.05, len(x)), s=13, color=col, marker=mk, alpha=0.9,
                       edgecolor='white', linewidth=0.4, zorder=3, label=lab if i == 0 else None)
        m = d[d['item'] == it]['diff'].mean()
        ax.plot([m, m], [i - 0.32, i + 0.32], color=INK, lw=1.6, zorder=4, label=f'Mean ({d.city.nunique()} stations)' if i == 0 else None)
    ax.set_yticks(range(len(items))); ax.set_yticklabels(items[::-1], fontsize=9.5); ax.tick_params(axis='x', labelsize=9.5)
    ax.set_xlabel('Calculated from the AMYs minus ASHRAE 2025 (°C unless marked)', fontsize=10)
    ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(-4.3, 4.3)
    ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.4, -0.12), ncol=3)
    fig.tight_layout(); fig.savefig(out / 'fig4_design.png', bbox_inches='tight', facecolor='white'); plt.close(fig)


def main():
    if '--bundled' in sys.argv:
        res, out = HERE / 'results', Path('figures').resolve()
    else:
        work = Path(sys.argv[1] if len(sys.argv) > 1 else 'amy_validation_work').resolve(); res, out = work / 'results', work / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for fig, needs in ((fig2, ['radiation_uscrn.csv', 'radiation_charlottesville_2022_hourly.csv']),
                       (fig3, ['holdout_interpolation.csv', 'holdout_sky_cover.csv']), (fig4, ['design_vs_ashrae.csv'])):
        missing = [n for n in needs if not (res / n).exists()]
        if missing:
            print(f'{fig.__name__}: skipped, run the scripts that write', ', '.join(missing))
        else:
            fig(res, out); print(f'{fig.__name__}: written to', out)


if __name__ == '__main__':
    main()
