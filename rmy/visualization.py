"""Extreme Event Explorer: a circle timeline of detected heat waves and cold spells.

Each circle is one event, placed at its start date (x = month, y = year).
Circle area is proportional to duration. Heat waves are colored by average
heat index; cold spells by average wind chill if available, otherwise by
average wind speed.

Ported from the notebook's "2.0 Visualization" cell.
"""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

# NOAA heat-index bands in degrees C: caution starts at 27, extreme danger at 51.
HEAT_INDEX_RANGE_C = (27, 51)
CIRCLE_SIZE = 58
# Event durations shown in the size legend.
LEGEND_DAYS = (3, 5, 7, 9, 11)
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def create_custom_colormap():
    """Return (heatwave_cmap, coldspell_cmap): reds for heat, blues for cold."""
    heatwave_colors = ['#F7DDE1', '#EBAFB9', '#DC7284', '#CF3952', '#B22C42']
    coldspell_colors = ['#b3e7f2', '#80d2e6', '#4dbeda', '#1aaacb', '#0086b3']
    heatwave_cmap = mcolors.LinearSegmentedColormap.from_list('heatwave_cmap', heatwave_colors)
    coldspell_cmap = mcolors.LinearSegmentedColormap.from_list('coldspell_cmap', coldspell_colors)
    return heatwave_cmap, coldspell_cmap


def _load_events(csv_path):
    """Read an events CSV and add year/month/day columns from begin_date."""
    if csv_path is None or not Path(csv_path).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty or 'begin_date' not in df.columns:
        return pd.DataFrame()
    begin = pd.to_datetime(df['begin_date'], format='%d/%m/%Y', errors='coerce')
    df = df.assign(year=begin.dt.year, month=begin.dt.month, day=begin.dt.day)
    return df.dropna(subset=['year']).astype({'year': int, 'month': int, 'day': int})


def visualize_extreme_events(heatwave_csv, coldspell_csv, output_path=None, show=True,
                             heat_index_range=HEAT_INDEX_RANGE_C,
                             title='Extreme Event Explorer: Heatwaves and Cold Spells Timeline'):
    """Plot every detected event as a circle on a month-by-year timeline.

    Parameters
    ----------
    heatwave_csv, coldspell_csv : events CSVs written by the detectors, for
        example `hotspells/heatwave_events.csv` and
        `coldspells/coldspells_events.csv` (all years), or the `_peak` files.
    output_path : if given, the figure is saved there (PNG, 150 dpi).
    show : call `plt.show()` (for notebooks). The figure is closed otherwise.
    heat_index_range : (min, max) heat index in degrees C for the color scale.
        Values outside are clipped. Pass None to use the data's own range.
    Returns the matplotlib Figure, or None if there are no events to plot.
    """
    hw = _load_events(heatwave_csv)
    cs = _load_events(coldspell_csv)
    if hw.empty and cs.empty:
        print('No events to plot: both event CSVs are missing or empty.')
        return None

    heatwave_cmap, coldspell_cmap = create_custom_colormap()
    fig, ax = plt.subplots(figsize=(20, 15))

    unique_years = sorted(set(hw.get('year', [])) | set(cs.get('year', [])))
    year_pos = {y: i for i, y in enumerate(unique_years)}

    for i in range(len(unique_years)):
        ax.hlines(y=i, xmin=1, xmax=13, color='gray', alpha=0.5, linestyle='--')
    for month in range(1, 14):
        ax.axvline(x=month, color='lightgray', linestyle='--', linewidth=0.7)

    # Heat waves: color by average heat index (degrees C).
    if not hw.empty and 'avg_heat_index' in hw.columns:
        if heat_index_range is None:
            heat_index_range = (hw['avg_heat_index'].min(), hw['avg_heat_index'].max())
        hi_min, hi_max = heat_index_range
        heatwave_norm = mcolors.Normalize(vmin=hi_min, vmax=hi_max)
        for _, row in hw.iterrows():
            ax.scatter(row['month'] + row['day'] / 30, year_pos[row['year']],
                       s=row['duration'] * CIRCLE_SIZE,
                       color=heatwave_cmap(heatwave_norm(min(max(row['avg_heat_index'], hi_min), hi_max))),
                       alpha=0.7, edgecolors='black', linewidth=0.5)
        sm = plt.cm.ScalarMappable(cmap=heatwave_cmap, norm=heatwave_norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.0001, shrink=0.8,
                     aspect=40).set_label('Heatwave Average Heat Index (°C)', fontsize=14)

    # Cold spells: color by wind chill if present, otherwise wind speed.
    cold_col = next((c for c in ('avg_wind_chill', 'avg_wind_speed') if c in cs.columns), None)
    if not cs.empty and cold_col is not None:
        coldspell_norm = mcolors.Normalize(vmin=cs[cold_col].min(), vmax=cs[cold_col].max())
        for _, row in cs.iterrows():
            ax.scatter(row['month'] + row['day'] / 30, year_pos[row['year']],
                       s=row['duration'] * CIRCLE_SIZE,
                       color=coldspell_cmap(coldspell_norm(row[cold_col])),
                       alpha=0.7, edgecolors='black', linewidth=0.5)
        sm = plt.cm.ScalarMappable(cmap=coldspell_cmap, norm=coldspell_norm)
        sm.set_array([])
        label = ('Cold Spell Wind Chill (°C)' if cold_col == 'avg_wind_chill'
                 else 'Cold Spell Wind Speed (m/s)')
        fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8,
                     aspect=40).set_label(label, fontsize=14)

    handles = [ax.scatter([], [], s=d * CIRCLE_SIZE, color='gray', alpha=0.5,
                          edgecolors='black', label=f'{d} days') for d in LEGEND_DAYS]
    legend = ax.legend(handles=handles, scatterpoints=1, frameon=False, labelspacing=2.2,
                       borderpad=1.2, handletextpad=1.5, title='Event Duration',
                       loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=14)
    legend.get_title().set_fontsize(16)

    ax.set_xlim(0.8, 13.2)
    ax.set_ylim(-1, len(unique_years))
    ax.set_yticks(range(len(unique_years)))
    ax.set_yticklabels(unique_years, fontsize=16)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=16)
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Year', fontsize=16)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f'🖼️  Timeline saved: {output_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
