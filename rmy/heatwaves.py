import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy.stats import genpareto

warnings.filterwarnings("ignore", category=UserWarning, message="Parsing dates.*")

def calculate_daily_tmin_tmax(epw_data):
    epw_data['temp_air'] = pd.to_numeric(epw_data['temp_air'], errors='coerce')
    epw_data['relative_humidity'] = pd.to_numeric(epw_data['relative_humidity'], errors='coerce')
    daily = epw_data.groupby(['year', 'month', 'day']).agg({
        'temp_air': ['min', 'max'],
        'relative_humidity': 'mean'
    }).reset_index()
    daily.columns = ['YEAR', 'MONTH', 'DAY', 'Tmin (oC)', 'Tmax (oC)', 'Humidity (%)']
    return daily

def calculate_heat_index(temp_air, rh):
    T_F = (temp_air * 9/5) + 32
    hi = (
        -42.379 + 2.04901523*T_F + 10.14333127*rh - 0.22475541*T_F*rh
        - 6.83783e-3*T_F**2 - 5.481717e-2*rh**2
        + 1.22874e-3*T_F**2*rh + 8.5282e-4*T_F*rh**2
        - 1.99e-6*T_F**2*rh**2
    )
    return hi

def identify_heatwaves_static(df, method='ensemble', tmin_th=20, tmax_th=35, max_duration=21):
    if method == 'ensemble':
        df1 = identify_heatwaves_static(df, method='percentile', max_duration=max_duration)
        df2 = identify_heatwaves_static(df, method='temperature', tmax_th=tmax_th, max_duration=max_duration)
        combined = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        return combined
    elif method == 'percentile':
    elif method == 'temperature':
    else:
        raise ValueError("Invalid method")

    series = condition.groupby((condition != condition.shift()).cumsum()).cumsum()
    starts = series[series == 1].index
    waves = []

    for start in starts:
        count = 1
        for next_day in range(start + 1, len(df)):
            if series[next_day] > count:
                count += 1
                end = next_day
                while end + 1 < len(df) and series[end + 1] > 0:
                    end += 1
                duration = end - start + 1
                if 3 <= duration <= max_duration:
                    waves.append((start, end))
                break
            else:
                break

    return pd.DataFrame(waves, columns=['Start', 'End']) if waves else pd.DataFrame(columns=['Start', 'End'])


def identify_evt_extremes(df, quantile_threshold=0.95, min_duration=3):
    """
    Identify extreme heatwave events using EVT (Peaks Over Threshold).
    """
    threshold = tmax_series.quantile(quantile_threshold)
    excesses = tmax_series[tmax_series > threshold] - threshold

    if len(excesses) < 5:
        return pd.DataFrame(columns=['Start', 'End'])  # Not enough data to fit GPD

    # Fit Generalized Pareto Distribution to excesses
    c, loc, scale = genpareto.fit(excesses, floc=0)

    # Define a high return level (e.g., 95th percentile of fitted GPD)
    return_level = threshold + genpareto.ppf(0.95, c, loc=0, scale=scale)

    # Flag days exceeding return level
    condition = condition.fillna(False)

    # Group consecutive days into events
    series = condition.groupby((condition != condition.shift()).cumsum()).cumsum()
    starts = series[series == 1].index
    events = []

    for start in starts:
        end = start
        while end + 1 < len(df) and condition.iloc[end + 1]:
            end += 1
        duration = end - start + 1
        if duration >= min_duration:
            events.append((start, end))

    return pd.DataFrame(events, columns=['Start', 'End']) if events else pd.DataFrame(columns=['Start', 'End'])


def apply_gnn_detection(df):
    df = df[['year', 'month', 'day', 'hour', 'temp_air']].dropna()
    daily_z = (daily_max - daily_max.mean()) / daily_max.std()
    threshold = 2.5
    extreme_days = daily_z[daily_z > threshold]
    return list(pd.to_datetime(extreme_days.index).year)

def calculate_event_stats(heatwave_df, base_df):
        return pd.DataFrame(), pd.DataFrame(), None

    events = []
    stats = {}

        s_idx, e_idx = row['Start'], row['End']
        year = int(spell.iloc[0]['YEAR'])
        duration = e_idx - s_idx + 1
        avg_tmax = round(spell['Tmax (oC)'].mean(), 1)
        std_tmax = round(spell['Tmax (oC)'].std(), 1)
        max_tmax = round(spell['Tmax (oC)'].max(), 1)
        avg_hi = round(calculate_heat_index(spell['Tmax (oC)'], spell['Humidity (%)']).mean(), 1)

        bdate = datetime(year, int(spell.iloc[0]['MONTH']), int(spell.iloc[0]['DAY']))
        edate = datetime(year, int(spell.iloc[-1]['MONTH']), int(spell.iloc[-1]['DAY']))

        events.append({
            'begin_date': bdate.strftime('%d/%m/%Y'),
            'end_date': edate.strftime('%d/%m/%Y'),
            'duration': duration,
            'avg_tmax': avg_tmax,
            'std_tmax': std_tmax,
            'max_tmax': max_tmax,
            'avg_heat_index': avg_hi,
            'year': year
        })

        if year not in stats:
            stats[year] = {'hwn': 0, 'hwf': 0, 'hwd': 0, 'hwdm': [], 'hwaa': -np.inf}
        stats[year]['hwn'] += 1
        stats[year]['hwf'] += duration
        stats[year]['hwd'] = max(stats[year]['hwd'], duration)
        stats[year]['hwdm'].append(duration)
        stats[year]['hwaa'] = max(stats[year]['hwaa'], max_tmax)

    for y in stats:
        stats[y]['hwdm'] = np.mean(stats[y]['hwdm'])

    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['hwn', 'hwf', 'hwd', 'hwdm', 'hwaa']).reset_index().rename(columns={'index': 'year'}).sort_values(by='year')
    events_df = pd.DataFrame(events).sort_values(by='begin_date')

    #print(f" Highest severity year: {peak_year}")

    return stats_df, events_df, peak_year

def process_file(epw_path, gnn_years=None, apply_gnn=True):
    #print(f" Processing file: {epw_path}")
        'year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air',
        'temp_dew', 'relative_humidity', 'atmospheric_pressure', 'etr', 'etrn',
        'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
        'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance',
        'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
        'visibility', 'ceiling_height', 'present_weather_observation',
        'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth',
        'snow_depth', 'days_since_last_snowfall', 'albedo',
        'liquid_precipitation_depth', 'liquid_precipitation_quantity'
    ])

    daily_df = calculate_daily_tmin_tmax(epw_df)
    daily_df = daily_df[(daily_df["Tmin (oC)"] != -9999) & (daily_df["Tmax (oC)"] != -9999)]

    # Static detection
    static_events = identify_heatwaves_static(daily_df, method='ensemble')
    static_stats, static_df, _ = calculate_event_stats(static_events, daily_df)

    # GNN filtering
    if apply_gnn and gnn_years:
        #print(f"🔎 GNN-filtered years: {gnn_years}")
    else:
        #print("⚠️ No GNN years or GNN skipped.")

    gnn_events = identify_heatwaves_static(filtered_df, method='temperature')
    gnn_stats, gnn_df, _ = calculate_event_stats(gnn_events, filtered_df)

    # EVT detection
    evt_events = identify_evt_extremes(daily_df)
    evt_stats, evt_df, _ = calculate_event_stats(evt_events, daily_df)

    # Combine all
    all_events = pd.concat([static_df, gnn_df, evt_df]).drop_duplicates()
    if 'begin_date' in all_events.columns and not all_events.empty:
        all_events = all_events.sort_values('begin_date').reset_index(drop=True)
    else:
        all_events = pd.DataFrame(columns=[
            'begin_date', 'end_date', 'duration', 'avg_tmax', 'std_tmax',
            'max_tmax', 'avg_heat_index', 'year'
        ])

    all_stats = pd.concat([static_stats, gnn_stats, evt_stats])
    if all_stats.empty:
        print(f" No valid stats found in {epw_path}. Skipping.")
        return all_stats, all_events, pd.DataFrame(), pd.DataFrame()

    all_stats = all_stats.groupby('year').agg({
        'hwn': 'sum', 'hwf': 'sum', 'hwd': 'max',
        'hwdm': 'mean', 'hwaa': 'max', 'severity': 'max'
    }).reset_index().sort_values('year')
    all_stats['severity'] = all_stats['hwf'] * all_stats['hwaa']

    if not all_stats.empty and 'severity' in all_stats.columns:
        peak_year = all_stats.loc[all_stats['severity'].idxmax()]['year']
        peak_events = all_events[all_events['year'] == peak_year]
        peak_stats = all_stats[all_stats['year'] == peak_year]
        #print(f" Highest severity year: {peak_year}")
    else:
        peak_events = pd.DataFrame()
        peak_stats = pd.DataFrame()

    return all_stats, all_events, peak_stats, peak_events, static_stats, static_df



def save_final_outputs(output_dir, base_stats, base_events, all_stats, all_events, peak_stats, peak_events):
    os.makedirs(output_dir, exist_ok=True)
    base_events.to_csv(os.path.join(output_dir, "heatwave_events_base.csv"), index=False)
    base_stats.to_csv(os.path.join(output_dir, "heatwave_stats_base.csv"), index=False)
    all_events.to_csv(os.path.join(output_dir, "heatwave_events.csv"), index=False)
    all_stats.to_csv(os.path.join(output_dir, "heatwave_stats.csv"), index=False)
    peak_events.to_csv(os.path.join(output_dir, "heatwave_events_peak.csv"), index=False)
    peak_stats.to_csv(os.path.join(output_dir, "heatwave_stats_peak.csv"), index=False)

def run_full_pipeline(epw_dir, base_dir, output_dir):
    print(f" Starting hybrid method on EPWs in {epw_dir}")

    # Load and process base EPW
    base_files = [f for f in os.listdir(base_dir) if f.endswith('.epw')]
    if not base_files:
        raise FileNotFoundError("No base EPW file found in base folder.")
    base_epw_path = os.path.join(base_dir, base_files[0])
    #print(f" Processing base file: {base_epw_path}")

    # Read for GNN detection
        'year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air',
        'temp_dew', 'relative_humidity', 'atmospheric_pressure', 'etr', 'etrn',
        'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
        'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance',
        'wind_direction', 'wind_speed', 'total_sky_cover', 'opaque_sky_cover',
        'visibility', 'ceiling_height', 'present_weather_observation',
        'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth',
        'snow_depth', 'days_since_last_snowfall', 'albedo',
        'liquid_precipitation_depth', 'liquid_precipitation_quantity'
    ])
    gnn_years = apply_gnn_detection(df_base)

    # Process base file
    base_stats, base_events, _, _, _, _ = process_file(
        base_epw_path, gnn_years, apply_gnn=True
    )
    if base_stats.empty:
        print(f"No valid stats found in {base_epw_path}. Skipping.")
        base_stats, base_events = pd.DataFrame(), pd.DataFrame()

    # Process all EPW files in the EPWs folder
    all_stats_df = []
    all_events_df = []

    for epw_file in sorted(os.listdir(epw_dir)):
        if not epw_file.endswith(".epw"):
            continue
        epw_path = os.path.join(epw_dir, epw_file)
        print(f" Processing file: {epw_path}")

        try:
            stats_df, events_df, _, _, _, _ = process_file(
                epw_path, gnn_years=None, apply_gnn=False
            )
        except Exception as e:
            print(f"Error processing {epw_path}: {e}")

    # Combine all stats and events
    epw_stats = pd.concat(all_stats_df, ignore_index=True) if all_stats_df else pd.DataFrame()
    epw_events = pd.concat(all_events_df, ignore_index=True) if all_events_df else pd.DataFrame()

    if not epw_stats.empty:
        epw_stats['severity'] = epw_stats['hwf'] * epw_stats['hwaa']
        print("\n📈 Severity scores across all years:")
        for _, row in epw_stats[['year', 'severity']].sort_values('year').iterrows():
            print(f"   - Year {int(row['year'])}: Severity = {row['severity']:.2f}")

        peak_year = int(epw_stats.loc[epw_stats['severity'].idxmax()]['year'])
        peak_events = epw_events[epw_events['year'] == peak_year]
        peak_stats = epw_stats[epw_stats['year'] == peak_year]

        print("\n Ensemble method run complete.")
        print(f" Peak severity year identified: {peak_year}")
        print("\n Event Stats for Peak Year:")
        for col in ['hwn', 'hwf', 'hwd', 'hwdm', 'hwaa', 'severity']:
            val = peak_stats.iloc[0][col]
            print(f"   - {col}: {val:.2f}" if isinstance(val, (float, int)) else f"   - {col}: {val}")
        print()

    else:
        print(" No heatwaves detected in EPWs.")
        peak_stats = pd.DataFrame()
        peak_events = pd.DataFrame()


    save_final_outputs(output_dir, base_stats, base_events, epw_stats, epw_events, peak_stats, peak_events)
    print("✅ Saved all 6 heat wave output CSVs")




import pandas as pd

# Function to create custom colormaps for heatwaves and cold spells
def create_custom_colormap():
    heatwave_colors = ['#F7DDE1', '#EBAFB9', '#DC7284', '#CF3952', '#B22C42']  # Heatwaves color scheme
    coldspell_colors = ['#b3e7f2', '#80d2e6', '#4dbeda', '#1aaacb', '#0086b3']  # Custom blue shades for cold spells
    heatwave_cmap = mcolors.LinearSegmentedColormap.from_list('heatwave_cmap', heatwave_colors)
    coldspell_cmap = mcolors.LinearSegmentedColormap.from_list('coldspell_cmap', coldspell_colors)
    return heatwave_cmap, coldspell_cmap

# Function to visualize heatwave and cold spell events on a timeline
    # Read the CSV files for heatwaves and cold spells

    # Convert 'begin_date' to datetime format



    # Create custom colormaps for heatwaves and cold spells
    heatwave_cmap, coldspell_cmap = create_custom_colormap()


    # Get unique years

    # Plot horizontal lines for each year
    for i, year in enumerate(unique_years):
        ax.hlines(y=i, xmin=1, xmax=13, color='gray', alpha=0.5, linestyle='--')

    # Add vertical dashed lines between months and after December
    for month in range(1, 14):  # Include 13 to add line after December
        ax.axvline(x=month, color='lightgray', linestyle='--', linewidth=0.7)

    # Adjust the normalization to better reflect the range of the data in the colormap
    # Set a fixed normalization range for heat index (85°C to 125°C)
    heatwave_norm = mcolors.Normalize(vmin=85, vmax=125)


        wind_chill_column = 'avg_wind_chill'
    else:
        wind_chill_column = 'avg_wind_speed'

    circle_size_factor = 50


        year_index = unique_years.index(row['year'])
        month_day = row['month'] + row['day'] / 30
        ax.scatter(month_day, year_index,
                   s=row['duration'] * circle_size_factor,
                   #color=heatwave_cmap(heatwave_norm(row['avg_heat_index'])),
                   # Cap heat index normalization between 85 and 125
                   color=heatwave_cmap(heatwave_norm(row['clipped_index'])),
                   alpha=0.7, edgecolors='black', linewidth=0.5)

        year_index = unique_years.index(row['year'])
        month_day = row['month'] + row['day'] / 30
        ax.scatter(month_day, year_index,
                   s=row['duration'] * circle_size_factor,
                   color=coldspell_cmap(coldspell_norm(row[wind_chill_column])),
                   alpha=0.7, edgecolors='black', linewidth=0.5)

    heatwave_sm.set_array([])
    coldspell_sm.set_array([])

    # Shift heatwave bar closer to coldspell bar by reducing pad
    cbar_heatwave = fig.colorbar(heatwave_sm, ax=ax, orientation='horizontal', pad=0.0001, shrink=0.8, aspect=40)
    cbar_coldspell = fig.colorbar(coldspell_sm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, aspect=40)

    cbar_heatwave.set_label('Heatwave Average Heat Index (°C)',fontsize=14)

    sizes = [1, 3, 5, 7]
    labels = [f'{size} days' for size in sizes]
    handles = [ax.scatter([], [], s=size * circle_size_factor, color='gray', alpha=0.5, edgecolors='black', label=label)
               for size, label in zip(sizes, labels)]

    # Removed bounding box from legend
    legend = ax.legend(
        handles=handles,
        scatterpoints=1,
        frameon=False,
        labelspacing=1,
        title='Event Duration',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=14  # label font size
    )

    # Now set the title font size explicitly
    legend.get_title().set_fontsize(16)


    ax.set_xlim(0.8, 13.2)
    ax.set_ylim(-1, len(unique_years))
    ax.set_yticks(ticks=range(len(unique_years)))
    ax.set_yticklabels(unique_years, fontsize=16)
    ax.set_xticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=16)

    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Year', fontsize=16)
    ax.set_title('Extreme Event Explorer: Heatwaves and Cold Spells Timeline')

    for spine in ax.spines.values():
        spine.set_visible(False)


# Example usage:



import pandas as pd
import numpy as np
import scipy.stats as stats

# --- Load data ---

# --- Extract and label ---
for df in (heatwave_df, coldspell_df):


# --- Combine & preprocess ---
events_df = pd.concat([
], ignore_index=True)

# Map years to indices to remove gaps
year_to_index = {yr: idx for idx, yr in enumerate(unique_years)}

# --- Color setup ---
hw_colors = ['#F7DDE1','#EBAFB9','#DC7284','#CF3952','#B22C42']
cs_colors = ['#b3e7f2','#80d2e6','#4dbeda','#1aaacb','#0086b3']
hw_cmap = mcolors.LinearSegmentedColormap.from_list('hw', hw_colors)
cs_cmap = mcolors.LinearSegmentedColormap.from_list('cs', cs_colors)


    else mcolors.to_hex(cs_cmap(cs_norm(row['intensity']))), axis=1)

# --- KDE setup ---
xs_hw_all = np.linspace(hw_all.min() - 2, hw_all.max() + 2, 200)
xs_cs_all = np.linspace(cs_all.min() - 2, cs_all.max() + 2, 200)
kde_hw_all = stats.gaussian_kde(hw_all)(xs_hw_all)
kde_cs_all = stats.gaussian_kde(cs_all)(xs_cs_all)
max_hw_pdf = kde_hw_all.max() * 1.6
max_cs_pdf = kde_cs_all.max() * 1.5

# --- Yearlines ---
background_yearlines = [
    go.Scatter(
        x=[0.8, 13.2], y=[i, i],
        mode='lines', line=dict(color='lightgrey', width=1, dash='dash'),
        showlegend=False, hoverinfo='none',
        xaxis="x", yaxis="y"
    ) for i in range(len(unique_years))
]

# --- Frames ---
frames = []
all_past = pd.DataFrame()

for yr in unique_years:
    all_past = pd.concat([all_past, df_year], ignore_index=True)

    pdf_hw = stats.gaussian_kde(past_hw)(xs_hw_all) if len(past_hw) > 1 else np.zeros_like(xs_hw_all)
    pdf_cs = stats.gaussian_kde(past_cs)(xs_cs_all) if len(past_cs) > 1 else np.zeros_like(xs_cs_all)

    # Peak values
    peak_hw_x = xs_hw_all[np.argmax(pdf_hw)]
    peak_hw_y = np.max(pdf_hw)
    peak_cs_x = xs_cs_all[np.argmax(pdf_cs)]
    peak_cs_y = np.max(pdf_cs)

    frames.append(go.Frame(
        data=[
            *background_yearlines,
            go.Scatter(
                x=all_past['month_day'], y=all_past['year_index'],
                mode='markers',
                marker=dict(size=all_past['duration'] * 2.7, color=all_past['color'], opacity=0.7,
                            line=dict(color='black', width=0.5)),
                hovertemplate="<b>%{customdata[0]}</b><br>Year: %{y}<br>Intensity: %{customdata[1]}<br>Duration: %{customdata[2]} days<extra></extra>",
                showlegend=False,
                xaxis="x", yaxis="y"
            ),
            # Heatwave KDEs
            go.Scatter(x=xs_hw_all, y=kde_hw_all, mode='lines', line=dict(color='#CF3952', width=3),
                       fill='tozeroy', fillcolor='rgba(207,57,82,0.05)', xaxis="x2", yaxis="y2", showlegend=False),
            go.Scatter(x=xs_hw_all, y=pdf_hw, mode='lines', line=dict(color='#CF3952', width=2, dash='dash'),
                       xaxis="x2", yaxis="y2", showlegend=False),
            go.Scatter(x=[peak_hw_x, peak_hw_x], y=[0, peak_hw_y], mode='lines',
                       line=dict(color='gray', width=2, dash='dot'), xaxis="x2", yaxis="y2", showlegend=False),
            go.Scatter(x=[peak_hw_x], y=[peak_hw_y], text=[f"{peak_hw_x:.1f}"], mode='text',
                       textposition='top center', xaxis="x2", yaxis="y2", showlegend=False, hoverinfo='skip'),
            # Coldspell KDEs
            go.Scatter(x=xs_cs_all, y=kde_cs_all, mode='lines', line=dict(color='#1aaacb', width=3),
                       fill='tozeroy', fillcolor='rgba(26,170,203,0.05)', xaxis="x3", yaxis="y3", showlegend=False),
            go.Scatter(x=xs_cs_all, y=pdf_cs, mode='lines', line=dict(color='#1aaacb', width=2, dash='dash'),
                       xaxis="x3", yaxis="y3", showlegend=False),
            go.Scatter(x=[peak_cs_x, peak_cs_x], y=[0, peak_cs_y], mode='lines',
                       line=dict(color='gray', width=2, dash='dot'), xaxis="x3", yaxis="y3", showlegend=False),
            go.Scatter(x=[peak_cs_x], y=[peak_cs_y], text=[f"{peak_cs_x:.1f}"], mode='text',
                       textposition='top center', xaxis="x3", yaxis="y3", showlegend=False, hoverinfo='skip'),
        ], name=str(yr)
    ))

# --- Colorbars ---
cb_hw = go.Scatter(x=[None], y=[None], mode='markers', marker=dict(
    colorscale=hw_colors, showscale=True, cmin=hw_norm.vmin, cmax=hw_norm.vmax, color=[hw_norm.vmin],
    colorbar=dict(orientation='h', x=0.275, len=0.55, xanchor='center', yanchor='top', y=-0.12, title="Heat Index (°C)")
), showlegend=False, hoverinfo='none')
cb_cs = go.Scatter(x=[None], y=[None], mode='markers', marker=dict(
    colorscale=cs_colors, showscale=True, cmin=cs_norm.vmin, cmax=cs_norm.vmax, color=[cs_norm.vmin],
    colorbar=dict(orientation='h', x=0.275, len=0.55, xanchor='center', yanchor='top', y=-0.24, title="Wind Chill (°C or m/s)")
), showlegend=False, hoverinfo='none')

# --- Layout ---
layout = go.Layout(
    title="Extreme Event Explorer: Heatwaves & Coldspells Timeline",
    width=1400, height=1000,
    margin=dict(b=300),
    xaxis=dict(domain=[0, 0.55], title="Month", tickmode='array',
               tickvals=list(range(1, 13)), ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
               range=[0.8, 13.2], linecolor='black'),
    yaxis=dict(domain=[0, 1], title="Year", tickvals=list(range(len(unique_years))),
               ticktext=[str(y) for y in unique_years], range=[-0.5, len(unique_years)-0.5], linecolor='black'),
    xaxis2=dict(domain=[0.65, 1], title="Heatwave Intensity (°C)", linecolor='black', anchor='y2'),
    yaxis2=dict(domain=[0.55, 1], title="Density (PDF)", linecolor='black', range=[0, max_hw_pdf], anchor='x2'),
    xaxis3=dict(domain=[0.65, 1], title="Coldspell Intensity (°C or m/s)", linecolor='black', anchor='y3'),
    yaxis3=dict(domain=[0, 0.45], title="Density (PDF)", linecolor='black', range=[0, max_cs_pdf], anchor='x3'),
    annotations=[dict(
        text="Solid = all-years PDF · Dashed = PDF(all past)",
        showarrow=False, xref="paper", yref="paper",
        x=0.85, y=1.07, font=dict(size=12)
    )],
    showlegend=False,
    updatemenus=[dict(
        type="buttons", direction="left", pad={"r": 10, "t": 10}, x=0.1, xanchor="right", y=-0.06, yanchor="top",
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, {"frame": {"duration": 600, "redraw": False}, "fromcurrent": True}]),
            dict(label="Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
        ]
    )]
)

# --- Build figure ---
    data=[*frames[0].data, cb_hw, cb_cs],
    layout=layout,
    frames=frames
)

fig.show()


import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Setup paths ---
base_epw_folder = Path('/content/base')
all_epw_folder = Path('/content/EPWs')
output_epw_path = Path('/content/final/RMY.epw')
coldspell_stats_path = Path('/content/coldspells/coldspells_stats_peak.csv')
heatwave_stats_path = Path('/content/hotspells/heatwave_stats_peak.csv')

def read_epw_file(epw_file_path):
        'year', 'month', 'day', 'hour', 'minute', 'data_source_unct', 'temp_air', 'temp_dew', 'relative_humidity',
        'atmospheric_pressure', 'etr', 'etrn', 'ghi_infrared', 'ghi', 'dni', 'dhi', 'global_hor_illum',
        'direct_normal_illum', 'diffuse_horizontal_illum', 'zenith_luminance', 'wind_direction', 'wind_speed',
        'total_sky_cover', 'opaque_sky_cover', 'visibility', 'ceiling_height', 'present_weather_observation',
        'present_weather_codes', 'precipitable_water', 'aerosol_optical_depth', 'snow_depth',
        'days_since_last_snowfall', 'albedo', 'liquid_precipitation_depth', 'liquid_precipitation_quantity'
    ])

def get_peak_year(stats_csv):

def find_epw_by_year(year, epw_folder):
    for fname in os.listdir(epw_folder):
        if fname.endswith('.epw') and str(year) in fname:
            return os.path.join(epw_folder, fname)
    raise FileNotFoundError(f"No EPW found for year {year} in {epw_folder}")

def interpolate_smooth_transitions(df, indices, columns):
    for col in columns:
    return df

def match_events(base_df, peak_df):
        best, min_diff = None, float('inf')
        for idx, peak in unmatched.iterrows():
            if base['begin_date'].month == peak['begin_date'].month and base['begin_date'].day == peak['begin_date'].day:
                diff = abs(base['duration'] - peak['duration'])
                if diff < min_diff:
                    best, min_diff = idx, diff
        if best is not None:
            matched.append((base, unmatched.loc[best]))
            unmatched = unmatched.drop(best)
    return matched, unmatched

def integrate_events(df, matched, unmatched, peak_epw, label):
    all_events = matched + [(None, e) for e in unmatched.itertuples()]
    replaced = set()
    for base, peak in all_events:
        for d in pd.date_range(start=peak.begin_date, end=peak.end_date, freq='D'):
            slice_peak = peak_epw[(peak_epw['month'] == d.month) & (peak_epw['day'] == d.day)].copy()
            if slice_peak.empty: continue
            smoothing_idx = list(range(max(0, idx.min()-8), min(len(df), idx.max()+9)))
            df = interpolate_smooth_transitions(df, smoothing_idx, slice_peak.columns.drop(['year', 'month', 'day', 'hour', 'minute']))
            for _, row in slice_peak.iterrows():
                replaced.add(f"{int(row['month']):02d}-{int(row['day']):02d}")
            df = interpolate_smooth_transitions(df, smoothing_idx, slice_peak.columns.drop(['year', 'month', 'day', 'hour', 'minute']))
    print(f"{label} events integrated. Days replaced: {len(replaced)}")
    return df, replaced

def calculate_monthly_avg_conditions(df, months):

def find_days_to_adjust_avg(epw_files, targets, months, cond):
    days, sources = {m: [] for m in months}, {m: [] for m in months}
    for epw in epw_files:
        df = read_epw_file(epw)
        for m in months:
            for d in range(1, 32):
                if not day.empty and cond(day, targets.loc[m]):
                    days[m].append(day)
                    sources[m].append(epw)
                if sum(len(v) for v in days.values()) >= 40:
                    return days, sources
    return days, sources

def integrate_days(df, days, replaced, sources, targets, months):
    inserted = set()
    for m in months:
        for i, day_df in enumerate(days[m]):
            if len(inserted) >= 30: break
                tag = f"{int(row['month']):02d}-{int(row['day']):02d}"
                if tag in replaced or tag in inserted: continue
                smoothing_idx = list(range(max(0, idx.min()-8), min(len(df), idx.max()+9)))
                inserted.add(tag)
    return df, inserted

def safe_load_events(path, cols):
    except: return pd.DataFrame(columns=cols)

# --- Begin process ---
base_epw_path = list(base_epw_folder.glob('*.epw'))[0]
#header = [next(open(base_epw_path)) for _ in range(8)]
# Read and store header lines safely
with open(base_epw_path, 'r') as f:
    header = [f.readline() for _ in range(8)]


base_epw = read_epw_file(base_epw_path)
epw_list = list(all_epw_folder.glob('*.epw'))

heat_peak = find_epw_by_year(get_peak_year(heatwave_stats_path), all_epw_folder)
cold_peak = find_epw_by_year(get_peak_year(coldspell_stats_path), all_epw_folder)
peak_heat = read_epw_file(heat_peak)
peak_cold = read_epw_file(cold_peak)

heat_base = safe_load_events('/content/hotspells/heatwave_events_base.csv', ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
heat_peak_df = safe_load_events('/content/hotspells/heatwave_events_peak.csv', ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
cold_base = safe_load_events('/content/coldspells/base/coldspells_events_base.csv', ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])
cold_peak_df = safe_load_events('/content/coldspells/coldspells_events_peak.csv', ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])

mh, uh = match_events(heat_base, heat_peak_df)
mc, uc = match_events(cold_base, cold_peak_df)

#old
final, rh = integrate_events(base_epw.copy(), mh, uh, peak_heat, 'Heatwave')
final, rc = integrate_events(final, mc, uc, peak_cold, 'Coldspell')

#NEW
# Step 1: Integrate Heatwaves on the base EPW
#heat_intermediate, rh = integrate_events(base_epw.copy(), mh, uh, peak_heat, 'Heatwave')

# Step 2: Use heat-adjusted file as input for coldspell integration
#final, rc = integrate_events(heat_intermediate.copy(), mc, uc, peak_cold, 'Coldspell')


# Adjust monthly averages***************************************************

#Southern Hemisphere
#summer = [12, 1, 2]
#winter = [6, 7, 8]

#Northern Hemisphere
summer = [6, 7, 8]
winter = [12, 1, 2]


base_summer = calculate_monthly_avg_conditions(base_epw, summer)
base_winter = calculate_monthly_avg_conditions(base_epw, winter)


sdays, sfiles = find_days_to_adjust_avg(epw_list, base_summer, summer, s_condition)
wdays, wfiles = find_days_to_adjust_avg(epw_list, base_winter, winter, w_condition)

final, ins_s = integrate_days(final, sdays, rh, sfiles, base_summer, summer)
final, ins_w = integrate_days(final, wdays, rc, wfiles, base_winter, winter)

# Write final EPW
with open(output_epw_path, 'w', newline='') as file:
    for line in header:
        file.write(line if line.endswith('\n') else line + '\n')
    final.to_csv(file, index=False, header=False, float_format='%g', na_rep='999')


print(f"✅ Final RMY saved to: {output_epw_path}")
print(f"📂 Heatwave Peak EPW used: {heat_peak}")
print(f"📂 Coldspell Peak EPW used: {cold_peak}")
