
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.stats import genpareto

def calculate_daily_tmax(epw_data):
    epw_data['temp_air'] = pd.to_numeric(epw_data['temp_air'], errors='coerce')
    epw_data['relative_humidity'] = pd.to_numeric(epw_data['relative_humidity'], errors='coerce')
    daily = epw_data.groupby(['year', 'month', 'day']).agg({
        'temp_air': ['max'],
        'relative_humidity': 'mean'
    }).reset_index()
    daily.columns = ['YEAR', 'MONTH', 'DAY', 'Tmax (oC)', 'RH (%)']
    return daily

def identify_evt_heat_extremes(df, quantile_threshold=0.95, min_duration=3):
    tmax_series = df['Tmax (oC)'].dropna()
    threshold = tmax_series.quantile(quantile_threshold)
    excesses = tmax_series[tmax_series > threshold] - threshold

    if len(excesses) < 5:
        return pd.DataFrame(columns=['Start', 'End'])

    c, loc, scale = genpareto.fit(excesses, floc=0)
    return_level = threshold + genpareto.ppf(0.98, c, loc=0, scale=scale)
    condition = df['Tmax (oC)'] > return_level
    condition = condition.fillna(False)

    spells = []
    grouped = condition.groupby((condition != condition.shift()).cumsum()).cumsum()
    for start in grouped[grouped == 1].index:
        end = start
        while end + 1 < len(grouped) and condition.iloc[end + 1]:
            end += 1
        duration = end - start + 1
        if duration >= min_duration:
            spells.append((start, end))

    return pd.DataFrame(spells, columns=['Start', 'End']) if spells else pd.DataFrame(columns=['Start', 'End'])

def calculate_heatwave_stats(events_df, base_df):
    stats, events = {}, []
    for _, row in events_df.iterrows():
        s_idx, e_idx = row['Start'], row['End']
        period = base_df.iloc[s_idx:e_idx + 1]
        year = int(period.iloc[0]['YEAR'])
        duration = e_idx - s_idx + 1
        avg_tmax = round(period['Tmax (oC)'].mean(), 1)
        std_tmax = round(period['Tmax (oC)'].std(), 1)
        max_tmax = round(period['Tmax (oC)'].max(), 1)
        avg_rh = round(period['RH (%)'].mean(), 1)
        bdate = datetime(year, int(period.iloc[0]['MONTH']), int(period.iloc[0]['DAY']))
        edate = datetime(year, int(period.iloc[-1]['MONTH']), int(period.iloc[-1]['DAY']))
        events.append({
            'begin_date': bdate.strftime('%d/%m/%Y'),
            'end_date': edate.strftime('%d/%m/%Y'),
            'duration': duration,
            'avg_tmax': avg_tmax,
            'std_tmax': std_tmax,
            'max_tmax': max_tmax,
            'avg_rh': avg_rh,
            'year': year
        })
        if year not in stats:
            stats[year] = {'hwn': 0, 'hwf': 0, 'hwd': 0, 'hwdm': [], 'hwaa': -np.inf, 'avg_rh': []}
        stats[year]['hwn'] += 1
        stats[year]['hwf'] += duration
        stats[year]['hwd'] = max(stats[year]['hwd'], duration)
        stats[year]['hwdm'].append(duration)
        stats[year]['hwaa'] = max(stats[year]['hwaa'], max_tmax)
        stats[year]['avg_rh'].append(avg_rh)
    for y in stats:
        stats[y]['hwdm'] = np.mean(stats[y]['hwdm'])
        stats[y]['avg_rh'] = np.mean(stats[y]['avg_rh'])
    stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index().rename(columns={'index': 'year'})
    stats_df['severity'] = stats_df['hwf'] * stats_df['hwaa']
    events_df_final = pd.DataFrame(events).sort_values(by='begin_date')
    return stats_df, events_df_final
