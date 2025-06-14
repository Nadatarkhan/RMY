
import os
from pathlib import Path
import pandas as pd
from rmy.heatwaves import run_full_pipeline as run_heatwaves
from rmy.coldspells import run_full_pipeline_cold as run_coldspells
from rmy.utils import (load_epw, extract_original_header, save_epw, 
    read_epw_file, get_peak_year, find_epw_by_year, 
    safe_load_events, match_events, integrate_events, 
    calculate_monthly_avg_conditions, find_days_to_adjust_avg, integrate_days)

def construct_final_rmy(base_epw_path, hot_events_path, cold_events_path, output_path):
    print("🔄 Constructing final RMY EPW...")
    base_df = read_epw_file(base_epw_path)
    header_lines = extract_original_header(base_epw_path)
    
    all_epw_folder = Path(base_epw_path).parent.parent / "EPWs"
    epw_list = list(all_epw_folder.glob('*.epw'))

    peak_hot_df = pd.read_csv(hot_events_path)
    peak_cold_df = pd.read_csv(cold_events_path)

    # Load stats
    heatwave_stats_path = Path(hot_events_path).parent / "heatwave_stats_peak.csv"
    coldspell_stats_path = Path(cold_events_path).parent / "coldspells_stats_peak.csv"
    heat_peak = find_epw_by_year(get_peak_year(heatwave_stats_path), all_epw_folder)
    cold_peak = find_epw_by_year(get_peak_year(coldspell_stats_path), all_epw_folder)
    peak_heat = read_epw_file(heat_peak)
    peak_cold = read_epw_file(cold_peak)

    heat_base = safe_load_events(str(Path(hot_events_path).parent / "heatwave_events_base.csv"), 
        ['begin_date','end_date','duration','avg_tmax','std_tmax','max_tmax'])
    cold_base = safe_load_events(str(Path(cold_events_path).parent / "base/coldspells_events_base.csv"), 
        ['begin_date','end_date','duration','avg_tmin','std_tmin','min_tmin'])

    mh, uh = match_events(heat_base, peak_hot_df)
    mc, uc = match_events(cold_base, peak_cold_df)

    final, rh = integrate_events(base_df.copy(), mh, uh, peak_heat, 'Heatwave')
    final, rc = integrate_events(final, mc, uc, peak_cold, 'Coldspell')

    summer = [6, 7, 8]
    winter = [12, 1, 2]

    base_summer = calculate_monthly_avg_conditions(base_df, summer)
    base_winter = calculate_monthly_avg_conditions(base_df, winter)

    s_condition = lambda df, t: df['temp_air'].mean() < t['temp_air'] and df['relative_humidity'].mean() < t['relative_humidity']
    w_condition = lambda df, t: df['temp_air'].mean() > t['temp_air'] and df['relative_humidity'].mean() > t['relative_humidity']

    sdays, sfiles = find_days_to_adjust_avg(epw_list, base_summer, summer, s_condition)
    wdays, wfiles = find_days_to_adjust_avg(epw_list, base_winter, winter, w_condition)

    final, ins_s = integrate_days(final, sdays, rh, sfiles, base_summer, summer)
    final, ins_w = integrate_days(final, wdays, rc, wfiles, base_winter, winter)

    os.makedirs(Path(output_path).parent, exist_ok=True)
    save_epw(final, header_lines, output_path)

    print(f"✅ Final RMY saved to: {output_path}")
    print(f"📂 Heatwave Peak EPW used: {heat_peak}")
    print(f"📂 Coldspell Peak EPW used: {cold_peak}")

def run_full_rmy_pipeline(epw_dir, base_dir, output_dir):
    print(f"📂 Running full RMY pipeline:")
    print(f"   AMY Folder: {epw_dir}")
    print(f"   Base File: {base_dir}")
    print(f"   Output Folder: {output_dir}")

    hot_output = os.path.join(output_dir, "hotspells")
    cold_output = os.path.join(output_dir, "coldspells")
    os.makedirs(hot_output, exist_ok=True)
    os.makedirs(cold_output, exist_ok=True)

    run_heatwaves(epw_dir, base_dir, hot_output)
    run_coldspells(epw_dir, base_dir, cold_output)

    base_file = [f for f in os.listdir(base_dir) if f.endswith('.epw')][0]
    base_epw_path = os.path.join(base_dir, base_file)
    hot_events_path = os.path.join(hot_output, "heatwave_events_peak.csv")
    cold_events_path = os.path.join(cold_output, "coldspells_events_peak.csv")
    rmy_output_path = os.path.join(output_dir, f"RMY_{base_file}")

    construct_final_rmy(base_epw_path, hot_events_path, cold_events_path, rmy_output_path)
