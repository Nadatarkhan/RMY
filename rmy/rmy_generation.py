import os
from pathlib import Path
import pandas as pd
from rmy.heatwaves import run_full_pipeline as run_heatwaves
from rmy.coldspells import run_full_pipeline_cold as run_coldspells
from rmy.utils import load_epw, save_epw, extract_original_header, seasonal_adjustment

def construct_final_rmy(base_epw_path, hot_events_path, cold_events_path, output_path):
    print("🔄 Constructing final RMY EPW...")

    base_df = load_epw(base_epw_path)
    header_lines = extract_original_header(base_epw_path)

    # Read in peak event files
    peak_hot_df = pd.read_csv(hot_events_path)
    peak_cold_df = pd.read_csv(cold_events_path)

    # Convert dates
    hot_days = pd.to_datetime(peak_hot_df['begin_date'], format="%d/%m/%Y").dt.date
    cold_days = pd.to_datetime(peak_cold_df['begin_date'], format="%d/%m/%Y").dt.date

    # Identify and replace event days
    base_df['date'] = pd.to_datetime(base_df[['year', 'month', 'day']])
    replacement_df = base_df.copy()

    def apply_event(df, event_days, variable):
        for day in event_days:
            mask = df['date'].dt.date == day
            if mask.sum() == 24:
                df.loc[mask, variable] += 3  # Simple bump, placeholder for insert
        return df

    replacement_df = apply_event(replacement_df, hot_days, 'temp_air')
    replacement_df = apply_event(replacement_df, cold_days, 'temp_air')

    # Smooth transitions and seasonally adjust
    replacement_df = seasonal_adjustment(replacement_df, base_df, variable='temp_air')

    # Save final RMY
    replacement_df.drop(columns=['date'], inplace=True)
    save_epw(replacement_df, header_lines, output_path)
    print(f"✅ RMY file saved to: {output_path}")

def run_full_rmy_pipeline(epw_dir, base_dir, output_dir):
    print(f"📂 Running full RMY pipeline:")
    print(f"   AMY Folder: {epw_dir}")
    print(f"   Base File: {base_dir}")
    print(f"   Output Folder: {output_dir}")

    # Run detection pipelines
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
