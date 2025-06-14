# RMY: Representative Meteorological Year Generator

Anomaly-driven generation of Representative Meteorological Year (RMY) weather files with embedded extreme events, including heatwaves and cold spells. This method preserves core TMY characteristics while adding realistic severe climate conditions to support robust design evaluations.

## 📁 Folder Structure

- `/content/base/` → contains the base TMY EPW file (only 1 file)
- `/content/EPWs/` → contains all AMY EPW files for detection
- `/content/hotspells/` → output folder for detected heatwave events
- `/content/coldspells/` → output folder for detected cold spell events
- `/content/final/` → final output RMY EPW file and summary CSVs


## ⚙️ Methods Used

The event detection pipeline includes:
- **Static Thresholding**: Identifies extremes based on fixed temperature or percentile thresholds.
- **GNN-Based Anomaly Detection**: Flags events using graph-based representations of temporal temperature anomalies.
- **Extreme Value Theory (EVT)**: Extracts statistically rare extremes using Peaks Over Threshold (POT) modeling.

Each method is used in a complementary ensemble to identify the most severe year and characteristic events.

## 🔄 Workflow Summary

1. Detect peak heatwaves and cold spells across 15+ years of EPW files.
2. Match extreme events to base-year dates using overlap logic.
3. Replace those dates with extreme-event days from the most severe year, using smoothing.
4. Rebalance monthly averages by inserting non-extreme days to maintain realism.
5. Output:
   - RMY file with embedded extremes
   - Summary CSVs for heatwaves and cold spells


![RMY Workflow](images/Fig1.png)

# RMY: Representative Meteorological Year Generator

Anomaly-driven generation of Representative Meteorological Year (RMY) weather files with embedded extreme events, including **heatwaves** and **cold spells**. This method preserves core TMY characteristics while adding realistic severe climate conditions to support robust design evaluations.

## Quick Start

Install required packages:
```bash
pip install -r requirements.txt
```

Then run the following from the repo root:
```bash
from rmy import run_full_rmy_pipeline
```

## Repository Structure

```
RMY/
├── rmy/
│   ├── __init__.py
│   ├── heatwaves.py
│   ├── coldspells.py
│   ├── utils.py
│   └── rmy_generation.py
├── examples/
│   └── RMY_Generation_Colab.ipynb
├── data/
│   ├── base/
│   └── epws/
├── final/
├── images/
│   └── event_timeline.png
├── README.md
├── LICENSE
└── requirements.txt
```

## Usage Guidance

You can run the full RMY pipeline via:

```bash
from rmy import run_full_rmy_pipeline
```

Make sure your folder structure matches:
- `EPWs/base/` → contains the base TMY file (1 file only)
- `EPWs/epws/` → contains full set of AMY EPWs
- `final/` → RMY weather file + event summaries will be saved here

## Try it on Google Colab

Run the full pipeline interactively on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nadatarkhan/RMY/blob/main/examples/RMY_Generation_Colab.ipynb)

## Sample Output

This timeline shows detected heatwaves and cold spells across years:

![Event Timeline](images/event_timeline.png)
## Citation

If you use this method, please cite:

Tarkhan, N., Crawley, D., Lawrie, L., & Reinhart, C.  
*Generation of representative meteorological years through anomaly-based detection of extreme events.*  
Journal of Building Performance Simulation, 2025.  
[https://doi.org/10.1080/19401493.2025.2499687](https://doi.org/10.1080/19401493.2025.2499687)


## License

This project is licensed under the MIT License. See the LICENSE file for details.