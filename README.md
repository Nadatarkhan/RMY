
# Extreme-Aware Meteorological Years: Generating RMYs and FRMYs

## Motivation
TMY weather files represent averaged conditions and hence do not represent the full spectrum of extremes. This limits their use for assessing the performance of buildings under realistic heat or cold stress conditions.

## Methodology
Anomaly-driven generation of Representative Meteorological Year (RMY) weather files with embedded extreme events, including heatwaves and cold spells. This method preserves core TMY characteristics while adding realistic severe climate conditions to support robust resilience evaluations, and produces standard EPW outputs that can be used across all major building simulation platforms.

### Methods Used
The event detection pipeline includes:

- **Static Thresholding**: Identifies extremes based on fixed temperature or percentile thresholds.
- **GNN-Based Anomaly Detection**: Flags events using graph-based representations of temporal temperature anomalies.
- **Extreme Value Theory (EVT)**: Extracts statistically rare extremes using Peaks Over Threshold (POT) modeling.

Each method is used in a complementary ensemble to identify the most severe year and characteristic events.

### Workflow Summary
1. Detect peak heatwaves and cold spells across 15+ years of EPW files.
2. Match extreme events to base-year dates using overlap logic.
3. Replace those dates with extreme-event days from the most severe year, using smoothing.
4. Rebalance monthly averages by inserting non-extreme days to maintain realism.

**Output:**
- RMY file with embedded extremes
- Summary CSVs for heatwaves and cold spells

## Key Components
- `rmy/` contains core functions for anomaly detection, integration, and output formatting.
- `examples/` includes a Google Colab notebook to run the pipeline interactively.
- `data/` folder contains subfolders for base TMY files and EPWs used for event detection.

## 📁 Folder Structure
```
data/
├── base/          # contains the base TMY EPW file (only 1 file)
├── epws/          # contains all AMY EPW files for detection
├── RMYs/          # generated RMY EPW files
├── FRMYs/         # generated FRMY EPW files

content/
├── hotspells/     # output folder for detected heatwave events
├── coldspells/    # output folder for detected cold spell events
├── final/         # final output RMY EPW file and summary CSVs
```

## Explore the Interactive Map

![map](images/map.gif)

Explore the map [**here**](https://svante.mit.edu/~pgiani/buildings/).

## Sample Output

This timeline shows detected heatwaves and cold spells across years:

![event timeline](images/event_timeline.png)

## Try it on Google Colab

Run the full pipeline interactively on Google Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nadatarkhan/RMY/blob/main/examples/RMY_Generation_Colab.ipynb)

## Usage Guidance
You can run the full RMY pipeline via:
```python
from rmy import run_full_rmy_pipeline
```

Make sure your folder structure matches:
```
EPWs/base/ → contains the base TMY file (1 file only)
EPWs/epws/ → contains full set of AMY EPWs
final/     → RMY weather file + event summaries will be saved here
```

## Citations

Tarkhan, N., Crawley, D., Lawrie, L., & Reinhart, C.  
**Generation of representative meteorological years through anomaly-based detection of extreme events**.  
*Journal of Building Performance Simulation*, 2025.  
https://doi.org/10.1080/19401493.2025.2499687

Tarkhan, N. & Reinhart, C.  
**Representing Climate Extremes: An Event-driven Approach to Urban Building Performance Assessments**.  
*Comfort at the Extremes Conference*, Seville, Nov. 2024.  
https://drive.google.com/file/d/14Kj9-jcL_SQGUaTvbdAzLVPOJHHWHLz0/view?usp=sharing

Giani, P.  
**Origin and Limits of Invariant Warming Patterns in Climate Models**.  
https://arxiv.org/abs/2411.14183

## License
This repository is licensed under the MIT License.
