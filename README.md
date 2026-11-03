
# Extreme-Aware Meteorological Years: Generating RMYs and FRMYs for Climate-Resilient Building Simulations

## Motivation

With the increasing *frequency*, *intensity*, and *duration* of extreme weather events worldwide, traditional simulation inputs no longer capture the conditions most critical to building resilience. TMY (Typical Meteorological Year) weather files represent averaged conditions and hence do not represent the full spectrum of extremes, which are critical to assessing thermal resilience, peak loads, overheating, and grid reliability under climate stress. 

This project adresses that gap by embedding realistic extreme events—both historical and future—into weather files that are fully compatible with building simulation tools like EnergyPlus, ClimateStudio, Rhino/Grasshopper, and more. All data, code, and workflows are open-source, enabling reproducibility and adaptation for global use.

## Key Components

- **RMYs (Representative Meteorological Years):** Weather files embedded with observed extreme events (heatwaves and cold spells), generated from historical AMYs (Actual Meteorological Years) data using anomaly detection.
- **FRMYs (Future Representative Meteorological Years):** Morphed weather files embedding future heatwaves and cold spells derived from climate emulator trajectories with annual resolution and embedded uncertainty.
- **Open-Source Anomaly Detection**: A modular, fully open-source detection framework that identifies climate extremes using an ensemble of methods—static thresholds, Graph Neural Networks (GNNs), and Extreme Value Theory (EVT). The codebase is public and reusable across projects aiming to detect and embed extremes in weather or environmental time-series data.
- **Event Integration in Accessible Weather files:** Temporal smoothing and seasonal averaging logic that inserts extremes while preserving monthly averages in weather files compatible with a wide range of simulation programs. 

## Extreme Events Explorer (EEE)
This timeline shows detected heatwaves and cold spells across years based on the detection algorithm in a sample city:

![Event Timeline](images/event_timeline.png)

---

## Methodology

This toolkit introduces a multi-method event detection and integration pipeline for constructing Representative Meteorological Year (RMY) and Future RMY (FRMY) weather files. These new formats restore historically observed or projected extremes into standard TMY files to enable realistic simulation for overheating, thermal resilience, and peak demand.

An Ensemble anomaly-detection method is used based on the following:

- **Static Thresholding:** Identifies extremes based on fixed temperature or percentile thresholds.
- **GNN-Based Anomaly Detection:** A graph autoencoder trained on all AMY days (nodes are days; edges link consecutive and similar days) flags days it cannot reconstruct well, relative to the time of year. Runs of at least three such days become events. See [docs/gnn.md](docs/gnn.md).
- **Extreme Value Theory (EVT):** Extracts statistically rare extremes using Peaks Over Threshold (POT) modeling.

Each method is used in a complementary ensemble to identify the most severe year and characteristic events.

**For FRMY generation**, future extremes are derived from annually-morphed climate emulator outputs that reflect global warming trajectories under different scenarios. These files embed projected heatwaves and cold spells with realistic variability, and are processed using the same anomaly detection and integration methods as RMYs.

![Emulator](images/Emulator.png)

## Workflow Summary

1. Detect peak heatwaves and cold spells across 15+ years of EPW files.
2. Match extreme events to base-year dates using overlap logic.
3. Replace those dates with extreme-event days from the most severe year, using smoothing.
4. Rebalance monthly averages by inserting non-extreme days to maintain realism.
5. Output:
   - RMY file with embedded extremes
   - Summary CSVs for heatwaves and cold spells


![RMY Workflow](images/Fig1.png) 


## Folder Structure

```
rmy/                    ← the Python package
├── heatwaves.py        ← heat-wave detection
├── coldspells.py       ← cold-spell detection
├── rmy_generation.py   ← RMY assembly (event insertion, compensation, smoothing)
├── gnn.py              ← graph neural network anomaly detector (paper Section 2.2.2)
├── seasons.py          ← hemisphere / equatorial season rules from EPW latitude
├── validation.py       ← EPW range and missing-value checks (`python -m rmy.audit`)
├── visualization.py    ← Extreme Event Explorer circle plot
├── utils.py            ← EPW reading and shared helpers
data/
├── base/               ← sample base TMY EPW (Boston, 1 file only)
├── EPWs/               ← sample AMY EPWs (Boston 2009-2023)
├── RMY/                ← published RMY EPWs
├── FRMY/               ← published FRMY EPWs
examples/               ← Google Colab notebook
tests/                  ← pytest checks
images/                 ← figures used in this README
```

---

## Explore the Interactive Map

![Map](images/map.gif)

Click below to explore an interactive dashboard where you can download RMY and FRMY weather files for cities worldwide:

👉 [Explore the Map here](https://svante.mit.edu/~pgiani/buildings/)

---

## Usage

Install required packages:

```bash
pip install -r requirements.txt
```

Then run the following from the repo root:

```python
from rmy import run_full_rmy_pipeline
from rmy.visualization import visualize_extreme_events

rmy_path = run_full_rmy_pipeline("data/EPWs", "data/base", "output")

visualize_extreme_events("output/hotspells/heatwave_events.csv",
                         "output/coldspells/coldspells_events.csv",
                         output_path="output/event_timeline.png")
```

Inputs:

```
data/EPWs/ → one AMY EPW per year; each file name must contain its year
data/base/ → exactly one base TMY EPW
```

Outputs, in the folder you pass as the third argument:

```
hotspells/   → heat-wave events and yearly stats (all years, base file, peak year)
coldspells/  → cold-spell events and yearly stats (all years, base file, peak year)
RMY_<base file name>.epw → the RMY weather file
```

Seasons come from the latitude in the base EPW's LOCATION line:

| Band | Latitude | Summer | Winter |
|---|---|---|---|
| Northern | ≥ 10° | Jun–Aug | Dec–Feb |
| Southern | ≤ −10° | Dec–Feb | Jun–Aug |
| Equatorial | between −10° and 10° | every month is compensated against its own base-TMY mean | cold spells are checked in all months |

The equatorial band is an extension to the published method, which defines seasons by hemisphere only.

Check EPW files for invalid values (netCDF fill values, out-of-range numbers) and missing data with `python -m rmy.audit FILE.epw ...`. The pipeline runs the same check on its inputs and treats invalid values as missing.

Run the tests with `python -m pytest tests/`.

## Try it on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nadatarkhan/RMY/blob/main/examples/RMY_Generation_Colab.ipynb)

---
## What's Next?

As climate conditions continue to shift, extreme events are evolving—not just in frequency, but in duration, intensity, and timing. Future extensions of this project will:

- Track how heatwaves and cold spells are shifting across decades
- Link these shifts to building performance risk and urban equity impacts
- Expand the tool to include dual temperature hazard assessments across cities

Below is a preview of how extreme events are changing over time:

![Events](images/Events.gif)

Stay tuned for the next phase: ***Extremes in Motion*** — a dynamic, multi-city comparison tool.


## Citations

Tarkhan, N., Crawley, D., Lawrie, L., & Reinhart, C.  
**Generation of representative meteorological years through anomaly-based detection of extreme events.**  
*Journal of Building Performance Simulation, 2025.*  
[https://doi.org/10.1080/19401493.2025.2499687](https://doi.org/10.1080/19401493.2025.2499687)

Giani, P., Fiore, A.M., Flierl, G., et al.␣␣  
**Origin and Limits of Invariant Warming Patterns in Climate Models.**  
*arXiv preprint, 2024.*  
[https://arxiv.org/abs/2411.14183](https://arxiv.org/abs/2411.14183)

Tarkhan, N., & Reinhart, C.  
**Representing Climate Extremes: An Event-driven Approach to Urban Building Performance Assessments.**  
*Comfort at the Extremes Conference, Seville, 2024.*  
[View PDF](https://drive.google.com/file/d/14Kj9-jcL_SQGUaTvbdAzLVPOJHHWHLz0/view?usp=sharing)

---

## License

This repository is released under the MIT License.
