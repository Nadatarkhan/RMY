# Extreme-Aware Meteorological Years (RMYs and FRMYs)

This repository provides tools and open datasets to generate and explore Representative Meteorological Years (RMYs) and Future RMYs (FRMYs) with embedded **extreme climate events**, such as heatwaves and cold spells. 

Typical Meteorological Year (TMY) weather files represent averaged conditions and hence do not represent the full spectrum of extremes, underestimating risks in overheating, grid failure, HVAC capacity, and passive survivability.

We address this limitation by introducing historically informed and climate-emulator-based future weather files that better reflect the intensifying risks cities face. All files are fully compatible with major simulation tools like **EnergyPlus**, **ClimateStudio**, **Rhino/Grasshopper**, and more.

---

## Methodology

RMYs embed **historically observed extremes** using a multi-method anomaly detection framework combining:

- **Static Thresholding**
- **Extreme Value Theory (EVT)**
- **Graph Neural Networks (GNNs)**

Events are matched to base TMY dates and inserted through a smoothing and seasonal averaging process that **restores peaks** while maintaining monthly averages.

FRMYs embed **future extreme events** using morphed climate scenarios from **emulators** developed by [Paolo Giani](https://eapsweb.mit.edu/people/pgiani), Postdoctoral Associate in EAPS at MIT, part of the **BC3 MIT Climate Grand Challenge**. These files reflect global warming trajectories under different scenarios and offer **annual-resolution** weather files with uncertainty.

---

## Key Components

- 🔍 **Anomaly Detection Algorithms** – detects extreme events using statistical and ML methods
- 🧠 **GNN-based Event Classifier** – learns spatio-temporal anomalies in EPW time series
- 📊 **Event Metrics & Visualization** – duration, magnitude, heat index, and wind chill
- 🌍 **Climate Emulator Integration** – generates annual future weather scenarios (FRMYs)
- 🧾 **EPW-Compatible Outputs** – usable in any standard simulation engine
- 📁 **Open-Source Pipeline** – reproducible across cities

---

## Explore the Interactive Map

View a global dashboard of RMY and FRMY files by location.

![Map GIF](images/map.gif)

👉 [**Explore the Map here**](https://svante.mit.edu/~pgiani/buildings/)

---

## RMY: Representative Meteorological Year Generator

Anomaly-driven generation of Representative Meteorological Year (RMY) weather files with embedded extreme events, including heatwaves and cold spells. This method preserves core TMY characteristics while adding realistic severe climate conditions to support robust resilience evaluations, and produces standard EPW outputs that can be used across all major building simulation platforms.

---

## 📁 Folder Structure

```
/data/RMY/        → Extreme-aware historical files
/data/FRMY/       → Emulator-based future weather files
/content/base/    → Base TMY EPW file (1 file)
/content/EPWs/    → AMY EPWs for detection
/content/hotspells/ → Output folder for heatwaves
/content/coldspells/ → Output folder for cold spells
/content/final/   → Final output RMY + stats
```

---

## Methods Used

**Event detection pipeline includes:**

- **Static Thresholding**: Fixed temperature and percentile rules
- **GNN-Based Anomaly Detection**: Graph-based temporal patterns
- **EVT (Extreme Value Theory)**: Peaks Over Threshold (POT)

These are combined in an **ensemble** to find the most representative extreme year and events.

---

## Workflow Summary

1. Detect peak heatwaves/cold spells from 15+ years of EPWs  
2. Match events to TMY dates  
3. Replace matched dates with peak events, smooth transitions  
4. Rebalance seasonal/monthly averages for realism  
5. Export:  
   - RMY EPW  
   - CSVs of event summaries

---

## RMY Workflow Diagram

![Timeline](images/event_timeline.png)

---

## 🔧 Quick Start

Install required packages:

```bash
pip install -r requirements.txt
```

Then run:

```python
from rmy import run_full_rmy_pipeline
```

---

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
│   ├── RMY/
│   └── FRMY/
├── final/
├── images/
│   └── map.gif
│   └── event_timeline.png
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 📈 Usage Guidance

```python
from rmy import run_full_rmy_pipeline
```

### Folder Setup:
- `data/base/`: base TMY
- `data/epws/`: all AMY EPWs
- `data/final/`: output RMYs and summaries

---

## 🚀 Try it on Google Colab

Click below to run the full pipeline interactively:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nadatarkhan/RMY/blob/main/examples/RMY_Generation_Colab.ipynb)

---

## Citations

Tarkhan, N., Crawley, D., Lawrie, L., & Reinhart, C.  
**Generation of representative meteorological years through anomaly-based detection of extreme events.**  
*Journal of Building Performance Simulation*, 2025.  
https://doi.org/10.1080/19401493.2025.2499687

Giani, P., et al.  
**Origin and Limits of Invariant Warming Patterns in Climate Models.**  
arXiv preprint, 2024.  
https://arxiv.org/abs/2411.14183

Tarkhan, N. & Reinhart, C.  
**Representing Climate Extremes: An Event-driven Approach to Urban Building Performance Assessments.**  
Comfort at the Extremes Conference, Seville, 2024.  
[View Paper](https://drive.google.com/file/d/14Kj9-jcL_SQGUaTvbdAzLVPOJHHWHLz0/view?usp=sharing)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
