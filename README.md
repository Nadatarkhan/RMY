# RMY: Representative Meteorological Years with Embedded Extremes

This repository provides tools and data for generating **Representative Meteorological Years (RMYs)** that integrate realistic climate extremes (heatwaves and cold spells) into typical weather files. These are designed for use in building and urban simulation workflows, and can be applied to both present-day and future climates (FRMYs).

---

## 📌 Motivation

Traditional TMY weather files fail to represent the full spectrum of extreme climate events that affect building performance, energy demand, and indoor comfort.

This repository introduces **RMYs** and **FRMYs** — weather files that embed extreme events using rigorous detection and replacement strategies — providing a more robust foundation for resilience and performance-based design.

---

## 🧩 Key Components

### 🔹 1. Event Detection
Detect extreme heatwaves and cold spells using:
- Temperature thresholds and percentile exceedance
- Ensemble approaches (hybrid of metrics)
- Event statistics extraction (duration, intensity, persistence)

### 🔹 2. RMY Construction
Embed detected events into base TMY files while preserving monthly/seasonal statistical structure.

### 🔹 3. FRMY Generation
Use future emulator-based weather files (e.g., FAMYs) to extract and integrate realistic future events under low/high emission scenarios (SSP245 / SSP585).

### 🔹 4. Visualization
Plot timelines, event characteristics, and RMY vs. TMY comparisons.

---

## 🧪 Sample Output: Extreme Events Explorer (EEE)

Visual timelines and statistics of embedded events — including seasonality, duration, and heat index — allow exploration and validation of extreme-aware files.

![event_timeline](images/event_timeline.png)

---

## 🌍 Explore the Interactive Map

<p align="center">
  <video src="images/map.mp4" controls width="100%"></video>
</p>

Click the link below to explore an interactive dashboard that allows you to navigate to any city and download its corresponding RMY or FRMY weather file:

🔗 https://svante.mit.edu/~pgiani/buildings/

---

## 📂 Repository Structure

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
│   ├── epws/
│   ├── RMYs/
│   └── FRMYs/
├── final/
├── images/
│   ├── event_timeline.png
│   └── map.mp4
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 📚 Citations

If you use this method or files, please cite:

Tarkhan, N., Crawley, D., Lawrie, L., & Reinhart, C.  
*Generation of representative meteorological years through anomaly-based detection of extreme events.*  
*Journal of Building Performance Simulation*, 2025.  
https://doi.org/10.1080/19401493.2025.2499687

Tarkhan, N., & Reinhart, C.  
*Representing Climate Extremes: An Event-driven Approach to Urban Building Performance Assessments.*  
Comfort at the Extremes Conference, Seville, Nov. 2024.  
https://drive.google.com/file/d/14Kj9-jcL_SQGUaTvbdAzLVPOJHHWHLz0/view?usp=sharing

Giani, P., Mbengue, C., & Gentine, P.  
*Origin and Limits of Invariant Warming Patterns in Climate Models.*  
arXiv preprint, 2024.  
https://arxiv.org/abs/2411.14183

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.
