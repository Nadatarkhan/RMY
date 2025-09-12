
# Extreme-Aware Meteorological Years (RMYs and FRMYs)

## Motivation

TMY weather files represent averaged conditions and hence do not represent the full spectrum of extremes. This limits their utility for resilience assessment, overheating simulations, and passive survivability design. This project introduces new weather files that embed observed or projected extremes—heatwaves, cold spells, and compound events—into EPW files that are fully compatible with simulation software like EnergyPlus and ClimateStudio.

## Methodology

RMYs embed observed extremes through a multi-method detection pipeline using thresholds, EVT, and GNNs. FRMYs incorporate climate projections using emulators and apply the same embedding methods. Events are integrated using a seasonal interpolation approach that preserves monthly means while restoring extreme events.

## Key Components

- **RMYs:** Representative past weather files that include historically observed extreme heat and cold events.
- **FRMYs:** Future weather files that embed extremes using emulator-projected trajectories under various climate scenarios.
- **Extreme Events Explorer (EEE):** Interactive visualization tool to inspect event frequency, intensity, duration, heat index, and wind chill over time.
- **Interactive Map:** Downloadable global library of present and future weather files for any location.
- **Codebase:** Fully reproducible scripts to generate and integrate events into EPWs.

## Explore the Interactive Map

![Interactive Map Animation](images/map.gif)

[Explore the Map here](https://svante.mit.edu/~pgiani/buildings/)

Click the link above to explore an interactive dashboard that allows you to navigate to any city and download its corresponding RMY or FRMY weather file.

## Repository Structure

```
├── data/
│   ├── RMY/
│   └── FRMY/
├── notebooks/
├── images/
└── README.md
```


## Citation

Tarkhan, N., Crawley, D., Lawrie, L., & Reinhart, C.  
*Generation of representative meteorological years through anomaly-based detection of extreme events*.  
Journal of Building Performance Simulation, 2025.  
https://doi.org/10.1080/19401493.2025.2499687

Tarkhan, N. and Reinhart, C.  
*Representing Climate Extremes: An Event-driven Approach to Urban Building Performance Assessments*.  
Comfort at the Extremes Conference, Seville, Nov. 2024.  
https://drive.google.com/file/d/14Kj9-jcL_SQGUaTvbdAzLVPOJHHWHLz0/view?usp=sharing

Giani, P., et al.  
*Origin and Limits of Invariant Warming Patterns in Climate Models*.  
https://arxiv.org/abs/2411.14183

## License

MIT License
