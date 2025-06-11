# Representative Meteorological Year (RMY) Generator

Anomaly-driven generation of Representative Meteorological Year (RMY) weather files with embedded extreme events, including **heat waves** and **cold spells**.

## Overview

This repository provides a set of tools to extract extreme climate events and integrate them into typical meteorological year weather files. The method combines multiple anomaly detection techniques to identify impactful events and supports both heat and cold extremes.

## Features

- Static and percentile-based threshold detection
- GNN-style anomaly detection
- Extreme Value Theory (EVT) using Peaks Over Threshold (POT)
- Event statistics including frequency, duration, and intensity
- Compatible with EnergyPlus `.epw` files
- Outputs ready-to-use CSV summaries

## Structure

```
RMY_Github/
├── rmy/
│   ├── heatwaves.py
│   ├── coldspells.py
│   ├── utils.py
├── examples/
│   └── example_usage.ipynb
├── data/
├── README.md
├── LICENSE
└── requirements.txt
```

## Usage

See `examples/example_usage.ipynb` for how to apply the detection methods to `.epw` files.

## Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.