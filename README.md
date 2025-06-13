# RMY: Representative Meteorological Year Generator

Anomaly-driven generation of Representative Meteorological Year (RMY) weather files with embedded extreme events, including **heatwaves** and **cold spells**. This method preserves core TMY characteristics while adding realistic severe climate conditions to support robust design evaluations.

## Quick Start

Install required packages:
```bash
pip install -r requirements.txt
```

Then run the following from the repo root:
```bash
python rmy_generation.py
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
├── EPWs/
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
python rmy_generation.py
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