# Validation of the AMY method

Scripts that reproduce the validation in the paper *Generating actual meteorological year weather
files: an open, quality-controlled method for building performance simulation* (Tarkhan and
Reinhart), and the results they produced (`results/`).

All inputs are public, and **no API key is needed**: NSRDB data are read from NREL's archive on
AWS. Each script takes one argument, a working folder. Downloads are cached in `<work>/cache` and
results are written to `<work>/results`, so a second run reuses the data.

```bash
pip install -r ../requirements.txt matplotlib
python 05_holdout_interpolation.py amy_validation_work
```

## What reproduces what

| Script | Paper | Output in `results/` |
|---|---|---|
| `01_nsrdb_cache.py` | Inputs for 4.2–4.4 | NSRDB 2022 for the ten U.S. stations, in the method's cache |
| `02_build_amys_2022.py` | Inputs for 4.2 | 2022 AMYs of the ten U.S. stations (`amy_2022_reports.csv`) |
| `03_radiation_uscrn.py` | 4.2, Fig. 2, Table 4 | `radiation_uscrn.csv`, `radiation_charlottesville_2022_hourly.csv` |
| `04_amy_timing.py` | 4.2 (end-to-end check) | `amy_timing.csv` |
| `05_holdout_interpolation.py` | 4.3, Table 5, Fig. 3a–c | `holdout_interpolation.csv` |
| `06_holdout_sky_cover.py` | 4.3, Fig. 3d | `holdout_sky_cover.csv` |
| `07_holdout_era5_temperature.py` | 4.3, Table 6 (temperature, dew point) | `holdout_era5_temperature.csv` |
| `08_holdout_era5_pressure_wind.py` | 4.3, Table 6 (pressure, wind) | `holdout_era5_pressure_wind.csv` |
| `09_holdout_extremes.py` | 4.3 (Extremes) | `holdout_extremes.csv` |
| `10_heat_dome.py` | 4.3 (Preserving an observed extreme) | `heat_dome_seattle_2021.csv` |
| `11_longwave_surfrad.py` | 4.4, Table 7 | `longwave_surfrad.csv` |
| `12_longwave_era5.py` | 4.4, Table 8 | `longwave_era5.csv` |
| `13_year_selection.py` | 4.1, Table 3 (input to 14) | per-station years in `<work>/years` |
| `14_design_conditions.py` | 4.1, 4.5, Tables 3 and 9, Fig. 4 | `years_selected.csv`, `era5_use_by_year.csv`, `design_vs_ashrae.csv`, `design_summary.csv` |
| `15_clear_sky_reports.py` | 4.6, Table 10 | `clear_sky_reports.csv` |
| `16_clear_sky_era5_cloud.py` | 4.6 (sensitivity) | `clear_sky_era5_cloud.csv` |
| `figures.py` | Figs. 2–4 | PNG files (`python figures.py --bundled` draws them from `results/`) |

Run `01` before `02`, `06` and `12`; `02` before `04`; and `13` before `14`. The others are
independent.

## Inputs

* `stations.json`: the twenty validation stations (NOAA number, city, standard UTC offset, region).
* `uscrn_sites.json`: the USCRN station used for each U.S. city in Section 4.2.
* `ashrae_design_lines.json`: the ASHRAE 2025 design-condition lines from the TMYx files
  (climate.onebuilding.org) of the stations whose TMYx is not in `data/cities`.
* Downloaded by the scripts: NOAA Global Hourly (AWS), the NSRDB (AWS), ERA5 (NSF NCAR Research
  Data Archive on AWS), USCRN hourly data (NOAA NCEI), SURFRAD (NOAA Global Monitoring
  Laboratory) and the NOAA station catalogue.

## Notes

* The method is imported from this repository; the paper describes commit `530e5d7`.
  `15_clear_sky_reports.py` loads the earlier version (commit `7b6d07e`) from the git history to
  show the effect of the clear-sky rule, so it needs a git checkout, not a downloaded archive.
* The scripts were collected and tidied for release after the analyses were run. Scripts
  `05`, `06`, `10`, `13`, `14`, `15` and `16` were re-run in this form and reproduce the bundled
  results exactly, and `figures.py --bundled` redraws Figs. 2–4; the others perform the same
  calculations but were not re-run after tidying.
* `12_longwave_era5.py` reads whole monthly ERA5 flux fields and takes several hours; the other
  scripts take minutes once the data are cached.
* Pressure errors are in hPa, temperatures in °C, wind speed in m/s, directions in degrees,
  sky cover in tenths and radiation in W/m².
