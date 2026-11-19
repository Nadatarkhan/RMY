# AMY sourcing

`generate_amys.py` builds actual meteorological year (AMY) EPW files for one NOAA weather station, ready for the TMY step (`tmy/`) and RMY generation (`rmy/`).

Give it a station number and it builds one EPW per year for the **15 most recent complete years**. The last complete year is read from the NOAA station record, so the period moves forward as new years are published. If the station has stopped reporting regularly while its record stays open (e.g. Lahore after 2018), the period ends in the most recent year with temperature reports in at least half of its hours. Years that fail, or have temperature, dew point or pressure gaps too long to fill, are replaced by earlier years.

## Run it

The easiest way is the end-to-end Colab notebook, `examples/RMY_End_to_End_Colab.ipynb` (AMYs → TMY → RMY, with plots). From Python:

```python
from amy_sourcing.generate_amys import generate_amys   # or load the file with importlib

run = generate_amys("72509014739", "output", nsrdb_api_key="...", nsrdb_email="...")
print(run["years"], run["amy_dir"])
```

From the command line (keys from environment variables):

```bash
NSRDB_API_KEY=... NSRDB_EMAIL=... python amy_sourcing/generate_amys.py --station 72509014739 --out output
```

Requirements: `pip install -r amy_sourcing/requirements.txt`.

## API key

Stations in the Americas use NSRDB solar radiation, which needs a free NREL API key and the email it is registered to: see <https://developer.nlr.gov/docs/climate/guide/>. Each user needs their own key; never put a key in code or in a shared notebook. The code removes keys and emails from every report and message it saves. Stations outside the Americas use PVGIS and need no key.

## Outputs

```
<out>/<City>/AMY/          <City>_<station>_<year>_AMY.epw, one per year
<out>/<City>/AMY_reports/  AMY_manifest.csv, per-year reports, hourly gap-filling provenance,
                           and years set aside in flagged/
<out>/_cache/              downloads, reused on later runs
```

`<City>` is found automatically from the station's coordinates: the largest town or city within 30 km in the station's own country (otherwise the nearest within 75 km), followed by the US state or the country code, for example `Las_Vegas_NV`, `Athens_OH` or `Dubai_AE`. To choose your own, pass `city="..."` (`--city` on the command line).

## Data and methods

- **Meteorology**: NOAA Global Hourly (ISD) station observations, from NOAA's public archive on AWS S3. Gaps of up to 12 h in dry bulb, dew point and pressure are interpolated, and wind speed and direction gaps are interpolated (direction around the compass). Gaps of 13 h or more (typically archive outages) are filled from the ERA5 reanalysis (NCAR copy on AWS, no key needed), corrected by the station's mean ERA5 bias for that month and hour of day; each variable is filled this way only if its long gaps cover no more than 5% of the year, the values are marked in the provenance file, and the ERA5 points are cached. If ERA5 cannot be reached the gaps are left as they are. A year that still has a gap longer than 24 h (72 h for pressure) is flagged and not used. A year with fewer than half its temperature hours observed (for example a station reporting only every 3 h) is used but graded `REVIEW_SPARSE`, since most of its hours are interpolated. The observed share is written in the EPW header (`COMMENTS 2`, "observed dry-bulb share"), and the TMY and RMY steps read it: the TMY takes its months from years with hourly records when at least three are available, and the RMY leaves sparse years out of heat-wave and cold-spell detection when at least five hourly years remain. In a test on Karachi 2018 thinned to 3-hourly reports, daily minima came out 1.3 °C too warm and heat-wave days fell from 33 to 28. Calm hours (0 m/s) always get wind direction 0°, also when the report gave a direction.
- **Time zone**: EPW hours are in local standard time (no daylight saving), with the offset found from the station coordinates (`timezonefinder`). Each NOAA report is assigned to the local clock hour in which it was made.
  - Half-hour and 45-minute offsets work like whole hours: India and Sri Lanka +5:30, Iran +3:30, Afghanistan +4:30, Nepal +5:45, Myanmar +6:30.
  - Hourly ERA5 values used to fill long gaps are interpolated to the half hour in those zones.
- **Solar radiation**: NSRDB GOES v4 (satellite, about 4 km) in the Americas; PVGIS-SARAH3, then PVGIS-ERA5, elsewhere. Hourly values are aligned to EPW hours in local standard time. PVGIS gives one snapshot per hour (SARAH: about 10 minutes past the hour, UTC); each EPW hour gets the value at its midpoint, by interpolating the clearness index between snapshots. PVGIS-ERA5 values are hourly means stamped at the half hour, already the midpoints, so they are used without a time shift.
- **Illuminance and zenith luminance**: Perez et al. (1990) luminous efficacy model.
- **Sky cover**: METARs reporting CAVOK, NSC, SKC, NCD or CLR without cloud groups count as clear sky (0), not missing. Remaining gaps are filled by linear interpolation (up to 12 h) or from an estimate, which is blended to the recorded values for 13-36 h and used directly for longer gaps, with EPW data-source flags B, C or E. The estimate is the NSRDB cloud type mapped to the station's own observed cover. Where there is no cloud type (PVGIS radiation, outside the Americas), it is the year's own recorded cover for the same clearness index by day, and for the same month and hour by night. Where the station reported no cloud at all (some automatic stations, e.g. Bern-Zollikofen), the estimate is ERA5 cloud cover at the station: total cloud for total cover, and low and medium cloud combined (random overlap) for opaque cover, shifted by its mean difference from any cover the station did record that year. No hour is left missing.
- **Corrupt temperature reports**: before gap filling, two kinds of report are set to missing and then filled like any other gap.
  - A dry bulb more than 1 °C below the dew point of the same hour. The value that jumped from its last valid reading is removed. These typically come from a garbled METAR temperature in an hour whose SYNOP has only the dew point.
  - Dry-bulb dips of up to 6 h more than 8 °C below both the hour before and the hour after, when the dew point also leaves its level. Single-hour rises of the same size are treated the same way.

  Fronts, which persist, and rain-cooled storm air, where the dew point holds, are kept. Dew points are not filtered: dry downslope and desert mixing events lower them by 10-20 °C for hours. The counts are in the AMY report (`DB_spikes_removed_hours`, `DB_below_DP_removed_hours`).
- **Precipitation**: NOAA accumulation reports of 1 to 6 hours, the shortest available each hour, with the period in the EPW precipitation quantity field. Longer (12 and 24 hour) totals are not used: they overlap the shorter ones and carry decoding errors.
- **Longwave (horizontal infrared)**: EnergyPlus Clark-Allen/Walton model from dry bulb, dew point and the filled opaque sky cover. (The Dilley-O'Brien clear-sky model is available as an option.)
- **City names**: GeoNames city list ([geonames.org](https://www.geonames.org/), CC BY 4.0), through the offline `geonamescache` package.

The functions are those of the AMY Generator V10.1 notebook (Nada Tarkhan); `generate_amys()` adds the station-to-years entry point.
