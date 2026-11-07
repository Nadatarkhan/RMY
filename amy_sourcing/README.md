# AMY sourcing

`generate_amys.py` builds actual meteorological year (AMY) EPW files for one NOAA weather station, ready for the TMY step (`tmy/`) and RMY generation (`rmy/`).

Give it a station number and it builds one EPW per year for the **15 most recent complete years**. The last complete year is read from the NOAA station record, so the period moves forward as new years are published. Years that fail, or have temperature, dew point or pressure gaps too long to fill, are replaced by earlier years.

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

- **Meteorology**: NOAA Global Hourly (ISD) station observations, from NOAA's public archive on AWS S3. Short gaps in dry bulb, dew point and pressure are interpolated; a year with a gap longer than 24 h (72 h for pressure) is flagged and not used. Wind speed and direction gaps are interpolated (direction around the compass).
- **Solar radiation**: NSRDB GOES v4 (satellite, about 4 km) in the Americas; PVGIS-SARAH3, then PVGIS-ERA5, elsewhere. Hourly values are aligned to EPW hours in local standard time.
- **Illuminance and zenith luminance**: Perez et al. (1990) luminous efficacy model.
- **Sky cover**: gaps filled by linear interpolation (up to 5 h) or from the NSRDB cloud type mapped to the station's own observed cover, with EPW data-source flags B, C or E.
- **Longwave (horizontal infrared)**: EnergyPlus Clark-Allen/Walton model from dry bulb, dew point and the filled opaque sky cover.
- **City names**: GeoNames city list ([geonames.org](https://www.geonames.org/), CC BY 4.0), through the offline `geonamescache` package.

The functions are those of the AMY Generator V10.1 notebook (Nada Tarkhan); `generate_amys()` adds the station-to-years entry point.
