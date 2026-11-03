"""Season definitions by climate band, derived from the EPW latitude.

The paper defines seasons by meteorological convention (JJA summer, DJF
winter) "applied according to the location's hemisphere". This module reads
the latitude from the EPW LOCATION line instead of relying on a list of city
names, and adds an equatorial band, which the paper does not define:

- Northern (lat >= +band): summer JJA, winter DJF.
- Southern (lat <= -band): summer DJF, winter JJA.
- Equatorial (|lat| < band, default 10 degrees): there is no meaningful
  summer or winter, so every month is treated on its own. Cold-spell
  detection considers all 12 months, and monthly compensation targets each
  calendar month's base-TMY mean separately.
"""
from dataclasses import dataclass

EQUATORIAL_BAND_DEG = 10.0
ALL_MONTHS = tuple(range(1, 13))
JJA = (6, 7, 8)
DJF = (12, 1, 2)


@dataclass(frozen=True)
class SeasonPlan:
    band: str                 # 'northern', 'southern' or 'equatorial'
    latitude: float
    summer_months: tuple      # months used for summer compensation (empty if equatorial)
    winter_months: tuple      # months used for winter compensation (empty if equatorial)
    cold_months: tuple        # months in which the cold-spell 'gnn' filter applies

    @property
    def is_equatorial(self):
        return self.band == 'equatorial'

    def describe(self):
        if self.is_equatorial:
            return (f'Equatorial (lat {self.latitude:+.2f}): all 12 months, '
                    'each compensated against its own monthly mean')
        return (f'{self.band.capitalize()} Hemisphere (lat {self.latitude:+.2f}): '
                f'summer={list(self.summer_months)}, winter={list(self.winter_months)}')


def read_epw_latitude(epw_path):
    """Latitude in decimal degrees from the EPW LOCATION line (7th field)."""
    with open(epw_path, 'r') as f:
        first = f.readline()
    fields = [x.strip() for x in first.split(',')]
    if not fields or fields[0].upper() != 'LOCATION' or len(fields) < 7:
        raise ValueError(f'{epw_path}: first line is not an EPW LOCATION record')
    try:
        lat = float(fields[6])
    except ValueError:
        raise ValueError(f'{epw_path}: latitude field is not a number: {fields[6]!r}')
    if not -90 <= lat <= 90:
        raise ValueError(f'{epw_path}: latitude out of range: {lat}')
    return lat


def season_plan_for_latitude(latitude, equatorial_band=EQUATORIAL_BAND_DEG):
    if abs(latitude) < equatorial_band:
        return SeasonPlan('equatorial', latitude, (), (), ALL_MONTHS)
    if latitude > 0:
        return SeasonPlan('northern', latitude, JJA, DJF, DJF)
    return SeasonPlan('southern', latitude, DJF, JJA, JJA)


def season_plan_for_epw(epw_path, equatorial_band=EQUATORIAL_BAND_DEG):
    return season_plan_for_latitude(read_epw_latitude(epw_path), equatorial_band)
