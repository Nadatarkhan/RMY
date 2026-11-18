# Quality assurance of the published RMYs and FRMYs

This report covers two groups of cities:
- **New**: 56 world capitals and 20 US state capitals.
- **Published**: the 40 cities published earlier, checked again with the same QA for comparison.

Each city gets about 85 checks across its AMYs, TMY, RMY and FRMY. Each check is graded PASS, WARN or FAIL, and the result records the measured value. The per-city results are in [`QA_summary_capitals.csv`](QA_summary_capitals.csv), and every individual check is in [`QA_all_checks_capitals.csv`](QA_all_checks_capitals.csv).

| | Cities | PASS | WARN | FAIL |
|---|---|---|---|---|
| New: 56 world capitals and 20 US state capitals | 76 | 6,081 | 562 | 14 |
| Published: the original 40 cities, checked again | 40 | 3,198 | 320 | 5 |

## What is checked

| Group | Checks |
|---|---|
| **S** station and AMYs | **Record length:** years of record.<br>**Location:** distance and elevation from the city centre.<br>**Data completeness:** observed share of hours; longest raw gap; ERA5-infilled share.<br>**AMY builder output:** QC grades from the AMY builder; solar source by year; solar hours borrowed from an adjacent year; dew points capped; wind gaps.<br>**Instrument:** whole-degree reporting (1 °C-resolution METAR stations). |
| **P** hourly physics, on every file | **File integrity:** row and field counts; missing-value codes.<br>**Temperature and humidity:** dew point above dry bulb; RH consistent with dry bulb and dew point; temperature range; jumps over 10 °C/h; isolated one-hour spikes; stuck values.<br>**Pressure:** pressure against station elevation.<br>**Solar geometry:** GHI at night; GHI above extraterrestrial; closure of GHI = DHI + DNI·cos z.<br>**Timing:** GHI centroid against solar noon (a timestamp check); hour of daily Tmax.<br>**Wind and sky:** wind ranges and calms; sky-cover ranges. |
| **C** independent cross-check | **Against PVGIS TMY at the city centre (ERA5 / SARAH):**<br>• mean temperature, adjusted for lapse rate;<br>• seasonal cycle;<br>• annual GHI;<br>• RH and wind;<br>• lag of the diurnal temperature cycle (a time-zone check).<br>**Internal:** TMY and RMY annual means against the station's own 15-year mean. |
| **R** TMY and RMY | **TMY sources:** TMY source months within the AMY period.<br>**Monthly balance:** RMY monthly means against the TMY, outside and inside event months.<br>**Inserted events:** heat waves and cold spells inserted; tails widened.<br>**Smoothness:** largest step at splice points against the largest step in any AMY. |
| **F** 2050 morph | **Header:** the header names CanESM5, SSP2-4.5 and a future period centred on 2050; location line unchanged.<br>**Warming applied:** global warming between 1.1 and 1.7 K; monthly ΔT and ΔRH within 0.06 °C and 0.6 %RH of the pattern-scaling coefficients; change in daily range; pressure change.<br>**Fields left alone:** solar, cloud and rain unchanged; hourly sequence preserved; design conditions blank and ground temperatures recalculated. |

The PVGIS comparisons are for review only. A PVGIS TMY takes one year per month from ERA5 grid cells of about 30 km, so differences of 2-5 °C appear wherever the cell mixes land and sea or mountain and valley. Juneau, Salt Lake City, Oslo and Taipei are examples. Those differences are not errors in the station files.

## Method changes made during this QA (now in `amy_sourcing`)

1. **Sky cover.** In 8 published cities and in most new cities outside the Americas, some hours had no total or opaque sky cover (EPW code 99). The cause was that long cloud gaps were filled only from NSRDB cloud types, which PVGIS-based years do not have.
   - Such gaps are now filled from the year's own recorded cover: by day, for the same clearness index; by night, for the same month and hour.
   - The 16 published files were filled the same way; only the sky-cover fields and their flags changed.
   - None of the 70 new cities has a missing sky-cover hour.
2. **Corrupt temperature reports** are removed before gap filling. These are dry bulbs below the same hour's dew point, and short dips that come back while the dew point moves too (e.g. Paris-Orly 2020: 2 °C readings in 15-24 °C afternoons).
   - A screen of 1,159 station-years flagged 15 hours, all of them corrupt. Fronts, storms and dry downslope events are kept.

## Findings to keep in mind when using the files

- **Cold-climate RMYs are colder than the station's 15-year mean.** The RMY inserts the peak year's cold spells, and in continental climates these can fill most of a winter month. In these cities the annual mean of the RMY, and so of the FRMY, is 0.6-1.5 °C below the station's 15-year mean:
  - Berlin −1.5
  - Oslo −1.0
  - Warsaw −0.9
  - Albany −0.8
  - Boise −0.7
  - Stockholm −0.7
  - Ulaanbaatar −0.7
  - Tallahassee −0.6

  The earlier 40 behave the same way: Minneapolis −0.9, Anchorage −0.9, Miami −0.9 (a long December cold spell), Helsinki −0.8, Fairbanks −0.8 and New York −0.8. This is by design: the files are for event-driven risk, peak loads and passive survivability. For annual energy use the TMY is the better baseline.
- **Station representativeness.**
  - **Distant stations:** Kampala uses Entebbe airport, 37 km away on Lake Victoria, and Santo Domingo uses Las Américas airport, 28 km away on the coast.
  - **Elevation differences:**
    - Kinshasa is 117 m below the city centre.
    - Lima (Callao) is 105 m below the city centre.
    - Santa Fe is 217 m below the city centre. It is the only long-record station there.
  - **Oslo** uses the Blindern city station, not Gardermoen airport, which is 37 km inland and colder.
- **Short or sparse records.**
  - Kinshasa and Kampala have 10 years, Accra and Maputo 11, and Baton Rouge 14.
  - Kinshasa's worst year has 64% of its hours observed.
  - Some stations report temperature in whole degrees: for example Berlin-Schönefeld, Rome-Ciampino and Havana.
- **Lima's solar** (NSRDB) is 18% below PVGIS-SARAH. The coastal stratus deck (garúa) is where satellite products disagree most.
- **Winter spikes at Astana and Ulaanbaatar.** These are single warm hours with falling dew point in calm, very cold air, consistent with mixing through strong surface inversions. They are kept as observed weather.
- **Not included.**
  - Abuja, Addis Ababa, Luanda and Windhoek have fewer than 10 complete years.
  - Tbilisi's station reports every 3 hours in all years.
  - Islamabad's station became sparse after 2019.
  - Naypyidaw (Myanmar) has only 3 usable years; Yangon represents Myanmar.

## Half-hour and 45-minute time zones

The AMY code now supports half-hour and 45-minute time zones. New Delhi and Colombo (+5:30), Tehran (+3:30),
Kabul (+4:30), Kathmandu (+5:45) and Yangon (+6:30) were built with it. Two timing checks pass in all six,
showing the hours are in local standard time:
- **Peak sunshine against solar noon:** within 0.07 h.
- **Diurnal temperature cycle against ERA5:** within 0 to 2 h, the same range as whole-hour cities (Paris +1 h).

Kabul's record ends in November 2021 (years 2006-2020). Kathmandu has 10 usable years, because 2013 and 2014 are
missing. Kathmandu's and Yangon's worst years have 52% and 64% of their hours observed.
