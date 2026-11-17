# Representative Meteorological Years (RMYs): 40 cities

20 US cities, one per US climate zone (ASHRAE 169, 1A to 8), and 20 international cities. Each is the city's ISO 15927-4 TMY with the heat
waves and cold spells of its peak years inserted (Tarkhan et al. 2025), built end to end from a NOAA
station number with this repository's code:

1. **AMYs**: the 15 most recent complete years of NOAA Global Hourly observations, with NSRDB (GOES
   v4) solar radiation in the US; Anchorage and Fairbanks, outside NSRDB's coverage, use PVGIS-ERA5.
   International cities use PVGIS-SARAH3 or PVGIS-ERA5 (NSRDB for Mexico City).
2. **TMY**: ISO 15927-4 from those AMYs (`tmy/`).
3. **RMY**: heat waves and cold spells detected with thresholds, GNN and EVT, the peak years' events
   inserted, and monthly means rebalanced within each month and, where a month is full of events,
   in the month before and after (at most 0.5 °C each; `rmy/`).

Months filled by long cold spells (e.g. Helena February, Boston February, Miami December) stay colder
than the TMY: their remaining difference is the inserted extreme itself. Each RMY's `COMMENTS 2` line
records the AMY years and the peak heat and cold years.

## United States

| City | Zone | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|
| Miami FL | 1A | 722020-12839 Miami International Airport | 2010-2024 | NSRDB | `Miami_FL_RMY_2010-2024.epw` |
| Houston TX | 2A | 722430-12960 G Bush Intercontinental Ap/Houston Ap | 2010-2024 | NSRDB | `Houston_TX_RMY_2010-2024.epw` |
| Phoenix AZ | 2B | 722780-23183 Phoenix Sky Harbor Intl Airport | 2010-2024 | NSRDB | `Phoenix_AZ_RMY_2010-2024.epw` |
| Atlanta GA | 3A | 722190-13874 Hartsfield-Jackson Atlanta Intl Ap | 2010-2024 | NSRDB | `Atlanta_GA_RMY_2010-2024.epw` |
| Las Vegas NV | 3B | 723860-23169 Mccarran International Airport | 2010-2024 | NSRDB | `Las_Vegas_NV_RMY_2010-2024.epw` |
| Los Angeles CA | 3B | 722950-23174 Los Angeles International Airport | 2010-2024 | NSRDB | `Los_Angeles_CA_RMY_2010-2024.epw` |
| San Francisco CA | 3C | 724940-23234 San Francisco International Airport | 2010-2024 | NSRDB | `San_Francisco_CA_RMY_2010-2024.epw` |
| New York NY | 4A | 744860-94789 John F Kennedy International Airport | 2010-2024 | NSRDB | `New_York_NY_RMY_2010-2024.epw` |
| Washington Dulles VA | 4A | 724030-93738 Washington Dulles International Ap | 2010-2024 | NSRDB | `Washington_Dulles_VA_RMY_2010-2024.epw` |
| Albuquerque NM | 4B | 723650-23050 Albuquerque Intl Sunport Airport | 2010-2024 | NSRDB | `Albuquerque_NM_RMY_2010-2024.epw` |
| Portland OR | 4C | 726980-24229 Portland International Airport | 2010-2024 | NSRDB | `Portland_OR_RMY_2010-2024.epw` |
| Seattle WA | 4C | 727930-24233 Seattle-Tacoma International Airport | 2010-2024 | NSRDB | `Seattle_WA_RMY_2010-2024.epw` |
| Boston MA | 5A | 725090-14739 Gen E L Logan International Airport | 2010-2024 | NSRDB | `Boston_MA_RMY_2010-2024.epw` |
| Chicago IL | 5A | 725300-94846 Chicago O'Hare International Airport | 2010-2024 | NSRDB | `Chicago_IL_RMY_2010-2024.epw` |
| Denver CO | 5B | 725650-03017 Denver International Airport | 2009-2024 | NSRDB | `Denver_CO_RMY_2009-2024.epw` |
| Minneapolis MN | 6A | 726580-14922 Minneapolis-St Paul International Ap | 2010-2024 | NSRDB | `Minneapolis_MN_RMY_2010-2024.epw` |
| Helena MT | 6B | 727720-24144 Helena Regional Airport | 2009-2024 | NSRDB | `Helena_MT_RMY_2009-2024.epw` |
| Anchorage AK | 7 | 702730-26451 Ted Stevens Anchorage Intl | 2008-2022 | PVGIS-ERA5 | `Anchorage_AK_RMY_2008-2022.epw` |
| Duluth MN | 7 | 727450-14913 Duluth International Airport | 2010-2024 | NSRDB | `Duluth_MN_RMY_2010-2024.epw` |
| Fairbanks AK | 8 | 702610-26411 Fairbanks International | 2007-2022 | PVGIS-ERA5 | `Fairbanks_AK_RMY_2007-2022.epw` |

## International

Lagos has 11 usable years and Dhaka 14; the others have 15. Mumbai and New Delhi are not included: the AMY
code does not yet support India's UTC+5:30 time zone, so Karachi and Dhaka represent South Asia. Climate
classes are the approximate Köppen-Geiger class of each city.

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Singapore | Singapore | Af | 486980-99999 Singapore Changi Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Singapore_SG_RMY_2009-2023.epw` |
| Lagos | Nigeria | Aw | 652010-99999 Murtala Muhammed | 2009-2023 (11 years) | PVGIS-SARAH3 | `Lagos_NG_RMY_2009-2023.epw` |
| Bangkok | Thailand | Aw | 484590-99999 Chon Buri/Sattahip / Suvarnabhumi Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Bangkok_TH_RMY_2009-2023.epw` |
| Dhaka | Bangladesh | Aw | 419220-99999 Hazrat Shahjalal Intl | 2008-2023 (14 years) | PVGIS-ERA5 | `Dhaka_BD_RMY_2008-2023.epw` |
| Nairobi | Kenya | Cwb | 637400-99999 Jkia / Nairobi Jomo Kenyatta Intl | 2007-2023 (15 years) | PVGIS-SARAH3 | `Nairobi_KE_RMY_2007-2023.epw` |
| Mexico City | Mexico | Cwb | 766793-99999 Licenciado Benito Juarez Intl / Mexico City Intl | 2010-2024 (15 years) | NSRDB | `Mexico_City_MX_RMY_2010-2024.epw` |
| Karachi | Pakistan | BWh | 417800-99999 Jinnah Intl | 2008-2023 (15 years) | PVGIS-ERA5 | `Karachi_PK_RMY_2008-2023.epw` |
| Riyadh | Saudi Arabia | BWh | 404370-99999 King Khaled Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Riyadh_SA_RMY_2009-2023.epw` |
| Dubai | United Arab Emirates | BWh | 411940-99999 Dubai Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Dubai_AE_RMY_2009-2023.epw` |
| Cairo | Egypt | BWh | 623660-99999 Cairo Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Cairo_EG_RMY_2009-2023.epw` |
| Madrid | Spain | Csa/BSk | 082210-99999 Adolfo Suarez Madrid Barajas | 2009-2023 (15 years) | PVGIS-SARAH3 | `Madrid_ES_RMY_2009-2023.epw` |
| Cape Town | South Africa | Csb | 688160-99999 Cape Town Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Cape_Town_ZA_RMY_2009-2023.epw` |
| Sao Paulo | Brazil | Cfa | 830750-99999 Guarulhos Gov Andre Franco Montoro | 2008-2022 (15 years) | PVGIS-SARAH3 | `Sao_Paulo_BR_RMY_2008-2022.epw` |
| Buenos Aires | Argentina | Cfa | 875760-99999 Ministro Pistarini | 2008-2022 (15 years) | PVGIS-SARAH3 | `Buenos_Aires_AR_RMY_2008-2022.epw` |
| Sydney | Australia | Cfa | 947670-99999 Sydney Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Sydney_AU_RMY_2009-2023.epw` |
| Tokyo | Japan | Cfa | 476710-99999 Tokyo Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Tokyo_JP_RMY_2009-2023.epw` |
| London | United Kingdom | Cfb | 037720-99999 London Heathrow | 2009-2023 (15 years) | PVGIS-SARAH3 | `London_GB_RMY_2009-2023.epw` |
| Beijing | China | Dwa | 545110-99999 Beijing - Capital International Airport | 2009-2023 (15 years) | PVGIS-ERA5 | `Beijing_CN_RMY_2009-2023.epw` |
| Moscow | Russia | Dfb | 275155-99999 Sheremetyevo | 2009-2023 (15 years) | PVGIS-SARAH3 | `Moscow_RU_RMY_2009-2023.epw` |
| Helsinki | Finland | Dfb | 029740-99999 Helsinki Vantaa | 2009-2023 (15 years) | PVGIS-SARAH3 | `Helsinki_FI_RMY_2009-2023.epw` |
