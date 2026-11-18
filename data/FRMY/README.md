# Future Representative Meteorological Years (FRMYs): 40 cities, 2050

20 US cities (one per climate zone) and 20 international cities. Each file is the city's RMY morphed to **2050** under **SSP2-4.5** (middle of the road)
with the **CanESM5** climate model: Track 2 of the FRMY method (RMY morph → future RMY), so the
observed peak years' heat waves and cold spells are kept and warmed. The RMYs of all 40 cities are in
`data/RMY`.

The morph is the CMIP6 pattern-scaling emulator of [wxbuild](https://github.com/kranke-git/wxbuild)
(Krane & Giani, MIT): for each month, local change = pattern-scaling slope × the model's global-mean
warming from the baseline (the RMY's years) to 2043-2057. Dry-bulb temperature (with the change in
daily range), relative humidity, pressure and wind are shifted; dew point is recalculated; solar
radiation, cloud and precipitation are not changed. The code is in `frmy/` on the branch
`nmt/frmy_demo`. Design conditions are left blank (one composite year is too short for the ASHRAE
statistics); ground temperatures and typical/extreme weeks are recalculated.

CanESM5 is a high-sensitivity model: its 2050 global warming is about twice that of other CMIP6
models (e.g. MPI-ESM1-2-LR), so these files are toward the warm end of SSP2-4.5 projections.

## United States

| City | Zone | File | Global warming (K) |
|---|---|---|---|
| Miami FL | 1A | `Miami_FL_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Houston TX | 2A | `Houston_TX_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Phoenix AZ | 2B | `Phoenix_AZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Atlanta GA | 3A | `Atlanta_GA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Las Vegas NV | 3B | `Las_Vegas_NV_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Los Angeles CA | 3B | `Los_Angeles_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| San Francisco CA | 3C | `San_Francisco_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| New York NY | 4A | `New_York_NY_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Washington Dulles VA | 4A | `Washington_Dulles_VA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Albuquerque NM | 4B | `Albuquerque_NM_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Portland OR | 4C | `Portland_OR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Seattle WA | 4C | `Seattle_WA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Boston MA | 5A | `Boston_MA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Chicago IL | 5A | `Chicago_IL_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Denver CO | 5B | `Denver_CO_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Minneapolis MN | 6A | `Minneapolis_MN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Helena MT | 6B | `Helena_MT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Anchorage AK | 7 | `Anchorage_AK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.43 |
| Duluth MN | 7 | `Duluth_MN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Fairbanks AK | 8 | `Fairbanks_AK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.47 |

## International

Built from the 15 most recent complete years at each NOAA station where available (Lagos has 11 usable
years, Dhaka 14). Solar radiation is PVGIS-SARAH3 where its satellite covers the city, PVGIS-ERA5
elsewhere, and NSRDB for Mexico City. Mumbai and New Delhi are not included: the AMY code does not yet
support India's UTC+5:30 time zone, so Karachi and Dhaka represent South Asia. Climate classes are the
approximate Köppen-Geiger class of each city.

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Singapore | Singapore | Af | 486980-99999 Singapore Changi Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Singapore_SG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Lagos | Nigeria | Aw | 652010-99999 Murtala Muhammed | 2009-2023 (11 years) | PVGIS-SARAH3 | `Lagos_NG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Bangkok | Thailand | Aw | 484590-99999 Chon Buri/Sattahip / Suvarnabhumi Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Bangkok_TH_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Dhaka | Bangladesh | Aw | 419220-99999 Hazrat Shahjalal Intl | 2008-2023 (14 years) | PVGIS-ERA5 | `Dhaka_BD_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Nairobi | Kenya | Cwb | 637400-99999 Jkia / Nairobi Jomo Kenyatta Intl | 2007-2023 (15 years) | PVGIS-SARAH3 | `Nairobi_KE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Mexico City | Mexico | Cwb | 766793-99999 Licenciado Benito Juarez Intl / Mexico City Intl | 2010-2024 (15 years) | NSRDB | `Mexico_City_MX_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Karachi | Pakistan | BWh | 417800-99999 Jinnah Intl | 2008-2023 (15 years) | PVGIS-ERA5 | `Karachi_PK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Riyadh | Saudi Arabia | BWh | 404370-99999 King Khaled Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Riyadh_SA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Dubai | United Arab Emirates | BWh | 411940-99999 Dubai Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Dubai_AE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Cairo | Egypt | BWh | 623660-99999 Cairo Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Cairo_EG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Madrid | Spain | Csa/BSk | 082210-99999 Adolfo Suarez Madrid Barajas | 2009-2023 (15 years) | PVGIS-SARAH3 | `Madrid_ES_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Cape Town | South Africa | Csb | 688160-99999 Cape Town Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Cape_Town_ZA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Sao Paulo | Brazil | Cfa | 830750-99999 Guarulhos Gov Andre Franco Montoro | 2008-2022 (15 years) | PVGIS-SARAH3 | `Sao_Paulo_BR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.43 |
| Buenos Aires | Argentina | Cfa | 875760-99999 Ministro Pistarini | 2008-2022 (15 years) | PVGIS-SARAH3 | `Buenos_Aires_AR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.43 |
| Sydney | Australia | Cfa | 947670-99999 Sydney Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Sydney_AU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Tokyo | Japan | Cfa | 476710-99999 Tokyo Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Tokyo_JP_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| London | United Kingdom | Cfb | 037720-99999 London Heathrow | 2009-2023 (15 years) | PVGIS-SARAH3 | `London_GB_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Beijing | China | Dwa | 545110-99999 Beijing - Capital International Airport | 2009-2023 (15 years) | PVGIS-ERA5 | `Beijing_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Moscow | Russia | Dfb | 275155-99999 Sheremetyevo | 2009-2023 (15 years) | PVGIS-SARAH3 | `Moscow_RU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Helsinki | Finland | Dfb | 029740-99999 Helsinki Vantaa | 2009-2023 (15 years) | PVGIS-SARAH3 | `Helsinki_FI_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

## Updates

**Sky cover filled (November 2026).** In 8 cities (Bangkok, Buenos Aires, Lagos, London, Moscow, Nairobi, Riyadh,
Singapore) 13 to 153 hours had no total or opaque sky cover (EPW code 99). The AMY builder filled long cloud gaps
only from NSRDB cloud types, which PVGIS-based years do not have. These hours are now filled with the AMY
builder's own method (`amy_sourcing`):
- gaps of up to 12 hours by linear interpolation;
- longer gaps by an estimate blended to the recorded values at both ends.
  - By day, the estimate is the file's typical cover for the same clearness index.
  - By night, it is the typical cover for the same month and hour.

The EPW source flag of each filled hour is set to B, C or E. No other field changed, and each RMY and its FRMY carry
identical sky cover. New builds fill these hours when the AMYs are made.
