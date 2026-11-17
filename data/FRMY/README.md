# Future Representative Meteorological Years (FRMYs): 20 US cities, 2050

Each file is the city's RMY in `data/RMY` morphed to **2050** under **SSP2-4.5** (middle of the road)
with the **CanESM5** climate model: Track 2 of the FRMY method (RMY morph → future RMY), so the
observed peak years' heat waves and cold spells are kept and warmed.

The morph is the CMIP6 pattern-scaling emulator of [wxbuild](https://github.com/kranke-git/wxbuild)
(Krane & Giani, MIT): for each month, local change = pattern-scaling slope × the model's global-mean
warming from the baseline (the RMY's years) to 2043-2057. Dry-bulb temperature (with the change in
daily range), relative humidity, pressure and wind are shifted; dew point is recalculated; solar
radiation, cloud and precipitation are not changed. The code is in `frmy/` on the branch
`nmt/frmy_demo`. Design conditions are left blank (one composite year is too short for the ASHRAE
statistics); ground temperatures and typical/extreme weeks are recalculated.

CanESM5 is a high-sensitivity model: its 2050 global warming is about twice that of other CMIP6
models (e.g. MPI-ESM1-2-LR), so these files are toward the warm end of SSP2-4.5 projections.

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
