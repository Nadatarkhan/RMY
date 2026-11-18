# Future Representative Meteorological Years (FRMYs): 110 cities, 2050

20 US cities (one per climate zone), 20 international cities, 50 world capitals and 20 US state capitals. Each file is the city's RMY morphed to **2050** under **SSP2-4.5** (middle of the road)
with the **CanESM5** climate model: Track 2 of the FRMY method (RMY morph → future RMY), so the
observed peak years' heat waves and cold spells are kept and warmed. The RMYs of all 110 cities are in
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

## World capitals (50)

Built in November 2026 with the fixed AMY code (sky cover filled in every hour, corrupt temperature reports removed;
see [`../QA.md`](../QA.md)). For each capital, the nearby NOAA stations were scored on the size of their yearly
records for 2009-2024. The main airport or city station with the most complete record was used; distances and
elevation differences from the city centre are in the QA summary. Solar radiation is PVGIS-SARAH3 or PVGIS-ERA5,
and NSRDB in the Americas, except Santiago (PVGIS-ERA5) and Montevideo and Asunción (PVGIS-SARAH3), which are
outside NSRDB. Kinshasa and Kampala have 10 usable years, Accra and Maputo 11; the others have 15. Capitals in
half-hour time zones (New Delhi, Tehran, Kabul, Kathmandu, Colombo, Yangon) are not included, and neither are
capitals whose stations had fewer than 10 complete years (Abuja, Addis Ababa, Luanda, Windhoek) or reported only
every 3 hours (Tbilisi, Islamabad after 2019).

### Europe (12)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Paris | France | Cfb | 071490-99999 Orly | 2008-2023 (15 years) | PVGIS-SARAH3 | `Paris_FR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Berlin | Germany | Cfb | 103850-99999 Schonefeld / Brandenburg | 2009-2023 (15 years) | PVGIS-SARAH3 | `Berlin_DE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Rome | Italy | Csa | 162390-99999 Ciampino | 2009-2023 (15 years) | PVGIS-SARAH3 | `Rome_IT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Vienna | Austria | Cfb | 110360-99999 Wien Schwechat | 2009-2023 (15 years) | PVGIS-SARAH3 | `Vienna_AT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Warsaw | Poland | Dfb | 123750-99999 Okecie / Warsaw Chopin | 2009-2023 (15 years) | PVGIS-SARAH3 | `Warsaw_PL_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Stockholm | Sweden | Dfb | 024640-99999 Bromma | 2009-2023 (15 years) | PVGIS-SARAH3 | `Stockholm_SE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Oslo | Norway | Dfb | 014920-99999 Oslo-Blindern | 2007-2023 (15 years) | PVGIS-SARAH3 | `Oslo_NO_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Athens | Greece | Csa | 167161-99999 Eleftherios Venizelos Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Athens_GR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Lisbon | Portugal | Csa | 085360-99999 Lisboa | 2009-2023 (15 years) | PVGIS-SARAH3 | `Lisbon_PT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Dublin | Ireland | Cfb | 039690-99999 Dublin | 2009-2023 (15 years) | PVGIS-SARAH3 | `Dublin_IE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Reykjavik | Iceland | Cfc | 040300-99999 Reykjavik | 2009-2023 (15 years) | PVGIS-SARAH3 | `Reykjavik_IS_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Bucharest | Romania | Dfa | 154200-99999 Baneasa Aurel Vlaicu | 2009-2023 (15 years) | PVGIS-SARAH3 | `Bucharest_RO_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### Middle East and Central Asia (8)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Ankara | Turkey | BSk | 171280-99999 Esenboga | 2009-2023 (15 years) | PVGIS-SARAH3 | `Ankara_TR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Amman | Jordan | BSh | 402700-99999 Marka Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Amman_JO_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Kuwait City | Kuwait | BWh | 405820-99999 Kuwait Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Kuwait_City_KW_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Doha | Qatar | BWh | 411700-99999 Doha Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Doha_QA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Muscat | Oman | BWh | 412560-99999 Seeb Intl / Muscat Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Muscat_OM_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Tashkent | Uzbekistan | Csa | 384570-99999 Yuzhny / Tashkent Islam Karimov Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Tashkent_UZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Astana | Kazakhstan | Dfb | 351880-99999 Astana Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Astana_KZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Baku | Azerbaijan | BSk | 378640-99999 Heydar Aliyev | 2009-2023 (15 years) | PVGIS-SARAH3 | `Baku_AZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### Africa (9)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Algiers | Algeria | Csa | 603900-99999 Houari Boumediene | 2009-2023 (15 years) | PVGIS-SARAH3 | `Algiers_DZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Rabat | Morocco | Csa | 601350-99999 Rabat Sale Intl | 2007-2023 (15 years) | PVGIS-SARAH3 | `Rabat_MA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Tunis | Tunisia | Csa | 607150-99999 Carthage | 2009-2023 (15 years) | PVGIS-SARAH3 | `Tunis_TN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Dakar | Senegal | BSh | 616410-99999 Leopold Sedar Senghor Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Dakar_SN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Accra | Ghana | Aw | 654720-99999 Kotoka Intl | 2010-2023 (11 years) | PVGIS-SARAH3 | `Accra_GH_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Kinshasa | DR Congo | Aw | 642100-99999 Ndjili Intl | 2011-2023 (10 years) | PVGIS-SARAH3 | `Kinshasa_CD_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Kampala | Uganda | Af | 637050-99999 Entebbe Intl | 2007-2023 (10 years) | PVGIS-SARAH3 | `Kampala_UG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Antananarivo | Madagascar | Cwb | 670830-99999 Ivato | 2009-2023 (15 years) | PVGIS-SARAH3 | `Antananarivo_MG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Maputo | Mozambique | Aw | 673410-99999 Maputo | 2007-2023 (11 years) | PVGIS-SARAH3 | `Maputo_MZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |

### Asia-Pacific (10)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Seoul | South Korea | Dwa | 471100-99999 Gimpo | 2009-2023 (15 years) | PVGIS-ERA5 | `Seoul_KR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Ulaanbaatar | Mongolia | Dwc | 442920-99999 Chinggis Khaan Intl / Buyant Ukhaa Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Ulaanbaatar_MN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Taipei | Taiwan | Cfa | 466960-99999 Songshan | 2009-2023 (15 years) | PVGIS-ERA5 | `Taipei_TW_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Hanoi | Vietnam | Cwa | 488200-99999 Noibai Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Hanoi_VN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Manila | Philippines | Aw | 984290-99999 Ninoy Aquino Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Manila_PH_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Kuala Lumpur | Malaysia | Af | 486470-99999 Sultan Abdul Aziz Shah Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Kuala_Lumpur_MY_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Jakarta | Indonesia | Am | 967490-99999 Soekarno Hatta Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Jakarta_ID_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Canberra | Australia | Cfb | 949260-99999 Canberra | 2009-2023 (15 years) | PVGIS-ERA5 | `Canberra_AU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Wellington | New Zealand | Cfb | 934360-00488 Wellington Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Wellington_NZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Phnom Penh | Cambodia | Aw | 489910-99999 Phnom Penh Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Phnom_Penh_KH_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### Americas (11)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Havana | Cuba | Aw | 782240-99999 Jose Marti Intl | 2010-2024 (15 years) | NSRDB | `Havana_CU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Ottawa | Canada | Dfb | 716280-99999 Ottawa Macdonald Cartier Intl | 2010-2024 (15 years) | NSRDB | `Ottawa_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Guatemala City | Guatemala | Cwb | 786410-99999 La Aurora | 2010-2024 (15 years) | NSRDB | `Guatemala_City_GT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Panama City | Panama | Am | 787920-99999 Tocumen Intl | 2009-2024 (15 years) | NSRDB | `Panama_City_PA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Bogota | Colombia | Cfb | 802220-99999 El Dorado Intl | 2009-2024 (15 years) | NSRDB | `Bogota_CO_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Lima | Peru | BWh | 846280-99999 Jorge Chavez Intl | 2007-2024 (15 years) | NSRDB | `Lima_PE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Brasilia | Brazil | Aw | 833780-99999 Presidente Juscelino Kubitschek | 2010-2024 (15 years) | NSRDB | `Brasilia_BR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Santo Domingo | Dominican Republic | Am | 784850-99999 De Las Americas Intl | 2009-2024 (15 years) | NSRDB | `Santo_Domingo_DO_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Santiago | Chile | Csb | 855740-99999 Arturo Merino Benitez Intl | 2007-2022 (15 years) | PVGIS-ERA5 | `Santiago_CL_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.47 |
| Montevideo | Uruguay | Cfa | 865800-99999 Carrasco Intl | 2008-2022 (15 years) | PVGIS-SARAH3 | `Montevideo_UY_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.43 |
| Asuncion | Paraguay | Cfa | 862180-99999 Silvio Pettirossi Intl | 2008-2022 (15 years) | PVGIS-SARAH3 | `Asuncion_PY_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.43 |

## US state capitals (20)

Twenty state capitals not already listed above, spread over ASHRAE climate zones 1A to 7, with NSRDB solar
radiation. Baton Rouge has 14 usable years (its station record starts in late 2010); the others have 15.

| City | Zone | File | Global warming (K) |
|---|---|---|---|
| Honolulu HI | 1A | `Honolulu_HI_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Austin TX | 2A | `Austin_TX_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Baton Rouge LA | 2A | `Baton_Rouge_LA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.32 |
| Tallahassee FL | 2A | `Tallahassee_FL_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Oklahoma City OK | 3A | `Oklahoma_City_OK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Sacramento CA | 3B | `Sacramento_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Nashville TN | 4A | `Nashville_TN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Richmond VA | 4A | `Richmond_VA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Salem OR | 4C | `Salem_OR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Columbus OH | 5A | `Columbus_OH_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Indianapolis IN | 5A | `Indianapolis_IN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Des Moines IA | 5A | `Des_Moines_IA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Albany NY | 5A | `Albany_NY_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Salt Lake City UT | 5B | `Salt_Lake_City_UT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Boise ID | 5B | `Boise_ID_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Santa Fe NM | 5B | `Santa_Fe_NM_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Madison WI | 6A | `Madison_WI_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Bismarck ND | 6A | `Bismarck_ND_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Cheyenne WY | 6B | `Cheyenne_WY_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Juneau AK | 7 | `Juneau_AK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |

## Quality assurance

Every city's AMYs, TMY, RMY and FRMY pass about 85 checks; see [`../QA.md`](../QA.md) for the method, the findings
(including the cold-climate RMY mean shift and station representativeness) and the per-city CSVs.

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
