# Future Representative Meteorological Years (FRMYs): 216 cities, 2050

20 US cities (one per climate zone), 20 international cities, 56 world capitals, 20 US state capitals and 100 more capitals and major cities. Each file is the city's RMY morphed to **2050** under **SSP2-4.5** (middle of the road)
with the **CanESM5** climate model: Track 2 of the FRMY method (RMY morph → future RMY), so the
observed peak years' heat waves and cold spells are kept and warmed. The RMYs of all 216 cities are in
`data/RMY`.

The morph is the CMIP6 pattern-scaling emulator of [wxbuild](https://github.com/kranke-git/wxbuild)
(Krane & Giani, MIT): for each month, local change = pattern-scaling slope × the model's global-mean
warming from the baseline (the RMY's years) to 2043-2057. Dry-bulb temperature (with the change in
daily range), relative humidity, pressure and wind are shifted; dew point is recalculated; solar
radiation, cloud and precipitation are not changed. Design conditions are left blank (one composite year
is too short for the ASHRAE statistics); ground temperatures and typical/extreme weeks are recalculated.

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
elsewhere, and NSRDB for Mexico City. New Delhi, added later with half-hour time-zone support, is listed under World capitals. Climate classes are the
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

## World capitals (56)

Built in November 2026 with the fixed AMY code (sky cover filled in every hour, corrupt temperature reports removed;
see [`../QA.md`](../QA.md)). For each capital, the nearby NOAA stations were scored on the size of their yearly
records for 2009-2024. The main airport or city station with the most complete record was used; distances and
elevation differences from the city centre are in the QA summary. Solar radiation is PVGIS-SARAH3 or PVGIS-ERA5,
and NSRDB in the Americas, except Santiago (PVGIS-ERA5) and Montevideo and Asunción (PVGIS-SARAH3), which are
outside NSRDB. Kinshasa, Kampala and Kathmandu have 10 usable years, Accra and Maputo 11; the others have 15. Kabul's station
stopped reporting in November 2021, so its years are 2006-2020. Myanmar is represented by Yangon, its largest city and
former capital: the station at Naypyidaw (Pyinmana) has only 3 usable years. The six capitals in half-hour and
45-minute time zones (New Delhi, Colombo, Tehran, Kabul, Kathmandu, Yangon) carry their fractional offset in the EPW
header (e.g. 5.5, 5.75). Not included: capitals whose stations had fewer than 10 complete years
(Abuja, Addis Ababa, Luanda, Windhoek) or reported only every 3 hours (Tbilisi, Islamabad after 2019).

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

### Middle East and Central Asia (10)

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
| Tehran | Iran | BSk | 407540-99999 Mehrabad Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Tehran_IR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Kabul | Afghanistan | BSk | 409480-99999 Kabul Intl | 2006-2020 (15 years) | PVGIS-ERA5 | `Kabul_AF_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.52 |

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

### Asia-Pacific (14)

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
| New Delhi | India | BSh | 421810-99999 Indira Gandhi Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `New_Delhi_IN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Kathmandu | Nepal | Cwa | 444540-99999 Tribhuvan Intl | 2011-2023 (10 years) | PVGIS-ERA5 | `Kathmandu_NP_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Colombo | Sri Lanka | Af | 434500-99999 Bandaranaike Intl Colombo | 2009-2023 (15 years) | PVGIS-ERA5 | `Colombo_LK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Yangon | Myanmar | Am | 480970-99999 Yangon Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Yangon_MM_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

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

## More capitals and major cities (100)

Built in November 2026 with the same code and QA as the world capitals above: national capitals not yet listed and
large cities, including 8 US metros outside the state-capital list. For each city, the NOAA stations within 45 km
with records from 2009 were scored on how many of their last 15 years have near-hourly reports. The nearest station
with at least 13 such years was used, and each choice was reviewed:
- In the US metros this gives the city airports (e.g. Dallas Love Field, Detroit City, St. Louis Downtown), nearer
  the urban core than the main airports.
- Toronto (Billy Bishop City Centre), Montreal (St-Hubert), Vancouver (Vancouver Intl), Prague (Ruzyně), Fukuoka and
  Monterrey use the best-scored station instead of the nearest.
- Brussels uses Brussels National: the Uccle city station has a year with only 25% of hours observed.
- Shenzhen uses its own airport (Bao'an), not Hong Kong's.

Solar radiation is PVGIS-SARAH3 or PVGIS-ERA5, and NSRDB in the Americas north of 20°S; Rio de Janeiro and Porto
Alegre use PVGIS-SARAH3 and Córdoba PVGIS-ERA5 (outside NSRDB, and for Córdoba outside SARAH-3). Most cities have 15
usable years. Lahore's station stopped reporting regularly in 2019, so its years are 2006-2018 (12 years); Kigali and
Bengaluru also have 12, Baghdad and Khartoum 13, Abu Dhabi, Tegucigalpa and Dallas 14.

Not included (see [`../QA.md`](../QA.md)): Conakry and Yaoundé (fewer than 10 complete years); Tripoli, Nouakchott,
Djibouti and Lilongwe (no station with 10 near-hourly years); Caracas, Medellín, San Salvador, La Paz and
Johannesburg, whose only long-record stations are 370-810 m above or below the city. Sixteen further cities were
built but left out to keep 100 with a regional balance, mostly for sparse records.

### Europe (17)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Amsterdam | Netherlands | Cfb | 062400-99999 Schiphol | 2009-2023 (15 years) | PVGIS-SARAH3 | `Amsterdam_NL_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Barcelona | Spain | Csa | 081810-99999 Barcelona | 2009-2023 (15 years) | PVGIS-SARAH3 | `Barcelona_ES_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Belgrade | Serbia | Cfa | 132720-99999 Beograd | 2009-2023 (15 years) | PVGIS-SARAH3 | `Belgrade_RS_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Bern | Switzerland | Cfb | 066310-99999 Bern-Zollikofen | 2009-2023 (15 years) | PVGIS-SARAH3 | `Bern_CH_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Brussels | Belgium | Cfb | 064510-99999 Brussels Natl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Brussels_BE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Budapest | Hungary | Cfa | 128430-99999 Budapest/Pestszentlorinc | 2009-2023 (15 years) | PVGIS-SARAH3 | `Budapest_HU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Copenhagen | Denmark | Cfb | 061800-99999 Kastrup | 2009-2023 (15 years) | PVGIS-SARAH3 | `Copenhagen_DK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Hamburg | Germany | Cfb | 101470-99999 Hamburg | 2009-2023 (15 years) | PVGIS-SARAH3 | `Hamburg_DE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Istanbul | Turkey | Csa | 170600-99999 Ataturk | 2009-2023 (15 years) | PVGIS-SARAH3 | `Istanbul_TR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Milan | Italy | Cfa | 160800-99999 Milano Linate | 2009-2023 (15 years) | PVGIS-SARAH3 | `Milan_IT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Prague | Czechia | Cfb | 115180-99999 Ruzyne | 2009-2023 (15 years) | PVGIS-SARAH3 | `Prague_CZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Riga | Latvia | Dfb | 264225-99999 Riga | 2009-2023 (15 years) | PVGIS-SARAH3 | `Riga_LV_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Saint Petersburg | Russia | Dfb | 260630-99999 Pulkovo | 2009-2023 (15 years) | PVGIS-SARAH3 | `Saint_Petersburg_RU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Sofia | Bulgaria | Cfb | 156140-99999 Sofia | 2009-2023 (15 years) | PVGIS-SARAH3 | `Sofia_BG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Tirana | Albania | Csa | 136150-99999 Tirana Rinas | 2009-2023 (15 years) | PVGIS-SARAH3 | `Tirana_AL_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Vilnius | Lithuania | Dfb | 267300-99999 Vilnius Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Vilnius_LT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Zagreb | Croatia | Cfb | 142400-99999 Zagreb\Maksimir | 2009-2023 (15 years) | PVGIS-SARAH3 | `Zagreb_HR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### Middle East and Central Asia (13)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Abu Dhabi | United Arab Emirates | BWh | 412160-99999 Al Bateen Executive | 2010-2023 (14 years) | PVGIS-SARAH3 | `Abu_Dhabi_AE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Ashgabat | Turkmenistan | BWk | 388800-99999 Ashgabat | 2009-2023 (15 years) | PVGIS-ERA5 | `Ashgabat_TM_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Baghdad | Iraq | BWh | 406500-99999 Baghdad Intl Airport | 2011-2023 (13 years) | PVGIS-SARAH3 | `Baghdad_IQ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Beirut | Lebanon | Csa | 401000-99999 Rafic Hariri Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Beirut_LB_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Bishkek | Kyrgyzstan | Dsa | 383531-99999 Manas | 2008-2023 (15 years) | PVGIS-ERA5 | `Bishkek_KG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Dushanbe | Tajikistan | Csa | 388360-99999 Dushanbe | 2009-2023 (15 years) | PVGIS-ERA5 | `Dushanbe_TJ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Isfahan | Iran | BWk | 408000-99999 Esfahan Shahid Beheshti Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Isfahan_IR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Izmir | Turkey | Csa | 172190-99999 Adnan Menderes | 2009-2023 (15 years) | PVGIS-SARAH3 | `Izmir_TR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Jeddah | Saudi Arabia | BWh | 410240-99999 King Abdulaziz Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Jeddah_SA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Manama | Bahrain | BWh | 411500-99999 Bahrain Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Manama_BH_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Mashhad | Iran | BSk | 407450-99999 Mashhad Intl / Shahid Hashemi Nejad Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Mashhad_IR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Tel Aviv | Israel | Csa | 401800-99999 Ben Gurion | 2009-2023 (15 years) | PVGIS-SARAH3 | `Tel_Aviv_IL_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Yerevan | Armenia | Dsa | 353946-99999 Zvartnots | 2009-2023 (15 years) | PVGIS-SARAH3 | `Yerevan_AM_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### Africa (13)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Abidjan | Cote d'Ivoire | Aw | 655780-99999 Abidjan Felix Houphouet Boigny Intl | 2008-2023 (15 years) | PVGIS-SARAH3 | `Abidjan_CI_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Alexandria | Egypt | BWh | 623180-99999 Alexandria Intl | 2008-2022 (15 years) | PVGIS-SARAH3 | `Alexandria_EG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.43 |
| Bamako | Mali | BSh | 612910-99999 Senou / President Modibo Keita | 2009-2023 (15 years) | PVGIS-SARAH3 | `Bamako_ML_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Dar es Salaam | Tanzania | Aw | 638940-99999 Mwalimu Julius K Nyerere Intl | 2007-2023 (15 years) | PVGIS-SARAH3 | `Dar_es_Salaam_TZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Gaborone | Botswana | BSh | 682400-99999 Sir Seretse Khama Intl | 2008-2023 (15 years) | PVGIS-SARAH3 | `Gaborone_BW_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Khartoum | Sudan | BWh | 627210-99999 Khartoum | 2008-2020 (13 years) | PVGIS-SARAH3 | `Khartoum_SD_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.48 |
| Kigali | Rwanda | Aw | 643870-99999 Kigali Intl | 2010-2023 (12 years) | PVGIS-SARAH3 | `Kigali_RW_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Libreville | Gabon | Am | 645000-99999 Leon M Ba | 2009-2023 (15 years) | PVGIS-SARAH3 | `Libreville_GA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Lomé | Togo | Aw | 653870-99999 Gnassingbe Eyadema Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Lome_TG_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Mombasa | Kenya | As | 638200-99999 Mombasa Moi Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Mombasa_KE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Niamey | Niger | BWh | 610520-99999 Diori Hamani | 2009-2023 (15 years) | PVGIS-SARAH3 | `Niamey_NE_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Ouagadougou | Burkina Faso | BSh | 655030-99999 Ouagadougou | 2009-2023 (15 years) | PVGIS-SARAH3 | `Ouagadougou_BF_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Port Louis | Mauritius | Am | 619900-99999 Sir Seewoosagur Ramgoolam Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Port_Louis_MU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### South Asia (8)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Ahmedabad | India | BSh | 426470-99999 Ahmedabad / Sardar Vallabhbhai Patel Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Ahmedabad_IN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Bengaluru | India | Aw | 427056-99999 Bangaluru Intl Airport / Kempegowda Intl | 2009-2023 (12 years) | PVGIS-ERA5 | `Bengaluru_IN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Chennai | India | As | 432790-99999 Chennai Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Chennai_IN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Hyderabad | India | BSh | 431280-99999 Begumpet Airport / Hyderabad Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Hyderabad_IN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Kolkata | India | Aw | 428090-99999 Netaji Subhash Chandra Bose Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Kolkata_IN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Lahore | Pakistan | BSh | 416410-99999 Allama Iqbal Intl | 2006-2018 (12 years) | PVGIS-ERA5 | `Lahore_PK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.57 |
| Malé | Maldives | Am | 435550-99999 Male Intl / Velana Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Male_MV_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Mumbai | India | Am | 430030-99999 Chhatrapati Shivaji Maharaj Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Mumbai_IN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### East Asia (14)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Busan | South Korea | Cfa | 471530-99999 Gimhae Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Busan_KR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Chengdu | China | Cwa | 562940-99999 Shuangliu | 2009-2023 (15 years) | PVGIS-ERA5 | `Chengdu_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Chongqing | China | Cfa | 575160-99999 Jiangbei | 2009-2023 (15 years) | PVGIS-ERA5 | `Chongqing_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Fukuoka | Japan | Cfa | 478080-99999 Fukuoka | 2009-2023 (15 years) | PVGIS-ERA5 | `Fukuoka_JP_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Guangzhou | China | Cfa | 592870-99999 Baiyun Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Guangzhou_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Harbin | China | Dwa | 541611-99999 Taiping | 2009-2023 (15 years) | PVGIS-ERA5 | `Harbin_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Hong Kong | Hong Kong | Cwa | 450070-99999 Hong Kong Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Hong_Kong_HK_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Osaka | Japan | Cfa | 477720-99999 Osaka | 2009-2023 (15 years) | PVGIS-ERA5 | `Osaka_JP_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Sapporo | Japan | Dfa | 474120-99999 Sapporo | 2009-2023 (15 years) | PVGIS-ERA5 | `Sapporo_JP_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Shanghai | China | Cfa | 583670-99999 Hongqiao Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Shanghai_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Shenzhen | China | Cwa | 594930-99999 Baoan Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Shenzhen_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Urumqi | China | BSk | 514635-99999 Diwopu | 2009-2023 (15 years) | PVGIS-ERA5 | `Urumqi_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Wuhan | China | Cfa | 574940-99999 Tianhe | 2009-2023 (15 years) | PVGIS-ERA5 | `Wuhan_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Xi'an | China | BSk | 570360-99999 Xianyang | 2009-2023 (15 years) | PVGIS-ERA5 | `Xian_CN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### Southeast Asia and Oceania (10)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Adelaide | Australia | Csa | 946720-99999 Adelaide Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Adelaide_AU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Auckland | New Zealand | Cfb | 931190-99999 Auckland Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Auckland_NZ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Bandar Seri Begawan | Brunei | Af | 963150-99999 Brunei Intl | 2008-2023 (15 years) | PVGIS-ERA5 | `Bandar_Seri_Begawan_BN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Brisbane | Australia | Cfa | 945780-99999 Brisbane Airport | 2009-2023 (15 years) | PVGIS-ERA5 | `Brisbane_AU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Cebu | Philippines | Am | 986460-99999 Mactan Cebu Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Cebu_PH_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Darwin | Australia | Aw | 941200-99999 Darwin Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Darwin_AU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Ho Chi Minh City | Vietnam | Aw | 489000-99999 Tan Son Nhat Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Ho_Chi_Minh_City_VN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Melbourne | Australia | Cfb | 948660-99999 Melbourne Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Melbourne_AU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Perth | Australia | Csa | 946100-99999 Perth Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Perth_AU_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |
| Suva | Fiji | Af | 916830-99999 Nausori Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Suva_FJ_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.38 |

### Americas (17)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File | Global warming (K) |
|---|---|---|---|---|---|---|---|
| Calgary | Canada | Dfb | 718770-99999 Calgary Intl | 2009-2024 (15 years) | NSRDB | `Calgary_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Georgetown | Guyana | Af | 810020-10502 Georgetown | 2010-2024 (15 years) | NSRDB | `Georgetown_GY_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Guadalajara | Mexico | Cwa | 766133-99999 Don Miguel Hidalgo Y Costilla Intl / Guadalajara Intl | 2010-2024 (15 years) | NSRDB | `Guadalajara_MX_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Kingston | Jamaica | Aw | 783970-99999 Norman Manley Intl | 2010-2024 (15 years) | NSRDB | `Kingston_JM_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Managua | Nicaragua | Aw | 787410-99999 Managua Intl / Augusto Cesar Sandino Intl | 2009-2024 (15 years) | NSRDB | `Managua_NI_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Manaus | Brazil | Af | 823320-99999 Campo Ponta Pelada | 2010-2024 (15 years) | NSRDB | `Manaus_BR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Monterrey | Mexico | BSh | 763943-99999 General Mariano Escobedo Intl / Monterrey Intl | 2008-2024 (15 years) | NSRDB | `Monterrey_MX_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Montreal | Canada | Dfb | 713710-99999 Montreal/St Hubert | 2010-2024 (15 years) | NSRDB | `Montreal_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Port of Spain | Trinidad and Tobago | Aw | 789700-11634 Piarco Intl Ap | 2008-2024 (15 years) | NSRDB | `Port_of_Spain_TT_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| Porto Alegre | Brazil | Cfa | 839710-99999 Salgado Filho | 2008-2022 (15 years) | PVGIS-SARAH3 | `Porto_Alegre_BR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.43 |
| Recife | Brazil | As | 828990-99999 Guararapes Gilberto Freyre Intl | 2010-2024 (15 years) | NSRDB | `Recife_BR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Rio de Janeiro | Brazil | Aw | 837460-99999 Galeao Antonio Carlos Jobim | 2008-2022 (15 years) | PVGIS-SARAH3 | `Rio_de_Janeiro_BR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.43 |
| Salvador | Brazil | Af | 832480-99999 Deputado Luis Eduardo Magalhaes | 2010-2024 (15 years) | NSRDB | `Salvador_BR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Tegucigalpa | Honduras | Aw | 787200-99999 Toncontin Intl | 2007-2024 (14 years) | NSRDB | `Tegucigalpa_HN_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.42 |
| Toronto | Canada | Dfa | 712650-99999 Toronto City Centre / Billy Bishop | 2010-2024 (15 years) | NSRDB | `Toronto_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Vancouver | Canada | Cfb | 718920-99999 Vancouver Intl | 2010-2024 (15 years) | NSRDB | `Vancouver_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Córdoba | Argentina | Cwa | 873440-99999 Ambrosio L V Taravella | 2006-2022 (15 years) | PVGIS-ERA5 | `Cordoba_AR_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.47 |

### United States (8)

Large US cities not already listed, with NSRDB solar radiation and their ASHRAE 169 climate zone.

| City | Zone | File | Global warming (K) |
|---|---|---|---|
| New Orleans LA | 2A | `New_Orleans_LA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |
| San Antonio TX | 2A | `San_Antonio_TX_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Dallas TX | 3A | `Dallas_TX_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.32 |
| San Diego CA | 3B | `San_Diego_CA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Kansas City MO | 4A | `Kansas_City_MO_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Philadelphia PA | 4A | `Philadelphia_PA_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| St. Louis MO | 4A | `St_Louis_MO_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.33 |
| Detroit MI | 5A | `Detroit_MI_FRMY_2050_SSP2-4.5_CanESM5.epw` | +1.37 |

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
