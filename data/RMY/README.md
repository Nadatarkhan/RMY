# Representative Meteorological Years (RMYs): 500 cities

20 US cities, one per US climate zone (ASHRAE 169, 1A to 8), 20 international cities, 56 world capitals, 20 US state capitals, 100 more capitals and major cities, 100 more US major cities and 184 more US cities. Each is the city's ISO 15927-4 TMY with the heat
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

Lagos has 11 usable years and Dhaka 14; the others have 15. New Delhi, added later with half-hour time-zone support, is listed under World capitals. Climate
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

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Paris | France | Cfb | 071490-99999 Orly | 2008-2023 (15 years) | PVGIS-SARAH3 | `Paris_FR_RMY_2008-2023.epw` |
| Berlin | Germany | Cfb | 103850-99999 Schonefeld / Brandenburg | 2009-2023 (15 years) | PVGIS-SARAH3 | `Berlin_DE_RMY_2009-2023.epw` |
| Rome | Italy | Csa | 162390-99999 Ciampino | 2009-2023 (15 years) | PVGIS-SARAH3 | `Rome_IT_RMY_2009-2023.epw` |
| Vienna | Austria | Cfb | 110360-99999 Wien Schwechat | 2009-2023 (15 years) | PVGIS-SARAH3 | `Vienna_AT_RMY_2009-2023.epw` |
| Warsaw | Poland | Dfb | 123750-99999 Okecie / Warsaw Chopin | 2009-2023 (15 years) | PVGIS-SARAH3 | `Warsaw_PL_RMY_2009-2023.epw` |
| Stockholm | Sweden | Dfb | 024640-99999 Bromma | 2009-2023 (15 years) | PVGIS-SARAH3 | `Stockholm_SE_RMY_2009-2023.epw` |
| Oslo | Norway | Dfb | 014920-99999 Oslo-Blindern | 2007-2023 (15 years) | PVGIS-SARAH3 | `Oslo_NO_RMY_2007-2023.epw` |
| Athens | Greece | Csa | 167161-99999 Eleftherios Venizelos Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Athens_GR_RMY_2009-2023.epw` |
| Lisbon | Portugal | Csa | 085360-99999 Lisboa | 2009-2023 (15 years) | PVGIS-SARAH3 | `Lisbon_PT_RMY_2009-2023.epw` |
| Dublin | Ireland | Cfb | 039690-99999 Dublin | 2009-2023 (15 years) | PVGIS-SARAH3 | `Dublin_IE_RMY_2009-2023.epw` |
| Reykjavik | Iceland | Cfc | 040300-99999 Reykjavik | 2009-2023 (15 years) | PVGIS-SARAH3 | `Reykjavik_IS_RMY_2009-2023.epw` |
| Bucharest | Romania | Dfa | 154200-99999 Baneasa Aurel Vlaicu | 2009-2023 (15 years) | PVGIS-SARAH3 | `Bucharest_RO_RMY_2009-2023.epw` |

### Middle East and Central Asia (10)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Ankara | Turkey | BSk | 171280-99999 Esenboga | 2009-2023 (15 years) | PVGIS-SARAH3 | `Ankara_TR_RMY_2009-2023.epw` |
| Amman | Jordan | BSh | 402700-99999 Marka Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Amman_JO_RMY_2009-2023.epw` |
| Kuwait City | Kuwait | BWh | 405820-99999 Kuwait Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Kuwait_City_KW_RMY_2009-2023.epw` |
| Doha | Qatar | BWh | 411700-99999 Doha Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Doha_QA_RMY_2009-2023.epw` |
| Muscat | Oman | BWh | 412560-99999 Seeb Intl / Muscat Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Muscat_OM_RMY_2009-2023.epw` |
| Tashkent | Uzbekistan | Csa | 384570-99999 Yuzhny / Tashkent Islam Karimov Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Tashkent_UZ_RMY_2009-2023.epw` |
| Astana | Kazakhstan | Dfb | 351880-99999 Astana Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Astana_KZ_RMY_2009-2023.epw` |
| Baku | Azerbaijan | BSk | 378640-99999 Heydar Aliyev | 2009-2023 (15 years) | PVGIS-SARAH3 | `Baku_AZ_RMY_2009-2023.epw` |
| Tehran | Iran | BSk | 407540-99999 Mehrabad Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Tehran_IR_RMY_2009-2023.epw` |
| Kabul | Afghanistan | BSk | 409480-99999 Kabul Intl | 2006-2020 (15 years) | PVGIS-ERA5 | `Kabul_AF_RMY_2006-2020.epw` |

### Africa (9)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Algiers | Algeria | Csa | 603900-99999 Houari Boumediene | 2009-2023 (15 years) | PVGIS-SARAH3 | `Algiers_DZ_RMY_2009-2023.epw` |
| Rabat | Morocco | Csa | 601350-99999 Rabat Sale Intl | 2007-2023 (15 years) | PVGIS-SARAH3 | `Rabat_MA_RMY_2007-2023.epw` |
| Tunis | Tunisia | Csa | 607150-99999 Carthage | 2009-2023 (15 years) | PVGIS-SARAH3 | `Tunis_TN_RMY_2009-2023.epw` |
| Dakar | Senegal | BSh | 616410-99999 Leopold Sedar Senghor Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Dakar_SN_RMY_2009-2023.epw` |
| Accra | Ghana | Aw | 654720-99999 Kotoka Intl | 2010-2023 (11 years) | PVGIS-SARAH3 | `Accra_GH_RMY_2010-2023.epw` |
| Kinshasa | DR Congo | Aw | 642100-99999 Ndjili Intl | 2011-2023 (10 years) | PVGIS-SARAH3 | `Kinshasa_CD_RMY_2011-2023.epw` |
| Kampala | Uganda | Af | 637050-99999 Entebbe Intl | 2007-2023 (10 years) | PVGIS-SARAH3 | `Kampala_UG_RMY_2007-2023.epw` |
| Antananarivo | Madagascar | Cwb | 670830-99999 Ivato | 2009-2023 (15 years) | PVGIS-SARAH3 | `Antananarivo_MG_RMY_2009-2023.epw` |
| Maputo | Mozambique | Aw | 673410-99999 Maputo | 2007-2023 (11 years) | PVGIS-SARAH3 | `Maputo_MZ_RMY_2007-2023.epw` |

### Asia-Pacific (14)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Seoul | South Korea | Dwa | 471100-99999 Gimpo | 2009-2023 (15 years) | PVGIS-ERA5 | `Seoul_KR_RMY_2009-2023.epw` |
| Ulaanbaatar | Mongolia | Dwc | 442920-99999 Chinggis Khaan Intl / Buyant Ukhaa Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Ulaanbaatar_MN_RMY_2009-2023.epw` |
| Taipei | Taiwan | Cfa | 466960-99999 Songshan | 2009-2023 (15 years) | PVGIS-ERA5 | `Taipei_TW_RMY_2009-2023.epw` |
| Hanoi | Vietnam | Cwa | 488200-99999 Noibai Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Hanoi_VN_RMY_2009-2023.epw` |
| Manila | Philippines | Aw | 984290-99999 Ninoy Aquino Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Manila_PH_RMY_2009-2023.epw` |
| Kuala Lumpur | Malaysia | Af | 486470-99999 Sultan Abdul Aziz Shah Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Kuala_Lumpur_MY_RMY_2009-2023.epw` |
| Jakarta | Indonesia | Am | 967490-99999 Soekarno Hatta Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Jakarta_ID_RMY_2009-2023.epw` |
| Canberra | Australia | Cfb | 949260-99999 Canberra | 2009-2023 (15 years) | PVGIS-ERA5 | `Canberra_AU_RMY_2009-2023.epw` |
| Wellington | New Zealand | Cfb | 934360-00488 Wellington Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Wellington_NZ_RMY_2009-2023.epw` |
| Phnom Penh | Cambodia | Aw | 489910-99999 Phnom Penh Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Phnom_Penh_KH_RMY_2009-2023.epw` |
| New Delhi | India | BSh | 421810-99999 Indira Gandhi Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `New_Delhi_IN_RMY_2009-2023.epw` |
| Kathmandu | Nepal | Cwa | 444540-99999 Tribhuvan Intl | 2011-2023 (10 years) | PVGIS-ERA5 | `Kathmandu_NP_RMY_2011-2023.epw` |
| Colombo | Sri Lanka | Af | 434500-99999 Bandaranaike Intl Colombo | 2009-2023 (15 years) | PVGIS-ERA5 | `Colombo_LK_RMY_2009-2023.epw` |
| Yangon | Myanmar | Am | 480970-99999 Yangon Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Yangon_MM_RMY_2009-2023.epw` |

### Americas (11)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Havana | Cuba | Aw | 782240-99999 Jose Marti Intl | 2010-2024 (15 years) | NSRDB | `Havana_CU_RMY_2010-2024.epw` |
| Ottawa | Canada | Dfb | 716280-99999 Ottawa Macdonald Cartier Intl | 2010-2024 (15 years) | NSRDB | `Ottawa_CA_RMY_2010-2024.epw` |
| Guatemala City | Guatemala | Cwb | 786410-99999 La Aurora | 2010-2024 (15 years) | NSRDB | `Guatemala_City_GT_RMY_2010-2024.epw` |
| Panama City | Panama | Am | 787920-99999 Tocumen Intl | 2009-2024 (15 years) | NSRDB | `Panama_City_PA_RMY_2009-2024.epw` |
| Bogota | Colombia | Cfb | 802220-99999 El Dorado Intl | 2009-2024 (15 years) | NSRDB | `Bogota_CO_RMY_2009-2024.epw` |
| Lima | Peru | BWh | 846280-99999 Jorge Chavez Intl | 2007-2024 (15 years) | NSRDB | `Lima_PE_RMY_2007-2024.epw` |
| Brasilia | Brazil | Aw | 833780-99999 Presidente Juscelino Kubitschek | 2010-2024 (15 years) | NSRDB | `Brasilia_BR_RMY_2010-2024.epw` |
| Santo Domingo | Dominican Republic | Am | 784850-99999 De Las Americas Intl | 2009-2024 (15 years) | NSRDB | `Santo_Domingo_DO_RMY_2009-2024.epw` |
| Santiago | Chile | Csb | 855740-99999 Arturo Merino Benitez Intl | 2007-2022 (15 years) | PVGIS-ERA5 | `Santiago_CL_RMY_2007-2022.epw` |
| Montevideo | Uruguay | Cfa | 865800-99999 Carrasco Intl | 2008-2022 (15 years) | PVGIS-SARAH3 | `Montevideo_UY_RMY_2008-2022.epw` |
| Asuncion | Paraguay | Cfa | 862180-99999 Silvio Pettirossi Intl | 2008-2022 (15 years) | PVGIS-SARAH3 | `Asuncion_PY_RMY_2008-2022.epw` |

## US state capitals (20)

Twenty state capitals not already listed above, spread over ASHRAE climate zones 1A to 7, with NSRDB solar
radiation. Baton Rouge has 14 usable years (its station record starts in late 2010); the others have 15.

| City | Zone | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|
| Honolulu HI | 1A | 911820-22521 Honolulu International Airport | 2010-2024 (15 years) | NSRDB | `Honolulu_HI_RMY_2010-2024.epw` |
| Austin TX | 2A | 722540-13904 Austin-Bergstrom Intl Airport | 2010-2024 (15 years) | NSRDB | `Austin_TX_RMY_2010-2024.epw` |
| Baton Rouge LA | 2A | 722320-13970 Baton Rouge Metro Ryan Field Airport | 2011-2024 (14 years) | NSRDB | `Baton_Rouge_LA_RMY_2011-2024.epw` |
| Tallahassee FL | 2A | 722140-93805 Tallahassee Regional Airport | 2010-2024 (15 years) | NSRDB | `Tallahassee_FL_RMY_2010-2024.epw` |
| Oklahoma City OK | 3A | 723530-13967 Will Rogers World Airport | 2010-2024 (15 years) | NSRDB | `Oklahoma_City_OK_RMY_2010-2024.epw` |
| Sacramento CA | 3B | 724830-23232 Sacramento Executive Airport | 2010-2024 (15 years) | NSRDB | `Sacramento_CA_RMY_2010-2024.epw` |
| Nashville TN | 4A | 723270-13897 Nashville International Airport | 2010-2024 (15 years) | NSRDB | `Nashville_TN_RMY_2010-2024.epw` |
| Richmond VA | 4A | 724010-13740 Richmond International Airport | 2010-2024 (15 years) | NSRDB | `Richmond_VA_RMY_2010-2024.epw` |
| Salem OR | 4C | 726940-24232 Mcnary Field Airport | 2010-2024 (15 years) | NSRDB | `Salem_OR_RMY_2010-2024.epw` |
| Columbus OH | 5A | 724280-14821 Port Columbus International Airport | 2010-2024 (15 years) | NSRDB | `Columbus_OH_RMY_2010-2024.epw` |
| Indianapolis IN | 5A | 724380-93819 Indianapolis International Airport | 2010-2024 (15 years) | NSRDB | `Indianapolis_IN_RMY_2010-2024.epw` |
| Des Moines IA | 5A | 725460-14933 Des Moines International Airport | 2009-2024 (15 years) | NSRDB | `Des_Moines_IA_RMY_2009-2024.epw` |
| Albany NY | 5A | 725180-14735 Albany International Airport | 2009-2024 (15 years) | NSRDB | `Albany_NY_RMY_2009-2024.epw` |
| Salt Lake City UT | 5B | 725720-24127 Salt Lake City International Airport | 2010-2024 (15 years) | NSRDB | `Salt_Lake_City_UT_RMY_2010-2024.epw` |
| Boise ID | 5B | 726810-24131 Boise Air Terminal/Gowen Fd Airport | 2010-2024 (15 years) | NSRDB | `Boise_ID_RMY_2010-2024.epw` |
| Santa Fe NM | 5B | 723656-23049 Santa Fe Municipal Airport | 2010-2024 (15 years) | NSRDB | `Santa_Fe_NM_RMY_2010-2024.epw` |
| Madison WI | 6A | 726410-14837 Dane Co Regional-Truax Field Airport | 2010-2024 (15 years) | NSRDB | `Madison_WI_RMY_2010-2024.epw` |
| Bismarck ND | 6A | 727640-24011 Bismarck Municipal Airport | 2010-2024 (15 years) | NSRDB | `Bismarck_ND_RMY_2010-2024.epw` |
| Cheyenne WY | 6B | 725640-24018 Cheyenne Airport | 2010-2024 (15 years) | NSRDB | `Cheyenne_WY_RMY_2010-2024.epw` |
| Juneau AK | 7 | 703810-25309 Juneau International | 2010-2024 (15 years) | NSRDB | `Juneau_AK_RMY_2010-2024.epw` |

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

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Amsterdam | Netherlands | Cfb | 062400-99999 Schiphol | 2009-2023 (15 years) | PVGIS-SARAH3 | `Amsterdam_NL_RMY_2009-2023.epw` |
| Barcelona | Spain | Csa | 081810-99999 Barcelona | 2009-2023 (15 years) | PVGIS-SARAH3 | `Barcelona_ES_RMY_2009-2023.epw` |
| Belgrade | Serbia | Cfa | 132720-99999 Beograd | 2009-2023 (15 years) | PVGIS-SARAH3 | `Belgrade_RS_RMY_2009-2023.epw` |
| Bern | Switzerland | Cfb | 066310-99999 Bern-Zollikofen | 2009-2023 (15 years) | PVGIS-SARAH3 | `Bern_CH_RMY_2009-2023.epw` |
| Brussels | Belgium | Cfb | 064510-99999 Brussels Natl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Brussels_BE_RMY_2009-2023.epw` |
| Budapest | Hungary | Cfa | 128430-99999 Budapest/Pestszentlorinc | 2009-2023 (15 years) | PVGIS-SARAH3 | `Budapest_HU_RMY_2009-2023.epw` |
| Copenhagen | Denmark | Cfb | 061800-99999 Kastrup | 2009-2023 (15 years) | PVGIS-SARAH3 | `Copenhagen_DK_RMY_2009-2023.epw` |
| Hamburg | Germany | Cfb | 101470-99999 Hamburg | 2009-2023 (15 years) | PVGIS-SARAH3 | `Hamburg_DE_RMY_2009-2023.epw` |
| Istanbul | Turkey | Csa | 170600-99999 Ataturk | 2009-2023 (15 years) | PVGIS-SARAH3 | `Istanbul_TR_RMY_2009-2023.epw` |
| Milan | Italy | Cfa | 160800-99999 Milano Linate | 2009-2023 (15 years) | PVGIS-SARAH3 | `Milan_IT_RMY_2009-2023.epw` |
| Prague | Czechia | Cfb | 115180-99999 Ruzyne | 2009-2023 (15 years) | PVGIS-SARAH3 | `Prague_CZ_RMY_2009-2023.epw` |
| Riga | Latvia | Dfb | 264225-99999 Riga | 2009-2023 (15 years) | PVGIS-SARAH3 | `Riga_LV_RMY_2009-2023.epw` |
| Saint Petersburg | Russia | Dfb | 260630-99999 Pulkovo | 2009-2023 (15 years) | PVGIS-SARAH3 | `Saint_Petersburg_RU_RMY_2009-2023.epw` |
| Sofia | Bulgaria | Cfb | 156140-99999 Sofia | 2009-2023 (15 years) | PVGIS-SARAH3 | `Sofia_BG_RMY_2009-2023.epw` |
| Tirana | Albania | Csa | 136150-99999 Tirana Rinas | 2009-2023 (15 years) | PVGIS-SARAH3 | `Tirana_AL_RMY_2009-2023.epw` |
| Vilnius | Lithuania | Dfb | 267300-99999 Vilnius Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Vilnius_LT_RMY_2009-2023.epw` |
| Zagreb | Croatia | Cfb | 142400-99999 Zagreb\Maksimir | 2009-2023 (15 years) | PVGIS-SARAH3 | `Zagreb_HR_RMY_2009-2023.epw` |

### Middle East and Central Asia (13)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Abu Dhabi | United Arab Emirates | BWh | 412160-99999 Al Bateen Executive | 2010-2023 (14 years) | PVGIS-SARAH3 | `Abu_Dhabi_AE_RMY_2010-2023.epw` |
| Ashgabat | Turkmenistan | BWk | 388800-99999 Ashgabat | 2009-2023 (15 years) | PVGIS-ERA5 | `Ashgabat_TM_RMY_2009-2023.epw` |
| Baghdad | Iraq | BWh | 406500-99999 Baghdad Intl Airport | 2011-2023 (13 years) | PVGIS-SARAH3 | `Baghdad_IQ_RMY_2011-2023.epw` |
| Beirut | Lebanon | Csa | 401000-99999 Rafic Hariri Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Beirut_LB_RMY_2009-2023.epw` |
| Bishkek | Kyrgyzstan | Dsa | 383531-99999 Manas | 2008-2023 (15 years) | PVGIS-ERA5 | `Bishkek_KG_RMY_2008-2023.epw` |
| Dushanbe | Tajikistan | Csa | 388360-99999 Dushanbe | 2009-2023 (15 years) | PVGIS-ERA5 | `Dushanbe_TJ_RMY_2009-2023.epw` |
| Isfahan | Iran | BWk | 408000-99999 Esfahan Shahid Beheshti Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Isfahan_IR_RMY_2009-2023.epw` |
| Izmir | Turkey | Csa | 172190-99999 Adnan Menderes | 2009-2023 (15 years) | PVGIS-SARAH3 | `Izmir_TR_RMY_2009-2023.epw` |
| Jeddah | Saudi Arabia | BWh | 410240-99999 King Abdulaziz Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Jeddah_SA_RMY_2009-2023.epw` |
| Manama | Bahrain | BWh | 411500-99999 Bahrain Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Manama_BH_RMY_2009-2023.epw` |
| Mashhad | Iran | BSk | 407450-99999 Mashhad Intl / Shahid Hashemi Nejad Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Mashhad_IR_RMY_2009-2023.epw` |
| Tel Aviv | Israel | Csa | 401800-99999 Ben Gurion | 2009-2023 (15 years) | PVGIS-SARAH3 | `Tel_Aviv_IL_RMY_2009-2023.epw` |
| Yerevan | Armenia | Dsa | 353946-99999 Zvartnots | 2009-2023 (15 years) | PVGIS-SARAH3 | `Yerevan_AM_RMY_2009-2023.epw` |

### Africa (13)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Abidjan | Cote d'Ivoire | Aw | 655780-99999 Abidjan Felix Houphouet Boigny Intl | 2008-2023 (15 years) | PVGIS-SARAH3 | `Abidjan_CI_RMY_2008-2023.epw` |
| Alexandria | Egypt | BWh | 623180-99999 Alexandria Intl | 2008-2022 (15 years) | PVGIS-SARAH3 | `Alexandria_EG_RMY_2008-2022.epw` |
| Bamako | Mali | BSh | 612910-99999 Senou / President Modibo Keita | 2009-2023 (15 years) | PVGIS-SARAH3 | `Bamako_ML_RMY_2009-2023.epw` |
| Dar es Salaam | Tanzania | Aw | 638940-99999 Mwalimu Julius K Nyerere Intl | 2007-2023 (15 years) | PVGIS-SARAH3 | `Dar_es_Salaam_TZ_RMY_2007-2023.epw` |
| Gaborone | Botswana | BSh | 682400-99999 Sir Seretse Khama Intl | 2008-2023 (15 years) | PVGIS-SARAH3 | `Gaborone_BW_RMY_2008-2023.epw` |
| Khartoum | Sudan | BWh | 627210-99999 Khartoum | 2008-2020 (13 years) | PVGIS-SARAH3 | `Khartoum_SD_RMY_2008-2020.epw` |
| Kigali | Rwanda | Aw | 643870-99999 Kigali Intl | 2010-2023 (12 years) | PVGIS-SARAH3 | `Kigali_RW_RMY_2010-2023.epw` |
| Libreville | Gabon | Am | 645000-99999 Leon M Ba | 2009-2023 (15 years) | PVGIS-SARAH3 | `Libreville_GA_RMY_2009-2023.epw` |
| Lomé | Togo | Aw | 653870-99999 Gnassingbe Eyadema Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Lome_TG_RMY_2009-2023.epw` |
| Mombasa | Kenya | As | 638200-99999 Mombasa Moi Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Mombasa_KE_RMY_2009-2023.epw` |
| Niamey | Niger | BWh | 610520-99999 Diori Hamani | 2009-2023 (15 years) | PVGIS-SARAH3 | `Niamey_NE_RMY_2009-2023.epw` |
| Ouagadougou | Burkina Faso | BSh | 655030-99999 Ouagadougou | 2009-2023 (15 years) | PVGIS-SARAH3 | `Ouagadougou_BF_RMY_2009-2023.epw` |
| Port Louis | Mauritius | Am | 619900-99999 Sir Seewoosagur Ramgoolam Intl | 2009-2023 (15 years) | PVGIS-SARAH3 | `Port_Louis_MU_RMY_2009-2023.epw` |

### South Asia (8)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Ahmedabad | India | BSh | 426470-99999 Ahmedabad / Sardar Vallabhbhai Patel Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Ahmedabad_IN_RMY_2009-2023.epw` |
| Bengaluru | India | Aw | 427056-99999 Bangaluru Intl Airport / Kempegowda Intl | 2009-2023 (12 years) | PVGIS-ERA5 | `Bengaluru_IN_RMY_2009-2023.epw` |
| Chennai | India | As | 432790-99999 Chennai Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Chennai_IN_RMY_2009-2023.epw` |
| Hyderabad | India | BSh | 431280-99999 Begumpet Airport / Hyderabad Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Hyderabad_IN_RMY_2009-2023.epw` |
| Kolkata | India | Aw | 428090-99999 Netaji Subhash Chandra Bose Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Kolkata_IN_RMY_2009-2023.epw` |
| Lahore | Pakistan | BSh | 416410-99999 Allama Iqbal Intl | 2006-2018 (12 years) | PVGIS-ERA5 | `Lahore_PK_RMY_2006-2018.epw` |
| Malé | Maldives | Am | 435550-99999 Male Intl / Velana Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Male_MV_RMY_2009-2023.epw` |
| Mumbai | India | Am | 430030-99999 Chhatrapati Shivaji Maharaj Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Mumbai_IN_RMY_2009-2023.epw` |

### East Asia (14)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Busan | South Korea | Cfa | 471530-99999 Gimhae Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Busan_KR_RMY_2009-2023.epw` |
| Chengdu | China | Cwa | 562940-99999 Shuangliu | 2009-2023 (15 years) | PVGIS-ERA5 | `Chengdu_CN_RMY_2009-2023.epw` |
| Chongqing | China | Cfa | 575160-99999 Jiangbei | 2009-2023 (15 years) | PVGIS-ERA5 | `Chongqing_CN_RMY_2009-2023.epw` |
| Fukuoka | Japan | Cfa | 478080-99999 Fukuoka | 2009-2023 (15 years) | PVGIS-ERA5 | `Fukuoka_JP_RMY_2009-2023.epw` |
| Guangzhou | China | Cfa | 592870-99999 Baiyun Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Guangzhou_CN_RMY_2009-2023.epw` |
| Harbin | China | Dwa | 541611-99999 Taiping | 2009-2023 (15 years) | PVGIS-ERA5 | `Harbin_CN_RMY_2009-2023.epw` |
| Hong Kong | Hong Kong | Cwa | 450070-99999 Hong Kong Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Hong_Kong_HK_RMY_2009-2023.epw` |
| Osaka | Japan | Cfa | 477720-99999 Osaka | 2009-2023 (15 years) | PVGIS-ERA5 | `Osaka_JP_RMY_2009-2023.epw` |
| Sapporo | Japan | Dfa | 474120-99999 Sapporo | 2009-2023 (15 years) | PVGIS-ERA5 | `Sapporo_JP_RMY_2009-2023.epw` |
| Shanghai | China | Cfa | 583670-99999 Hongqiao Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Shanghai_CN_RMY_2009-2023.epw` |
| Shenzhen | China | Cwa | 594930-99999 Baoan Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Shenzhen_CN_RMY_2009-2023.epw` |
| Urumqi | China | BSk | 514635-99999 Diwopu | 2009-2023 (15 years) | PVGIS-ERA5 | `Urumqi_CN_RMY_2009-2023.epw` |
| Wuhan | China | Cfa | 574940-99999 Tianhe | 2009-2023 (15 years) | PVGIS-ERA5 | `Wuhan_CN_RMY_2009-2023.epw` |
| Xi'an | China | BSk | 570360-99999 Xianyang | 2009-2023 (15 years) | PVGIS-ERA5 | `Xian_CN_RMY_2009-2023.epw` |

### Southeast Asia and Oceania (10)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Adelaide | Australia | Csa | 946720-99999 Adelaide Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Adelaide_AU_RMY_2009-2023.epw` |
| Auckland | New Zealand | Cfb | 931190-99999 Auckland Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Auckland_NZ_RMY_2009-2023.epw` |
| Bandar Seri Begawan | Brunei | Af | 963150-99999 Brunei Intl | 2008-2023 (15 years) | PVGIS-ERA5 | `Bandar_Seri_Begawan_BN_RMY_2008-2023.epw` |
| Brisbane | Australia | Cfa | 945780-99999 Brisbane Airport | 2009-2023 (15 years) | PVGIS-ERA5 | `Brisbane_AU_RMY_2009-2023.epw` |
| Cebu | Philippines | Am | 986460-99999 Mactan Cebu Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Cebu_PH_RMY_2009-2023.epw` |
| Darwin | Australia | Aw | 941200-99999 Darwin Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Darwin_AU_RMY_2009-2023.epw` |
| Ho Chi Minh City | Vietnam | Aw | 489000-99999 Tan Son Nhat Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Ho_Chi_Minh_City_VN_RMY_2009-2023.epw` |
| Melbourne | Australia | Cfb | 948660-99999 Melbourne Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Melbourne_AU_RMY_2009-2023.epw` |
| Perth | Australia | Csa | 946100-99999 Perth Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Perth_AU_RMY_2009-2023.epw` |
| Suva | Fiji | Af | 916830-99999 Nausori Intl | 2009-2023 (15 years) | PVGIS-ERA5 | `Suva_FJ_RMY_2009-2023.epw` |

### Americas (17)

| City | Country | Climate (Köppen) | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|---|
| Calgary | Canada | Dfb | 718770-99999 Calgary Intl | 2009-2024 (15 years) | NSRDB | `Calgary_CA_RMY_2009-2024.epw` |
| Georgetown | Guyana | Af | 810020-10502 Georgetown | 2010-2024 (15 years) | NSRDB | `Georgetown_GY_RMY_2010-2024.epw` |
| Guadalajara | Mexico | Cwa | 766133-99999 Don Miguel Hidalgo Y Costilla Intl / Guadalajara Intl | 2010-2024 (15 years) | NSRDB | `Guadalajara_MX_RMY_2010-2024.epw` |
| Kingston | Jamaica | Aw | 783970-99999 Norman Manley Intl | 2010-2024 (15 years) | NSRDB | `Kingston_JM_RMY_2010-2024.epw` |
| Managua | Nicaragua | Aw | 787410-99999 Managua Intl / Augusto Cesar Sandino Intl | 2009-2024 (15 years) | NSRDB | `Managua_NI_RMY_2009-2024.epw` |
| Manaus | Brazil | Af | 823320-99999 Campo Ponta Pelada | 2010-2024 (15 years) | NSRDB | `Manaus_BR_RMY_2010-2024.epw` |
| Monterrey | Mexico | BSh | 763943-99999 General Mariano Escobedo Intl / Monterrey Intl | 2008-2024 (15 years) | NSRDB | `Monterrey_MX_RMY_2008-2024.epw` |
| Montreal | Canada | Dfb | 713710-99999 Montreal/St Hubert | 2010-2024 (15 years) | NSRDB | `Montreal_CA_RMY_2010-2024.epw` |
| Port of Spain | Trinidad and Tobago | Aw | 789700-11634 Piarco Intl Ap | 2008-2024 (15 years) | NSRDB | `Port_of_Spain_TT_RMY_2008-2024.epw` |
| Porto Alegre | Brazil | Cfa | 839710-99999 Salgado Filho | 2008-2022 (15 years) | PVGIS-SARAH3 | `Porto_Alegre_BR_RMY_2008-2022.epw` |
| Recife | Brazil | As | 828990-99999 Guararapes Gilberto Freyre Intl | 2010-2024 (15 years) | NSRDB | `Recife_BR_RMY_2010-2024.epw` |
| Rio de Janeiro | Brazil | Aw | 837460-99999 Galeao Antonio Carlos Jobim | 2008-2022 (15 years) | PVGIS-SARAH3 | `Rio_de_Janeiro_BR_RMY_2008-2022.epw` |
| Salvador | Brazil | Af | 832480-99999 Deputado Luis Eduardo Magalhaes | 2010-2024 (15 years) | NSRDB | `Salvador_BR_RMY_2010-2024.epw` |
| Tegucigalpa | Honduras | Aw | 787200-99999 Toncontin Intl | 2007-2024 (14 years) | NSRDB | `Tegucigalpa_HN_RMY_2007-2024.epw` |
| Toronto | Canada | Dfa | 712650-99999 Toronto City Centre / Billy Bishop | 2010-2024 (15 years) | NSRDB | `Toronto_CA_RMY_2010-2024.epw` |
| Vancouver | Canada | Cfb | 718920-99999 Vancouver Intl | 2010-2024 (15 years) | NSRDB | `Vancouver_CA_RMY_2010-2024.epw` |
| Córdoba | Argentina | Cwa | 873440-99999 Ambrosio L V Taravella | 2006-2022 (15 years) | PVGIS-ERA5 | `Cordoba_AR_RMY_2006-2022.epw` |

### United States (8)

Large US cities not already listed, with NSRDB solar radiation and their ASHRAE 169 climate zone.

| City | Zone | NOAA station | AMY years | Solar | File |
|---|---|---|---|---|---|
| New Orleans LA | 2A | 722315-53917 Lakefront Airport | 2009-2024 (15 years) | NSRDB | `New_Orleans_LA_RMY_2009-2024.epw` |
| San Antonio TX | 2A | 722523-12970 Stinson Minicipal Airport | 2010-2024 (15 years) | NSRDB | `San_Antonio_TX_RMY_2010-2024.epw` |
| Dallas TX | 3A | 722580-13960 Dallas Love Field Airport | 2011-2024 (14 years) | NSRDB | `Dallas_TX_RMY_2011-2024.epw` |
| San Diego CA | 3B | 722900-23188 San Diego International Airport | 2010-2024 (15 years) | NSRDB | `San_Diego_CA_RMY_2010-2024.epw` |
| Kansas City MO | 4A | 724463-13988 C.R. Wheeler Downtown Arpt | 2010-2024 (15 years) | NSRDB | `Kansas_City_MO_RMY_2010-2024.epw` |
| Philadelphia PA | 4A | 724080-13739 Philadelphia International Airport | 2010-2024 (15 years) | NSRDB | `Philadelphia_PA_RMY_2010-2024.epw` |
| St. Louis MO | 4A | 725314-03960 St Louis Downtown Airport | 2010-2024 (15 years) | NSRDB | `St_Louis_MO_RMY_2010-2024.epw` |
| Detroit MI | 5A | 725375-14822 Detroit City Airport | 2008-2024 (15 years) | NSRDB | `Detroit_MI_RMY_2008-2024.epw` |

## More US major cities (100)

Built in November 2026 with the current code of this repository (`amy_sourcing`, `tmy`, `rmy`), as in
`examples/RMY_End_to_End_Colab.ipynb`: the 100 largest US metro areas not listed above, each from its main airport
station, with NSRDB solar radiation. Candidates were screened for NOAA records covering 2010-2024 with near-hourly
reports in every year. Where a year was too incomplete to fill, the builder replaced it with an earlier one (Fort Myers,
Lakeland, Salinas); every city has 15 usable years. FRMYs have not been made for these cities.

Checks on all 100 (per-city results in the full packages, `QA_summary.csv`): 15 complete AMYs and no high-risk or
sparse year used; solar timing of the TMY and RMY within 1.2 min of solar noon; no missing solar, longwave or sky-cover
hours; relative humidity consistent with dry bulb and dew point; no dew point above dry bulb; every hourly temperature
step over 10 °C lies inside observed weather, none at an inserted block. ERA5 filled long gaps at 54 stations, at most
377 hours of dry-bulb temperature over 15 years (Tucson).

| City | NOAA station | AMY years | Solar | Peak heat / cold year | File |
|---|---|---|---|---|---|
| Birmingham AL | 722280-13876 Birmingham International Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Birmingham_AL_RMY_2010-2024.epw` |
| Huntsville AL | 723230-03856 Huntsville Intl/C.T.Jones Field Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2010 | `Huntsville_AL_RMY_2010-2024.epw` |
| Mobile AL | 722230-13894 Mobile Regional Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Mobile_AL_RMY_2010-2024.epw` |
| Montgomery AL | 722260-13895 Montgomery Rgnl (Dannelly Fd) Ap | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Montgomery_AL_RMY_2010-2024.epw` |
| Fayetteville AR | 723436-53922 Nw Arkansas Regional Arpt | 2010-2024 (15 years) | NSRDB | 2011 / 2014 | `Fayetteville_AR_RMY_2010-2024.epw` |
| Little Rock AR | 723403-13963 Adams Field Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Little_Rock_AR_RMY_2010-2024.epw` |
| Tucson AZ | 722740-23160 Tucson International Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2013 | `Tucson_AZ_RMY_2010-2024.epw` |
| Bakersfield CA | 723840-23155 Meadows Field Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Bakersfield_CA_RMY_2010-2024.epw` |
| Fresno CA | 723890-93193 Fresno Yosemite International Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Fresno_CA_RMY_2010-2024.epw` |
| Modesto CA | 724926-23258 Mdsto Cty-Co H Sham Fd Apt | 2010-2024 (15 years) | NSRDB | 2017 / 2013 | `Modesto_CA_RMY_2010-2024.epw` |
| Oxnard CA | 723927-93110 Oxnard Airport | 2010-2024 (15 years) | NSRDB | 2017 / 2013 | `Oxnard_CA_RMY_2010-2024.epw` |
| Riverside-San Bernardino CA | 747040-03102 Ontario International Arpt | 2010-2024 (15 years) | NSRDB | 2016 / 2011 | `Riverside_San_Bernardino_CA_RMY_2010-2024.epw` |
| Salinas CA | 725930-23233 Salinas Municipal Airport | 2009-2024 (15 years) | NSRDB | 2017 / 2013 | `Salinas_CA_RMY_2009-2024.epw` |
| San Jose CA | 724945-23293 N Y. Mineta Sn Jo Intl Apt | 2010-2024 (15 years) | NSRDB | 2017 / 2013 | `San_Jose_CA_RMY_2010-2024.epw` |
| Santa Rosa CA | 724957-23213 Sonoma County Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `Santa_Rosa_CA_RMY_2010-2024.epw` |
| Stockton CA | 724920-23237 Stockton Metropolitan Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Stockton_CA_RMY_2010-2024.epw` |
| Colorado Springs CO | 724660-93037 City Of Colorado Springs Municipal Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Colorado_Springs_CO_RMY_2010-2024.epw` |
| Bridgeport CT | 725040-94702 Igor I Sikorsky Memorial Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Bridgeport_CT_RMY_2010-2024.epw` |
| Hartford CT | 725080-14740 Bradley International Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2015 | `Hartford_CT_RMY_2010-2024.epw` |
| New Haven CT | 725045-14758 Tweed-New Haven Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `New_Haven_CT_RMY_2010-2024.epw` |
| Washington DC | 724050-13743 Ronald Reagan Washington Natl Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2015 | `Washington_DC_RMY_2010-2024.epw` |
| Daytona Beach FL | 747870-12834 Daytona Beach International Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2010 | `Daytona_Beach_FL_RMY_2010-2024.epw` |
| Fort Myers FL | 722108-12894 Sw Florida Intn Airport | 2009-2023 (15 years) | NSRDB | 2019 / 2010 | `Fort_Myers_FL_RMY_2009-2023.epw` |
| Gainesville FL | 747560-12816 Gainesville Regional Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Gainesville_FL_RMY_2010-2024.epw` |
| Jacksonville FL | 722060-13889 Jacksonville International Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2010 | `Jacksonville_FL_RMY_2010-2024.epw` |
| Lakeland FL | 722119-12883 Lakeland Linder Regional Airport | 2008-2024 (15 years) | NSRDB | 2011 / 2018 | `Lakeland_FL_RMY_2008-2024.epw` |
| Melbourne FL | 722040-12838 Melbourne Intl Ap | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Melbourne_FL_RMY_2010-2024.epw` |
| Orlando FL | 722050-12815 Orlando International Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2010 | `Orlando_FL_RMY_2010-2024.epw` |
| Pensacola FL | 722223-13899 Pensacola Regional Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Pensacola_FL_RMY_2010-2024.epw` |
| Sarasota FL | 722115-12871 Sarasota/Bradenton Intl Ap | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Sarasota_FL_RMY_2010-2024.epw` |
| Tampa FL | 722110-12842 Tampa International Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Tampa_FL_RMY_2010-2024.epw` |
| Augusta GA | 722180-03820 Augusta Regional At Bush Field Ap | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Augusta_GA_RMY_2010-2024.epw` |
| Savannah GA | 722070-03822 Savannah/Hilton Head Intl Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Savannah_GA_RMY_2010-2024.epw` |
| Peoria IL | 725320-14842 Greater Peoria Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Peoria_IL_RMY_2010-2024.epw` |
| Rockford IL | 725430-94822 Greater Rockford Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Rockford_IL_RMY_2010-2024.epw` |
| Fort Wayne IN | 725330-14827 Fort Wayne International Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Fort_Wayne_IN_RMY_2010-2024.epw` |
| Wichita KS | 724500-03928 Wichita Eisenhower National | 2010-2024 (15 years) | NSRDB | 2011 / 2014 | `Wichita_KS_RMY_2010-2024.epw` |
| Lexington KY | 724220-93820 Blue Grass Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Lexington_KY_RMY_2010-2024.epw` |
| Louisville KY | 724230-93821 Louisville Intl-Standiford Field Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Louisville_KY_RMY_2010-2024.epw` |
| Lafayette LA | 722405-13976 Lafayette Regional Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Lafayette_LA_RMY_2010-2024.epw` |
| Shreveport LA | 722480-13957 Shreveport Regional Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Shreveport_LA_RMY_2010-2024.epw` |
| Worcester MA | 725100-94746 Worcester Regional Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2015 | `Worcester_MA_RMY_2010-2024.epw` |
| Baltimore MD | 724060-93721 Baltimore-Washington Intl Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Baltimore_MD_RMY_2010-2024.epw` |
| Portland ME | 726060-14764 Portland International Jetport Ap | 2010-2024 (15 years) | NSRDB | 2020 / 2015 | `Portland_ME_RMY_2010-2024.epw` |
| Grand Rapids MI | 726350-94860 Gerald R Ford International Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Grand_Rapids_MI_RMY_2010-2024.epw` |
| Lansing MI | 725390-14836 Capital City Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Lansing_MI_RMY_2010-2024.epw` |
| Springfield MO | 724400-13995 Springfield-Branson Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Springfield_MO_RMY_2010-2024.epw` |
| Jackson MS | 722350-03940 Jackson International Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Jackson_MS_RMY_2010-2024.epw` |
| Billings MT | 726770-24033 Billings Logan International Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2022 | `Billings_MT_RMY_2010-2024.epw` |
| Great Falls MT | 727750-24143 Great Falls International Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2019 | `Great_Falls_MT_RMY_2010-2024.epw` |
| Asheville NC | 723150-03812 Asheville Regional Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Asheville_NC_RMY_2010-2024.epw` |
| Charlotte NC | 723140-13881 Charlotte/Douglas International Ap | 2010-2024 (15 years) | NSRDB | 2015 / 2010 | `Charlotte_NC_RMY_2010-2024.epw` |
| Greensboro NC | 723170-13723 Piedmont Triad International Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2010 | `Greensboro_NC_RMY_2010-2024.epw` |
| Raleigh NC | 723060-13722 Raleigh-Durham International Ap | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Raleigh_NC_RMY_2010-2024.epw` |
| Wilmington NC | 723020-13748 Wilmington International Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Wilmington_NC_RMY_2010-2024.epw` |
| Winston-Salem NC | 723193-93807 Smith Reynolds Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2010 | `Winston_Salem_NC_RMY_2010-2024.epw` |
| Fargo ND | 727530-14914 Hector International Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2022 | `Fargo_ND_RMY_2010-2024.epw` |
| Lincoln NE | 725510-14939 Lincoln Municipal Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Lincoln_NE_RMY_2010-2024.epw` |
| Omaha NE | 725500-14942 Eppley Airfield Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Omaha_NE_RMY_2010-2024.epw` |
| Reno NV | 724880-23185 Reno/Tahoe International Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Reno_NV_RMY_2010-2024.epw` |
| Buffalo NY | 725280-14733 Buffalo Niagara International Airpor | 2010-2024 (15 years) | NSRDB | 2020 / 2015 | `Buffalo_NY_RMY_2010-2024.epw` |
| Rochester NY | 725290-14768 Greater Rochester International Ap | 2010-2024 (15 years) | NSRDB | 2016 / 2015 | `Rochester_NY_RMY_2010-2024.epw` |
| Syracuse NY | 725190-14771 Syracuse Hancock International Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2015 | `Syracuse_NY_RMY_2010-2024.epw` |
| Akron OH | 725210-14895 Akron-Canton Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Akron_OH_RMY_2010-2024.epw` |
| Cincinnati OH | 724210-93814 Cincinnati/Northern Kentucky Intl Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Cincinnati_OH_RMY_2010-2024.epw` |
| Cleveland OH | 725240-14820 Cleveland-Hopkins International Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2015 | `Cleveland_OH_RMY_2010-2024.epw` |
| Dayton OH | 724290-93815 J.M.Cox Dayton International Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Dayton_OH_RMY_2010-2024.epw` |
| Toledo OH | 725360-94830 Toledo Express Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2014 | `Toledo_OH_RMY_2010-2024.epw` |
| Tulsa OK | 723560-13968 Tulsa International Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2014 | `Tulsa_OK_RMY_2010-2024.epw` |
| Eugene OR | 726930-24221 Mahlon Sweet Field Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2013 | `Eugene_OR_RMY_2010-2024.epw` |
| Medford OR | 725970-24225 Rogue Valley Intl-Medford Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2013 | `Medford_OR_RMY_2010-2024.epw` |
| Allentown PA | 725170-14737 Lehigh Valley International Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2015 | `Allentown_PA_RMY_2010-2024.epw` |
| Harrisburg PA | 723990-14711 Harrisburg International Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2015 | `Harrisburg_PA_RMY_2010-2024.epw` |
| Lancaster PA | 725116-54737 Lancaster Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2014 | `Lancaster_PA_RMY_2010-2024.epw` |
| Pittsburgh PA | 725200-94823 Pittsburgh International Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2014 | `Pittsburgh_PA_RMY_2010-2024.epw` |
| Scranton PA | 725130-14777 Wilkes-Barre/Scranton Intl Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2014 | `Scranton_PA_RMY_2010-2024.epw` |
| Providence RI | 725070-14765 Theodore F Green State Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2015 | `Providence_RI_RMY_2010-2024.epw` |
| Charleston SC | 722080-13880 Charleston Afb/International Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2010 | `Charleston_SC_RMY_2010-2024.epw` |
| Columbia SC | 723100-13883 Columbia Metropolitan Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2010 | `Columbia_SC_RMY_2010-2024.epw` |
| Greenville SC | 723120-03870 Greenville-Spartanburg Intl Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Greenville_SC_RMY_2010-2024.epw` |
| Sioux Falls SD | 726510-14944 Joe Foss Field Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Sioux_Falls_SD_RMY_2010-2024.epw` |
| Chattanooga TN | 723240-13882 Lovell Field Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2010 | `Chattanooga_TN_RMY_2010-2024.epw` |
| Knoxville TN | 723260-13891 Mc Ghee Tyson Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Knoxville_TN_RMY_2010-2024.epw` |
| Memphis TN | 723340-13893 Memphis International Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2014 | `Memphis_TN_RMY_2010-2024.epw` |
| Amarillo TX | 723630-23047 Amarillo Rick Husband Intl Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2011 | `Amarillo_TX_RMY_2010-2024.epw` |
| Brownsville TX | 722500-12919 Brownsville/S Padre Island Intl Ap | 2010-2024 (15 years) | NSRDB | 2023 / 2014 | `Brownsville_TX_RMY_2010-2024.epw` |
| Corpus Christi TX | 722510-12924 Corpus Christi International Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2022 | `Corpus_Christi_TX_RMY_2010-2024.epw` |
| El Paso TX | 722700-23044 El Paso International Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2011 | `El_Paso_TX_RMY_2010-2024.epw` |
| Laredo TX | 722520-12907 Laredo International Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2022 | `Laredo_TX_RMY_2010-2024.epw` |
| Lubbock TX | 722670-23042 Lubbock International Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2022 | `Lubbock_TX_RMY_2010-2024.epw` |
| McAllen TX | 722506-12959 Mc Allen Miller Intl Arpt | 2010-2024 (15 years) | NSRDB | 2023 / 2022 | `McAllen_TX_RMY_2010-2024.epw` |
| Midland TX | 722650-23023 Midland International Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2011 | `Midland_TX_RMY_2010-2024.epw` |
| Waco TX | 722560-13959 Waco Regional Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2022 | `Waco_TX_RMY_2010-2024.epw` |
| Roanoke VA | 724110-13741 Roanoke Regional/Woodrum Field Ap | 2010-2024 (15 years) | NSRDB | 2024 / 2014 | `Roanoke_VA_RMY_2010-2024.epw` |
| Virginia Beach-Norfolk VA | 723080-13737 Norfolk International Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2010 | `Virginia_Beach_Norfolk_VA_RMY_2010-2024.epw` |
| Burlington VT | 726170-14742 Burlington International Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2015 | `Burlington_VT_RMY_2010-2024.epw` |
| Spokane WA | 727850-24157 Spokane International Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2017 | `Spokane_WA_RMY_2010-2024.epw` |
| Green Bay WI | 726450-14898 Austin Straubel International Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Green_Bay_WI_RMY_2010-2024.epw` |
| Milwaukee WI | 726400-14839 General Mitchell International Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Milwaukee_WI_RMY_2010-2024.epw` |
| Charleston WV | 724140-13866 Yeager Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2014 | `Charleston_WV_RMY_2010-2024.epw` |
## More US cities (184)

Built in November 2026 with the current code of this repository (`amy_sourcing`, `tmy`, `rmy`), as in
`examples/RMY_End_to_End_Colab.ipynb`: the most populous US cities not listed above (GeoNames populations, down to
71,970), each from the nearest NOAA station within 25 km that no other city here uses, with NSRDB solar radiation.
Candidates were screened for NOAA records covering 2010-2024 with near-hourly reports in every year. Cities were built
in population order (210 in all) and the 184 most populous that passed every check were kept. Left out: Chula Vista
CA, Arlington VA, Rancho Cucamonga CA, Killeen TX, Roseville CA, Chico CA, Round Rock TX, North Charleston SC, Daly
City CA, Wichita Falls TX, Edinburg TX, Cranston RI, Springdale AR, Auburn WA, Brick NJ, Missouri City TX, Jonesboro
AR and Temple TX (fewer than 15 usable years); Riverside CA, St. Petersburg FL, Oceanside CA, Fort Collins CO,
Hollywood FL, Bellevue WA, Sterling Heights MI and Troy MI (single-hour dry-bulb readings 11 to 28 °C away from the
hours around them, e.g. -35.8 °C at Sterling Heights in December, which the AMY builder keeps as observed). Virginia
Beach, Norfolk and San Bernardino use their own stations (Oceana NAS, Norfolk NAS, March ARB), not those of the
metro-area files above. Where a year was too incomplete to fill, the builder replaced it with an earlier one (39
cities, e.g. New York City, Central Park, 2005-2024); every city has 15 usable years. FRMYs have not been made for
these cities.

Checks on all 184 (per-city results in the full packages, `QA_summary.csv`): 15 complete AMYs and no high-risk year
used (one sparse year each at Elizabeth NJ, Appleton WI and Waukesha WI, kept out of event detection and TMY
selection); solar timing of the TMY and RMY within 1.7 min of solar noon; no missing solar, longwave or sky-cover
hours; relative humidity consistent with dry bulb and dew point; no dew point above dry bulb; calm hours with
direction 0; no hourly dry-bulb value more than 10 °C from its 9-hour running median except two fronts (Edmond OK, 1
January 2022; Kenosha WI, a lake-breeze front in April 2014). ERA5 filled long gaps at 152 stations, at most 1,714
hours of dry-bulb temperature over 15 years (Bend OR).

| City | NOAA station | AMY years | Solar | Peak heat / cold year | File |
|---|---|---|---|---|---|
| Tuscaloosa AL | 722286-93806 Tuscaloosa Municipal Arpt | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Tuscaloosa_AL_RMY_2010-2024.epw` |
| Fort Smith AR | 723440-13964 Fort Smith Regional Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2014 | `Fort_Smith_AR_RMY_2010-2024.epw` |
| Glendale AZ | 722785-23111 Luke Afb Airport | 2009-2024 (15 years) | NSRDB | 2023 / 2011 | `Glendale_AZ_RMY_2009-2024.epw` |
| Mesa AZ | 722786-23104 Williams Gateway Airport | 2009-2024 (15 years) | NSRDB | 2020 / 2013 | `Mesa_AZ_RMY_2009-2024.epw` |
| Peoria AZ | 722784-03184 Phoenix Deer Valley Arpt | 2010-2024 (15 years) | NSRDB | 2023 / 2013 | `Peoria_AZ_RMY_2010-2024.epw` |
| Scottsdale AZ | 722789-03192 Scottsdale Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `Scottsdale_AZ_RMY_2010-2024.epw` |
| Yuma AZ | 740035-03145 Yuma Mcas | 2010-2024 (15 years) | NSRDB | 2023 / 2011 | `Yuma_AZ_RMY_2010-2024.epw` |
| Anaheim CA | 722976-03166 Fullerton Municipal Arpt | 2010-2024 (15 years) | NSRDB | 2012 / 2011 | `Anaheim_CA_RMY_2010-2024.epw` |
| Carlsbad CA | 722926-03154 Marine Corps Air Station | 2007-2024 (15 years) | NSRDB | 2020 / 2013 | `Carlsbad_CA_RMY_2007-2024.epw` |
| El Cajon CA | 722903-03131 Montgomery Field Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `El_Cajon_CA_RMY_2010-2024.epw` |
| Escondido CA | 722927-03177 Mcclellan-Palomar Airport | 2009-2024 (15 years) | NSRDB | 2022 / 2013 | `Escondido_CA_RMY_2009-2024.epw` |
| Fairfield CA | 745160-23202 Travis Air Force Base | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Fairfield_CA_RMY_2010-2024.epw` |
| Fontana CA | 722899-03179 Chino Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `Fontana_CA_RMY_2010-2024.epw` |
| Fremont CA | 724927-23285 Livermore Municipal Arpt | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Fremont_CA_RMY_2010-2024.epw` |
| Glendale CA | 722880-23152 Burbank-Glendale-Pasa Arpt | 2010-2024 (15 years) | NSRDB | 2016 / 2011 | `Glendale_CA_RMY_2010-2024.epw` |
| Huntington Beach CA | 722975-53141 Los Alamitos Aaf Airport | 2009-2024 (15 years) | NSRDB | 2016 / 2013 | `Huntington_Beach_CA_RMY_2009-2024.epw` |
| Indio CA | 747187-03104 Desert Resorts Rgnl Arpt | 2009-2024 (15 years) | NSRDB | 2019 / 2013 | `Indio_CA_RMY_2009-2024.epw` |
| Lancaster CA | 723816-03159 General William J. Fox Airfield Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2013 | `Lancaster_CA_RMY_2010-2024.epw` |
| Long Beach CA | 722970-23129 Long Beach / Daugherty Field / Airport | 2010-2024 (15 years) | NSRDB | 2015 / 2013 | `Long_Beach_CA_RMY_2010-2024.epw` |
| Merced CA | 724815-23257 Mrcd Muni/Macready Fld Apt | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Merced_CA_RMY_2010-2024.epw` |
| Oakland CA | 724930-23230 Metro Oakland Intl Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `Oakland_CA_RMY_2010-2024.epw` |
| Palmdale CA | 723820-23182 Pdle Prodn Flt/Tst Af Plt | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Palmdale_CA_RMY_2010-2024.epw` |
| Redding CA | 725920-24257 Redding Municipal Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Redding_CA_RMY_2010-2024.epw` |
| San Bernardino CA | 722860-23119 March Air Reserve Base | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `San_Bernardino_CA_RMY_2010-2024.epw` |
| Santa Ana CA | 722977-93184 J. Wayne Apt-Orange Co Apt | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `Santa_Ana_CA_RMY_2010-2024.epw` |
| Santa Barbara CA | 723925-23190 Santa Barbara Municipal Airport | 2010-2024 (15 years) | NSRDB | 2017 / 2013 | `Santa_Barbara_CA_RMY_2010-2024.epw` |
| Santa Clarita CA | 722886-23130 Van Nuys Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2013 | `Santa_Clarita_CA_RMY_2010-2024.epw` |
| Santa Maria CA | 723940-23273 Santa Maria Pub/Capt G A Hancock Ap | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `Santa_Maria_CA_RMY_2010-2024.epw` |
| Thousand Oaks CA | 723926-23136 Camarillo Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `Thousand_Oaks_CA_RMY_2010-2024.epw` |
| Torrance CA | 722885-93197 Santa Monica Muni Airport | 2010-2024 (15 years) | NSRDB | 2017 / 2011 | `Torrance_CA_RMY_2010-2024.epw` |
| Vacaville CA | 724828-93241 Nut Tree Airport | 2010-2024 (15 years) | NSRDB | 2017 / 2013 | `Vacaville_CA_RMY_2010-2024.epw` |
| Vallejo CA | 724955-93227 Napa County Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2013 | `Vallejo_CA_RMY_2010-2024.epw` |
| Ventura CA | 723910-93111 Point Mugu | 2010-2024 (15 years) | NSRDB | 2017 / 2011 | `Ventura_CA_RMY_2010-2024.epw` |
| Visalia CA | 723896-93144 Visalia Municipal Airport | 2008-2024 (15 years) | NSRDB | 2024 / 2013 | `Visalia_CA_RMY_2008-2024.epw` |
| Aurora CO | 724695-23036 Buckley Air Force Base | 2008-2024 (15 years) | NSRDB | 2012 / 2014 | `Aurora_CO_RMY_2008-2024.epw` |
| Centennial CO | 724666-93067 Centennial Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Centennial_CO_RMY_2010-2024.epw` |
| Greeley CO | 724768-24051 Greeley-Weld County Arpt | 2007-2024 (15 years) | NSRDB | 2012 / 2007 | `Greeley_CO_RMY_2007-2024.epw` |
| Lakewood CO | 724699-03065 Atct Rocky Mountain Metro | 2008-2024 (15 years) | NSRDB | 2012 / 2014 | `Lakewood_CO_RMY_2008-2024.epw` |
| Pueblo CO | 724640-93058 Pueblo Memorial Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2011 | `Pueblo_CO_RMY_2010-2024.epw` |
| Danbury CT | 725086-54734 Danbury Municipal Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2015 | `Danbury_CT_RMY_2010-2024.epw` |
| New Britain CT | 725087-14752 Hartford-Brainard Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2015 | `New_Britain_CT_RMY_2010-2024.epw` |
| Waterbury CT | 725029-64707 Waterbury-Oxford Airport | 2008-2024 (15 years) | NSRDB | 2021 / 2015 | `Waterbury_CT_RMY_2008-2024.epw` |
| Cape Coral FL | 722106-12835 Page Field Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2010 | `Cape_Coral_FL_RMY_2010-2024.epw` |
| Clearwater FL | 722116-12873 St Pete-Clwtr Intl Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Clearwater_FL_RMY_2010-2024.epw` |
| Coral Springs FL | 722049-92805 Pompano Beach Airpark Arpt | 2009-2024 (15 years) | NSRDB | 2023 / 2010 | `Coral_Springs_FL_RMY_2009-2024.epw` |
| Doral FL | 722029-12888 Kendall-Tamiami Exec Arpt | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Doral_FL_RMY_2010-2024.epw` |
| Fort Lauderdale FL | 747830-12849 Ft Laud/Hollywood Intl Apt | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Fort_Lauderdale_FL_RMY_2010-2024.epw` |
| Hialeah FL | 722024-12882 Opa Locka Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2010 | `Hialeah_FL_RMY_2010-2024.epw` |
| Homestead FL | 722026-12826 Homestead Afb Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Homestead_FL_RMY_2010-2024.epw` |
| Palm Bay FL | 747950-12867 Patrick Afb Airport | 2007-2023 (15 years) | NSRDB | 2023 / 2018 | `Palm_Bay_FL_RMY_2007-2023.epw` |
| Pembroke Pines FL | 722037-92809 North Perry Airport | 2009-2024 (15 years) | NSRDB | 2024 / 2010 | `Pembroke_Pines_FL_RMY_2009-2024.epw` |
| Port Saint Lucie FL | 722103-12895 St Lucie County Intl Arpt | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Port_Saint_Lucie_FL_RMY_2010-2024.epw` |
| West Palm Beach FL | 722030-12844 Palm Beach International Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `West_Palm_Beach_FL_RMY_2010-2024.epw` |
| Albany GA | 722160-13869 Sw Georgia Regional Arpt | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Albany_GA_RMY_2010-2024.epw` |
| Athens GA | 723110-13873 Athens/Ben Epps Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Athens_GA_RMY_2010-2024.epw` |
| Columbus GA | 722255-93842 Columbus Metro Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Columbus_GA_RMY_2010-2024.epw` |
| Macon GA | 722170-03813 Middle Georgia Regional Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Macon_GA_RMY_2010-2024.epw` |
| Roswell GA | 722270-13864 Dobbins Air Reserve Base Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Roswell_GA_RMY_2010-2024.epw` |
| Sandy Springs GA | 722196-53863 Dekalb-Peachtree Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2010 | `Sandy_Springs_GA_RMY_2010-2024.epw` |
| South Fulton GA | 722195-03888 Fulton Co-Brown Fld Arpt | 2010-2024 (15 years) | NSRDB | 2010 / 2010 | `South_Fulton_GA_RMY_2010-2024.epw` |
| Cedar Rapids IA | 725450-14990 Eastern Iowa Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Cedar_Rapids_IA_RMY_2010-2024.epw` |
| Davenport IA | 725440-14923 Quad City International Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Davenport_IA_RMY_2010-2024.epw` |
| Iowa City IA | 725462-14937 Iowa City Municipal Arpt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Iowa_City_IA_RMY_2010-2024.epw` |
| Sioux City IA | 725570-14943 Sioux Gateway/Col Bud Day Field Ap | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Sioux_City_IA_RMY_2010-2024.epw` |
| Arlington Heights IL | 744665-04838 Palwaukee Municipal Arpt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Arlington_Heights_IL_RMY_2010-2024.epw` |
| Aurora IL | 744655-04808 Aurora Municipal Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Aurora_IL_RMY_2010-2024.epw` |
| Bloomington IL | 724397-54831 Central Illinois Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Bloomington_IL_RMY_2010-2024.epw` |
| Bolingbrook IL | 725348-04831 Lewis University Airport | 2009-2024 (15 years) | NSRDB | 2012 / 2014 | `Bolingbrook_IL_RMY_2009-2024.epw` |
| Champaign IL | 725315-94870 Universi Of Il Willard Apt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Champaign_IL_RMY_2010-2024.epw` |
| Decatur IL | 725316-03887 Decatur Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Decatur_IL_RMY_2010-2024.epw` |
| Joliet IL | 725345-14834 Joliet Regional Airport | 2009-2024 (15 years) | NSRDB | 2012 / 2014 | `Joliet_IL_RMY_2009-2024.epw` |
| Naperville IL | 725305-94892 Dupage Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Naperville_IL_RMY_2010-2024.epw` |
| Springfield IL | 724390-93822 Abraham Lincoln Capital Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Springfield_IL_RMY_2010-2024.epw` |
| Waukegan IL | 725347-14880 Waukegan Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Waukegan_IL_RMY_2010-2024.epw` |
| Bloomington IN | 724375-03893 Monroe County Airport | 2009-2024 (15 years) | NSRDB | 2012 / 2014 | `Bloomington_IN_RMY_2009-2024.epw` |
| Carmel IN | 724384-53842 Eagle Creek Airpark Arpt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Carmel_IN_RMY_2010-2024.epw` |
| Evansville IN | 724320-93817 Evansville Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Evansville_IN_RMY_2010-2024.epw` |
| Hammond IN | 722126-04879 Lansing Municipal Ap | 2008-2024 (15 years) | NSRDB | 2012 / 2014 | `Hammond_IN_RMY_2008-2024.epw` |
| South Bend IN | 725350-14848 South Bend Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `South_Bend_IN_RMY_2010-2024.epw` |
| Kansas City KS | 724460-03947 Kansas City International Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Kansas_City_KS_RMY_2010-2024.epw` |
| Lawrence KS | 724508-03997 Lawrence Municipal Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Lawrence_KS_RMY_2010-2024.epw` |
| Olathe KS | 724475-93909 New Century Aircenter Arpt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Olathe_KS_RMY_2010-2024.epw` |
| Overland Park KS | 724468-03967 Johnson Co Executive Arpt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Overland_Park_KS_RMY_2010-2024.epw` |
| Topeka KS | 724560-13996 Philip Billard Municipal Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Topeka_KS_RMY_2010-2024.epw` |
| Lake Charles LA | 722404-13941 Chennault International Ap | 2009-2024 (15 years) | NSRDB | 2023 / 2014 | `Lake_Charles_LA_RMY_2009-2024.epw` |
| Brockton MA | 725068-54777 Taunton Municipal Airport | 2008-2024 (15 years) | NSRDB | 2020 / 2015 | `Brockton_MA_RMY_2008-2024.epw` |
| Cambridge MA | 725059-14702 Laurence G Hanscom Fld Apt | 2010-2024 (15 years) | NSRDB | 2022 / 2015 | `Cambridge_MA_RMY_2010-2024.epw` |
| Fall River MA | 725079-14787 Newport State Airport | 2010-2024 (15 years) | NSRDB | 2018 / 2015 | `Fall_River_MA_RMY_2010-2024.epw` |
| Lowell MA | 744904-94723 Lawrence Municipal Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2015 | `Lowell_MA_RMY_2010-2024.epw` |
| Lynn MA | 725088-54733 Beverly Municipal Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2015 | `Lynn_MA_RMY_2010-2024.epw` |
| New Bedford MA | 725065-94726 New Bedford Rgnl Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2015 | `New_Bedford_MA_RMY_2010-2024.epw` |
| Springfield MA | 744910-14703 Westover Afb/Metropolitan Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Springfield_MA_RMY_2010-2024.epw` |
| Ann Arbor MI | 725374-94889 Ann Arbor Municipal Arpt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Ann_Arbor_MI_RMY_2010-2024.epw` |
| Dearborn MI | 725370-94847 Detroit Metro Wayne County Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Dearborn_MI_RMY_2010-2024.epw` |
| Flint MI | 726370-14826 Bishop International Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2014 | `Flint_MI_RMY_2010-2024.epw` |
| Kalamazoo MI | 726357-94815 Klmazo/Btl Creek Intl Arpt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Kalamazoo_MI_RMY_2010-2024.epw` |
| Livonia MI | 725376-14853 Willow Run Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Livonia_MI_RMY_2010-2024.epw` |
| Warren MI | 720113-54829 Oakland/Troy Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Warren_MI_RMY_2010-2024.epw` |
| Bloomington MN | 726579-94963 Flying Cloud Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Bloomington_MN_RMY_2010-2024.epw` |
| Brooklyn Park MN | 726575-94960 Crystal Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2014 | `Brooklyn_Park_MN_RMY_2010-2024.epw` |
| Plymouth MN | 726577-94974 Anoka Co-Blne Ap(Jns Fd) Ap | 2009-2024 (15 years) | NSRDB | 2021 / 2014 | `Plymouth_MN_RMY_2009-2024.epw` |
| Rochester MN | 726440-14925 Rochester International Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Rochester_MN_RMY_2010-2024.epw` |
| Saint Paul MN | 726584-14927 St Paul Dwtwn Holman Fd Ap | 2010-2024 (15 years) | NSRDB | 2021 / 2014 | `Saint_Paul_MN_RMY_2010-2024.epw` |
| Independence MO | 720306-53879 Lee'S Summit Municipal Apt | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Independence_MO_RMY_2010-2024.epw` |
| O'Fallon MO | 724345-03966 Spirit Of St Louis Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `OFallon_MO_RMY_2010-2024.epw` |
| Saint Joseph MO | 724490-13993 Rosecrans Memorial Airport | 2009-2024 (15 years) | NSRDB | 2012 / 2014 | `Saint_Joseph_MO_RMY_2009-2024.epw` |
| Concord NC | 722132-63801 Concord Regional Airport | 2008-2024 (15 years) | NSRDB | 2011 / 2015 | `Concord_NC_RMY_2008-2024.epw` |
| Fayetteville NC | 723035-93740 Fayett Rgnl/Grannis Fld Ap | 2010-2024 (15 years) | NSRDB | 2011 / 2010 | `Fayetteville_NC_RMY_2010-2024.epw` |
| Gastonia NC | 723147-53870 Gastonia Municipal Airport | 2010-2024 (15 years) | NSRDB | 2015 / 2010 | `Gastonia_NC_RMY_2010-2024.epw` |
| Greenville NC | 723065-13783 Pitt-Greenville Airport | 2010-2024 (15 years) | NSRDB | 2019 / 2010 | `Greenville_NC_RMY_2010-2024.epw` |
| Manchester NH | 743945-14710 Manchester Airport | 2010-2024 (15 years) | NSRDB | 2020 / 2015 | `Manchester_NH_RMY_2010-2024.epw` |
| Nashua NH | 743946-54754 Boire Field Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Nashua_NH_RMY_2010-2024.epw` |
| Camden NJ | 724085-94732 Ne Philadelphia Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Camden_NJ_RMY_2010-2024.epw` |
| Edison NJ | 722247-54785 Somerset Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2015 | `Edison_NJ_RMY_2010-2024.epw` |
| Elizabeth NJ | 724097-54738 Morristown Municipal Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2015 | `Elizabeth_NJ_RMY_2010-2024.epw` |
| Jersey City NJ | 725025-94741 Teterboro Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Jersey_City_NJ_RMY_2010-2024.epw` |
| Newark NJ | 725020-14734 Newark Liberty International Ap | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Newark_NJ_RMY_2010-2024.epw` |
| Paterson NJ | 724094-54743 Essex County Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2015 | `Paterson_NJ_RMY_2010-2024.epw` |
| Trenton NJ | 724095-14792 Trenton Mercer Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Trenton_NJ_RMY_2010-2024.epw` |
| Las Cruces NM | 722695-93041 Las Cruces Intl Airport | 2008-2024 (15 years) | NSRDB | 2023 / 2011 | `Las_Cruces_NM_RMY_2008-2024.epw` |
| Rio Rancho NM | 723647-03034 Double Eagle Ii Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2011 | `Rio_Rancho_NM_RMY_2010-2024.epw` |
| Henderson NV | 746140-23112 Nellis Afb Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2011 | `Henderson_NV_RMY_2010-2024.epw` |
| North Las Vegas NV | 724846-53123 North Las Vegas Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `North_Las_Vegas_NV_RMY_2010-2024.epw` |
| Amherst NY | 725287-04724 Niagara Falls Intl Airport | 2010-2024 (15 years) | NSRDB | 2016 / 2015 | `Amherst_NY_RMY_2010-2024.epw` |
| New York City NY | 725053-94728 Central Park | 2005-2024 (15 years) | NSRDB | 2010 / 2015 | `New_York_City_NY_RMY_2005-2024.epw` |
| Yonkers NY | 725037-94745 Westchester County Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2015 | `Yonkers_NY_RMY_2010-2024.epw` |
| Parma OH | 725245-04853 Cleveland Burke Lakefnt Ap | 2009-2024 (15 years) | NSRDB | 2024 / 2014 | `Parma_OH_RMY_2009-2024.epw` |
| Broken Arrow OK | 723564-53908 Richard Lloyd Jones Jr Apt | 2010-2024 (15 years) | NSRDB | 2011 / 2011 | `Broken_Arrow_OK_RMY_2010-2024.epw` |
| Edmond OK | 723544-03954 Wiley Post Airport | 2009-2024 (15 years) | NSRDB | 2011 / 2022 | `Edmond_OK_RMY_2009-2024.epw` |
| Lawton OK | 723550-13945 Henry Post Aaf Airport | 2008-2023 (15 years) | NSRDB | 2011 / 2014 | `Lawton_OK_RMY_2008-2023.epw` |
| Norman OK | 723570-03948 University Of Oklahoma Westheimer Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2022 | `Norman_OK_RMY_2010-2024.epw` |
| Bend OR | 720638-00224 Bend Municipal Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2013 | `Bend_OR_RMY_2010-2024.epw` |
| Gresham OR | 726985-24242 Portland-Troutdale Airport | 2010-2024 (15 years) | NSRDB | 2015 / 2013 | `Gresham_OR_RMY_2010-2024.epw` |
| Hillsboro OR | 726986-94261 Portland-Hillsboro Airport | 2010-2024 (15 years) | NSRDB | 2015 / 2013 | `Hillsboro_OR_RMY_2010-2024.epw` |
| Bethlehem PA | 720324-64753 Quakertown Airport | 2006-2023 (15 years) | NSRDB | 2016 / 2015 | `Bethlehem_PA_RMY_2006-2023.epw` |
| Erie PA | 725260-14860 Erie Intl/T. Ridge Field Airport | 2010-2024 (15 years) | NSRDB | 2018 / 2014 | `Erie_PA_RMY_2010-2024.epw` |
| Reading PA | 725103-14712 Rdng Rgnl/C A Spaatz Fd Ap | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Reading_PA_RMY_2010-2024.epw` |
| Rapid City SD | 726625-24006 Ellsworth Air Force Base | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Rapid_City_SD_RMY_2010-2024.epw` |
| Clarksville TN | 746710-13806 Campbell Aaf Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2010 | `Clarksville_TN_RMY_2010-2024.epw` |
| Murfreesboro TN | 723273-13827 Smyrna Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2010 | `Murfreesboro_TN_RMY_2010-2024.epw` |
| Abilene TX | 722660-13962 Abilene Regional Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2014 | `Abilene_TX_RMY_2010-2024.epw` |
| Arlington TX | 722094-53984 Grand Prairie Municipal Airport | 2009-2024 (15 years) | NSRDB | 2011 / 2022 | `Arlington_TX_RMY_2009-2024.epw` |
| Beaumont TX | 722410-12917 Southeast Texas Regional Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Beaumont_TX_RMY_2010-2024.epw` |
| College Station TX | 747460-03904 Easterwood Field Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2014 | `College_Station_TX_RMY_2010-2024.epw` |
| Denton TX | 722589-03991 Denton Municipal Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2022 | `Denton_TX_RMY_2010-2024.epw` |
| Fort Worth TX | 747390-13961 Ft Worth Meacham Intl Arpt | 2009-2024 (15 years) | NSRDB | 2023 / 2022 | `Fort_Worth_TX_RMY_2009-2024.epw` |
| Garland TX | 720291-53970 Mesquite Metropolitan Arpt | 2010-2024 (15 years) | NSRDB | 2023 / 2022 | `Garland_TX_RMY_2010-2024.epw` |
| Grand Prairie TX | 722599-03971 Dallas Executive Airport | 2010-2024 (15 years) | NSRDB | 2011 / 2022 | `Grand_Prairie_TX_RMY_2010-2024.epw` |
| Irving TX | 722590-03927 Dallas/Ft Worth International Ap | 2010-2024 (15 years) | NSRDB | 2011 / 2014 | `Irving_TX_RMY_2010-2024.epw` |
| Longview TX | 722470-03901 East Texas Regional Arpt | 2010-2024 (15 years) | NSRDB | 2011 / 2022 | `Longview_TX_RMY_2010-2024.epw` |
| McKinney TX | 722541-53914 Mckinney Municipal Arpt | 2010-2024 (15 years) | NSRDB | 2011 / 2011 | `McKinney_TX_RMY_2010-2024.epw` |
| Mesquite TX | 722143-53975 Lancaster Airport | 2009-2024 (15 years) | NSRDB | 2011 / 2014 | `Mesquite_TX_RMY_2009-2024.epw` |
| Odessa TX | 722648-03031 Odessa-Schlemeyer Fld Arpt | 2008-2023 (15 years) | NSRDB | 2023 / 2011 | `Odessa_TX_RMY_2008-2023.epw` |
| Pasadena TX | 722440-12918 William P. Hobby Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Pasadena_TX_RMY_2010-2024.epw` |
| Pearland TX | 722427-12975 Clover Field Airport | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Pearland_TX_RMY_2010-2024.epw` |
| Pharr TX | 722159-12980 Mid Valley Airport | 2008-2023 (15 years) | NSRDB | 2009 / 2022 | `Pharr_TX_RMY_2008-2023.epw` |
| Plano TX | 722598-03970 Addison Airport | 2009-2024 (15 years) | NSRDB | 2011 / 2014 | `Plano_TX_RMY_2009-2024.epw` |
| San Angelo TX | 722630-23034 San Angelo Regional/Maths Field Ap | 2010-2024 (15 years) | NSRDB | 2023 / 2022 | `San_Angelo_TX_RMY_2010-2024.epw` |
| Sugar Land TX | 722543-12977 Sugar Land Regional Arpt | 2010-2024 (15 years) | NSRDB | 2023 / 2010 | `Sugar_Land_TX_RMY_2010-2024.epw` |
| Tyler TX | 722448-13972 Tyler Pounds Regional Arpt | 2010-2024 (15 years) | NSRDB | 2011 / 2022 | `Tyler_TX_RMY_2010-2024.epw` |
| Layton UT | 725755-24101 Hill Afb Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2013 | `Layton_UT_RMY_2010-2024.epw` |
| Ogden UT | 725750-24126 Ogden-Hinckley Airport | 2010-2024 (15 years) | NSRDB | 2024 / 2013 | `Ogden_UT_RMY_2010-2024.epw` |
| Provo UT | 725724-24174 Provo Municipal Airport | 2008-2024 (15 years) | NSRDB | 2020 / 2013 | `Provo_UT_RMY_2008-2024.epw` |
| Alexandria VA | 724037-93728 Davison Aaf Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Alexandria_VA_RMY_2010-2024.epw` |
| Chesapeake VA | 724006-03701 Chesapeake Regional Arpt | 2009-2024 (15 years) | NSRDB | 2010 / 2010 | `Chesapeake_VA_RMY_2009-2024.epw` |
| Hampton VA | 723086-93741 Nwpt News/Wimburg Intl Apt | 2010-2024 (15 years) | NSRDB | 2010 / 2022 | `Hampton_VA_RMY_2010-2024.epw` |
| Lynchburg VA | 724100-13733 Lynchburg Rgnl/Preston Glenn Fld Ap | 2010-2024 (15 years) | NSRDB | 2010 / 2010 | `Lynchburg_VA_RMY_2010-2024.epw` |
| Newport News VA | 745980-13702 Langley Afb Airport | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Newport_News_VA_RMY_2010-2024.epw` |
| Norfolk VA | 723085-13750 Norfolk Nas | 2010-2024 (15 years) | NSRDB | 2010 / 2015 | `Norfolk_VA_RMY_2010-2024.epw` |
| Suffolk VA | 724007-03719 Suffolk Municipal Airport | 2009-2024 (15 years) | NSRDB | 2010 / 2010 | `Suffolk_VA_RMY_2009-2024.epw` |
| Virginia Beach VA | 723075-13769 Oceana Nas | 2010-2024 (15 years) | NSRDB | 2016 / 2010 | `Virginia_Beach_VA_RMY_2010-2024.epw` |
| Bellingham WA | 727976-24217 Bellingham Intl Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2022 | `Bellingham_WA_RMY_2010-2024.epw` |
| Everett WA | 727937-24222 Snohomish Co (Paine Fd) Ap | 2010-2024 (15 years) | NSRDB | 2021 / 2017 | `Everett_WA_RMY_2010-2024.epw` |
| Federal Way WA | 742060-24207 Mcchord Afb Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2014 | `Federal_Way_WA_RMY_2010-2024.epw` |
| Kennewick WA | 727845-24163 Tri-Cities Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2017 | `Kennewick_WA_RMY_2010-2024.epw` |
| Kent WA | 727934-94248 Renton Municipal Airport | 2010-2024 (15 years) | NSRDB | 2022 / 2013 | `Kent_WA_RMY_2010-2024.epw` |
| Spokane Valley WA | 727856-94176 Felts Field Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2017 | `Spokane_Valley_WA_RMY_2010-2024.epw` |
| Tacoma WA | 727938-94274 Tacoma Narrows Airport | 2010-2024 (15 years) | NSRDB | 2021 / 2022 | `Tacoma_WA_RMY_2010-2024.epw` |
| Vancouver WA | 726836-04201 Scappoose Indus Airpk Arpt | 2009-2023 (15 years) | NSRDB | 2021 / 2009 | `Vancouver_WA_RMY_2009-2023.epw` |
| Yakima WA | 727810-24243 Yakima Air Terminal/Mcalsr Field Ap | 2010-2024 (15 years) | NSRDB | 2015 / 2022 | `Yakima_WA_RMY_2010-2024.epw` |
| Appleton WI | 726457-04825 Outagamie Co Rgnl Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Appleton_WI_RMY_2010-2024.epw` |
| Kenosha WI | 726505-04845 Kenosha Regional Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Kenosha_WI_RMY_2010-2024.epw` |
| Racine WI | 726424-94818 John H Batten Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Racine_WI_RMY_2010-2024.epw` |
| Waukesha WI | 726409-04897 Waukesha County Airport | 2010-2024 (15 years) | NSRDB | 2012 / 2014 | `Waukesha_WI_RMY_2010-2024.epw` |

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
