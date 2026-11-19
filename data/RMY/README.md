# Representative Meteorological Years (RMYs): 216 cities

20 US cities, one per US climate zone (ASHRAE 169, 1A to 8), 20 international cities, 56 world capitals, 20 US state capitals and 100 more capitals and major cities. Each is the city's ISO 15927-4 TMY with the heat
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
