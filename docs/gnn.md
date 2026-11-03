# The GNN detector: paper versus code

`rmy/gnn.py` implements the unsupervised graph neural network of the paper
(Tarkhan et al. 2025, Section 2.2.2) and its place in the hierarchical
ensemble (Section 2.3). It replaces the notebook's stand-ins: a z-score on
the base file's daily maximum (heat waves) and a rolling z-score on Tmin
(cold spells). Neither was a GNN.

## Design

| Element | Paper | Code |
|---|---|---|
| Nodes | Each day | Each day of every AMY, pooled ("across the 15 years analyzed") |
| Features | Daily Tmin, Tmax, wind speed, relative humidity, heat index, wind chill | Same six. Heat index from Tmax and mean RH; wind chill from Tmin and mean wind (Osczevski and Bluestein 2005) |
| Feature scaling | Not specified | Relative to the time of year: minus the multi-year mean for that calendar day (31-day smoothing), divided by that month's spread |
| Temporal edges | Consecutive days | Consecutive calendar days (no edge across a missing year) |
| Similarity edges | Days with similar multivariate patterns | 5 nearest neighbours in feature space, made symmetric |
| Layers | Two GCN layers with ReLU (Eq. 1); layer 1 to a higher dimension | GCN with normalized adjacency and self-loops: 6 → 16 → 8, ReLU after each |
| Decoder | Not specified | One linear layer, 8 → 6 (needed to reconstruct the features) |
| Loss | Mean squared reconstruction error (Eq. 2) | Same |
| Training | Not specified | Full batch, Adam, learning rate 0.01, 300 epochs, fixed seed |
| Anomaly threshold | Mean + k × std of the errors ("dynamic threshold") | k = 2, applied to the error averaged over a centred 3-day window |
| Hot or cold | Not specified | Hot if Tmax is above normal for the time of year, cold if Tmin is below |
| Events | Runs of anomalous days; duration is a feature | Runs of at least 3 days, as for the other methods |
| Role in the ensemble | Applied after thresholds, refining candidates and adding gradual events; methods "contribute additively to the final event set" | GNN events are added to the threshold and EVT events, then overlaps are resolved as before |

All settings are in `GNNConfig`. The model is written in NumPy/SciPy with
explicit gradients (checked against finite differences in the tests), so it
needs no deep-learning framework and gives identical results for identical
inputs.

## Why two of the open choices matter

Both were tested on the Boston AMYs (2009–2023):

- **Seasonal scaling.** With features scaled over the whole year, every
  summer day looks alike, and the GNN's "hot" anomalies were warm spells in
  winter. Relative to the time of year, its runs include known Boston heat:
  22 July 2011 (record 103 F), 20–22 June 2012, 11 September 2013,
  August 2016, May 2017 and 2018, 19–21 July 2019 and August 2022.
- **Three-day scoring.** A reconstruction error picks out single unusual
  days; within a heat wave usually only the peak day passes the threshold.
  Averaging over three days, as the paper's emphasis on "structured temporal
  patterns rather than isolated outliers" suggests, gives runs that can form
  events. Cold runs then include 14 February 2016 and 4 February 2023
  (both record cold), Christmas 2022 and January 2022.

## Effect on the Boston sample

Against the notebook baseline:

- Heat waves: one event added (11–14 August 2016); peak year unchanged (2022).
- Cold spells: 27 December events from the old rolling z-score are gone and
  10 shorter ones added. The peak cold year moves from 2015 to 2014.

The move is caused by the 21-day cap, not the GNN: 31 January to
21 February 2015 is one 22-day run below the thresholds, so the cap drops it
whole (the same problem as review item 1 for Phoenix heat). The old z-score
happened to detect a 4-day slice of it, which kept 2015 on top. The EVT
stage that followed detects 14–16 February 2015 and restores 2015 as the
peak year.

## Sensitivity

Settings left open by the paper were varied one at a time on the Boston
sample (detection only, no 21-day cap). Every variant gives the same peak
years (heat 2022, cold 2015) and the same peak-year events, so the RMY is
unchanged. Across all years, at most two events differ from the default.

| Variant | Heat events | Cold events | Events differing from default |
|---|---|---|---|
| default (k = 2, 5 neighbours, 16/8 units, 300 epochs, seed 0) | 58 | 74 | 0 |
| threshold k = 1.5 | 58 | 75 | 1 cold |
| threshold k = 2.5 | 58 | 74 | 2 heat |
| 3 neighbours | 58 | 74 | 0 |
| 10 neighbours | 57 | 74 | 1 heat |
| seed 1, seed 2 | 58 | 74 | 0 |
| 32/16 units, 12/4 units | 58 | 74 | 0 |
| 600 epochs | 58 | 74 | 0 |

Pass a `GNNConfig` to `run_full_rmy_pipeline(..., gnn_config=GNNConfig(threshold_k=2.5))`
to change these settings.
