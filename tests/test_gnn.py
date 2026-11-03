from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rmy import gnn
from rmy.gnn import GCNAutoencoder, GNNConfig, build_graph, detect_gnn_anomalies, wind_chill_c
from rmy.utils import list_epws

REPO = Path(__file__).resolve().parents[1]
AMYS = list_epws(REPO / 'data' / 'cities' / 'Boston' / 'AMY')


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    dates = pd.Series(pd.date_range('2020-01-01', periods=12))
    x = rng.normal(size=(12, 6))
    a = build_graph(dates, x, k_neighbors=2)
    model = GCNAutoencoder(6, GNNConfig(hidden_dim=5, embed_dim=3, seed=3))
    _, grads = model.loss_and_grads(a, x)
    eps = 1e-6
    for name, param in model.params.items():
        for idx in [tuple(rng.integers(0, s) for s in param.shape) for _ in range(3)]:
            old = param[idx]
            param[idx] = old + eps
            up, _ = model.loss_and_grads(a, x)
            param[idx] = old - eps
            down, _ = model.loss_and_grads(a, x)
            param[idx] = old
            assert grads[name][idx] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-8), name


def test_graph_links_consecutive_days_only_and_is_normalized():
    dates = pd.Series(pd.to_datetime(['2020-12-30', '2020-12-31', '2022-01-01', '2022-01-02']))
    a = build_graph(dates, np.zeros((4, 6)), k_neighbors=0).toarray()
    assert np.allclose(a, a.T)
    assert a[0, 1] > 0 and a[2, 3] > 0
    assert a[1, 2] == 0            # 2020-12-31 and 2022-01-01 are not consecutive
    # Row sums of D^-1/2 (A+I) D^-1/2 for a two-node path are 1.
    assert np.allclose(a.sum(axis=1), 1.0)


def test_wind_chill_matches_published_table():
    # Environment Canada / NWS table: -10 C air, 20 km/h wind -> about -18 C.
    assert wind_chill_c(-10.0, 20 / 3.6) == pytest.approx(-17.9, abs=0.1)
    # Outside the formula's range the air temperature is returned.
    assert wind_chill_c(15.0, 10.0) == 15.0


def test_training_reduces_loss_and_is_deterministic():
    gnn._CACHE.clear()
    first = detect_gnn_anomalies(AMYS)
    gnn._CACHE.clear()
    second = detect_gnn_anomalies(AMYS)
    assert first.attrs['final_loss'] < 0.1 * first.attrs['initial_loss']
    pd.testing.assert_series_equal(first['error'], second['error'])
    assert first['anomaly'].mean() < 0.06        # a few percent of days


@pytest.mark.parametrize('kind, start, end', [
    ('hot', '2011-07-20', '2011-07-22'),     # Boston record 103 F on 22 July 2011
    ('hot', '2016-08-11', '2016-08-14'),
    ('cold', '2016-02-13', '2016-02-15'),    # record cold, 14 Feb 2016
    ('cold', '2023-02-03', '2023-02-05'),    # record cold, 4 Feb 2023
])
def test_flags_known_boston_events(kind, start, end):
    d = detect_gnn_anomalies(AMYS)
    window = d[(d.date >= start) & (d.date <= end)]
    assert window[kind].all(), window[['date', 'score', kind]]
