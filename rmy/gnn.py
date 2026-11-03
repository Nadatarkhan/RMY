"""Unsupervised graph neural network (GNN) anomaly detector, paper Section 2.2.2.

What the paper specifies, and how it is implemented here:

- Nodes: one per day, pooled across all AMY years ("across the 15 years
  analyzed", Section 2.3).
- Node features: daily Tmin, Tmax, mean wind speed, mean relative humidity,
  heat index (from Tmax and mean RH) and wind chill (from Tmin and mean wind
  speed, Osczevski and Bluestein 2005).
- Edges: consecutive days, plus "days with similar multivariate weather
  patterns", implemented as the k nearest neighbours in feature space.
- Model: two GCN layers with ReLU (Eq. 1). Layer 1 maps the features to a
  higher-dimensional space, layer 2 aggregates neighbours. Both layers use
  the standard normalized adjacency with self-loops.
- Loss: mean squared reconstruction error of the input features (Eq. 2).
- Anomalies: days whose reconstruction error exceeds mean + k * std of all
  errors (the paper's "dynamic threshold").

Choices the paper leaves open, all set in GNNConfig:
- Feature scaling. Each feature is expressed relative to its time of year:
  the multi-year mean for that calendar day (smoothed over a month) is
  subtracted and the result divided by that month's spread. With plain
  annual scaling every summer day looks alike and summer heat waves are not
  anomalous; relative to the season they are.
- Persistence. The paper says the GNN keeps anomalies that "exhibit
  structured temporal patterns rather than isolated outliers" and uses the
  "duration of detected anomalies". Each day is therefore scored by the mean
  reconstruction error over a centred 3-day window (the minimum event
  length) before the mean + k * std threshold is applied.
- A linear decoder from the embedding back to the features (needed to
  compute a reconstruction; the paper does not describe one).
- Layer widths, neighbour count, epochs, learning rate and the threshold
  multiplier k.
- Direction: an anomalous day counts toward heat waves if its Tmax is above
  normal for the time of year, and toward cold spells if its Tmin is below
  normal. The paper does not say how GNN anomalies are split.

The model is written in NumPy/SciPy with explicit gradients and the Adam
optimizer, so it needs no deep-learning framework and gives the same result
for the same inputs and seed.
"""
from dataclasses import dataclass, astuple
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .utils import calculate_heat_index, read_epw_raw

FEATURES = ['Tmin', 'Tmax', 'wind_speed', 'relative_humidity', 'heat_index', 'wind_chill']


@dataclass(frozen=True)
class GNNConfig:
    hidden_dim: int = 16        # layer 1 width (higher than the 6 input features)
    embed_dim: int = 8          # layer 2 width
    k_neighbors: int = 5        # similarity edges per day
    epochs: int = 300
    learning_rate: float = 0.01
    threshold_k: float = 2.0    # anomaly if score > mean + threshold_k * std
    persistence_days: int = 3   # window for averaging the daily error
    climatology_window: int = 31  # days, for the seasonal baseline
    seed: int = 0


def wind_chill_c(temp_c, wind_ms):
    """Wind chill in degrees C (Osczevski and Bluestein 2005, as used by NWS/MSC).

    Defined for temperatures at or below 10 C and wind of at least 4.8 km/h;
    elsewhere the air temperature is returned.
    """
    t = np.asarray(temp_c, dtype=float)
    v = np.asarray(wind_ms, dtype=float) * 3.6
    wc = 13.12 + 0.6215 * t - 11.37 * v ** 0.16 + 0.3965 * t * v ** 0.16
    return np.where((t <= 10) & (v >= 4.8), wc, t)


def daily_features(epw_path):
    """Daily GNN features for one EPW. Days with any missing feature are dropped."""
    df = read_epw_raw(epw_path)
    day = df.groupby(['year', 'month', 'day']).agg(
        Tmin=('temp_air', 'min'), Tmax=('temp_air', 'max'),
        wind_speed=('wind_speed', 'mean'),
        relative_humidity=('relative_humidity', 'mean'),
    ).reset_index()
    day = day.rename(columns={'year': 'YEAR', 'month': 'MONTH', 'day': 'DAY'})
    day['heat_index'] = calculate_heat_index(day['Tmax'], day['relative_humidity']).to_numpy()
    day['wind_chill'] = wind_chill_c(day['Tmin'], day['wind_speed'])
    day['date'] = pd.to_datetime(dict(year=day.YEAR, month=day.MONTH, day=day.DAY), errors='coerce')
    return day.dropna(subset=FEATURES + ['date']).reset_index(drop=True)


def build_graph(dates, x_std, k_neighbors):
    """Normalized adjacency D^-1/2 (A + I) D^-1/2 with temporal and kNN edges."""
    n = len(dates)
    order = np.argsort(dates.to_numpy())
    d_sorted = dates.to_numpy()[order]
    consecutive = (d_sorted[1:] - d_sorted[:-1]) == np.timedelta64(1, 'D')
    rows = [order[:-1][consecutive]]
    cols = [order[1:][consecutive]]
    if k_neighbors > 0 and n > k_neighbors:
        _, nbr = cKDTree(x_std).query(x_std, k=k_neighbors + 1)
        rows.append(np.repeat(np.arange(n), k_neighbors))
        cols.append(nbr[:, 1:].ravel())
    r, c = np.concatenate(rows), np.concatenate(cols)
    a = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(float) + sp.identity(n, format='csr')
    deg = np.asarray(a.sum(axis=1)).ravel()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    return (d_inv_sqrt @ a @ d_inv_sqrt).tocsr()


class GCNAutoencoder:
    """Two GCN layers with ReLU (paper Eq. 1) and a linear decoder.

    H1 = ReLU(A X W1 + b1), H2 = ReLU(A H1 W2 + b2), X_hat = H2 Wd + bd.
    """

    def __init__(self, n_features, cfg: GNNConfig):
        rng = np.random.default_rng(cfg.seed)

        def glorot(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, size=(fan_in, fan_out))

        self.params = {
            'W1': glorot(n_features, cfg.hidden_dim), 'b1': np.zeros(cfg.hidden_dim),
            'W2': glorot(cfg.hidden_dim, cfg.embed_dim), 'b2': np.zeros(cfg.embed_dim),
            'Wd': glorot(cfg.embed_dim, n_features), 'bd': np.zeros(n_features),
        }

    def forward(self, a, x):
        p = self.params
        ax = a @ x
        z1 = ax @ p['W1'] + p['b1']
        h1 = np.maximum(z1, 0)
        ah1 = a @ h1
        z2 = ah1 @ p['W2'] + p['b2']
        h2 = np.maximum(z2, 0)
        x_hat = h2 @ p['Wd'] + p['bd']
        return x_hat, (ax, z1, h1, ah1, z2, h2)

    def loss_and_grads(self, a, x):
        """Loss of Eq. 2, (1/N) sum_i ||x_i - x_hat_i||^2, and its gradients."""
        p = self.params
        x_hat, (ax, z1, h1, ah1, z2, h2) = self.forward(a, x)
        n = x.shape[0]
        diff = x_hat - x
        loss = float((diff ** 2).sum() / n)
        d_xhat = 2.0 * diff / n
        g = {'Wd': h2.T @ d_xhat, 'bd': d_xhat.sum(axis=0)}
        d_z2 = (d_xhat @ p['Wd'].T) * (z2 > 0)
        g['W2'], g['b2'] = ah1.T @ d_z2, d_z2.sum(axis=0)
        d_h1 = a.T @ (d_z2 @ p['W2'].T)
        d_z1 = d_h1 * (z1 > 0)
        g['W1'], g['b1'] = ax.T @ d_z1, d_z1.sum(axis=0)
        return loss, g

    def fit(self, a, x, cfg: GNNConfig):
        """Full-batch Adam. Returns the loss per epoch."""
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(val) for k, val in self.params.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        for t in range(1, cfg.epochs + 1):
            loss, g = self.loss_and_grads(a, x)
            history.append(loss)
            for k in self.params:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                m_hat, v_hat = m[k] / (1 - b1 ** t), v[k] / (1 - b2 ** t)
                self.params[k] -= cfg.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return history


def _climatology(daily, col, window):
    """Multi-year mean of `col` for each calendar day, smoothed over `window` days."""
    doy = daily['date'].dt.strftime('%m-%d')
    clim = daily.groupby(doy)[col].mean()
    k = len(clim)
    padded = pd.concat([clim.iloc[-window:], clim, clim.iloc[:window]])
    smooth = padded.rolling(window, center=True, min_periods=1).mean().iloc[window:window + k]
    return doy.map(smooth)


def seasonal_anomalies(daily, window):
    """Features relative to the time of year, in units of that month's spread."""
    out = np.empty((len(daily), len(FEATURES)))
    month = daily['date'].dt.month
    for i, col in enumerate(FEATURES):
        resid = daily[col] - _climatology(daily, col, window)
        spread = resid.groupby(month).transform(lambda r: r.std(ddof=0)).replace(0, 1.0)
        out[:, i] = (resid / spread).to_numpy()
    return out


_CACHE = {}


def detect_gnn_anomalies(epw_paths, cfg: GNNConfig = GNNConfig()):
    """Train the GNN on all days of the given AMYs and flag anomalous days.

    Returns one row per day with YEAR, MONTH, DAY, date, the features, the
    reconstruction `error`, its 3-day mean `score`, and boolean columns
    `anomaly`, `hot` and `cold`.
    Results are cached for the same files and settings, so heat-wave and
    cold-spell detection share one trained model.
    """
    paths = tuple(sorted(Path(p) for p in epw_paths))
    key = (tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths), astuple(cfg))
    if key in _CACHE:
        return _CACHE[key].copy()

    daily = pd.concat([daily_features(p) for p in paths], ignore_index=True)
    daily = daily.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    if len(daily) < 30:
        raise ValueError(f'Too few days for the GNN ({len(daily)})')

    x_std = seasonal_anomalies(daily, cfg.climatology_window)

    a = build_graph(daily['date'], x_std, cfg.k_neighbors)
    model = GCNAutoencoder(len(FEATURES), cfg)
    history = model.fit(a, x_std, cfg)
    x_hat, _ = model.forward(a, x_std)

    daily['error'] = ((x_std - x_hat) ** 2).sum(axis=1)
    # Average over consecutive calendar days, so isolated outliers score low.
    span = pd.date_range(daily['date'].min(), daily['date'].max())
    by_day = daily.set_index('date')['error'].reindex(span)
    daily['score'] = (by_day.rolling(cfg.persistence_days, center=True, min_periods=2)
                      .mean().reindex(daily['date']).to_numpy())
    threshold = daily['score'].mean() + cfg.threshold_k * daily['score'].std(ddof=0)
    daily['anomaly'] = daily['score'] > threshold
    tmin_idx, tmax_idx = FEATURES.index('Tmin'), FEATURES.index('Tmax')
    daily['hot'] = daily['anomaly'] & (x_std[:, tmax_idx] > 0)
    daily['cold'] = daily['anomaly'] & (x_std[:, tmin_idx] < 0)
    daily.attrs.update(threshold=float(threshold), final_loss=history[-1],
                       initial_loss=history[0], n_edges=int((a.nnz - len(daily)) / 2))

    print(f'GNN: {len(daily)} days from {len(paths)} files; loss {history[0]:.3f} -> '
          f'{history[-1]:.3f}; {int(daily.anomaly.sum())} anomalous days '
          f'({int(daily.hot.sum())} hot, {int(daily.cold.sum())} cold)')
    _CACHE[key] = daily
    return daily.copy()


def gnn_flags_for(daily_df, gnn_daily, kind):
    """Align the GNN `hot` or `cold` flags with a detector's daily table.

    `daily_df` must have YEAR, MONTH and DAY columns. Days the GNN did not
    see are not flagged.
    """
    flags = gnn_daily[['YEAR', 'MONTH', 'DAY', kind]]
    merged = daily_df[['YEAR', 'MONTH', 'DAY']].merge(flags, on=['YEAR', 'MONTH', 'DAY'], how='left')
    return merged[kind].fillna(False).astype(bool).to_numpy()
