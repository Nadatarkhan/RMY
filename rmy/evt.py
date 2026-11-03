"""Extreme Value Theory, Peaks Over Threshold (paper Section 2.2.3).

The paper's procedure:
- Variable: the heat index for heat waves; daily minimum temperature for
  cold spells.
- "A grid search was conducted to systematically test various quantile
  thresholds ... and identify the most appropriate threshold where heat waves
  could be detected."
- A Generalized Pareto Distribution (GPD) is fitted to the excesses over that
  threshold by maximum likelihood (Eq. 3).
- "Heat waves were defined as periods of at least three consecutive days
  during which the heat index exceeded the selected threshold." Cold spells
  likewise, with Tmin below the threshold.

Implementation choices where the paper is silent:
- The threshold is selected on all AMY days pooled, like the GNN ("across
  the 15 years analyzed"), so a mild year does not get its own top 5 %.
- Grid: quantiles 0.99 down to 0.90 in steps of 0.005 (0.01 up to 0.10 for
  cold). The selected threshold is the most extreme quantile at which the
  GPD fit is valid (at least 30 excesses, finite parameters) and at least
  one run of 3 consecutive days exists in the pooled record.
- Each threshold is reported with its GPD shape and scale and the mean
  number of exceedance days per year.

The notebook previously flagged only days beyond a GPD return level well
inside the tail, computed separately for each year, which almost never gave
3 consecutive days (review note item 4).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import genpareto

MIN_EXCESSES = 30
MIN_RUN = 3


@dataclass(frozen=True)
class POTThreshold:
    threshold: float
    quantile: float
    shape: float
    scale: float
    n_excesses: int
    exceedance_days_per_year: float
    tail: str  # 'upper' or 'lower'

    def flags(self, values):
        values = pd.Series(values, dtype=float)
        if self.tail == 'upper':
            return (values > self.threshold).to_numpy()
        return (values < self.threshold).to_numpy()

    def describe(self, name):
        side = '>' if self.tail == 'upper' else '<'
        return (f'EVT {name}: threshold {side} {self.threshold:.2f} (quantile {self.quantile:.3f}), '
                f'GPD shape {self.shape:+.3f}, scale {self.scale:.3f}, '
                f'{self.exceedance_days_per_year:.1f} exceedance days per year')


def _has_run(flags, dates, min_run=MIN_RUN):
    """True if `flags` holds `min_run` consecutive calendar days."""
    s = pd.Series(flags, index=pd.DatetimeIndex(dates)).sort_index()
    s = s[~s.index.duplicated()]
    s = s.reindex(pd.date_range(s.index.min(), s.index.max()), fill_value=False)
    return bool((s.astype(int).rolling(min_run).sum() >= min_run).any())


def select_pot_threshold(values, dates, tail='upper', quantiles=None):
    """Grid-search the POT threshold and fit a GPD to the excesses.

    `values` and `dates` are daily values and their dates, for all years.
    Returns the most extreme valid POTThreshold, or None if none is valid.
    """
    v = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates)).dropna()
    if v.empty:
        return None
    if quantiles is None:
        upper = np.round(np.arange(0.99, 0.8999, -0.005), 3)
        quantiles = upper if tail == 'upper' else np.round(1 - upper, 3)
    n_years = max(len(v) / 365.25, 1.0)

    for q in quantiles:
        u = float(v.quantile(q))
        excess = (v - u) if tail == 'upper' else (u - v)
        excess = excess[excess > 0]
        if len(excess) < MIN_EXCESSES or excess.std(ddof=0) == 0:
            continue
        shape, _, scale = genpareto.fit(excess.to_numpy(), floc=0)
        if not (np.isfinite(shape) and np.isfinite(scale) and scale > 0):
            continue
        cand = POTThreshold(u, float(q), float(shape), float(scale), int(len(excess)),
                            float(len(excess) / n_years), tail)
        if _has_run(cand.flags(v), v.index):
            return cand
    return None
