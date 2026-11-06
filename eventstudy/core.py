"""
Short-window event study on daily returns.

For each (asset, event date) pair the class

1. finds event day 0 (the first trading day on or after the event date),
2. fits a normal-return model on an estimation window that ends well before
   the event (default: days -250 to -11),
3. computes abnormal returns (AR) over the event window and cumulates them
   into CARs, and
4. tests whether the average CAR differs from zero.

Normal-return models
--------------------
``'market'``   OLS market model  R_it = a_i + b_i R_mt + e_it  (default)
``'mean'``     constant-mean model  R_it = mu_i + e_it
``'adjusted'`` market-adjusted  AR_it = R_it - R_mt  (no estimation)

Tests
-----
* Cross-sectional t-test on CARs (Brown & Warner 1985).
* Standardised cross-sectional test (Boehmer, Musumeci & Poulsen 1991):
  each CAR is divided by its estimation-period standard deviation, which is
  robust to event-induced changes in volatility.
* Sign test: share of positive CARs against 50%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ('market', 'mean', 'adjusted')


@dataclass(frozen=True)
class EventWindow:
    """Relative trading-day window [start, end], inclusive."""
    start: int
    end: int

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError('window start must be <= end')

    @property
    def days(self) -> range:
        return range(self.start, self.end + 1)

    def __str__(self) -> str:
        return f'[{self.start:+d}, {self.end:+d}]'


class EventStudy:
    """
    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, DatetimeIndex, one column per asset. Must
        include the market column when ``model`` is 'market' or 'adjusted'.
    events : pd.DataFrame
        Columns ``asset`` and ``date`` (one row per event). An optional
        ``label`` column is kept for reference.
    market : str, optional
        Column of ``returns`` holding the market/benchmark return.
    estimation : tuple(int, int)
        Estimation window relative to day 0, e.g. (-250, -11).
    event : tuple(int, int)
        Event window relative to day 0, e.g. (-5, 5).
    model : {'market', 'mean', 'adjusted'}
    min_obs : float
        Minimum share of non-missing estimation-window returns required.
    """

    def __init__(self, returns: pd.DataFrame, events: pd.DataFrame,
                 market: str | None = None,
                 estimation: tuple[int, int] = (-250, -11),
                 event: tuple[int, int] = (-5, 5),
                 model: str = 'market',
                 min_obs: float = 0.8):
        if model not in MODELS:
            raise ValueError(f'model must be one of {MODELS}')
        if model in ('market', 'adjusted') and market not in returns.columns:
            raise ValueError(f'market column {market!r} not found in returns')
        self.returns = returns.sort_index()
        self.events = events.reset_index(drop=True).copy()
        self.events['date'] = pd.to_datetime(self.events['date'])
        self.market = market
        self.est = EventWindow(*estimation)
        self.win = EventWindow(*event)
        if self.est.end >= self.win.start:
            raise ValueError('estimation window must end before the event window starts')
        self.model = model
        self.min_obs = min_obs
        self.ar, self.sigma, self.skipped = self._compute()

    # ── core ──────────────────────────────────────────────────────────────

    def _compute(self):
        idx = self.returns.index
        ar_cols, sigmas, skipped, kept = {}, {}, [], []

        for i, ev in self.events.iterrows():
            asset, date = ev['asset'], ev['date']
            if asset not in self.returns.columns:
                skipped.append((i, 'asset not in returns')); continue
            t0 = idx.searchsorted(date)                     # first day >= date
            lo_e, hi_e = t0 + self.est.start, t0 + self.est.end
            lo_w, hi_w = t0 + self.win.start, t0 + self.win.end
            if lo_e < 0 or hi_w >= len(idx):
                skipped.append((i, 'not enough data around event')); continue

            r_i = self.returns[asset]
            est_i = r_i.iloc[lo_e:hi_e + 1]
            win_i = r_i.iloc[lo_w:hi_w + 1].to_numpy()

            if self.model == 'adjusted':
                r_m = self.returns[self.market]
                est_ar = (est_i - r_m.iloc[lo_e:hi_e + 1]).dropna()
                ar = win_i - r_m.iloc[lo_w:hi_w + 1].to_numpy()
                dof = 0
            elif self.model == 'mean':
                est_ar = est_i.dropna() - est_i.mean()
                ar = win_i - est_i.mean()
                dof = 1
            else:
                r_m = self.returns[self.market]
                df_est = pd.concat([est_i, r_m.iloc[lo_e:hi_e + 1]], axis=1).dropna()
                y, x = df_est.iloc[:, 0].to_numpy(), df_est.iloc[:, 1].to_numpy()
                if len(y) < 3:
                    skipped.append((i, 'too few estimation observations')); continue
                beta, alpha = np.polyfit(x, y, 1)
                est_ar = pd.Series(y - (alpha + beta * x))
                ar = win_i - (alpha + beta * r_m.iloc[lo_w:hi_w + 1].to_numpy())
                dof = 2

            n_est = len(est_ar)
            if n_est < self.min_obs * len(self.est.days):
                skipped.append((i, 'too few estimation observations')); continue

            ar_cols[i] = ar
            sigmas[i] = float(np.sqrt((est_ar ** 2).sum() / max(n_est - dof, 1)))
            kept.append(i)

        ar_df = pd.DataFrame(ar_cols, index=list(self.win.days))
        ar_df.index.name = 'tau'
        return ar_df, pd.Series(sigmas), skipped

    # ── outputs ───────────────────────────────────────────────────────────

    @property
    def n_events(self) -> int:
        return self.ar.shape[1]

    @property
    def car(self) -> pd.DataFrame:
        """Cumulative abnormal returns from the start of the event window."""
        return self.ar.cumsum()

    def car_window(self, start: int, end: int) -> pd.Series:
        """CAR over [start, end] for every event (start/end relative to day 0)."""
        if start < self.win.start or end > self.win.end:
            raise ValueError(f'window must lie inside the event window {self.win}')
        return self.ar.loc[start:end].sum()

    def test(self, start: int, end: int) -> dict:
        """Mean CAR over [start, end] with cross-sectional, BMP and sign tests."""
        car = self.car_window(start, end)
        n = len(car)
        length = end - start + 1
        mean = car.mean()

        sd = car.std(ddof=1)
        t_cs = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan

        scar = car / (self.sigma[car.index] * np.sqrt(length))
        s_sd = scar.std(ddof=1)
        t_bmp = scar.mean() / (s_sd / np.sqrt(n)) if n > 1 and s_sd > 0 else np.nan

        n_pos = int((car > 0).sum())
        p_sign = stats.binomtest(n_pos, n, 0.5).pvalue if n > 0 else np.nan

        def p_of(t):
            return float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan

        return {'window': str(EventWindow(start, end)), 'N': n, 'CAAR': mean,
                't (cross-sec.)': t_cs, 'p (cross-sec.)': p_of(t_cs),
                't (BMP)': t_bmp, 'p (BMP)': p_of(t_bmp),
                'share positive': n_pos / n if n else np.nan, 'p (sign)': p_sign}

    def summary(self, windows=((0, 0), (-1, 1), (0, 1), (-5, 5))) -> pd.DataFrame:
        """Test table for several CAR windows."""
        rows = [self.test(s, e) for s, e in windows
                if s >= self.win.start and e <= self.win.end]
        return pd.DataFrame(rows).set_index('window')

    def plot(self, ax=None, title: str | None = None, color: str = '#4C72B0'):
        """Average CAR across events with a 95% cross-sectional band."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        car = self.car
        mean = car.mean(axis=1)
        se = car.std(axis=1, ddof=1) / np.sqrt(car.shape[1])
        crit = stats.t.ppf(0.975, df=max(car.shape[1] - 1, 1))
        ax.fill_between(car.index, mean - crit * se, mean + crit * se,
                        color=color, alpha=0.18, lw=0, label='95% band')
        ax.plot(car.index, mean, color=color, lw=2, marker='o', ms=3,
                label=f'Average CAR (N = {car.shape[1]})')
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='grey', lw=0.8, ls='--')
        ax.set_xlabel('Trading days relative to event (day 0)')
        ax.set_ylabel('Cumulative abnormal return')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1%}'))
        if title:
            ax.set_title(title, loc='left', fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        return ax
