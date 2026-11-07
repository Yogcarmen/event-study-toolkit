"""
Price download (with a local CSV cache) and return construction.

Downloading uses ``yfinance`` (Yahoo Finance) and is only needed for the
examples; the core library and tests work on any price DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(tickers: list[str], start: str, end: str,
                cache_dir: str | Path | None = 'data/cache') -> pd.DataFrame:
    """
    Daily adjusted close prices, one column per ticker.

    Results are cached as CSV so that re-running an example does not hit the
    network again. Delete the cache file to refresh.
    """
    tickers = list(tickers)
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = '_'.join(t.replace('^', 'IDX').replace('.', '-') for t in tickers)
        if len(name) > 80:  # keep file names short (Windows path limits)
            import hashlib
            name = f'{len(tickers)}tickers_' + hashlib.md5(name.encode()).hexdigest()[:12]
        cache = cache_dir / f'{name}_{start}_{end}.csv'
        if cache.exists():
            return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf  # optional dependency
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)
    prices = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    if not isinstance(raw.columns, pd.MultiIndex):
        prices.columns = tickers
    prices = prices[tickers].dropna(how='all')

    if cache is not None:
        prices.to_csv(cache)
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a price panel (first row dropped)."""
    return prices.sort_index().pct_change().iloc[1:]
