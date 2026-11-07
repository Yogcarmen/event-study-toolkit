"""Cache naming and return construction (offline)."""

import pandas as pd

from eventstudy import load_prices, to_returns


def test_long_ticker_lists_use_short_cache_names(tmp_path):
    tickers = [f'T{i:03d}' for i in range(200)]
    idx = pd.bdate_range('2024-01-01', periods=3)
    prices = pd.DataFrame(1.0, index=idx, columns=tickers)
    name = '_'.join(tickers)
    import hashlib
    short = f'200tickers_{hashlib.md5(name.encode()).hexdigest()[:12]}_2024-01-01_2024-01-05.csv'
    prices.to_csv(tmp_path / short)
    # served from cache, so no network call is made
    out = load_prices(tickers, '2024-01-01', '2024-01-05', cache_dir=tmp_path)
    assert list(out.columns) == tickers
    assert len(short) < 80


def test_to_returns_drops_first_row():
    idx = pd.bdate_range('2024-01-01', periods=3)
    r = to_returns(pd.DataFrame({'A': [100.0, 110.0, 99.0]}, index=idx))
    assert len(r) == 2
    assert abs(r['A'].iloc[0] - 0.10) < 1e-12
