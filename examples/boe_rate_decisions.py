"""
Example: UK banks vs housebuilders around Bank of England rate decisions.

Question: during the 2022-23 hiking cycle, did rate-sensitive UK sectors earn
abnormal returns on Monetary Policy Committee announcement days?

* Banks tend to benefit from higher rates (wider net interest margins).
* Housebuilders tend to suffer (higher mortgage rates, weaker demand).

Design choices
--------------
* Each sector is an equal-weighted portfolio, so every announcement date is
  one observation. Stocks reacting to the same announcement are correlated,
  and treating them as independent events would overstate significance.
* Market model against the FTSE 100, estimated over days -250 to -11.
* Announcements are at 12:00 UK time, so day 0 close-to-close returns
  capture the reaction.

Run:  python examples/boe_rate_decisions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from eventstudy import EventStudy, load_prices, to_returns  # noqa: E402

SECTORS = {
    'UK banks': ['LLOY.L', 'BARC.L', 'NWG.L', 'HSBA.L', 'STAN.L'],
    # Barratt Developments (BDEV.L) now trades as Barratt Redrow (BTRW.L)
    'UK housebuilders': ['PSN.L', 'BTRW.L', 'TW.L', 'BKG.L'],
}
MARKET = '^FTSE'
OUT = ROOT / 'output'


def main():
    decisions = pd.read_csv(ROOT / 'examples' / 'data' / 'boe_mpc_decisions.csv',
                            parse_dates=['date'])
    tickers = [t for ts in SECTORS.values() for t in ts] + [MARKET]
    prices = load_prices(tickers, start='2021-01-01', end='2024-03-01',
                         cache_dir=ROOT / 'data' / 'cache')
    rets = to_returns(prices)

    # Equal-weighted sector portfolios + market
    port = pd.DataFrame({name: rets[ts].mean(axis=1) for name, ts in SECTORS.items()})
    port[MARKET] = rets[MARKET]

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    tables = []
    colors = {'UK banks': '#4C72B0', 'UK housebuilders': '#C44E52'}

    for ax, name in zip(axes, SECTORS):
        events = pd.DataFrame({'asset': name, 'date': decisions['date'],
                               'label': decisions['change_bp']})
        es = EventStudy(port, events, market=MARKET,
                        estimation=(-250, -11), event=(-5, 5))
        table = es.summary(windows=[(0, 0), (-1, 1), (0, 5)])
        table.insert(0, 'portfolio', name)
        tables.append(table)
        es.plot(ax=ax, title=name, color=colors[name])

        # Larger hikes (>= 50bp) vs the rest, day-0 abnormal return
        ar0 = es.ar.loc[0]
        big = decisions.loc[ar0.index, 'change_bp'] >= 50
        print(f'{name}: day-0 AR, hikes >= 50bp: {ar0[big].mean():+.2%} '
              f'(n={big.sum()}), smaller/holds: {ar0[~big].mean():+.2%} (n={(~big).sum()})')

    fig.suptitle('Average cumulative abnormal return around Bank of England rate decisions, '
                 '2022-23 (15 announcements)', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'boe_car.png', dpi=150, bbox_inches='tight')
    print(f'Saved {OUT / "boe_car.png"}')

    result = pd.concat(tables)
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(result.to_string())
    result.to_csv(OUT / 'boe_summary.csv')


if __name__ == '__main__':
    main()
