"""A small, tested toolkit for short-window event studies on daily returns."""

from eventstudy.core import EventStudy, EventWindow
from eventstudy.data import load_prices, to_returns

__all__ = ['EventStudy', 'EventWindow', 'load_prices', 'to_returns']
__version__ = '0.1.0'
