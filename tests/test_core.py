"""Tests on simulated returns with a known abnormal return injected."""

import numpy as np
import pandas as pd
import pytest

from eventstudy import EventStudy

N_ASSETS = 30
DAYS = pd.bdate_range('2020-01-01', periods=800)


def simulate(shock=0.0, seed=0, shock_day=0):
    """Market-model returns for 30 assets; one event per asset at a random date."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.0003, 0.01, len(DAYS))
    betas = rng.uniform(0.6, 1.4, N_ASSETS)
    alphas = rng.normal(0, 0.0002, N_ASSETS)
    data = {'MKT': mkt}
    events = []
    for j in range(N_ASSETS):
        r = alphas[j] + betas[j] * mkt + rng.normal(0, 0.015, len(DAYS))
        t0 = int(rng.integers(300, 750))
        r[t0 + shock_day] += shock
        data[f'A{j}'] = r
        events.append({'asset': f'A{j}', 'date': DAYS[t0]})
    return pd.DataFrame(data, index=DAYS), pd.DataFrame(events)


def test_recovers_injected_abnormal_return():
    rets, events = simulate(shock=0.03)
    es = EventStudy(rets, events, market='MKT')
    assert es.n_events == N_ASSETS
    res = es.test(0, 0)
    assert res['CAAR'] == pytest.approx(0.03, abs=0.008)
    assert res['p (cross-sec.)'] < 0.01
    assert res['p (BMP)'] < 0.01
    # the shock is on day 0 only: pre-event window should be ~0
    assert abs(es.test(-5, -1)['CAAR']) < 0.015


def test_no_effect_under_the_null():
    rets, events = simulate(shock=0.0, seed=1)
    res = EventStudy(rets, events, market='MKT').test(-1, 1)
    assert abs(res['CAAR']) < 0.01
    assert res['p (cross-sec.)'] > 0.01


@pytest.mark.parametrize('model', ['market', 'mean', 'adjusted'])
def test_all_models_detect_large_shock(model):
    rets, events = simulate(shock=0.05, seed=2)
    res = EventStudy(rets, events, market='MKT', model=model).test(0, 0)
    assert res['CAAR'] > 0.03
    assert res['share positive'] > 0.8


def test_car_is_cumulative_ar():
    rets, events = simulate(shock=0.02, seed=3)
    es = EventStudy(rets, events, market='MKT', event=(-3, 3))
    assert np.allclose(es.car.iloc[-1], es.ar.sum())
    assert np.allclose(es.car_window(-3, 3), es.ar.sum())


def test_event_too_close_to_sample_start_is_skipped():
    rets, events = simulate(seed=4)
    events = pd.concat([events, pd.DataFrame({'asset': ['A0'], 'date': [DAYS[50]]})],
                       ignore_index=True)
    es = EventStudy(rets, events, market='MKT')
    assert es.n_events == N_ASSETS
    assert len(es.skipped) == 1


def test_event_on_non_trading_day_maps_to_next_trading_day():
    rets, _ = simulate(seed=5)
    saturday = pd.Timestamp('2021-06-05')
    es = EventStudy(rets, pd.DataFrame({'asset': ['A1'], 'date': [saturday]}),
                    market='MKT', event=(0, 0))
    day0 = rets.index[rets.index.searchsorted(saturday)]
    assert day0 == pd.Timestamp('2021-06-07')
    assert es.n_events == 1


def test_overlapping_windows_are_rejected():
    rets, events = simulate(seed=6)
    with pytest.raises(ValueError):
        EventStudy(rets, events, market='MKT', estimation=(-100, -2), event=(-5, 5))
