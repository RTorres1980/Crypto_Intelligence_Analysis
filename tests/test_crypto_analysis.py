import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crypto_analysis as ca  # noqa: E402


@pytest.fixture(scope="module")
def ranked():
    universe, prices = ca.sample_data()
    return ca.rank(ca.metrics(prices, universe)), prices


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(1, 2, 60))
    down = pd.Series(np.linspace(2, 1, 60))
    assert ca.rsi(up).iloc[-1] > 99
    assert ca.rsi(down).iloc[-1] < 1


def test_max_drawdown():
    s = pd.Series([100, 120, 90, 110, 60, 80])
    assert ca.max_drawdown(s) == pytest.approx(60 / 120 - 1)


def test_score_range_and_order(ranked):
    r, _ = ranked
    assert r["score"].between(0, 100).all()
    assert r["score"].is_monotonic_decreasing


def test_no_metric_dominates_by_units(ranked):
    # Each weighted component is a percentile, so the best coin on every metric scores 100.
    r, _ = ranked
    best = r.copy()
    for col in ["return_90d", "sharpe_90d", "sma50_vs_sma200", "max_drawdown_90d", "volume_to_mcap"]:
        best.loc[best.index[0], col] = best[col].max() + 1
    assert ca.rank(best)["score"].iloc[0] == pytest.approx(100.0)


def test_stablecoins_excluded_list():
    assert {"USDT", "USDC", "DAI"} <= ca.STABLECOINS


def test_correlation_excludes_anchor(ranked):
    _, prices = ranked
    c = ca.correlation_to(prices, "SAMPLE-01")
    assert "SAMPLE-01" not in c.index
    assert c.between(-1, 1).all()
