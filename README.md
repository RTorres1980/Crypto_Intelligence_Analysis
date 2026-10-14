# Crypto Intelligence Analysis

![CI](https://github.com/RTorres1980/Crypto_Intelligence_Analysis/actions/workflows/ci.yml/badge.svg)

Ranks the largest cryptocurrencies on recent momentum, risk-adjusted return, trend, drawdown and liquidity, and shows how closely they move together. The analysis is a tested Python module (`crypto_analysis.py`) and a notebook that walks through the results with charts.

## Method

1. **Universe**: the top 50 coins by market cap from the CoinGecko API, excluding stablecoins (USDT, USDC, …) and wrapped or staked tokens (WBTC, stETH, …), which only track another asset.
2. **Prices**: daily closes from Yahoo Finance. Coins with incomplete history are dropped.
3. **Metrics** (last 90 days): return, annualized volatility, return-to-volatility (Sharpe-style), maximum drawdown, 50-day vs 200-day moving average, 14-day RSI, and trading volume as a share of market cap.
4. **Score (0–100)**: each metric is converted to a percentile rank across coins, then weighted: Sharpe 30%, momentum 25%, trend 15%, drawdown 15%, liquidity 15%. Using ranks keeps a metric measured on a 0–100 scale from overwhelming ratios measured around 0–1.
5. **RSI flag**: RSI is reported as Overbought (>70) or Oversold (<30) next to the score rather than mixed into it.
6. **Correlation**: 90-day daily-return correlations to the market leader and among the top 10.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook Crypto_Intelligence_Analysis.ipynb
pytest -q
```

The notebook's first cell has `LIVE = False`. With that setting it uses **synthetic sample coins** (SAMPLE-01 …), so it runs without network access. The outputs saved in the notebook come from that sample. Set `LIVE = True` to pull current data from CoinGecko and Yahoo Finance.

From Python:

```python
import crypto_analysis as ca
universe = ca.fetch_universe(50)
prices = ca.fetch_prices(universe["symbol"], start="2024-01-01")
ranked = ca.rank(ca.metrics(prices, universe))
```

## Tests

The tests check RSI at its extremes, the max-drawdown calculation, that scores fall between 0 and 100 and are sorted, that a coin best on every metric scores exactly 100 (so no single metric dominates), and the correlation output.

## Limits

The score describes recent behavior; it is not a forecast or investment advice. A 90-day window is short for an asset class this volatile, so try a longer `lookback` to see how stable the ranking is.

**Author:** Robert Torres · MIT License
