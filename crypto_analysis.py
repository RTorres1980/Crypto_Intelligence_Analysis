"""Rank large-cap cryptocurrencies on momentum, risk-adjusted return, trend and liquidity.

Data:
  live    top coins by market cap from the CoinGecko API, daily prices from Yahoo Finance
  sample  synthetic coins (SAMPLE-01 ...) for offline runs, tests and documentation

Every metric is converted to a percentile rank before weighting, so no single metric
dominates because of its units.
"""
import numpy as np
import pandas as pd

STABLECOINS = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "PYUSD", "USDD", "BUSD", "USD1"}
WRAPPED_OR_STAKED = {"WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBBTC", "WBETH", "RETH"}
WEIGHTS = {"momentum_90d": 0.25, "sharpe_90d": 0.30, "trend": 0.15, "drawdown": 0.15, "liquidity": 0.15}
TRADING_DAYS = 365  # crypto trades every day


# ---------- Data ----------

def fetch_universe(limit=50):
    """Top coins by market cap from CoinGecko, excluding stablecoins and wrapped/staked tokens."""
    import requests
    r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                     params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1},
                     timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())[["symbol", "name", "market_cap", "total_volume"]]
    df["symbol"] = df["symbol"].str.upper()
    return df[~df["symbol"].isin(STABLECOINS | WRAPPED_OR_STAKED)].reset_index(drop=True)


def fetch_prices(symbols, start):
    """Daily closes from Yahoo Finance; symbols Yahoo doesn't list are dropped."""
    import yfinance as yf
    tickers = [f"{s}-USD" for s in symbols]
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    raw.columns = [c.replace("-USD", "") for c in raw.columns]
    return raw.dropna(axis=1, thresh=int(0.9 * len(raw)))


def sample_data(n_coins=20, days=540, seed=11):
    """Synthetic market: correlated random walks with a shared market factor."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end="2025-09-30", periods=days, freq="D")
    market = rng.normal(0.0008, 0.03, days)
    cols = {}
    for i in range(n_coins):
        beta = rng.uniform(0.6, 1.5)
        drift = rng.normal(0.0004, 0.0012)
        idio = rng.normal(0, rng.uniform(0.015, 0.05), days)
        cols[f"SAMPLE-{i + 1:02d}"] = 10 ** rng.uniform(-1, 4) * np.exp(np.cumsum(drift + beta * market + idio))
    prices = pd.DataFrame(cols, index=dates)
    universe = pd.DataFrame({"symbol": list(cols), "name": [f"Sample coin {i + 1}" for i in range(n_coins)],
                             "market_cap": np.sort(10 ** rng.uniform(8.5, 12, n_coins))[::-1]})
    universe["total_volume"] = universe["market_cap"] * rng.uniform(0.01, 0.15, n_coins)
    return universe, prices


# ---------- Metrics ----------

def rsi(series, window=14):
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def max_drawdown(series):
    return float((series / series.cummax() - 1).min())


def metrics(prices, universe, lookback=90):
    recent = prices.iloc[-lookback:]
    rets = recent.pct_change().dropna()
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    m = pd.DataFrame({
        "price": prices.iloc[-1],
        "return_90d": recent.iloc[-1] / recent.iloc[0] - 1,
        "volatility_90d": vol,
        "sharpe_90d": rets.mean() * TRADING_DAYS / vol,
        "max_drawdown_90d": recent.apply(max_drawdown),
        "above_sma50": prices.iloc[-1] > prices.rolling(50).mean().iloc[-1],
        "sma50_vs_sma200": prices.rolling(50).mean().iloc[-1] / prices.rolling(200).mean().iloc[-1] - 1,
        "rsi_14": prices.apply(lambda s: rsi(s).iloc[-1]),
    })
    u = universe.set_index("symbol")
    m = m.join(u[["name", "market_cap", "total_volume"]])
    m["volume_to_mcap"] = m["total_volume"] / m["market_cap"]
    return m


def rank(m, weights=WEIGHTS):
    """Composite 0-100 score from percentile ranks of each metric."""
    pr = pd.DataFrame({
        "momentum_90d": m["return_90d"].rank(pct=True),
        "sharpe_90d": m["sharpe_90d"].rank(pct=True),
        "trend": m["sma50_vs_sma200"].rank(pct=True),
        "drawdown": m["max_drawdown_90d"].rank(pct=True),         # less negative is better
        "liquidity": m["volume_to_mcap"].rank(pct=True),
    })
    out = m.copy()
    out["score"] = (100 * sum(pr[k] * w for k, w in weights.items()) / sum(weights.values())).round(1)
    out["rsi_flag"] = np.select([out["rsi_14"] > 70, out["rsi_14"] < 30], ["Overbought", "Oversold"], "")
    return out.sort_values("score", ascending=False)


def correlation_to(prices, anchor, lookback=90):
    rets = prices.iloc[-lookback:].pct_change().dropna()
    return rets.corr()[anchor].drop(anchor).sort_values(ascending=False)
