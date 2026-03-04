#!/usr/bin/env python3
"""
All Weather Portfolio Backtesting Tool
======================================
Simulates a Ray Dalio-inspired All Weather portfolio using ASX-listed ETFs
and direct equities. Pulls historical price data via yfinance, runs a
backtest with periodic rebalancing, and outputs performance metrics + charts.
"""

import os
import io
import time
import datetime
import warnings
import contextlib

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

STARTING_CAPITAL = 100_000  # AUD
REBALANCE_FREQ = "quarterly"  # "monthly", "quarterly", "annual"
RISK_FREE_RATE = 0.04  # annualised, for Sharpe / Sortino
TRANSACTION_COST_BPS = 0  # basis points per trade (0 = no cost)
START_DATE = None  # None = as far back as data allows
END_DATE = None  # None = latest available
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DOWNLOAD_DELAY = 2.0   # seconds between successful yfinance calls
RATE_LIMIT_WAIT = 15   # base seconds to wait when rate-limited (multiplied by attempt)
MAX_ATTEMPTS = 5       # max retries per ticker

# ---------------------------------------------------------------------------
# Portfolio sleeve definitions
# ---------------------------------------------------------------------------

# Non-equity sleeves: ticker -> total portfolio weight
SLEEVE_ALLOCATIONS = {
    "Long-Term Bonds":      {"tickers": {"GGOV.AX": 1.0}, "weight": 0.40},
    "Intermediate Bonds":   {"tickers": {"US10.AX": 1.0}, "weight": 0.15},
    "Broad Commodities":    {"tickers": {"BCOM.AX": 1.0}, "weight": 0.065},
    "Gold":                 {"tickers": {"GOLD.AX": 1.0}, "weight": 0.065},
    "Bitcoin":              {"tickers": {"BTC-AUD": 1.0}, "weight": 0.02},
}

# Equity sleeve (30% of total)
EQUITY_WEIGHT = 0.30
EQUITY_HOLDINGS = {
    "CBA.AX": 0.080, "CSL.AX": 0.060, "WBC.AX": 0.060, "BHP.AX": 0.060,
    "MQG.AX": 0.050, "WES.AX": 0.050, "WOW.AX": 0.050, "TCL.AX": 0.050,
    "TLS.AX": 0.045, "VGS.AX": 0.040, "COL.AX": 0.040, "AMC.AX": 0.040,
    "ALL.AX": 0.040, "GMG.AX": 0.040, "APA.AX": 0.030, "ATEC.AX": 0.030,
    "BXB.AX": 0.030, "EDV.AX": 0.025, "CHC.AX": 0.020, "COH.AX": 0.020,
    "CPU.AX": 0.020, "NDQ.AX": 0.020, "PME.AX": 0.020, "REA.AX": 0.020,
    "RMD.AX": 0.020, "RIO.AX": 0.020, "WTC.AX": 0.020,
}

# Proxy map: ASX ETF -> US equivalent for backfill
PROXY_MAP = {
    "GGOV.AX": "TLT",
    "US10.AX": "IEF",
    "BCOM.AX": "DBC",
    "GOLD.AX": "GLD",
}

# Benchmarks
BENCHMARKS = {
    "ASX 200": {"STW.AX": 1.0},
    "S&P 500": {"SPY": 1.0},
    "Classic 60/40": {"STW.AX": 0.60, "GGOV.AX": 0.40},
    "All Weather (VAS)": None,  # built dynamically below
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_all_weather_vas():
    """Build the pure All Weather variant that replaces equities with VAS."""
    alloc = {}
    for sleeve in SLEEVE_ALLOCATIONS.values():
        for t, w in sleeve["tickers"].items():
            alloc[t] = sleeve["weight"] * w
    alloc["VAS.AX"] = EQUITY_WEIGHT
    return alloc


def build_target_weights():
    """Return a dict {ticker: total_portfolio_weight} for the main portfolio."""
    weights = {}
    for sleeve in SLEEVE_ALLOCATIONS.values():
        for t, w in sleeve["tickers"].items():
            weights[t] = sleeve["weight"] * w
    for t, w in EQUITY_HOLDINGS.items():
        weights[t] = EQUITY_WEIGHT * w
    return weights


def ticker_to_sleeve(ticker):
    """Map a ticker to its sleeve name for contribution charts."""
    for name, sleeve in SLEEVE_ALLOCATIONS.items():
        if ticker in sleeve["tickers"]:
            return name
    if ticker in EQUITY_HOLDINGS:
        return "Stocks"
    if ticker == "VAS.AX":
        return "Stocks"
    return "Other"


def _output_has_rate_limit(text):
    """Check if captured yfinance output contains rate-limit messages."""
    t = text.lower().replace(" ", "")
    return "ratelimit" in t or "429" in t or "toomany" in t


def _download_single(ticker, start, end):
    """Download a single ticker using yf.Ticker().history() API.

    Returns a Series or None. Uses the Ticker API instead of yf.download()
    because it tends to be more resilient to rate limiting.
    """
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), \
             contextlib.redirect_stderr(captured):
            tkr = yf.Ticker(ticker)
            df = tkr.history(start=start, end=end, auto_adjust=True)
    except Exception:
        return None, captured.getvalue()

    yf_output = captured.getvalue()

    if df is not None and not df.empty and "Close" in df.columns:
        series = df["Close"].dropna().squeeze()
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series.name = ticker
        return series, yf_output
    return None, yf_output


def download_all_prices(tickers, start="2005-01-01", end=None):
    """Download prices one at a time with adaptive rate-limit handling.

    Uses yf.Ticker().history() instead of yf.download() for better
    rate-limit resilience.  Downloads in small batches with pauses.
    """
    proxy_log = {}
    unique_tickers = sorted(set(tickers))

    # Also download proxies we might need
    proxies_needed = {v for k, v in PROXY_MAP.items() if k in unique_tickers}
    all_to_download = sorted(set(unique_tickers) | proxies_needed)

    total = len(all_to_download)
    print(f"Downloading data for {total} tickers...")
    print(f"  yfinance version: {yf.__version__}")

    prices = {}
    failed_tickers = []
    consecutive_fails = 0

    for i, t in enumerate(all_to_download, 1):
        print(f"  [{i}/{total}] {t}...", end=" ", flush=True)

        series, yf_output = _download_single(t, start, end)

        if series is not None and len(series) > 0:
            prices[t] = series
            print(f"OK ({len(series)} days)")
            consecutive_fails = 0
        elif _output_has_rate_limit(yf_output):
            print("rate-limited")
            failed_tickers.append(t)
            consecutive_fails += 1
            # Adaptive backoff: longer waits as consecutive failures mount
            wait = RATE_LIMIT_WAIT * min(consecutive_fails, 4)
            print(f"    Cooling down {wait}s...", flush=True)
            time.sleep(wait)
        else:
            # Print what yfinance said for debugging
            snippet = yf_output.strip()[:200] if yf_output.strip() else "(empty)"
            print(f"no data  [yf: {snippet}]")
            failed_tickers.append(t)
            consecutive_fails = 0

        # Small delay between tickers to stay under radar
        if i < total:
            time.sleep(DOWNLOAD_DELAY)

    # Retry pass for tickers that were rate-limited
    if failed_tickers:
        print(f"\n  Waiting {RATE_LIMIT_WAIT * 2}s before retrying "
              f"{len(failed_tickers)} failed tickers...")
        time.sleep(RATE_LIMIT_WAIT * 2)
        still_failed = []
        for i, t in enumerate(failed_tickers, 1):
            print(f"  [retry {i}/{len(failed_tickers)}] {t}...",
                  end=" ", flush=True)
            series, yf_output = _download_single(t, start, end)
            if series is not None and len(series) > 0:
                prices[t] = series
                print(f"OK ({len(series)} days)")
            else:
                print("FAILED")
                still_failed.append(t)
            time.sleep(DOWNLOAD_DELAY * 2)
        failed_tickers = still_failed

    if failed_tickers:
        print(f"\n  No data for: {', '.join(failed_tickers)}")

    # Proxy backfill: extend ASX ETF history with US equivalent
    for asx_ticker, us_proxy in PROXY_MAP.items():
        if asx_ticker not in unique_tickers:
            continue
        if asx_ticker not in prices:
            # No ASX data at all — use proxy entirely
            if us_proxy in prices:
                print(f"  Using {us_proxy} as full proxy for {asx_ticker}")
                prices[asx_ticker] = prices[us_proxy].rename(asx_ticker)
                proxy_log[asx_ticker] = {
                    "proxy": us_proxy,
                    "from": str(prices[us_proxy].index.min().date()),
                    "to": str(prices[us_proxy].index.max().date()),
                    "type": "full",
                }
            continue

        asx_start = prices[asx_ticker].index.min()
        if us_proxy in prices:
            proxy_data = prices[us_proxy]
            proxy_before = proxy_data.loc[proxy_data.index < asx_start]
            if len(proxy_before) > 0:
                # Scale proxy so its last value matches the ASX ETF's first value
                scale = prices[asx_ticker].iloc[0] / proxy_before.iloc[-1]
                proxy_scaled = proxy_before * scale
                combined = pd.concat([proxy_scaled, prices[asx_ticker]])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                combined.name = asx_ticker
                prices[asx_ticker] = combined
                proxy_log[asx_ticker] = {
                    "proxy": us_proxy,
                    "from": str(proxy_before.index.min().date()),
                    "to": str(proxy_before.index.max().date()),
                    "type": "backfill",
                }
                print(f"  Backfilled {asx_ticker} with {us_proxy} "
                      f"({proxy_log[asx_ticker]['from']} to "
                      f"{proxy_log[asx_ticker]['to']})")

    return prices, proxy_log


def build_returns_df(prices, tickers):
    """Build an aligned daily returns DataFrame for the given tickers."""
    frames = []
    available = []
    for t in tickers:
        if t in prices:
            frames.append(prices[t])
            available.append(t)
        else:
            print(f"  [warn] No price data for {t} — skipping")
    if not frames:
        return pd.DataFrame(), []
    df = pd.concat(frames, axis=1)
    df.columns = available
    # Ensure tz-naive DatetimeIndex (mixed tz from yf.Ticker().history())
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.sort_index().ffill()
    # Trim to common start
    df = df.dropna(how="any")
    returns = df.pct_change().iloc[1:]
    return returns, available


def get_rebalance_dates(index, freq):
    """Return rebalance dates from a DatetimeIndex."""
    if freq == "monthly":
        groups = index.to_period("M")
    elif freq == "quarterly":
        groups = index.to_period("Q")
    elif freq == "annual":
        groups = index.to_period("Y")
    else:
        raise ValueError(f"Unknown frequency: {freq}")

    dates = []
    seen = set()
    for dt, period in zip(index, groups):
        if period not in seen:
            dates.append(dt)
            seen.add(period)
    return dates


def run_backtest(returns, target_weights, starting_capital, rebalance_freq,
                 tx_cost_bps=0):
    """
    Run a backtest given daily returns and target weights.

    Returns:
        portfolio_value: pd.Series of daily portfolio value
        sleeve_values: pd.DataFrame of daily value per-sleeve (for contribution)
    """
    available = [t for t in target_weights if t in returns.columns]
    if not available:
        return pd.Series(dtype=float), pd.DataFrame()

    # Normalise weights to available tickers
    raw = {t: target_weights[t] for t in available}
    total_w = sum(raw.values())
    if total_w == 0:
        return pd.Series(dtype=float), pd.DataFrame()
    weights = {t: w / total_w for t, w in raw.items()}

    ret = returns[available].copy()
    dates = ret.index
    rebal_dates = set(get_rebalance_dates(dates, rebalance_freq))

    n_assets = len(available)
    w = np.array([weights[t] for t in available])
    port_values = np.empty(len(dates))
    sleeve_matrix = np.empty((len(dates), n_assets))
    current_values = w * starting_capital
    tx_frac = tx_cost_bps / 10_000

    for i, dt in enumerate(dates):
        day_ret = ret.iloc[i].values
        current_values = current_values * (1 + day_ret)
        port_val = current_values.sum()
        port_values[i] = port_val
        sleeve_matrix[i] = current_values

        if dt in rebal_dates and i < len(dates) - 1:
            new_values = w * port_val
            turnover = np.abs(new_values - current_values).sum()
            cost = turnover * tx_frac
            port_val -= cost
            current_values = w * port_val

    portfolio_value = pd.Series(port_values, index=dates, name="Portfolio")
    sleeve_values = pd.DataFrame(sleeve_matrix, index=dates, columns=available)
    return portfolio_value, sleeve_values


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calc_metrics(portfolio_value, risk_free_rate=RISK_FREE_RATE):
    """Calculate a dict of performance metrics from a portfolio value series."""
    if portfolio_value.empty or len(portfolio_value) < 2:
        return {}

    total_ret = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
    n_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    n_years = n_days / 365.25
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    daily_ret = portfolio_value.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)

    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0
    sortino = (cagr - risk_free_rate) / downside if downside > 0 else 0

    # Drawdown
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Monthly returns for best/worst month and % positive
    monthly = portfolio_value.resample("ME").last().pct_change().dropna()
    best_month = monthly.max() if len(monthly) > 0 else 0
    worst_month = monthly.min() if len(monthly) > 0 else 0
    pct_positive_months = (monthly > 0).mean() * 100 if len(monthly) > 0 else 0

    # Annual returns for best/worst year
    annual = portfolio_value.resample("YE").last().pct_change().dropna()
    best_year = annual.max() if len(annual) > 0 else 0
    worst_year = annual.min() if len(annual) > 0 else 0

    return {
        "Total Return": f"{total_ret:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Volatility": f"{vol:.2%}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Best Year": f"{best_year:.2%}",
        "Worst Year": f"{worst_year:.2%}",
        "Best Month": f"{best_month:.2%}",
        "Worst Month": f"{worst_month:.2%}",
        "% Positive Months": f"{pct_positive_months:.1f}%",
        "Period": f"{portfolio_value.index[0].date()} to {portfolio_value.index[-1].date()}",
    }


# ============================================================================
# CHART GENERATION
# ============================================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def chart_equity_curves(results, filename="equity_curves.png"):
    """Chart 1: Equity curves for all portfolios (log scale)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10.colors
    for i, (name, pv) in enumerate(results.items()):
        ax.plot(pv.index, pv.values, label=name, linewidth=1.4,
                color=colors[i % len(colors)])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))
    ax.set_title("Equity Curves (Log Scale)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (AUD)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def chart_drawdowns(results, filename="drawdowns.png"):
    """Chart 2: Drawdown from peak."""
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10.colors
    for i, (name, pv) in enumerate(results.items()):
        running_max = pv.cummax()
        dd = (pv - running_max) / running_max * 100
        ax.plot(dd.index, dd.values, label=name, linewidth=1.2,
                color=colors[i % len(colors)])
    ax.set_title("Drawdown from Peak", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def chart_rolling_returns(results, filename="rolling_12m_returns.png"):
    """Chart 3: Rolling 12-month returns — All Weather vs ASX 200."""
    fig, ax = plt.subplots(figsize=(14, 6))
    window = 252  # approx trading days in a year
    for name in ["All Weather", "ASX 200"]:
        if name not in results:
            continue
        pv = results[name]
        rolling = pv.pct_change(window).dropna() * 100
        ax.plot(rolling.index, rolling.values, label=name, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Rolling 12-Month Returns", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def chart_sleeve_contribution(sleeve_values, filename="sleeve_contribution.png"):
    """Chart 4: Stacked area chart of sleeve contributions."""
    if sleeve_values.empty:
        print("  [skip] No sleeve data for contribution chart")
        return
    # Group columns by sleeve
    sleeve_map = {}
    for col in sleeve_values.columns:
        s = ticker_to_sleeve(col)
        sleeve_map.setdefault(s, []).append(col)

    grouped = pd.DataFrame(index=sleeve_values.index)
    for sleeve_name, cols in sleeve_map.items():
        grouped[sleeve_name] = sleeve_values[cols].sum(axis=1)

    # Resample to weekly for cleaner chart
    grouped = grouped.resample("W").last().dropna()

    fig, ax = plt.subplots(figsize=(14, 7))
    sleeve_order = ["Long-Term Bonds", "Intermediate Bonds",
                    "Broad Commodities", "Gold", "Bitcoin", "Stocks"]
    ordered_cols = [c for c in sleeve_order if c in grouped.columns]
    remaining = [c for c in grouped.columns if c not in ordered_cols]
    ordered_cols += remaining

    ax.stackplot(grouped.index, *[grouped[c] for c in ordered_cols],
                 labels=ordered_cols, alpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))
    ax.set_title("Asset Sleeve Contribution to Portfolio Value", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (AUD)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def chart_annual_returns(results, filename="annual_returns.png"):
    """Chart 5: Grouped bar chart of calendar-year returns."""
    annual_data = {}
    for name, pv in results.items():
        ann = pv.resample("YE").last().pct_change().dropna() * 100
        ann.index = ann.index.year
        annual_data[name] = ann

    all_years = sorted(
        set().union(*(a.index.tolist() for a in annual_data.values())))
    if not all_years:
        return

    fig, ax = plt.subplots(figsize=(max(14, len(all_years) * 1.5), 7))
    n_ports = len(annual_data)
    bar_width = 0.8 / n_ports
    x = np.arange(len(all_years))
    colors = plt.cm.tab10.colors

    for i, (name, ann) in enumerate(annual_data.items()):
        vals = [ann.get(y, 0) for y in all_years]
        ax.bar(x + i * bar_width, vals, bar_width, label=name,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + bar_width * (n_ports - 1) / 2)
    ax.set_xticklabels(all_years, rotation=45)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Calendar Year Returns (%)", fontsize=14)
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def chart_monthly_heatmap(portfolio_value, filename="monthly_heatmap.png"):
    """Chart 6: Monthly returns heatmap (months x years) for All Weather."""
    monthly = portfolio_value.resample("ME").last().pct_change().dropna() * 100
    years = sorted(monthly.index.year.unique())
    months = list(range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    grid = np.full((len(years), 12), np.nan)
    for dt, val in monthly.items():
        yi = years.index(dt.year)
        mi = dt.month - 1
        grid[yi, mi] = val

    fig, ax = plt.subplots(figsize=(12, max(4, len(years) * 0.45)))
    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(grid, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_title("All Weather — Monthly Returns Heatmap (%)", fontsize=14)

    # Annotate cells
    for yi in range(len(years)):
        for mi in range(12):
            val = grid[yi, mi]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(mi, yi, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Return (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ALL WEATHER PORTFOLIO BACKTEST")
    print("=" * 70)
    print(f"Starting capital:  ${STARTING_CAPITAL:,.0f}")
    print(f"Rebalance freq:    {REBALANCE_FREQ}")
    print(f"Risk-free rate:    {RISK_FREE_RATE:.2%}")
    print(f"Transaction cost:  {TRANSACTION_COST_BPS} bps")
    print()

    ensure_output_dir()

    # Build target weights
    aw_weights = build_target_weights()
    vas_weights = build_all_weather_vas()

    # Collect all tickers we need
    all_tickers = set(aw_weights.keys())
    all_tickers.add("STW.AX")   # ASX 200 benchmark
    all_tickers.add("SPY")      # S&P 500 benchmark
    all_tickers.add("VAS.AX")   # Pure All Weather benchmark
    for bm_tickers in BENCHMARKS.values():
        if bm_tickers:
            all_tickers.update(bm_tickers.keys())

    # Download all prices
    start = START_DATE or "2005-01-01"
    end = END_DATE
    prices, proxy_log = download_all_prices(list(all_tickers), start=start,
                                            end=end)
    print(f"\nDownloaded {len(prices)} tickers successfully.\n")

    if proxy_log:
        print("Proxy backfill log:")
        for ticker, info in proxy_log.items():
            print(f"  {ticker} -> {info['proxy']} ({info['type']}: "
                  f"{info['from']} to {info['to']})")
        print()

    # Build returns for All Weather
    aw_tickers = list(aw_weights.keys())
    aw_returns, aw_available = build_returns_df(prices, aw_tickers)
    if aw_returns.empty:
        print("[FATAL] No usable data for All Weather portfolio.")
        return

    print(f"All Weather backtest period: {aw_returns.index[0].date()} to "
          f"{aw_returns.index[-1].date()}")
    print(f"Available tickers: {len(aw_available)} / {len(aw_tickers)}")
    missing = set(aw_tickers) - set(aw_available)
    if missing:
        print(f"Missing tickers: {missing}")
    print()

    # --- Run All Weather backtest ---
    print("Running All Weather backtest...")
    aw_value, aw_sleeves = run_backtest(
        aw_returns, aw_weights, STARTING_CAPITAL, REBALANCE_FREQ,
        TRANSACTION_COST_BPS)

    results = {"All Weather": aw_value}

    # --- Benchmarks ---
    # We need to align all benchmarks to the same date range as All Weather
    date_range = aw_value.index

    # ASX 200
    print("Running ASX 200 benchmark...")
    bm_returns, bm_avail = build_returns_df(prices, ["STW.AX"])
    if not bm_returns.empty:
        bm_ret_aligned = bm_returns.reindex(date_range).dropna()
        if not bm_ret_aligned.empty:
            bm_val, _ = run_backtest(bm_ret_aligned, {"STW.AX": 1.0},
                                     STARTING_CAPITAL, REBALANCE_FREQ)
            results["ASX 200"] = bm_val

    # S&P 500
    print("Running S&P 500 benchmark...")
    bm_returns, bm_avail = build_returns_df(prices, ["SPY"])
    if not bm_returns.empty:
        bm_ret_aligned = bm_returns.reindex(date_range).dropna()
        if not bm_ret_aligned.empty:
            bm_val, _ = run_backtest(bm_ret_aligned, {"SPY": 1.0},
                                     STARTING_CAPITAL, REBALANCE_FREQ)
            results["S&P 500"] = bm_val

    # Classic 60/40
    print("Running Classic 60/40 benchmark...")
    bm_60_40 = {"STW.AX": 0.60, "GGOV.AX": 0.40}
    bm_returns, bm_avail = build_returns_df(prices, list(bm_60_40.keys()))
    if not bm_returns.empty:
        bm_ret_aligned = bm_returns.reindex(date_range).dropna()
        if not bm_ret_aligned.empty:
            bm_val, _ = run_backtest(bm_ret_aligned, bm_60_40,
                                     STARTING_CAPITAL, REBALANCE_FREQ)
            results["Classic 60/40"] = bm_val

    # All Weather (VAS)
    print("Running All Weather (VAS) benchmark...")
    vas_tickers = list(vas_weights.keys())
    bm_returns, bm_avail = build_returns_df(prices, vas_tickers)
    if not bm_returns.empty:
        bm_ret_aligned = bm_returns.reindex(date_range).dropna()
        if not bm_ret_aligned.empty:
            bm_val, _ = run_backtest(bm_ret_aligned, vas_weights,
                                     STARTING_CAPITAL, REBALANCE_FREQ)
            results["All Weather (VAS)"] = bm_val

    # Align all results to a common date range
    common_start = max(pv.index[0] for pv in results.values())
    common_end = min(pv.index[-1] for pv in results.values())
    for name in list(results.keys()):
        pv = results[name]
        pv = pv.loc[common_start:common_end]
        # Re-scale so all start at STARTING_CAPITAL
        if len(pv) > 0:
            pv = pv / pv.iloc[0] * STARTING_CAPITAL
        results[name] = pv

    # Also trim sleeve values to common range
    aw_sleeves = aw_sleeves.loc[common_start:common_end]
    # Rescale sleeves proportionally
    if len(aw_sleeves) > 0 and "All Weather" in results:
        scale = results["All Weather"].iloc[0] / aw_sleeves.iloc[0].sum()
        aw_sleeves = aw_sleeves * scale

    print(f"\nCommon backtest period: {common_start.date()} to "
          f"{common_end.date()}")

    # --- Metrics ---
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    metrics_dict = {}
    for name, pv in results.items():
        metrics_dict[name] = calc_metrics(pv)

    metrics_df = pd.DataFrame(metrics_dict)
    print(metrics_df.to_string())

    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "performance_metrics.csv")
    metrics_df.to_csv(csv_path)
    print(f"\nMetrics saved to {csv_path}")

    # --- Charts ---
    print("\nGenerating charts...")
    chart_equity_curves(results)
    chart_drawdowns(results)
    chart_rolling_returns(results)
    chart_sleeve_contribution(aw_sleeves)
    chart_annual_returns(results)
    if "All Weather" in results:
        chart_monthly_heatmap(results["All Weather"])

    # --- Proxy log to file ---
    if proxy_log:
        proxy_path = os.path.join(OUTPUT_DIR, "proxy_log.csv")
        pd.DataFrame(proxy_log).T.to_csv(proxy_path)
        print(f"\nProxy log saved to {proxy_path}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
