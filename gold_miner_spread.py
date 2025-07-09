import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pysr import PySRRegressor
import matplotlib.ticker as mtick
import warnings

warnings.filterwarnings("ignore")


# 1. Most Recent Trading Day


def get_most_recent_trading_day():
    """Return last trading day as an ISO date string."""
    today = datetime.today()
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    return today.strftime('%Y-%m-%d')


# 2. Load Price Data


def load_data(ticker, start="2006-06-01"):
    """Download price data for a ticker and return a clean price series."""
    end = get_most_recent_trading_day()
    # Use auto_adjust=False so the old column names (including ``Adj Close``)
    # are available. yfinance now returns a MultiIndex by default, so handle
    # that here and extract a Series instead of a DataFrame.
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        # Prefer adjusted close prices if present
        if ("Adj Close", ticker) in data.columns:
            return data[("Adj Close", ticker)].dropna()
        if ("Close", ticker) in data.columns:
            return data[("Close", ticker)].dropna()
    else:
        if "Adj Close" in data.columns:
            return data["Adj Close"].dropna()
        if "Close" in data.columns:
            return data["Close"].dropna()

    raise ValueError(f"'Adj Close' and 'Close' not found for {ticker}")

# 3. Create Lag Matrix


def lag_matrix(series, lags=10):
    """Return DataFrame of lagged values for a series."""
    lagged_data = []
    for i in range(1, lags + 1):
        lag = series.shift(i)
        lag.name = f"lag_{i}"
        lagged_data.append(lag)
    df = pd.concat(lagged_data, axis=1)
    df.dropna(inplace=True)
    return df

# 4. Rolling Correlation Change Filter


def rolling_corr_change(series1, series2, window=15):
    """Return 1 when correlation decreases, otherwise 0."""
    corr = series1.rolling(window).corr(series2)
    corr_change = corr.diff()
    return (corr_change < 0).astype(int)


def optimal_corr_window(series1, series2, windows):
    """Return the window length with the highest mean absolute correlation."""
    best_window = None
    best_score = -np.inf
    for w in windows:
        corr = series1.rolling(w).corr(series2).abs().mean()
        if corr > best_score:
            best_score = corr
            best_window = w
    return best_window


# 5. Load GDX and GLD data
gdx = load_data("GDX")
gld = load_data("GLD")

# Rolling correlation change between GLD and GDX returns
gdx_ret = gdx.pct_change()
gld_ret = gld.pct_change()
candidate_windows = range(10, 31, 5)
best_window = optimal_corr_window(gld_ret, gdx_ret, candidate_windows)
corr_filter_series = rolling_corr_change(gld_ret, gdx_ret, window=best_window)
print(f"Optimal correlation window: {best_window}")

# 6. Calculate spread using formula provided
aligned = pd.concat([gld, gdx], axis=1, join='inner')
aligned.columns = ['GLD', 'GDX']
spread = aligned['GLD'].pct_change() - aligned['GDX'].pct_change()
spread = spread.dropna()

# 7. Split into Train and Test
split_point = int(len(spread) * 0.7)
train_series = spread[:split_point]
test_series = spread[split_point:]

lags = 10
train_features = lag_matrix(train_series, lags=lags)
train_targets = train_series[train_features.index]
test_features = lag_matrix(test_series, lags=lags)
test_targets = test_series[test_features.index]

# 8. Symbolic Regressor
model = PySRRegressor(
    niterations=10000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log"],
    population_size=1000,
    model_selection="best",
    loss="L2DistLoss()",
    maxsize=7,
    tournament_selection_n=20,
    verbosity=1,
    random_state=42
)

model.fit(train_features.values, train_targets.values)

# 9. Predict using the 10 best models and apply correlation change filter
top_equations = model.equations_.sort_values("loss").head(10)
pred_list = []
for idx in top_equations.index:
    pred_list.append(model.predict(test_features.values, index=idx))
raw_preds = np.mean(np.column_stack(pred_list), axis=1)

# Align correlation filter to prediction index
filter_values = (
    corr_filter_series.loc[test_targets.index]
    .shift(1)
    .fillna(0)
    .to_numpy()
    .ravel()
)

# 10. Backtest Strategy
returns = test_targets.diff().fillna(0).values
n_preds = len(raw_preds)
aligned_returns = returns[-n_preds:]

# Determine daily position using correlation filter
pred_sign = np.sign(raw_preds)
positions = np.zeros_like(pred_sign)
for i in range(1, n_preds):
    if filter_values[i] == 1:
        # Follow model prediction when correlation change is negative
        positions[i] = pred_sign[i]
    else:
        # Otherwise keep previous day's position
        positions[i] = positions[i - 1]

signals = positions[1:]
assert len(signals) == len(aligned_returns[1:]), (
    "signals and returns must be equal length"
)
strategy_returns = signals * aligned_returns[1:]

# Transaction costs (5 bps per buy/sell) applied only when the position changes
tc_rate = 0.0005
trade_costs = tc_rate * np.abs(np.diff(positions))
strategy_returns -= trade_costs

# Track the number of transactions executed by the strategy and the
# benchmark. A position change from long to short counts as two
# transactions, matching the transaction cost calculation above.
num_transactions_strategy = int(np.sum(np.abs(np.diff(positions))))
num_transactions_benchmark = 1

# Buy and hold: open a long position on the spread on day 1 and keep it
# for the rest of the period. We subtract the transaction cost for the
# initial trade only.
benchmark_returns = aligned_returns[1:].copy()
benchmark_returns[0] -= tc_rate


def annualized_return(returns, periods_per_year=252):
    """Annualized compounded return from a series of returns."""
    total_return = np.prod(1 + returns) - 1
    n_years = len(returns) / periods_per_year
    return (1 + total_return) ** (1 / n_years) - 1


def annualized_std(returns, periods_per_year=252):
    """Annualized standard deviation of returns."""
    return np.std(returns) * np.sqrt(periods_per_year)


def sharpe_ratio_func(returns, periods_per_year=252):
    """Return Sharpe ratio for a series of returns."""
    if np.std(returns) == 0:
        return np.nan
    return annualized_return(returns, periods_per_year) / annualized_std(
        returns, periods_per_year
    )


def max_drawdown(returns):
    """Return the maximum drawdown of a series of returns."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = 1 - cumulative / running_max
    return np.max(drawdowns)


def calmar_ratio(returns, periods_per_year=252):
    """Return the Calmar ratio for a series of returns."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns, periods_per_year) / mdd


# Evaluate performance after transaction costs
sharpe_ratio = sharpe_ratio_func(strategy_returns)

# 11. Plot Cumulative Returns in Percent with Background Signal Coloring
cumulative_strategy = np.cumprod(1 + strategy_returns) - 1
cumulative_benchmark = np.cumprod(1 + benchmark_returns) - 1

# Plot returns
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cumulative_strategy))
ax.plot(x, cumulative_strategy * 100, label="Strategy")
ax.plot(x, cumulative_benchmark * 100, label="Buy & Hold")

# Background shading based on signal
signal_colors = {1: 'lightgreen', -1: 'lightcoral', 0: 'white'}
prev_sig = signals[0]
start_idx = 0

for i in range(1, len(signals)):
    if signals[i] != prev_sig or i == len(signals) - 1:
        end_idx = i
        ax.axvspan(
            start_idx,
            end_idx,
            color=signal_colors.get(prev_sig, 'white'),
            alpha=0.3,
        )
        start_idx = i
        prev_sig = signals[i]

ax.set_title(f"Cumulative Returns (%) | Sharpe Ratio: {sharpe_ratio:.2f}")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("Trading Days")
ax.legend()
ax.grid(True)
plt.tight_layout()


# 12. Output Return Table

metrics_df = pd.DataFrame({
    "Cumulative Return (%)": [
        cumulative_strategy[-1] * 100,
        cumulative_benchmark[-1] * 100
    ],
    "Annualized Return (%)": [
        annualized_return(strategy_returns) * 100,
        annualized_return(benchmark_returns) * 100
    ],
    "Max Drawdown (%)": [
        max_drawdown(strategy_returns) * 100,
        max_drawdown(benchmark_returns) * 100,
    ],
    "Annualized Std Dev (%)": [
        annualized_std(strategy_returns) * 100,
        annualized_std(benchmark_returns) * 100
    ],
    "Sharpe Ratio": [
        sharpe_ratio_func(strategy_returns),
        sharpe_ratio_func(benchmark_returns)
    ],
    "Calmar Ratio": [
        calmar_ratio(strategy_returns),
        calmar_ratio(benchmark_returns),
    ],
    "Transactions": [
        num_transactions_strategy,
        num_transactions_benchmark,
    ]
}, index=["Strategy", "Buy & Hold"])

# Show table
print("\nStrategy Performance Summary:")
print(metrics_df.round(2))
plt.show()

# 13. Print Equation
print("\nBest Discovered Equation:")
print(model.sympy())
