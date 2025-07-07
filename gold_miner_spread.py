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


def load_data(ticker, start='2006-06-01'):
    """Download price data for a ticker and return a clean series."""
    end = get_most_recent_trading_day()
    data = yf.download(ticker, start=start, end=end, progress=False)
    if "Adj Close" in data.columns:
        return data["Adj Close"].dropna()
    elif "Close" in data.columns:
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


# 5. Load GDX and GLD data
gdx = load_data("GDX")
gld = load_data("GLD")

# Rolling correlation change between GLD and GDX returns
gdx_ret = gdx.pct_change()
gld_ret = gld.pct_change()
corr_filter_series = rolling_corr_change(gld_ret, gdx_ret)

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
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log"],
    population_size=1000,
    model_selection="best",
    loss="L2DistLoss()",
    maxsize=20,
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
filter_values = corr_filter_series.loc[test_targets.index].fillna(0).values

# 10. Backtest Strategy
returns = test_targets.diff().fillna(0).values
n_preds = len(raw_preds)
aligned_returns = returns[-n_preds:]
signals = np.sign(raw_preds[1:]) * filter_values[1:]
assert len(signals) == len(aligned_returns[1:]), (
    "signals and returns must be equal length"
)
strategy_returns = signals * aligned_returns[1:]
benchmark_returns = aligned_returns[1:]


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


sharpe_ratio = sharpe_ratio_func(strategy_returns)

# 11. Plot Cumulative Returns in Percent with Background Signal Coloring
cumulative_strategy = np.cumsum(strategy_returns)
cumulative_benchmark = np.cumsum(benchmark_returns)

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
    "Annualized Std Dev (%)": [
        annualized_std(strategy_returns) * 100,
        annualized_std(benchmark_returns) * 100
    ],
    "Sharpe Ratio": [
        sharpe_ratio_func(strategy_returns),
        sharpe_ratio_func(benchmark_returns)
    ]
}, index=["Strategy", "Buy & Hold"])

# Show table
print("\nStrategy Performance Summary:")
print(metrics_df.round(2))
plt.show()

# 13. Print Equation
print("\nBest Discovered Equation:")
print(model.sympy())
