import numpy as np
import pandas as pd
import yfinance as yf
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data(ticker: str) -> pd.Series:
    data = yf.download(ticker, progress=False)
    return data['Adj Close']

def compute_spread(xau: pd.Series, gold: pd.Series) -> pd.Series:
    spread = np.log(xau / gold)
    spread = spread.dropna()
    return spread

def lag_matrix(series: pd.Series, lags: int) -> np.ndarray:
    df = pd.concat([series.shift(i) for i in range(lags)], axis=1)
    df.columns = [f'lag_{i}' for i in range(lags)]
    return df.iloc[lags:]

def correlation_filter(preds: np.ndarray, actuals: np.ndarray, window: int = 50) -> np.ndarray:
    filtered = []
    for t in range(window, len(preds)):
        past_preds = preds[t-window:t]
        past_actuals = actuals[t-window:t]
        corr = np.corrcoef(past_preds, past_actuals)[0, 1]
        filtered.append(corr * preds[t])
    return np.array(filtered)

def main():
    xau = load_data('^XAU')
    gold = load_data('GC=F')
    spread = compute_spread(xau, gold)

    train, test = train_test_split(spread, train_size=0.7, shuffle=False)

    lags = 10
    train_features = lag_matrix(train, lags).values
    train_targets = train.values[lags:]

    gp = SymbolicRegressor(
        function_set=['add', 'sub', 'mul', 'div', 'log', 'sin', 'cos', 'exp'],
        population_size=1000,
        generations=20,
        parsimony_coefficient=0.01,
        metric='rmse',
        random_state=42,
        verbose=0,
    )
    gp.fit(train_features[:-1], train_targets[1:])

    full = pd.concat([train, test])
    features_full = lag_matrix(full, lags).values
    preds = gp.predict(features_full)
    preds_series = pd.Series(preds, index=full.index[lags:])

    actuals = full.iloc[lags:]
    filtered = correlation_filter(preds_series.values, actuals.values)
    filtered_index = actuals.index[50:]
    filtered_series = pd.Series(filtered, index=filtered_index)

    returns = actuals.diff().iloc[1:].reindex(filtered_index)
    signals = np.sign(filtered_series.shift(1)).reindex(filtered_index)
    strategy_returns = signals * returns
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

    cumulative = strategy_returns.cumsum()
    buy_hold = returns.cumsum()

    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Total Return: {cumulative.iloc[-1]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative, label='Strategy')
    plt.plot(buy_hold, label='Buy & Hold')
    plt.legend()
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.tight_layout()
    plt.savefig('results.png')
    plt.close()

if __name__ == '__main__':
    main()
