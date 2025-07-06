# Gold Miner Spread Modeling

This repository demonstrates how to model the spread between the Philadelphia Gold and Silver Index (`^XAU`) and the spot gold price (`GC=F`) using genetic programming and a rolling correlation filter. The approach implements a simple backtest and outputs performance metrics.

## Features

- Downloads daily price data from Yahoo Finance using `yfinance`.
- Computes the log spread between the index and gold futures.
- Uses a symbolic regression model from `gplearn` to forecast the spread.
- Applies a rolling correlation filter to the raw model forecasts.
- Backtests a simple long/short strategy based on the filtered forecasts.
- Reports Sharpe ratio and cumulative returns.

## Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `yfinance`
- `gplearn`
- `scikit-learn`
- `matplotlib`

## Usage

1. Install the dependencies:

```bash
pip install -r SETUP
```

2. Run the model:

```bash
python gold_spread_gp.py
```

The script downloads data, trains the model on the first 70% of observations, then tests on the remaining data while applying the correlation filter. It outputs risk metrics and saves a plot of cumulative returns as `results.png`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
