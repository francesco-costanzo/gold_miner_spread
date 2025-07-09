# Gold Miner Spread

This repository contains a Python script that backtests a gold mining strategy using 
ETF data for GLD and GDX. The script relies on the PySR symbolic regression
library to discover trading rules.

## Setup

1. Run the provided `setup.sh` script to create a virtual environment and
   install dependencies:
   ```bash
   ./setup.sh
   ```

   If installation fails due to restricted network access, manually install the
   required packages listed in `requirements.txt`.

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Usage

Execute the strategy with:
```bash
python gold_miner_spread.py
```

The script downloads price data from Yahoo Finance and outputs a plot and summary
statistics of the backtested strategy.

The annualized return metric now uses a compounding approach based on the
product of period returns. The strategy also subtracts transaction costs of
5 basis points per trade from daily returns to better approximate real-world
execution.

The correlation filter now automatically selects the best rolling window
length by testing multiple candidates and choosing the one with the highest
average absolute correlation between GLD and GDX returns.

When the model forecasts the spread direction, the position is only updated if
the prior day's change in correlation is negative. Otherwise the strategy
maintains its existing exposure. Transaction costs are only charged when this
position changes.

The symbolic regressor now uses tournament selection with a tournament size of
20 and a 75% mutation probability. The search runs for up to 10,000
iterations with a reduced maximum equation size of 6.
