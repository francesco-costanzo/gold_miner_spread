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
