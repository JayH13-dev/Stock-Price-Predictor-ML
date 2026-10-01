# Stock Predictions ML

An LSTM-based model that predicts the next day's closing stock price from the previous 29 days of closing prices.

## How it works

The notebook ([Main.ipynb](Main.ipynb)) follows this pipeline:

1. **Download data** — pulls daily stock price history for a given ticker from Yahoo Finance via `yfinance` (starting 2020-01-01).
2. **Scale** — the closing price series is standardized with `StandardScaler` so the network trains on normalized values rather than raw dollar amounts.
3. **Window the data** — a sliding window of 30 days is used to build overlapping sequences: the first 29 days form the input, and the 30th day is the target to predict.
4. **Split** — 80% of the windows are used for training, 20% for testing.
5. **Model** — a 2-layer LSTM (`hidden_dim=32`) followed by a linear output layer, implemented in PyTorch.
6. **Train** — trained for 200 epochs (full-batch, no mini-batching) using MSE loss and the Adam optimizer (`lr=0.01`).
7. **Evaluate** — predictions are inverse-transformed back to real price values, and RMSE is computed on both the training and test sets.
8. **Visualize** — plots actual vs. predicted price, plus a prediction-error chart against the RMSE threshold.

## Setup

```bash
pip install -r requirements.txt
```

Then open [Main.ipynb](Main.ipynb) in Jupyter and run the cells top to bottom. Change the `ticker` variable in the "Selecting Stock" cell to predict a different stock (any valid Yahoo Finance ticker symbol).

## Results

<!-- Fill in after running the notebook, e.g.:
- Ticker: AAPL
- Train RMSE: ...
- Test RMSE: ...
-->

The model is a reasonable predictor of short-term price movement, with the predicted curve generally tracking the actual price. However, it tends to lag — effectively anchoring each prediction on the current day's price rather than genuinely forecasting the next move. This becomes especially visible during periods of sharp price change, where prediction error spikes above the RMSE threshold.

## Known limitations

- **Lag/shift effect**: the model's predictions trail the actual price rather than leading it, a common failure mode for single-feature sequence models trained only on past prices.
- **Single feature**: only closing price is used — no volume, technical indicators, or other signals.
- **No validation split or early stopping**: the model always trains for the full 200 epochs regardless of whether loss has plateaued, which risks overfitting.
- **Full-batch training**: no mini-batching, which limits scalability to longer histories or multiple tickers.

## Possible improvements

- Add more input features (volume, moving averages, RSI, etc.)
- Introduce a validation set and early stopping
- Try mini-batch training and/or learning rate scheduling
- Experiment with sequence length, hidden size, and number of LSTM layers
