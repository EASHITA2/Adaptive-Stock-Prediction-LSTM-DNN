import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import plotly.graph_objects as go
import plotly.offline as py
import warnings

warnings.filterwarnings("ignore")

# --- Parameters ---
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
end_date = datetime.now().date()
start_date = end_date - timedelta(days=3 * 365)

# --- RSI Function ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Load data directly from yfinance ---

full_df = yf.download(tickers, start=start_date, end=end_date)
if full_df.empty:
    raise SystemExit(" No data downloaded.")


fig = go.Figure()

# --- Loop through tickers ---
for ticker in tickers:
    print(f"\nGARCH- {ticker} ---")

    ts = pd.DataFrame(full_df['Close'][ticker].dropna())
    ts.rename(columns={ticker: 'Close'}, inplace=True)

    if ts.empty:
        print(f"No data for {ticker}")
        continue

    ts["RSI14"] = calculate_rsi(ts["Close"])
    ts["Returns"] = 100 * ts["Close"].pct_change()
    ts.dropna(inplace=True)

    train_size = int(len(ts) * 0.8)
    if train_size < 10:
        print(f"⚠️  Not enough data for {ticker}.")
        continue

    train_returns = ts["Returns"][:train_size]
    test_returns = ts["Returns"][train_size:]

    model = arch_model(train_returns, vol="Garch", p=1, q=1)
    model_fit = model.fit(disp="off")

    forecast = model_fit.forecast(horizon=len(test_returns), reindex=False)
    predicted_returns = forecast.mean.values[-1]

    last_train_price = ts["Close"].iloc[train_size - 1]
    predicted_prices = [last_train_price]
    for r in predicted_returns:
        predicted_prices.append(predicted_prices[-1] * (1 + r / 100))
    
    predicted_series = pd.Series(predicted_prices[1:], index=ts.index[train_size:])
    actual_prices = ts["Close"][train_size:]

    # --- PERFORMANCE METRICS ---
    r2 = r2_score(actual_prices, predicted_series)
    mae = mean_absolute_error(actual_prices, predicted_series)
    mse = mean_squared_error(actual_prices, predicted_series)

    actual_dir = np.sign(np.diff(actual_prices.values))
    pred_dir = np.sign(np.diff(predicted_series.values))
    
    if len(actual_dir) == 0 or len(pred_dir) == 0:
        print("Cannot calculate directional metrics: not enough test data points.")
        continue

    acc = accuracy_score(actual_dir, pred_dir)
    prec = precision_score(actual_dir, pred_dir, average="macro", zero_division=0)
    rec = recall_score(actual_dir, pred_dir, average="macro", zero_division=0)
    f1 = f1_score(actual_dir, pred_dir, average="macro", zero_division=0)
    directional_acc = np.mean(actual_dir == pred_dir)
    
    latest_rsi = ts["RSI14"].iloc[-1] if not ts["RSI14"].empty else np.nan

    try:
        next_pred_return = model_fit.forecast(horizon=len(test_returns) + 1, reindex=False).mean.values[-1][-1]
        next_day_price = predicted_series.values[-1] * (1 + next_pred_return / 100)
        next_date = ts.index[-1] + pd.tseries.offsets.BDay(1)
    except Exception as e:
        next_day_price = np.nan
        next_date = ts.index[-1] + pd.Timedelta(days=1)

    # --- PRINT METRICS ---
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    
    if not np.isnan(next_day_price):
        print(f"Predicted closing price on {next_date.strftime('%d-%m-%Y')}: ₹{next_day_price:.2f}")

    # --- PLOTLY TRACES ---
    fig.add_trace(go.Scatter(x=ts.index[:train_size], y=ts["Close"][:train_size], mode="lines", name=f"{ticker} Train"))
    fig.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices, mode="lines", name=f"{ticker} Test Actual"))
    fig.add_trace(go.Scatter(x=predicted_series.index, y=predicted_series, mode="lines", name=f"{ticker} Predicted", line=dict(dash="dash")))

fig.update_layout(title="GARCH Price Forecasts", xaxis_title="Date", yaxis_title="Price (₹)", hovermode="x unified", template="plotly_white", height=700)
py.plot(fig, filename="garch_forecast.html", auto_open=True)