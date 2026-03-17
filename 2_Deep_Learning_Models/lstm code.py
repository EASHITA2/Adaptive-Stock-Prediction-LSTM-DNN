import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import plotly.offline as py
import warnings

warnings.filterwarnings("ignore")

# --- REPRODUCIBILITY SEED ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- CONFIGURATION ---
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
end_date = datetime.now().date()
start_date = end_date - timedelta(days=3 * 365)
sequence_length = 60

# === LOAD DATA FROM YFINANCE ===

full_df = yf.download(tickers, start=start_date, end=end_date)
if full_df.empty:
    raise SystemExit("No data downloaded.")

combined_plot_data = {}

# === PROCESS EACH TICKER ===
for ticker in tickers:
    print(f"\nLSTM- {ticker}")

    df = full_df.xs(ticker, level=1, axis=1).copy()

    # === FEATURE ENGINEERING using pandas_ta ===
    df.ta.ema(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.dropna(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_20', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9']
    target = 'Close'

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(df[features])
    y_scaled = y_scaler.fit_transform(df[[target]])

    X_seq, y_seq = [], []
    for i in range(sequence_length, len(df)):
        X_seq.append(X_scaled[i-sequence_length:i])
        y_seq.append(y_scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    
    if len(X_seq) == 0:
        print(f"Not enough data for {ticker}. Skipping.")
        continue

    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    dates_train = df.index[sequence_length:split_idx+sequence_length]
    dates_test = df.index[split_idx+sequence_length:]

    # === LSTM MODEL ARCHITECTURE ===
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, X_seq.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0004), loss='huber')
    es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    
    
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[es], verbose=0)
    

    y_pred = model.predict(X_test, verbose=0)
    y_pred_actual = y_scaler.inverse_transform(y_pred)
    y_true_actual = y_scaler.inverse_transform(y_test)

    combined_plot_data[ticker] = {
        "train_dates": dates_train,
        "train_actual": y_scaler.inverse_transform(y_train),
        "test_dates": dates_test,
        "test_actual": y_true_actual,
        "predicted": y_pred_actual
    }

    # === PERFORMANCE METRICS ===
    r2 = r2_score(y_true_actual, y_pred_actual)
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    mse = mean_squared_error(y_true_actual, y_pred_actual)

    actual_dir = np.sign(np.diff(y_true_actual.flatten()))
    pred_dir = np.sign(np.diff(y_pred_actual.flatten()))
    
    acc = accuracy_score(actual_dir, pred_dir)
    prec = precision_score(actual_dir, pred_dir, average='macro', zero_division=0)
    rec = recall_score(actual_dir, pred_dir, average='macro', zero_division=0)
    f1 = f1_score(actual_dir, pred_dir, average='macro', zero_division=0)
    directional_acc = np.mean(actual_dir == pred_dir)

    # === NEXT BUSINESS DAY PREDICTION ===
    last_seq = X_scaled[-sequence_length:].reshape(1, sequence_length, X_seq.shape[2])
    y_next_scaled = model.predict(last_seq, verbose=0)
    y_next = y_scaler.inverse_transform(y_next_scaled)
    next_date = df.index[-1] + pd.tseries.offsets.BDay(1)

    # --- Print Results ---
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    
    print(f"Predicted closing price on {next_date.strftime('%d-%m-%Y')}: ₹{y_next[0][0]:.2f}")

# === COMBINED PLOTLY FORECAST ===
fig = go.Figure()
for ticker, d in combined_plot_data.items():
    fig.add_trace(go.Scatter(x=d["train_dates"], y=d["train_actual"].flatten(), mode='lines', name=f'{ticker} Train'))
    fig.add_trace(go.Scatter(x=d["test_dates"], y=d["test_actual"].flatten(), mode='lines', name=f'{ticker} Test Actual'))
    fig.add_trace(go.Scatter(x=d["test_dates"], y=d["predicted"].flatten(), mode='lines', name=f'{ticker} Predicted', line=dict(dash='dash')))

fig.update_layout(title="LSTM Price Forecasts", xaxis_title="Date", yaxis_title="Price (₹)", template="plotly_white", hovermode="x unified", height=700)
py.plot(fig, filename="lstm_forecast.html", auto_open=True)