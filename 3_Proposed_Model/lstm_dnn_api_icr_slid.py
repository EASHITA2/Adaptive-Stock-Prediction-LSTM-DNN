import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import os
import random

# === AGGRESSIVE DETERMINISM SETUP (RUN THIS FIRST) ===
# This block aggressively sets seeds and environment variables for reproducibility.
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)

import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import plotly.offline as py
import pandas.tseries.offsets as pto

# Set TensorFlow's random seed and enable deterministic operations
tf.random.set_seed(SEED)
# The next line uses a TF API that may change across TF versions; keep it to follow your original determinism intent
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# === SUPPRESS WARNINGS AND TF INFO ===
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")

# === CONFIGURATION ===
tickers = {"RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS"}
end_date = datetime.now().date()
start_date = end_date - timedelta(days=3*365)
TRAIN_SPLIT_RATIO = 0.8

# --- Sliding window / incremental learning hyperparameters ---
WINDOW_SIZE = 60               # number of recent samples used to fine-tune (sliding window)
INCREMENTAL_EPOCHS = 1         # epochs to fine-tune per incremental step (keep small)
INCREMENTAL_BATCH_SIZE = 16    # batch size during incremental fine-tuning
UPDATE_AFTER_EVERY = 1         # update model after every N predictions (1 => after each prediction)

def build_enhanced_hybrid_model(input_shape):
    """
    Builds an enhanced and more robust integrated LSTM-DNN model,
    inspired by the paper's deep architecture.
    """
    model = Sequential()

    # LSTM Block: Captures temporal dependencies
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    # Deep DNN Block: For extracting complex, non-linear patterns
    model.add(Dense(units=128))
    model.add(LeakyReLU(alpha=0.0005))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=64))
    model.add(LeakyReLU(alpha=0.0005))
    model.add(Dropout(0.1))

    # Output Layer
    model.add(Dense(units=1))

    # Compile the model using the Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    return model

# === DOWNLOAD DATA ===

data = yf.download(list(tickers.values()), start=start_date, end=end_date, progress=False)
if data.empty:
    raise SystemExit("No data downloaded.")

combined_plot_data = {}

# === MODELING ===
for name, ticker in tickers.items():
    print(f"\nLSTM-DNN HYBRID- {name}")

    # --- 1. Data Preparation & Feature Engineering ---
    df = data['Close'][ticker].to_frame(name='Close')
    df['Open'] = data['Open'][ticker]
    df['High'] = data['High'][ticker]
    df['Low'] = data['Low'][ticker]
    df['Volume'] = data['Volume'][ticker]
    df.dropna(inplace=True)

    # Add indicators (pandas_ta appends columns with the naming convention used below)
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

    # --- 2. Chronological Train-Test Split ---
    split_idx = int(len(X_scaled) * TRAIN_SPLIT_RATIO)
    X_train_2d, X_test_2d = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # --- 3. Reshape Data for LSTM (Sequence of length 1) ---
    X_train = X_train_2d.reshape((X_train_2d.shape[0], 1, X_train_2d.shape[1]))
    X_test = X_test_2d.reshape((X_test_2d.shape[0], 1, X_test_2d.shape[1]))
    
    
    model = build_enhanced_hybrid_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64, # Using batch size as mentioned in the paper
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )
   
    # --- 5. Incremental / Sliding-window prediction + online fine-tuning ---
    # We'll predict test samples chronologically, and after each prediction (or after UPDATE_AFTER_EVERY predictions),
    # we fine-tune the model on the last WINDOW_SIZE samples (sliding window) including the newly observed true sample.
    incremental_predictions = []
    # We'll also maintain a dynamic pool of scaled features and targets for incremental windowing
    # X_scaled and y_scaled are the whole dataset arrays (chronological).
    total_len = len(X_scaled)
    # test indexes are [split_idx, split_idx+1, ..., total_len-1]
    for relative_i, idx in enumerate(range(split_idx, total_len)):
        # prepare single input (shape: 1,1,num_features)
        x_input = X_scaled[idx].reshape((1, 1, X_scaled.shape[1]))
        y_pred_scaled = model.predict(x_input, verbose=0)
        incremental_predictions.append(y_pred_scaled)

        # After prediction, adapt (fine-tune) using a sliding window of recent samples up to current idx (inclusive)
        if ((relative_i + 1) % UPDATE_AFTER_EVERY) == 0:
            # define window start
            start_idx = max(0, idx - WINDOW_SIZE + 1)
            window_X = X_scaled[start_idx: idx + 1]   # shape (n_window, n_features)
            window_y = y_scaled[start_idx: idx + 1]   # shape (n_window, 1)
            if len(window_X) >= 2:  # need at least 2 samples to run a meaningful batch update
                window_X_seq = window_X.reshape((window_X.shape[0], 1, window_X.shape[1]))
                # fine-tune with small epochs to simulate incremental learning
                try:
                    model.fit(window_X_seq, window_y, epochs=INCREMENTAL_EPOCHS, batch_size=INCREMENTAL_BATCH_SIZE, verbose=0)
                except Exception as e:
                    # In case of any runtime errors (e.g., too small batch), fallback to single-sample fit
                    try:
                        model.fit(window_X_seq, window_y, epochs=1, batch_size=1, verbose=0)
                    except Exception:
                        pass

    # convert predictions list to array and inverse transform
    y_pred_scaled_array = np.vstack(incremental_predictions)  # shape (n_test, 1)
    y_pred_actual = y_scaler.inverse_transform(y_pred_scaled_array)
    y_true_actual = y_scaler.inverse_transform(y_test)

    # --- 6. Store Data for Plotting ---
    y_train_actual = y_scaler.inverse_transform(y_train)
    combined_plot_data[name] = {
        "dates_train": df.index[:split_idx], "train_actual": y_train_actual,
        "dates_test": df.index[split_idx:], "test_actual": y_true_actual,
        "predicted": y_pred_actual
    }

    # --- 7. Evaluation metrics ---
    r2 = r2_score(y_true_actual, y_pred_actual)
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    mse = mean_squared_error(y_true_actual, y_pred_actual)
    print("--- Regression Metrics (After incremental/sliding-window updates) ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    y_true_flat = y_true_actual.flatten()
    y_pred_flat = y_pred_actual.flatten()
    # For directional accuracy, need diffs of length n-1; if test set has <2 samples this fails, so guard:
    if len(y_true_flat) >= 2 and len(y_pred_flat) >= 2:
        actual_direction = np.sign(np.diff(y_true_flat))
        predicted_direction = np.sign(np.diff(y_pred_flat))
        try:
            accuracy = accuracy_score(actual_direction, predicted_direction)
            precision = precision_score(actual_direction, predicted_direction, average='macro', zero_division=0)
            recall = recall_score(actual_direction, predicted_direction, average='macro', zero_division=0)
            f1 = f1_score(actual_direction, predicted_direction, average='macro', zero_division=0)
        except Exception:
            accuracy = precision = recall = f1 = np.nan
    else:
        accuracy = precision = recall = f1 = np.nan

    print("--- Directional / Classification-like Metrics ---")
    print(f"Directional Accuracy: {accuracy}")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

    # --- 8. Predict the Next Business Day's Price using the latest updated model ---
    last_day_features = X_scaled[-1].reshape((1, 1, len(features)))
    predicted_price_scaled = model.predict(last_day_features, verbose=0)
    predicted_price_actual = y_scaler.inverse_transform(predicted_price_scaled)
    next_date = df.index[-1] + pto.BDay(1)
    print(f"Predicted closing for {name} on {next_date.strftime('%d-%m-%Y')}: ₹{predicted_price_actual[0][0]:.2f}")


# === COMBINED PLOT ===
fig = go.Figure()
for name, d in combined_plot_data.items():
    fig.add_trace(go.Scatter(x=d["dates_train"], y=d["train_actual"].flatten(), mode='lines', name=f'{name} Historical Train Data'))
    fig.add_trace(go.Scatter(x=d["dates_test"], y=d["test_actual"].flatten(), mode='lines', name=f'{name} Actual Test Data'))
    fig.add_trace(go.Scatter(x=d["dates_test"], y=d["predicted"].flatten(), mode='lines', name=f'{name} Predicted Price', line=dict(dash='dash')))

fig.update_layout(
    title="LSTM-DNN Hybrid forecast (with incremental sliding-window updates)",
    xaxis_title="Date", yaxis_title="Stock Price (₹)",
    template="plotly_white", hovermode="x unified",
    legend_title="Legend", height=700
)
py.plot(fig, filename="LSTM-DNN_Hybrid_forecast_incremental.html", auto_open=True)
