import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
import plotly.graph_objects as go
import os

# --- Parameters ---
folder_path = r"C:\Users\erpde\Downloads\dataset_3_india_5yrs\dataset_3_india_5yrs - Copy"
tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
arima_order = (2, 1, 2)

# --- Load all CSVs ---
all_data = []
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(file.replace(".csv", ""))  # From filename
        all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)

# --- Create Plotly Figure ---
fig = go.Figure()

# --- Loop for each stock ---
for ticker in tickers:
    stock_df = full_df[full_df["Ticker"] == ticker].sort_values("Date")

    if stock_df.empty:
        print(f"❌ No data for {ticker}")
        continue

    ts = stock_df[["Date", "Close"]].dropna().copy()
    ts.set_index("Date", inplace=True)

    # Train-Test Split
    train_size = int(len(ts) * 0.8)
    train = ts.iloc[:train_size]
    test = ts.iloc[train_size:]

    # --- Fit ARIMA ---
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test)).values

    # --- MSE ---
    mse = mean_squared_error(test["Close"].values, forecast)

    # --- Classification Metrics (Direction-based) ---
    actual_direction = np.sign(np.diff(test["Close"].values, prepend=train["Close"].values[-1]))  # 1 for up, -1 for down
    predicted_direction = np.sign(np.diff(forecast, prepend=train["Close"].values[-1]))

    accuracy = accuracy_score(actual_direction > 0, predicted_direction > 0)
    precision = precision_score(actual_direction > 0, predicted_direction > 0, zero_division=0)
    recall = recall_score(actual_direction > 0, predicted_direction > 0, zero_division=0)

    print(f"📊 {ticker} | MSE: {mse:.2f} | Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")

    # --- Plot training data ---
    fig.add_trace(go.Scatter(
        x=train.index, y=train["Close"],
        mode='lines',
        name=f"{ticker} Train"
    ))

    # --- Plot test data ---
    fig.add_trace(go.Scatter(
        x=test.index, y=test["Close"],
        mode='lines',
        name=f"{ticker} Test"
    ))

    # --- Plot forecast ---
    fig.add_trace(go.Scatter(
        x=test.index, y=forecast,
        mode='lines',
        name=f"{ticker} Forecast",
        line=dict(dash='dash')
    ))

# --- Final Layout ---
fig.update_layout(
    title="📈 ARIMA Forecast & Classification Metrics (Close Price Direction)",
    xaxis_title="Date",
    yaxis_title="Closing Price",
    hovermode="x unified"
)

fig.show()
