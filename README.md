# Adaptive-Stock-Prediction-LSTM-DNN
End-to-end adaptive stock price prediction system using ARIMA, GARCH, CNN-LSTM, BiLSTM-Transformer, and a novel LSTM-DNN hybrid model with incremental learning and sliding window techniques, achieving high accuracy and real-time adaptability.

## 📊 Baseline Models: ARIMA & GARCH

### 🔹 ARIMA Model (Trend Forecasting)

ARIMA (AutoRegressive Integrated Moving Average) is used as a statistical baseline model to capture linear trends in stock price time-series data.

#### 📸 Results

![ARIMA Forecast](1_Baseline_Models/arimaforecast.png)
![ARIMA Metrics](1_Baseline_Models/arimaresults.png)
![ARIMA Metrics](1_Baseline_Models/arimaresults1.png)

#### 🔍 Inference – ARIMA

* Accuracy ranges between ~49–59% (close to random baseline)
* Strong bias toward predicting downward trends
* Very low recall for upward movements (fails to capture bullish signals)
* Produces smooth forecasts that miss sudden market fluctuations
* Captures linear patterns but fails in nonlinear environments

👉 **Conclusion:** ARIMA is useful for basic trend analysis but not suitable for real-world stock prediction.



### 🔹 GARCH Model (Volatility Modeling)

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is used to model time-varying volatility in stock returns.

#### 📸 Results

![GARCH Forecast](1_Baseline_Models/garch_forecast.png)
![GARCH Metrics](1_Baseline_Models/garch_metrics.png)
![GARCH Metrics](1_Baseline_Models/garch_metrics1.png)

#### 🔍 Inference – GARCH

* Successfully captures market volatility
* Produces smooth and nearly static forecasts
* Assumes mean return ≈ 0, limiting predictive capability
* Fails to respond to sudden price changes
* Weak directional prediction performance

👉 **Conclusion:** GARCH is effective for volatility estimation but not suitable for accurate stock price prediction.



## 🔥 Key Insight

* ARIMA → Captures linear trends but fails in nonlinear markets
* GARCH → Models volatility but not actual price movement
* Both models show poor directional accuracy

👉 These limitations motivate the use of deep learning models such as DNN and LSTM-DNN.

## 🧠 DNN Model (Deep Neural Network)

---

### ⚙️ Methodology

* **Data Collection:**
  Three-year OHLCV data for RELIANCE.NS, TCS.NS, and INFY.NS fetched using `yfinance`

* **Feature Engineering:**
  Technical indicators added:

  * EMA (20)
  * RSI (14)
  * MACD (12,26,9)
  * MACD Signal

  Final features:

  ```
  Open, High, Low, Close, Volume, EMA_20, RSI_14, MACD, MACD Signal
  ```

* **Preprocessing:**

  * MinMaxScaler normalization
  * Chronological split (80% train, 20% test)
  * Target variable: Closing Price



### 🧱 Model Architecture

* Dense layers: **64 → 32 neurons**
* Activation: **LeakyReLU**
* Dropout: **0.3, 0.2**
* Optimizer: **Adam (lr = 0.004)**
* Loss Function: **Huber Loss**
* EarlyStopping applied
* Training: up to 200 epochs with validation

---

### 📊 Performance

| Stock    | R² Score | Insight            |
| -------- | -------- | ------------------ |
| TCS      | ~0.965   | Highest accuracy   |
| RELIANCE | ~0.894   | Strong performance |
| INFY     | ~0.894   | Stable predictions |

* **MAE:** ~₹25 → Low average prediction error
* **MSE:** Indicates variability in prediction stability



### 🔍 Inference

* Learns **nonlinear relationships** between price and indicators

* Achieves high accuracy, especially for stable stocks

* Shows **trend-following behavior** (extends recent trends)

* Performs well for **short-term predictions**

* Struggles with:

  * Sudden reversals
  * High volatility spikes

* Model consistency varies:

  * **INFY:** Most stable (lowest MSE)
  * **TCS:** High accuracy but occasional large errors

👉 **Conclusion:**
DNN significantly improves prediction accuracy over traditional models by capturing nonlinear patterns, but lacks temporal awareness, limiting performance in highly volatile conditions.



### 📸 Results

```markdown
![DNN Forecast](dnn_forecast.png)
![DNN Metrics](dnn_metrics.png)

```
## 🔹 LSTM Model (Long Short-Term Memory)



### ⚙️ Methodology

* **Data Collection:**
  Three-year OHLCV data for RELIANCE.NS, TCS.NS, and INFY.NS collected using `yfinance`

* **Feature Engineering:**
  Technical indicators added:

  * EMA (20)
  * RSI (14)
  * MACD (12,26,9)
  * MACD Signal

  Final features include OHLCV + technical indicators

* **Preprocessing:**

  * MinMaxScaler normalization
  * 60-day sliding window for sequence creation
  * Chronological split (80% train, 20% test)
  * Target variable: Closing Price



### 🧱 Model Architecture

* Stacked LSTM layers: **128 → 64 units**
* Dropout: **0.3, 0.2**
* Output: Dense(1)
* Optimizer: **Adam (lr = 0.0004)**
* Loss Function: **Huber Loss**
* EarlyStopping applied
* Training: up to 200 epochs



### 📊 Performance

| Stock    | R² Score | Insight              |
| -------- | -------- | -------------------- |
| RELIANCE | ~0.826   | Best performance     |
| INFY     | ~0.772   | Moderate performance |
| TCS      | ~0.754   | Weak performance     |

* **MAE:**

  * RELIANCE: ~₹24
  * INFY: ~₹24
  * TCS: ~₹64 (high error)

* **MSE:**

  * RELIANCE: Lowest → most stable
  * TCS: Highest → frequent large errors



### 🔍 Inference

* Captures **temporal dependencies** using sequential data

* Performs well for:

  * Stable stocks
  * Clear trending patterns

* Struggles with:

  * High volatility
  * Sudden trend reversals

* Observations:

  * **Lagging behavior** in predictions (especially for TCS)
  * Predictions fail to keep up with rapid price changes
  * Short-term predictions reasonable for RELIANCE and INFY
  * Performance varies significantly across stocks

👉 **Conclusion:**
LSTM effectively models time-series dependencies but suffers from lag and inconsistency in highly volatile conditions, limiting its standalone performance.

---

### 📸 Results

```markdown
![LSTM Forecast](2_Deep_Learning_Models/lstm_forecast.png)
![LSTM Metrics](2_Deep_Learning_Models/lstm_metrics.png)
```





## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Statsmodels (ARIMA)
* ARCH (GARCH)
* Scikit-learn
* Plotly

---

## 👩‍💻 Author

Eashita Prabhudesai

