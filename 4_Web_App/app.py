"""
app_streamlit_v4.py
LSTM-DNN v4 Logic + Full Streamlit UI
======================================
Model logic: v4 (log-return prediction, focal loss, residual DNN, batch walk-forward)
UI: Full Streamlit webapp with candlestick chart, multi-ticker, FX conversion
Buy/Sell signal: 100% model-driven (log-return + direction head probabilities)
"""

import os
import random
import warnings
import time
from datetime import datetime, timedelta, date
import pytz

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import pandas_market_calendars as mcal
import joblib
import requests

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM,
    BatchNormalization, LayerNormalization,
    LeakyReLU, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.tseries.offsets import BDay

tf.get_logger().setLevel("ERROR")

# ─────────────────── SEEDS ───────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ─────────────────── CONFIG ──────────────────────────────────
WINDOW        = 40
TRAIN_SPLIT   = 0.80
EPOCHS        = 100
INC_STEP      = 5
INC_EPOCHS    = 8
BATCH         = 64
LR            = 6e-4
LR_INC        = 2e-5
TX_COST       = 0.001
MAX_MOVE      = 0.05

MODEL_DIR = "saved_models_v4"
os.makedirs(MODEL_DIR, exist_ok=True)

def mpath(n): return os.path.join(MODEL_DIR, f"v4_{n}.keras")
def spath(n): return os.path.join(MODEL_DIR, f"v4_sc_{n}.pkl")
def sr(x, n=4): return round(float(x), n)

# ─────────────────── FOCAL LOSS ──────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    def _loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        pt  = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return K.mean(alpha * K.pow(1.0 - pt, gamma) * bce)
    return _loss

# ─────────────────── MODEL BUILD ─────────────────────────────
def build_model(shape, lr):
    inp = Input(shape=shape)

    x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(inp)
    x = LayerNormalization()(x)
    x = LSTM(96,  return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(x)
    x = LayerNormalization()(x)
    x = LSTM(64,  dropout=0.15)(x)
    x = BatchNormalization()(x)

    h1   = Dense(128)(x);   h1 = LeakyReLU(0.01)(h1);   h1 = Dropout(0.25)(h1)
    h2   = Dense(64)(h1);   h2 = LeakyReLU(0.01)(h2);   h2 = Dropout(0.15)(h2)
    skip = Dense(64)(x)
    shared = Add()([h2, skip])
    shared = LayerNormalization()(shared)
    h3 = Dense(32)(shared); h3 = LeakyReLU(0.01)(h3)

    r_out = Dense(16, activation="relu")(h3)
    r_out = Dense(1, name="logret_output")(r_out)

    d_out = Dense(64, activation="relu")(shared)
    d_out = Dropout(0.2)(d_out)
    d_out = Dense(32, activation="relu")(d_out)
    d_out = Dense(16, activation="relu")(d_out)
    d_out = Dense(1, activation="sigmoid", name="dir_output")(d_out)

    m = Model(inputs=inp, outputs=[r_out, d_out])
    _compile_model(m, lr)
    return m

def _compile_model(m, lr):
    m.compile(
        optimizer=Adam(lr, clipnorm=0.5),
        loss={
            "logret_output": tf.keras.losses.Huber(delta=0.3),
            "dir_output":    focal_loss(gamma=2.0, alpha=0.25),
        },
        loss_weights={"logret_output": 0.4, "dir_output": 1.6},
        metrics={"dir_output": "accuracy"}
    )

# ─────────────────── FEATURES ────────────────────────────────
def make_features(df):
    d = df.copy()
    c, h, l, o, v = d["Close"], d["High"], d["Low"], d["Open"], d["Volume"]

    d.ta.ema(length=9,  append=True)
    d.ta.ema(length=21, append=True)
    d.ta.ema(length=50, append=True)
    d.ta.rsi(length=14, append=True)
    d.ta.macd(fast=12, slow=26, signal=9, append=True)
    d.ta.atr(length=14, append=True)
    d.ta.bbands(length=20, std=2, append=True)

    cols = d.columns.tolist()

    def col(*keys, excl=None):
        excl = excl or []
        return next((c for c in cols
                     if all(k.upper() in c.upper() for k in keys)
                     and not any(e.upper() in c.upper() for e in excl)), None)

    ema9  = col("EMA","9")
    ema21 = col("EMA","21")
    ema50 = col("EMA","50")
    rsi14 = col("RSI","14")
    macd  = col("MACD", excl=["MACDs","MACDh"])
    macds = col("MACDs")
    atr14 = col("ATR","14")
    bbl   = col("BBL")
    bbu   = col("BBU")
    bbm   = col("BBM")

    required = dict(ema21=ema21, ema50=ema50, rsi14=rsi14, macd=macd, macds=macds)
    missing  = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d["log_ret"]   = np.log(c / c.shift(1))
    d["ret2"]      = c.pct_change(2)
    d["ret5"]      = c.pct_change(5)
    d["vol5"]      = c.rolling(5).std()
    d["vol10"]     = c.rolling(10).std()
    d["hl_ratio"]  = (h - l) / (c + 1e-9)
    d["oc_ratio"]  = (c - o) / (o + 1e-9)
    d["close_pos"] = (c - l) / (h - l + 1e-9)
    d["vol_ratio"] = v / (v.rolling(10).mean() + 1e-9)
    d["tx_cost"]   = TX_COST

    if ema21: d["c_ema21"] = c / d[ema21]
    if ema50: d["c_ema50"] = c / d[ema50]
    if bbu and bbl and bbm:
        d["bb_pos"]   = (c - d[bbl]) / (d[bbu] - d[bbl] + 1e-9)
        d["bb_width"] = (d[bbu] - d[bbl]) / (d[bbm] + 1e-9)
    if atr14: d["atr_ratio"] = d[atr14] / (c + 1e-9)

    d["target_ret"] = np.log(c.shift(-1) / c)
    d["target_dir"] = (d["target_ret"] > 0).astype(int)

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.dropna(inplace=True)

    feats = [
        "Open","High","Low","Close","Volume",
        "log_ret","ret2","ret5",
        "vol5","vol10",
        "hl_ratio","oc_ratio","close_pos","vol_ratio",
        "tx_cost",
    ]
    optional = [ema9, ema21, ema50, rsi14, macd, macds, atr14,
                "c_ema21","c_ema50","bb_pos","bb_width","atr_ratio"]
    feats += [f for f in optional if f and f in d.columns]
    feats  = list(dict.fromkeys(feats))

    # Extra cols needed for chart overlays (kept in df, not in model features)
    extra_chart = [c for c in [
        col("EMA","20"), col("EMA","50"), col("EMA","100"),
        rsi14, macd, macds, atr14,
    ] if c and c in d.columns]

    return d, feats

# ─────────────────── SEQUENCES ───────────────────────────────
def make_seqs(X, yr, yd, w):
    n = len(X) - w
    Xs  = np.array([X[i:i+w] for i in range(n)])
    Yrs = yr[w:]
    Yds = yd[w:]
    return Xs, Yrs, Yds

# ─────────────────── AI SIGNAL (MODEL-DRIVEN) ────────────────
def generate_model_signal(dir_prob, pred_logret, confidence_threshold=0.55):
    """
    Pure model-driven signal:
    - dir_prob: sigmoid output of direction head (0-1, >0.5 = up)
    - pred_logret: predicted log-return
    - Returns: signal, confidence, reasons
    """
    reasons = []
    confidence = float(dir_prob)

    move_pct = (np.exp(pred_logret) - 1) * 100

    if dir_prob > confidence_threshold and pred_logret > 0:
        signal = "BUY"
        reasons.append(f"Direction head: {dir_prob*100:.1f}% probability of upward move")
        reasons.append(f"Predicted return: +{move_pct:.2f}%")
        if dir_prob > 0.70:
            reasons.append("High confidence bullish signal from LSTM pattern")
        elif dir_prob > 0.60:
            reasons.append("Moderate confidence bullish signal")
        else:
            reasons.append("Mild bullish bias — proceed with caution")

    elif dir_prob < (1 - confidence_threshold) and pred_logret < 0:
        signal = "SELL"
        confidence = 1 - float(dir_prob)
        reasons.append(f"Direction head: {(1-dir_prob)*100:.1f}% probability of downward move")
        reasons.append(f"Predicted return: {move_pct:.2f}%")
        if dir_prob < 0.30:
            reasons.append("High confidence bearish signal from LSTM pattern")
        elif dir_prob < 0.40:
            reasons.append("Moderate confidence bearish signal")
        else:
            reasons.append("Mild bearish bias — proceed with caution")
    else:
        signal = "HOLD"
        confidence = 1 - abs(dir_prob - 0.5) * 2
        reasons.append(f"Direction head probability: {dir_prob*100:.1f}% (near neutral)")
        reasons.append(f"Predicted return: {move_pct:.2f}% (within noise band)")
        reasons.append("Model indicates consolidation / insufficient conviction")

    confidence = round(float(min(confidence, 1.0)), 4)
    return signal, confidence, reasons


def generate_smart_insight(signal, confidence, reasons, last_price, next_price, dir_prob, pred_logret):
    direction = (np.exp(pred_logret) - 1) * 100
    lines = []
    lines.append(f"The LSTM-DNN v4 model forecasts a **{direction:+.2f}%** move for the next session.")
    lines.append(f"Direction head confidence: **{dir_prob*100:.1f}%** probability of upward close.")
    lines.append(f"\n**Overall AI signal: {signal}**\n")
    lines.append("**Key model-driven factors:**")
    for r in reasons:
        lines.append(f"• {r}")
    if signal == "BUY":
        lines.append("\nThe model's regression and classification heads are aligned bullish. "
                     "Consider entry with appropriate risk management.")
    elif signal == "SELL":
        lines.append("\nBoth model outputs indicate downside risk. "
                     "Consider reducing exposure or defensive positioning.")
    else:
        lines.append("\nModel outputs show mixed or low-conviction signals. "
                     "Best to wait for a clearer setup before acting.")
    lines.append(f"\n*Confidence score: {confidence*100:.1f}%*")
    lines.append("\n⚠️ *This is a model forecast, not financial advice. Always manage risk.*")
    return "\n".join(lines)

# ─────────────────── TICKERS ─────────────────────────────────
DEFAULT_TICKERS = {
    # US
    "S&P 500 (US)": "^GSPC", "DOW 30 (US)": "^DJI", "NASDAQ (US)": "^IXIC",
    "APPLE (US)": "AAPL", "MICROSOFT (US)": "MSFT", "GOOGLE (US)": "GOOGL",
    "TESLA (US)": "TSLA", "AMAZON (US)": "AMZN", "META (US)": "META",
    "NVIDIA (US)": "NVDA", "JPM CHASE (US)": "JPM", "NETFLIX (US)": "NFLX",
    "AMD (US)": "AMD", "INTEL (US)": "INTC",
    # India
    "NIFTY 50 (IN)": "^NSEI", "SENSEX (IN)": "^BSESN",
    "RELIANCE IND (IN)": "RELIANCE.NS", "TCS (IN)": "TCS.NS",
    "INFOSYS (IN)": "INFY.NS", "HDFC BANK (IN)": "HDFCBANK.NS",
    "ICICI BANK (IN)": "ICICIBANK.NS", "L&T (IN)": "LT.NS",
    "BAJAJ FIN (IN)": "BAJFINANCE.NS", "SBI (IN)": "SBIN.NS",
    "WIPRO (IN)": "WIPRO.NS", "AXIS BANK (IN)": "AXISBANK.NS",
    "BHARTI AIRTEL (IN)": "BHARTIARTL.NS", "TITAN (IN)": "TITAN.NS",
    "HCL TECH (IN)": "HCLTECH.NS",
    # Europe
    "LSE FTSE (UK)": "^FTSE", "SHELL PLC (UK)": "SHEL.L",
    "SAP (DE)": "SAP.DE", "DE DAX (DE)": "^GDAXI",
    "LVMH (FR)": "MC.PA", "FR CAC (FR)": "^FCHI",
    "NESTLE (CH)": "NESN.SW", "ASML (NL)": "ASML.AS",
    # Asia/Pacific
    "NIKKEI (JP)": "^N225", "TOYOTA (JP)": "7203.T",
    "TENCENT (HK)": "0700.HK", "HK HANG SENG (HK)": "^HSI",
    "SAMSUNG (KR)": "005930.KS", "AU ASX (AU)": "^AXJO",
    "SHANGHAI (CN)": "000001.SS", "DBS GROUP (SG)": "D05.SI",
    "TSMC (TW)": "2330.TW",
    # Americas
    "B3 IBOVESPA (BR)": "^BVSP", "PETROBRAS (BR)": "PBR",
    "CA TSX (CA)": "^GSPTSE", "SHOPIFY (CA)": "SHOP.TO",
    "MEXICO IPC (MX)": "^MXX",
}

EXCHANGE_MAP = {
    ".NS": {"calendar": "NSE", "tz": "Asia/Kolkata", "close_time": "15:30"},
    ".BO": {"calendar": "BSE", "tz": "Asia/Kolkata", "close_time": "15:30"},
    "NYSE": {"calendar": "NYSE", "tz": "America/New_York", "close_time": "16:00"},
    "US":   {"calendar": "NYSE", "tz": "America/New_York", "close_time": "16:00"},
    ".L":  {"calendar": "LSE",  "tz": "Europe/London",    "close_time": "16:30"},
    ".T":  {"calendar": "TSE",  "tz": "Asia/Tokyo",       "close_time": "15:00"},
    ".DE": {"calendar": "XETR", "tz": "Europe/Berlin",    "close_time": "17:30"},
    ".PA": {"calendar": "Euronext", "tz": "Europe/Paris", "close_time": "17:30"},
    ".HK": {"calendar": "HKEX", "tz": "Asia/Hong_Kong",  "close_time": "16:00"},
    ".AX": {"calendar": "ASX",  "tz": "Australia/Sydney", "close_time": "16:00"},
    ".KS": {"calendar": "KOR",  "tz": "Asia/Seoul",       "close_time": "15:30"},
    ".SW": {"calendar": "SIX",  "tz": "Europe/Zurich",    "close_time": "17:30"},
    ".SS": {"calendar": "SSE",  "tz": "Asia/Shanghai",    "close_time": "15:00"},
    ".SZ": {"calendar": "SSE",  "tz": "Asia/Shanghai",    "close_time": "15:00"},
    ".TO": {"calendar": "XTSE", "tz": "America/Toronto",  "close_time": "16:00"},
    ".SI": {"calendar": "XSES", "tz": "Asia/Singapore",   "close_time": "17:00"},
    ".TW": {"calendar": "XTPE", "tz": "Asia/Taipei",      "close_time": "13:30"},
    ".SA": {"calendar": "BVMF", "tz": "America/Sao_Paulo","close_time": "17:00"},
    ".MC": {"calendar": "XMAD", "tz": "Europe/Madrid",    "close_time": "17:30"},
    ".MI": {"calendar": "XMIL", "tz": "Europe/Rome",      "close_time": "17:30"},
    "^MXX":{"calendar": "BMV",  "tz": "America/Mexico_City","close_time": "15:00"},
}

PMC_CAL_MAP = {
    "NSE": "XNSE", "BSE": "XBOM", "NYSE": "NYSE", "LSE": "LSE",
    "TSE": "XJPX", "XETR": "XETR", "Euronext": "XAMS",
    "HKEX": "XHKG", "ASX": "XASX", "KOR": "XKRX", "SIX": "XSWX",
    "SSE": "XSHG", "XTSE": "XTSE", "XSES": "XSES", "XTPE": "XTPE",
    "BVMF": "BVMF", "XMAD": "XMAD", "XMIL": "XMIL", "BMV": "BMV",
}

CURRENCY_META = {
    "INR": {"symbol": "₹"}, "USD": {"symbol": "$"}, "GBP": {"symbol": "£"},
    "JPY": {"symbol": "¥"}, "EUR": {"symbol": "€"}, "CHF": {"symbol": "CHF"},
    "HKD": {"symbol": "HK$"}, "AUD": {"symbol": "A$"}, "KRW": {"symbol": "₩"},
    "CNY": {"symbol": "¥"}, "CAD": {"symbol": "C$"}, "SGD": {"symbol": "S$"},
    "TWD": {"symbol": "NT$"}, "BRL": {"symbol": "R$"}, "MXN": {"symbol": "Mex$"},
}

# ─────────────────── EXCHANGE HELPERS ────────────────────────
def detect_exchange_from_ticker(ticker):
    t = ticker.upper()
    INDEX_MAP = {
        "^GSPC":"US","^DJI":"US","^IXIC":"US","^RUT":"US",
        "^NSEI":".NS","^BSESN":".BO",
        "^FTSE":".L","^GDAXI":".DE","^FCHI":".PA","^N100":".PA",
        "^N225":".T","^TOPIX":".T","^HSI":".HK","^KS11":".KS",
        "^AXJO":".AX","000001.SS":".SS","^STI":".SI",
        "^BVSP":".SA","^GSPTSE":".TO","^MXX":"^MXX",
    }
    if t in INDEX_MAP:
        key = INDEX_MAP[t]
        return EXCHANGE_MAP[key] if key in EXCHANGE_MAP else EXCHANGE_MAP["US"]
    for suf, meta in EXCHANGE_MAP.items():
        if suf.startswith(".") and t.endswith(suf):
            return meta
    return EXCHANGE_MAP["US"]

def get_market_calendar(ticker):
    exch = detect_exchange_from_ticker(ticker)
    cal_key = exch.get("calendar")
    tz = exch.get("tz", "UTC")
    if cal_key in PMC_CAL_MAP:
        try:
            cal = mcal.get_calendar(PMC_CAL_MAP[cal_key])
            return cal, tz
        except Exception:
            return None, tz
    return None, tz

def last_completed_trading_date(ticker):
    cal, tz_str = get_market_calendar(ticker)
    tz = pytz.timezone(tz_str)
    today = datetime.now(tz).date()
    if cal is None:
        return (pd.Timestamp(today) - BDay(1)).date()
    schedule = cal.schedule(
        start_date=today - timedelta(days=20),
        end_date=today
    )
    if schedule.empty:
        return (pd.Timestamp(today) - BDay(1)).date()
    return schedule.index[-1].date()

@st.cache_data(ttl=60*60, show_spinner=False)
def get_fx_rate_to_inr(currency_code):
    currency_code = currency_code.upper()
    if currency_code == "INR":
        return 1.0
    try:
        fx_symbol = f"{currency_code}INR=X"
        df = yf.download(fx_symbol, period="5d", interval="1d", progress=False)
        if not df.empty:
            rate = df["Close"].dropna().iloc[-1]
            if isinstance(rate, pd.Series):
                rate = rate.iloc[0]
            if float(rate) > 0:
                return float(rate)
    except Exception:
        pass
    try:
        url = "https://api.exchangerate.host/latest"
        r = requests.get(url, params={"base": currency_code, "symbols": "INR"}, timeout=5)
        if r.status_code == 200:
            rate = r.json().get("rates", {}).get("INR")
            if rate and float(rate) > 0:
                return float(rate)
    except Exception:
        pass
    return None

def get_ticker_currency(ticker):
    try:
        tk = yf.Ticker(ticker)
        try:
            info = tk.fast_info or tk.info
        except Exception:
            info = tk.info
        cur = info.get("currency", None)
        if cur:
            return cur.upper()
    except Exception:
        pass
    exch = detect_exchange_from_ticker(ticker)
    return "INR" if exch["tz"] == "Asia/Kolkata" else "USD"

@st.cache_data(ttl=24*60*60)
def get_company_name(ticker):
    try:
        tk = yf.Ticker(ticker)
        try:
            info = tk.fast_info or tk.info
        except Exception:
            info = tk.info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

def format_price_dual_currency(price, currency_code):
    currency_code = currency_code.upper()
    meta = CURRENCY_META.get(currency_code, {"symbol": currency_code})
    symbol = meta["symbol"]
    try:
        fx_rate = get_fx_rate_to_inr(currency_code)
        price_inr = price * fx_rate if fx_rate else None
    except Exception:
        price_inr = None
    local_str = f"{symbol}{price:,.2f} ({currency_code})"
    inr_str = f"₹{price_inr:,.2f} (INR)" if price_inr is not None else "INR N/A"
    return local_str, inr_str, price_inr

# ─────────────────── DATA DOWNLOAD ───────────────────────────
@st.cache_data(show_spinner=False)
def download_full_data(ticker, start_date, end_date):
    df = yf.download(
        ticker, start=start_date, end=end_date,
        progress=False, auto_adjust=True, timeout=60
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open","High","Low","Close","Volume"]].dropna()

# ─────────────────── EXTRA CHART INDICATORS ──────────────────
def add_chart_indicators(df):
    """Add visual-only indicators for the candlestick chart."""
    d = df.copy()
    d.ta.ema(length=20, append=True)
    d.ta.ema(length=50, append=True)
    d.ta.ema(length=100, append=True)
    d.ta.ema(length=200, append=True)
    d.ta.sma(length=20, append=True)
    d.ta.sma(length=50, append=True)
    d.ta.rsi(length=14, append=True)
    d.ta.macd(append=True)
    d.ta.bbands(length=20, append=True)
    d.ta.atr(length=14, append=True)
    d.ta.adx(length=14, append=True)
    d.ta.vwap(append=True)
    d.ta.supertrend(length=10, multiplier=3, append=True)
    d.ta.psar(append=True)
    d.ta.kc(length=20, append=True)
    d.ta.donchian(length=20, append=True)
    d.ta.cci(length=20, append=True)
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    return d

# ─────────────────── STREAMLIT UI ────────────────────────────
st.set_page_config(page_title="LSTM-DNN v4 Stock Forecast", layout="wide")
st.title("LSTM-DNN v4 — Stock Price Forecast")
st.caption("Log-return prediction · Focal loss · Residual DNN · Walk-forward batch mode")

if "run_model" not in st.session_state:
    st.session_state.run_model = False

# ─── SIDEBAR ───
st.sidebar.header("Configuration")

tickers = DEFAULT_TICKERS.copy()
user_ticker = st.sidebar.text_input("Add custom ticker (e.g. MSFT, RELIANCE.NS)")
if user_ticker.strip():
    tickers[user_ticker.upper()] = user_ticker.upper()

ticker_choice = st.sidebar.selectbox("Select Ticker", list(tickers.keys()))
selected_ticker = tickers[ticker_choice]

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = selected_ticker
if st.session_state.last_ticker != selected_ticker:
    st.session_state.run_model = False
    st.session_state.last_ticker = selected_ticker

# Technical indicators selector
st.sidebar.subheader("Candlestick Indicators")
indicator_options = [
    "EMA 20","EMA 50","EMA 100","EMA 200",
    "SMA 20","SMA 50",
    "Bollinger Bands","Keltner Channel","Donchian Channel",
    "VWAP","Supertrend","Parabolic SAR",
    "RSI (14) Overlay","MACD Overlay","CCI (20) Overlay",
    "RSI (14)","MACD","CCI (20)","ADX",
]
selected_indicators = st.sidebar.multiselect(
    "Select indicators", indicator_options, default=["EMA 20","VWAP"]
)

years = st.sidebar.slider("Years of historical data", 1, 10, 3)

if st.sidebar.button("Run / Refresh Model"):
    st.session_state.run_model = True

# ─── METADATA ───
end_date        = date.today()
start_date      = end_date - timedelta(days=365 * years)
currency        = get_ticker_currency(selected_ticker)
exchange_meta   = detect_exchange_from_ticker(selected_ticker)
company_name    = ticker_choice if ticker_choice in DEFAULT_TICKERS else get_company_name(selected_ticker)
cal, tz_str     = get_market_calendar(selected_ticker)
tz              = pytz.timezone(tz_str)
last_trade_date = last_completed_trading_date(selected_ticker)

if cal is not None:
    today_sched = cal.schedule(start_date=date.today(), end_date=date.today())
    market_close_time = (
        today_sched["market_close"].iloc[0].astimezone(tz).strftime("%H:%M")
        if not today_sched.empty
        else exchange_meta.get("close_time","N/A")
    )
else:
    market_close_time = exchange_meta.get("close_time","N/A")

st.markdown(
    f"**Company:** {company_name}  \n"
    f"**Ticker:** {selected_ticker}  \n"
    f"**Exchange Timezone:** {exchange_meta['tz']}  \n"
    f"**Market Close Time:** {market_close_time}  \n"
    f"**Currency:** {currency}  \n"
    f"**Period:** {start_date} → {end_date}"
)

fx_rate = get_fx_rate_to_inr(currency)
fx_available = fx_rate is not None
if not fx_available:
    fx_rate = 1.0

if fx_available:
    st.caption(f"FX rate: 1 {currency} = {fx_rate:.4f} INR (live daily close)")
else:
    st.warning(f"Live FX rate unavailable. Using native {currency} pricing.")

# ─── DOWNLOAD DATA ───
with st.spinner("Downloading historical data..."):
    yahoo_end = min(last_trade_date, date.today()) + timedelta(days=1)
    raw_df = download_full_data(selected_ticker, str(start_date), str(yahoo_end))

st.markdown(f"**Last completed trading day:** {last_trade_date}")

if raw_df is None or raw_df.empty:
    st.error("No historical data returned. Check the ticker or date range.")
    st.stop()

if len(raw_df) < WINDOW * 5:
    st.error(f"Not enough data ({len(raw_df)} rows). Try a wider date range.")
    st.stop()

# ─── FEATURES ───
try:
    df, feats = make_features(raw_df)
except Exception as e:
    st.error(f"Feature engineering failed: {e}")
    st.stop()

st.success(f"Loaded {len(df)} trading days · {len(feats)} model features.")

# ─── CANDLESTICK CHART ───
st.markdown("---")
st.subheader("Candlestick Chart with Technical Indicators")

plot_raw = raw_df.iloc[-252:].copy()
with st.spinner("Computing chart indicators..."):
    plot_df = add_chart_indicators(plot_raw)

# Build dynamic subplots
panel_titles, panel_keys = ["Price"], ["price"]
if "RSI (14)" in selected_indicators: panel_titles.append("RSI");  panel_keys.append("rsi")
if "MACD"    in selected_indicators: panel_titles.append("MACD"); panel_keys.append("macd")
if "CCI (20)" in selected_indicators: panel_titles.append("CCI");  panel_keys.append("cci")
if "ADX"     in selected_indicators: panel_titles.append("ADX");  panel_keys.append("adx")

rows = len(panel_titles)
row_heights = [0.55] + [(0.45/(rows-1)) if rows > 1 else 0.0] * (rows-1)
fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=row_heights,
                    subplot_titles=panel_titles)
row_map = {key: i+1 for i, key in enumerate(panel_keys)}
rp = row_map["price"]

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
    low=plot_df["Low"], close=plot_df["Close"], name="Price"
), row=rp, col=1)

def add_line(col_name, label):
    if col_name in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[col_name],
            mode="lines", name=label
        ), row=rp, col=1)

def scale_to_price(series, price):
    mn, mx = series.min(), series.max()
    pmn, pmx = price.min(), price.max()
    if mx - mn == 0: return np.full(len(series), np.nan)
    return ((series - mn)/(mx - mn)) * (pmx - pmn) + pmn

if "EMA 20"  in selected_indicators: add_line("EMA_20",  "EMA 20")
if "EMA 50"  in selected_indicators: add_line("EMA_50",  "EMA 50")
if "EMA 100" in selected_indicators: add_line("EMA_100", "EMA 100")
if "EMA 200" in selected_indicators: add_line("EMA_200", "EMA 200")
if "SMA 20"  in selected_indicators: add_line("SMA_20",  "SMA 20")
if "SMA 50"  in selected_indicators: add_line("SMA_50",  "SMA 50")

# VWAP
if "VWAP" in selected_indicators:
    vwap_col = next((c for c in plot_df.columns if c.startswith("VWAP")), None)
    if vwap_col: add_line(vwap_col, "VWAP")

# Bollinger Bands
if "Bollinger Bands" in selected_indicators:
    for c in plot_df.columns:
        if c.startswith("BBU"): add_line(c, "BB Upper")
        elif c.startswith("BBM"): add_line(c, "BB Mid")
        elif c.startswith("BBL"): add_line(c, "BB Lower")

# Keltner
if "Keltner Channel" in selected_indicators:
    for c in plot_df.columns:
        if c.startswith("KCU"): add_line(c, "KC Upper")
        elif c.startswith("KCM"): add_line(c, "KC Mid")
        elif c.startswith("KCL"): add_line(c, "KC Lower")

# Donchian
if "Donchian Channel" in selected_indicators:
    for c in plot_df.columns:
        if c.startswith("DCU"): add_line(c, "DC Upper")
        if c.startswith("DCL"): add_line(c, "DC Lower")

# Supertrend
if "Supertrend" in selected_indicators:
    st_col = next((c for c in plot_df.columns if c.startswith("SUPERT_")), None)
    if st_col: add_line(st_col, "Supertrend")

# Parabolic SAR
if "Parabolic SAR" in selected_indicators:
    psar_col = next((c for c in plot_df.columns if c.startswith("PSAR")), None)
    if psar_col:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[psar_col],
            mode="markers", marker=dict(size=4), name="PSAR"
        ), row=rp, col=1)

# Overlays (scaled)
for ind_name, col_hint, color in [
    ("RSI (14) Overlay", "RSI_14", "purple"),
    ("MACD Overlay", "MACD_12_26_9", "orange"),
    ("CCI (20) Overlay", None, "brown"),
]:
    if ind_name in selected_indicators:
        if col_hint and col_hint in plot_df.columns:
            src = plot_df[col_hint]
        elif col_hint is None:
            cci_c = next((c for c in plot_df.columns if c.startswith("CCI_")), None)
            src = plot_df[cci_c] if cci_c else None
        else:
            src = None
        if src is not None:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=scale_to_price(src, plot_df["Close"]),
                mode="lines", name=ind_name,
                line=dict(color=color, dash="dot"), showlegend=False
            ), row=rp, col=1)

# Sub-panels
if "RSI (14)" in selected_indicators and "rsi" in row_map:
    if "RSI_14" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["RSI_14"], name="RSI",
            line=dict(color="purple")
        ), row=row_map["rsi"], col=1)
        fig.add_hline(y=70, line_dash="dash", row=row_map["rsi"], col=1)
        fig.add_hline(y=30, line_dash="dash", row=row_map["rsi"], col=1)
        fig.update_yaxes(fixedrange=True, row=row_map["rsi"])

if "MACD" in selected_indicators and "macd" in row_map:
    if "MACD_12_26_9" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["MACD_12_26_9"], name="MACD"
        ), row=row_map["macd"], col=1)
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df.get("MACDs_12_26_9", pd.Series()), name="Signal"
        ), row=row_map["macd"], col=1)
        fig.update_yaxes(fixedrange=True, row=row_map["macd"])

if "CCI (20)" in selected_indicators and "cci" in row_map:
    cci_col = next((c for c in plot_df.columns if c.startswith("CCI_")), None)
    if cci_col:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[cci_col], name="CCI"
        ), row=row_map["cci"], col=1)
        fig.add_hline(y=100,  line_dash="dash", row=row_map["cci"], col=1)
        fig.add_hline(y=-100, line_dash="dash", row=row_map["cci"], col=1)
        fig.update_yaxes(fixedrange=True, row=row_map["cci"])

if "ADX" in selected_indicators and "adx" in row_map:
    if "ADX_14" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["ADX_14"], name="ADX"
        ), row=row_map["adx"], col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray",
                      row=row_map["adx"], col=1)
        fig.update_yaxes(fixedrange=True, row=row_map["adx"])

sym = CURRENCY_META.get(currency, {}).get("symbol", currency)
fig.update_layout(
    height=900, template="plotly_white", hovermode="x unified",
    xaxis_rangeslider_visible=False,
    yaxis_title=f"Price ({currency})", yaxis_tickprefix=sym
)
st.plotly_chart(fig, use_container_width=True)

# ─────────────────── MODEL SECTION ───────────────────────────
if st.session_state.run_model:

    # ── Scalers & sequences
    nf       = len(feats)
    sp_idx   = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:sp_idx]
    test_df  = df.iloc[sp_idx:]

    Xtr = train_df[feats].values
    Xte = test_df[feats].values
    ytr = train_df["target_ret"].values.reshape(-1,1)
    yte = test_df["target_ret"].values.reshape(-1,1)
    dtr = train_df["target_dir"].values
    dte = test_df["target_dir"].values

    Xsc = RobustScaler(); ysc = RobustScaler()
    Xsc.fit(Xtr); ysc.fit(ytr)
    Xtr_s = Xsc.transform(Xtr)
    Xte_s = Xsc.transform(Xte)
    ytr_s = ysc.transform(ytr).flatten()

    Xseq, yseq_r, yseq_d = make_seqs(Xtr_s, ytr_s, dtr, WINDOW)

    # ── Safe name for model files
    safe_name = selected_ticker.replace("^","").replace(".","_").replace("-","_")
    mp, sp = mpath(safe_name), spath(safe_name)
    model = None

    if os.path.exists(mp) and os.path.exists(sp):
        st.info("Loading saved model...")
        try:
            model = load_model(mp, custom_objects={"_loss": focal_loss()})
            sv = joblib.load(sp)
            Xsc, ysc = sv["X"], sv["y"]
            # Re-transform with loaded scalers
            Xtr_s = Xsc.transform(Xtr)
            Xte_s = Xsc.transform(Xte)
            ytr_s = ysc.transform(ytr).flatten()
            Xseq, yseq_r, yseq_d = make_seqs(Xtr_s, ytr_s, dtr, WINDOW)
            _compile_model(model, LR_INC)
        except Exception as e:
            st.warning(f"Load failed ({e}), retraining...")
            model = None
            for p in [mp, sp]:
                if os.path.exists(p): os.remove(p)

    if model is None:
        st.info(f"Training new model — {nf} features, window={WINDOW}...")
        prog = st.progress(0, text="Training...")
        model = build_model((WINDOW, nf), LR)
        cbs = [
            EarlyStopping(monitor="val_dir_output_accuracy",
                          patience=18, restore_best_weights=True, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                              patience=8, min_lr=1e-6, verbose=0),
        ]
        model.fit(
            Xseq,
            {"logret_output": yseq_r, "dir_output": yseq_d},
            epochs=EPOCHS, batch_size=BATCH,
            validation_split=0.12, callbacks=cbs, verbose=0
        )
        prog.progress(100, text="Training complete.")
        model.save(mp)
        joblib.dump({"X": Xsc, "y": ysc}, sp)
        st.success(f"Model saved: {mp}")

    # ── Walk-forward batch prediction (v4 style)
    st.info("Running walk-forward evaluation...")
    wf_prog = st.progress(0, text="Walk-forward...")

    buf_X = list(Xtr_s)
    buf_r = list(ytr_s)
    buf_d = list(dtr)

    pred_prices, actual_prices = [], []
    pred_dirs,   actual_dirs   = [], []
    pred_log_returns           = []
    dir_probs_all              = []

    N      = len(Xte_s)
    chunks = list(range(0, N, INC_STEP))

    all_closes   = list(train_df["Close"].values) + list(test_df["Close"].values)
    close_offset = sp_idx

    for ci, chunk_start in enumerate(chunks):
        chunk_end = min(chunk_start + INC_STEP, N)

        # Build batch windows
        batch_wins = []
        for i in range(chunk_start, chunk_end):
            combined = buf_X + list(Xte_s[chunk_start:i])
            batch_wins.append(np.array(combined[-WINDOW:]))

        batch_arr = np.array(batch_wins)
        ret_preds, dir_probs = model.predict(batch_arr, verbose=0, batch_size=64)

        for j, i in enumerate(range(chunk_start, chunk_end)):
            pred_logret = float(ysc.inverse_transform([[ret_preds[j][0]]])[0][0])
            pred_logret = np.clip(pred_logret, np.log(1-MAX_MOVE), np.log(1+MAX_MOVE))
            prev_close  = float(all_closes[close_offset + i])
            pred_price  = prev_close * np.exp(pred_logret)
            pred_dir    = int(dir_probs[j][0] > 0.5)
            dir_prob_val= float(dir_probs[j][0])

            true_logret = float(yte[i][0])
            true_price  = (float(test_df["Close"].iloc[i+1])
                           if i+1 < len(test_df)
                           else prev_close * np.exp(true_logret))
            true_dir    = int(dte[i])

            pred_prices.append(pred_price)
            actual_prices.append(true_price)
            pred_dirs.append(pred_dir)
            actual_dirs.append(true_dir)
            pred_log_returns.append(pred_logret)
            dir_probs_all.append(dir_prob_val)

        # Reveal true data
        for i in range(chunk_start, chunk_end):
            buf_X.append(Xte_s[i])
            buf_r.append(float(ysc.transform([[float(yte[i][0])]])[0][0]))
            buf_d.append(int(dte[i]))

        # Incremental retrain
        if len(buf_X) > WINDOW * 5:
            rX = np.array(buf_X[-WINDOW*10:])
            rr = np.array(buf_r[-WINDOW*10:])
            rd = np.array(buf_d[-WINDOW*10:])
            iX, ir, id_ = make_seqs(rX, rr, rd, WINDOW)
            if len(iX) >= BATCH:
                model.fit(
                    iX,
                    {"logret_output": ir, "dir_output": id_},
                    epochs=INC_EPOCHS, batch_size=BATCH, verbose=0
                )

        wf_prog.progress(int((ci+1)/len(chunks)*100),
                         text=f"Walk-forward: chunk {ci+1}/{len(chunks)}")

    wf_prog.progress(100, text="Walk-forward complete.")
    model.save(mp)

    # ── Metrics
    preds   = np.array(pred_prices)
    actuals = np.array(actual_prices)

    mse_v  = sr(mean_squared_error(actuals, preds))
    rmse_v = sr(np.sqrt(mse_v))
    mae_v  = sr(mean_absolute_error(actuals, preds))
    mape_v = sr(np.mean(np.abs((actuals - preds) / (actuals + 1e-9))) * 100)
    r2_v   = sr(r2_score(actuals, preds))
    da_v   = sr(accuracy_score(actual_dirs, pred_dirs))

    # ── Next-day forecast
    last_win = np.array(buf_X[-WINDOW:]).reshape(1, WINDOW, nf)
    ret_next, dir_next = model.predict(last_win, verbose=0)
    next_logret = float(ysc.inverse_transform([[ret_next[0][0]]])[0][0])
    next_logret = np.clip(next_logret, np.log(1-MAX_MOVE), np.log(1+MAX_MOVE))
    next_dir_prob = float(dir_next[0][0])
    last_price  = float(df["Close"].iloc[-1])
    next_price  = last_price * np.exp(next_logret)

    # Next trading date
    if cal is not None:
        last_session  = df.index[-1].date()
        future_sched  = cal.schedule(
            start_date=last_session,
            end_date=last_session + timedelta(days=10)
        )
        future_dates = future_sched.index.date
        future_dates = future_dates[future_dates > last_session]
        next_date = pd.Timestamp(future_dates[0]) if len(future_dates) > 0 else df.index[-1] + BDay(1)
    else:
        next_date = df.index[-1] + BDay(1)

    # ── Model-driven signal
    signal, confidence, reasons = generate_model_signal(next_dir_prob, next_logret)
    insight = generate_smart_insight(signal, confidence, reasons,
                                     last_price, next_price,
                                     next_dir_prob, next_logret)

    # ── DISPLAY
    st.markdown("---")
    st.subheader("Forecast & Evaluation Metrics")

    last_local, last_inr, _ = format_price_dual_currency(last_price, currency)
    next_local, next_inr, _ = format_price_dual_currency(next_price, currency)
    pct_change = (next_price / last_price - 1) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Last Close Price", last_local)
        st.caption(f"INR equivalent: {last_inr}")
        st.markdown(f"**Forecast Date:** {next_date.date()}")
        st.metric("Next Day Forecast", next_local,
                  delta=f"{pct_change:+.2f}%")
        st.caption(f"INR equivalent: {next_inr}")
        st.markdown(f"**Direction head probability (up):** {next_dir_prob*100:.1f}%")

    with col2:
        st.write(f"R² Score: **{r2_v}**")
        st.write(f"MAE: **{mae_v}**")
        st.write(f"MSE: **{mse_v}**")
        st.write(f"RMSE: **{rmse_v}**")
        st.write(f"MAPE (%): **{mape_v:.4f}**")
        st.write(f"Directional Accuracy: **{da_v*100:.2f}%**")

    # ── AI Signal (model-driven)
    st.markdown("---")
    st.subheader("🤖 AI Trading Recommendation")
    st.caption("Signal derived purely from LSTM-DNN v4 outputs (regression + direction head)")

    if signal == "BUY":
        st.success(f"AI Recommendation: **BUY**")
    elif signal == "SELL":
        st.error(f"AI Recommendation: **SELL**")
    else:
        st.warning(f"AI Recommendation: **HOLD**")

    st.progress(int(confidence * 100))
    st.write(f"Model Confidence: **{confidence*100:.1f}%**")

    # ── Smart Insight
    st.markdown("### 📊 Model-Driven Market Insight")
    st.info(insight)

    # ── Directional probability gauge (walk-forward)
    st.markdown("---")
    st.subheader("Walk-Forward Direction Probability History")
    dp_df = pd.DataFrame({
        "Date": test_df.index[:len(dir_probs_all)],
        "Dir Prob (Up)": dir_probs_all,
        "Actual Direction": actual_dirs,
    })
    fig_dp = go.Figure()
    fig_dp.add_trace(go.Scatter(
        x=dp_df["Date"], y=dp_df["Dir Prob (Up)"],
        mode="lines", name="Model P(Up)",
        line=dict(color="royalblue", width=1.5)
    ))
    fig_dp.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig_dp.add_hline(y=0.55, line_dash="dot", line_color="green",
                     annotation_text="BUY threshold")
    fig_dp.add_hline(y=0.45, line_dash="dot", line_color="red",
                     annotation_text="SELL threshold")
    fig_dp.update_layout(
        height=300, template="plotly_white",
        yaxis_title="P(Up)", xaxis_title="Date",
        hovermode="x unified"
    )
    st.plotly_chart(fig_dp, use_container_width=True)

    # ── Predicted vs Actual chart
    st.markdown("---")
    st.subheader("Predicted vs Actual Price 📉")

    full_actual = df["Close"].values
    pred_series = np.full(len(df), np.nan)
    start_pred  = sp_idx + 1
    end_pred    = start_pred + len(preds)
    if end_pred <= len(pred_series):
        pred_series[start_pred:end_pred] = preds
    else:
        pred_series[start_pred:] = preds[:len(pred_series)-start_pred]

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df.index, y=full_actual,
        mode="lines", name="Actual Price"
    ))
    fig_pred.add_trace(go.Scatter(
        x=df.index, y=pred_series,
        mode="lines", name="Predicted Price",
        line=dict(color="red", dash="dot")
    ))
    fig_pred.add_trace(go.Scatter(
        x=[df.index[-1], next_date],
        y=[last_price, next_price],
        mode="lines+markers", name="Next Day Forecast",
        line=dict(color="green", dash="dash"),
        marker=dict(size=9)
    ))
    fig_pred.update_layout(
        height=700, template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date", yaxis_title=f"Price ({currency})"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Save model
    try:
        model.save(mp)
        st.success("Model saved successfully.")
    except Exception as e:
        st.warning(f"Model save failed: {e}")

else:
    st.info(
        "Configure parameters from the sidebar and click "
        "**Run / Refresh Model** to train and forecast."
    )