import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Predict Stocks",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    "<style>footer{visibility:hidden;}</style>",
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("📊 Predict Stocks")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Fundamental Info",
        "Technical Indicators",
        "Screener",
        "Pattern Recognition",
        "Next-Day Forecasting"
    ]
)

# -------------------------------------------------
# COMMON UTIL
# -------------------------------------------------
@st.cache_data(ttl=300)
def load_data(ticker, years=5):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.download(ticker, start, end)
    df.dropna(inplace=True)
    return df

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------------------------------------
# HOME
# -------------------------------------------------
if page == "Home":
    st.title("📈 Stock Market Screener & Prediction")

    st.markdown("""
**Predict Stocks** is an all‑in‑one platform for retail investors to analyze  
**NSE‑listed stocks** using:

✅ Fundamental analysis  
✅ Technical indicators  
✅ Screeners  
✅ Pattern signals  
✅ Machine‑learning‑based forecasting  

Data Source: **Yahoo Finance**
    """)

    st.subheader("🧭 Modules")
    st.markdown("""
- **Fundamental Info** – company details & financials  
- **Technical Indicators** – RSI, EMA, MACD‑lite  
- **Screener** – breakout & momentum signals  
- **Pattern Recognition** – bullish / bearish logic  
- **Next‑Day Forecasting** – ML regression model  
    """)

# -------------------------------------------------
# FUNDAMENTAL INFO
# -------------------------------------------------
elif page == "Fundamental Info":
    st.title("🏢 Fundamental Information")

    ticker = st.text_input("Enter NSE Symbol", "RELIANCE.NS")
    stock = yf.Ticker(ticker)
    info = stock.info

    st.subheader(info.get("longName", ticker))

    col1, col2 = st.columns(2)
    col1.metric("Market Cap", info.get("marketCap", "NA"))
    col2.metric("52W High", info.get("fiftyTwoWeekHigh", "NA"))

    st.markdown(f"**Sector:** {info.get('sector','NA')}")
    st.markdown(f"**Industry:** {info.get('industry','NA')}")

    with st.expander("Business Summary"):
        st.write(info.get("longBusinessSummary", "Not Available"))

# -------------------------------------------------
# TECHNICAL INDICATORS
# -------------------------------------------------
elif page == "Technical Indicators":
    st.title("📈 Technical Indicators")

    ticker = st.text_input("Enter NSE Symbol", "TCS.NS")
    df = load_data(ticker)

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = rsi(df["Close"])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50"))

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Latest RSI", round(df["RSI"].iloc[-1], 2))

# -------------------------------------------------
# SCREENER
# -------------------------------------------------
elif page == "Screener":
    st.title("🔎 Stock Screener")

    ticker = st.text_input("Enter NSE Symbol", "INFY.NS")
    df = load_data(ticker)

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    breakout = "YES" if df["Close"].iloc[-1] > df["Close"].rolling(20).max().iloc[-2] else "NO"
    trend = "Bullish" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "Bearish"

    col1, col2 = st.columns(2)
    col1.metric("Trend", trend)
    col2.metric("Breakout", breakout)

# -------------------------------------------------
# PATTERN RECOGNITION (DEPLOY‑SAFE)
# -------------------------------------------------
elif page == "Pattern Recognition":
    st.title("🕯️ Pattern Recognition")

    ticker = st.text_input("Enter NSE Symbol", "HDFCBANK.NS")
    df = load_data(ticker, 1)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = "Neutral"
    if last["Close"] > last["Open"] and prev["Close"] < prev["Open"]:
        signal = "Bullish Engulfing"
    elif last["Close"] < last["Open"] and prev["Close"] > prev["Open"]:
        signal = "Bearish Engulfing"

    st.metric("Detected Pattern", signal)

# -------------------------------------------------
# NEXT‑DAY FORECASTING (ML‑SAFE)
# -------------------------------------------------
elif page == "Next-Day Forecasting":
    st.title("🤖 Next‑Day Forecasting")

    ticker = st.text_input("Enter NSE Symbol", "TRIDENT.NS")
    df = load_data(ticker)

    df["Day"] = np.arange(len(df))
    X = df[["Day"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[len(df)]])
    prediction = model.predict(next_day)[0]

    st.metric("Predicted Next Close", f"₹ {round(prediction,2)}")

    st.caption("Model: Linear Regression (deploy‑safe baseline)")
