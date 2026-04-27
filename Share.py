import streamlit as st

stimport pandas as pdst.set_page_config(
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.title("📈 Simple Stock Market App")

st.caption("Data is for educational purposes only. Not investment advice.")

# ---------------------------
# Helper function
# ---------------------------
@st.cache_data
def load_data(symbol):
    try:
        df = yf.download(symbol, period="1y", progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.dropna()
    except Exception:
        return pd.DataFrame()

# ---------------------------
# Sidebar
# ---------------------------
symbol = st.sidebar.text_input("Enter Stock Symbol", "RELIANCE.NS")

# ---------------------------
# Load data
# ---------------------------
df = load_data(symbol)

if df.empty:
    st.error("No data found. Check symbol.")
    st.stop()

# ---------------------------
# Price section
# ---------------------------
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])

change_pct = ((last_close - prev_close) / prev_close) * 100

st.metric(
    label="Last Close Price",
    value=f"{last_close:.2f}",
    delta=f"{change_pct:.2f}%"
)

# ---------------------------
# Chart
# ---------------------------
st.subheader("Price Chart")
st.line_chart(df["Close"])

# ---------------------------
# Technical Indicators
# ---------------------------
st.subheader("Technical Indicators")

df["EMA20"] = df["Close"].ewm(span=20).mean()
df["EMA50"] = df["Close"].ewm(span=50).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

col1, col2, col3 = st.columns(3)

col1.metric("EMA 20", f"{float(df['EMA20'].iloc[-1]):.2f}")
col2.metric("EMA 50", f"{float(df['EMA50'].iloc[-1]):.2f}")
col3.metric("RSI", f"{float(df['RSI'].iloc[-1]):.2f}")

# ---------------------------
# Screener
# ---------------------------
st.subheader("Screener")

trend = "Bullish" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "Bearish"

recent_high = float(df["Close"].rolling(20).max().iloc[-1])
breakout = "YES" if last_close >= recent_high else "NO"

c1, c2 = st.columns(2)
c1.metric("Trend", trend)
c2.metric("20-Day Breakout", breakout)

# ---------------------------
# Forecast
# ---------------------------
st.subheader("Next Day Forecast (Simple ML)")

X = np.arange(len(df)).reshape(-1, 1)
y = df["Close"].values

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[len(df)]])[0]

st.metric("Predicted Next Close", f"{prediction:.2f}")
    page_title="Stock Market App",
    layout="wide"
)

import yfinance as yf
