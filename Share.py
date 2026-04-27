# ===============================
# app.py
# ===============================lit config MUST come immediately after import# ===============================
st.set_page_config(
    page_title="Predict Stocks - India",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Standard imports
# -----------------------------
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Optional NSE options support
try:
    from pnsea import NSE
except Exception:
    NSE = None

st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)

st.sidebar.caption(
    "⚠️ Data is for informational purposes only. Not investment advice."
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(ttl=300)
def load_stock(symbol, years=5):
    try:
        end = datetime.today()
        start = end - timedelta(days=365 * years)
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# -----------------------------
# Sidebar navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Technical Analysis",
        "Screener",
        "Forecast"
    ]
)

# -----------------------------
# Pages
# -----------------------------
if page == "Home":
    st.title("Indian Stock Market Dashboard")
    st.write("Stable, cloud-safe stock analytics app.")

elif page == "Technical Analysis":
    st.title("📈 Technical Analysis")

    symbol = st.text_input("NSE Symbol", "RELIANCE.NS")
    df = load_stock(symbol)

    if df.empty:
        st.error("No data available.")
    else:
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
        fig.add_scatter(x=df.index, y=df["EMA20"], name="EMA 20")
        fig.add_scatter(x=df.index, y=df["EMA50"], name="EMA 50")
        st.plotly_chart(fig, use_container_width=True)

        last_close = safe_float(df["Close"].iloc[-1])
        if last_close:
            st.metric("Last Close", f"{last_close:.2f}")

elif page == "Screener":
    st.title("🧮 Screener")

    symbol = st.text_input("NSE Symbol", "INFY.NS")
    df = load_stock(symbol)

    if df.empty or len(df) < 20:
        st.error("Insufficient data.")
    else:
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        trend = "Bullish" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "Bearish"
        st.metric("Trend", trend)

elif page == "Forecast":
    st.title("🤖 Next-Day Forecast")

    symbol = st.text_input("NSE Symbol", "RELIANCE.NS")
    df = load_stock(symbol)

    if df.empty or len(df) < 5:
        st.error("Not enough data.")
    else:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["Close"].astype(float).values

        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict([[len(df)]])[0]

        st.metric("Predicted Next Close", f"{pred:.2f}")

import streamlit as st

