# =========================
# MASTER STOCK ANALYSIS APP
# =========================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Master Trading Dashboard",
    layout="wide"
)

st.title("📊 Master Trading & Analysis Reversion",st.title("📊 Master Trading & Analysis Dashboard")
        "Breakout",
        "Swing"
    ]
)

# -------------------------
# DATA FETCH
# -------------------------
@st.cache_data(ttl=300)
def fetch_data(symbol, period):
    df = yf.download(symbol, period=period)
    df.dropna(inplace=True)
    return df

# -------------------------
# INDICATORS
# -------------------------
def add_indicators(df):
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean()))
    df["Return"] = df["Close"].pct_change()
    return df

# -------------------------
# PRICE CHART
# -------------------------
def price_chart(df):
    fig = go.Figure()

    fig.add_candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    )

    fig.add_scatter(x=df.index, y=df["EMA_20"], name="EMA 20")
    fig.add_scatter(x=df.index, y=df["EMA_50"], name="EMA 50")

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# STRATEGY SIGNAL PLACEHOLDER
# -------------------------
def generate_signal(df):
    if df["EMA_20"].iloc[-1] > df["EMA_50"].iloc[-1]:
        return "✅ Bullish"
    else:
        return "⚠️ Bearish"

# -------------------------
# TOP 5 TRADE IDEAS
# -------------------------
@st.cache_data(ttl=300)
def top5_assets(asset_type):
    if asset_type == "Stocks":
        universe = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

    elif asset_type == "Commodities":
        universe = ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]

    else:  # Options proxy via indices
        universe = ["^NSEI", "^NSEBANK", "^BSESN", "^NSEFIN", "^CNXIT"]

    data = []
    for u in universe:
        df = yf.download(u, period="5d")
        ret = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
        data.append([u, round(ret, 2)])

    return pd.DataFrame(data, columns=["Symbol", "5D % Move"]).sort_values(
        "5D % Move", ascending=False
    )

# -------------------------
# CHATBOT (RULE-BASED)
# -------------------------
def chatbot_response(question, df):
    q = question.lower()

    if "trend" in q:
        return f"Current trend looks {generate_signal(df)}"

    if "rsi" in q:
        return f"Latest RSI is {round(df['RSI'].iloc[-1],2)}"

    if "price" in q:
        return f"Last close price is {round(df['Close'].iloc[-1],2)}"

    return "I can help with trend, RSI, and price. More intelligence can be added."

# =========================
# MAIN DASHBOARD LAYOUT
# =========================

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📈 Live Market Analysis")

    try:
        df = fetch_data(symbol_input, timeframe)
        df = add_indicators(df)

        price_chart(df)

        st.markdown("### 📌 Strategy Summary")
        st.write("Active Strategies:", strategy_selected)
        st.write("Overall Signal:", generate_signal(df))

        st.markdown("### 📊 Key Metrics")
        st.metric("Last Price", round(df["Close"].iloc[-1], 2))
        st.metric("RSI", round(df["RSI"].iloc[-1], 2))
        st.metric("Volatility", round(df["Return"].std() * np.sqrt(252) * 100, 2))

    except Exception as e:
        st.error("Invalid symbol or data not available.")

with col2:
    st.subheader("🔥 Top 5 Trade Ideas")

    top5 = top5_assets(asset_type)
    st.dataframe(top5, use_container_width=True)

# -------------------------
# CHATBOT SECTION
# -------------------------
st.markdown("---")
st.subheader("💬 Trading Assistant Chatbot")

user_q = st.text_input("Ask something (trend, RSI, price)")

if user_q:
    response = chatbot_response(user_q, df)
    st.success(response)

# -------------------------
# FOOTER
# -------------------------
st.markdown(
    f"""
    ---
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
    This is an extensible trading dashboard. Strategies can be plugged in one by one.
    """
)

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("🔧 Controls")

asset_type = st.sidebar.selectbox(
    "Asset Type",
    ["Stocks", "Commodities", "Options (Index Proxy)"]
)

symbol_input = st.sidebar.text_input(
    "Enter Symbol",
    "RELIANCE.NS"
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1mo", "3mo", "6mo", "1y", "2y"]
)

strategy_selected = st.sidebar.multiselect(
    "Select Strategies (enable one by one later)",
    [
        "Trend Following",
        "Momentum",
