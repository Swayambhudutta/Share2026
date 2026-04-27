# =========================================
# MASTER TRADING & ANALYSIS STREAMLIT APP
# =========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Master Trading Dashboard",
    layout="wide",
)

st.title("📊 Master Trading & Analysis Dashboard")

# -----------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------
st.sidebar.header("🔧 Controls")

asset_type = st.sidebar.selectbox(
    "Asset Type",
    ["Stocks", "Commodities", "Options (Index Proxy)"],
)

symbol_input = st.sidebar.text_input(
    "Enter Symbol",
    "RELIANCE.NS",
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1mo", "3mo", "6mo", "1y", "2y"],
)

strategy_selected = st.sidebar.multiselect(
    "Select Strategies (add logic later)",
    [
        "Trend Following",
        "Momentum",
        "Mean Reversion",
        "Breakout",
        "Swing",
    ],
)

# -----------------------------------------
# DATA FETCH
# -----------------------------------------
@st.cache_data(ttl=300)
def fetch_data(symbol, period):
    df = yf.download(symbol, period=period)
    df.dropna(inplace=True)
    return df

# -----------------------------------------
# INDICATORS
# -----------------------------------------
def add_indicators(df):
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Return"] = df["Close"].pct_change()
    return df

# -----------------------------------------
# PRICE CHART
# -----------------------------------------
def price_chart(df):
    fig = go.Figure()

    fig.add_candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    )

    fig.add_scatter(x=df.index, y=df["EMA_20"], name="EMA 20")
    fig.add_scatter(x=df.index, y=df["EMA_50"], name="EMA 50")

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# BASIC SIGNAL (PLACEHOLDER)
# -----------------------------------------
def generate_signal(df):
    if df["EMA_20"].iloc[-1] > df["EMA_50"].iloc[-1]:
        return "✅ Bullish Trend"
    else:
        return "⚠️ Bearish Trend"

# -----------------------------------------
# TOP 5 IDEAS
# -----------------------------------------
@st.cache_data(ttl=300)
def top5_assets(asset_type):
    if asset_type == "Stocks":
        universe = [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
        ]

    elif asset_type == "Commodities":
        universe = ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F"]

    else:
        universe = ["^NSEI", "^NSEBANK", "^BSESN", "^NSEFIN", "^CNXIT"]

    rows = []

    for u in universe:
        try:
            df = yf.download(u, period="5d")
            ret = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
            rows.append([u, round(ret, 2)])
        except:
            pass

    return (
        pd.DataFrame(rows, columns=["Symbol", "5D % Move"])
        .sort_values("5D % Move", ascending=False)
        .head(5)
    )

# -----------------------------------------
# CHATBOT (RULE BASED)
# -----------------------------------------
def chatbot_response(question, df):
    q = question.lower()

    if "trend" in q:
        return generate_signal(df)

    if "rsi" in q:
        return f"Latest RSI: {round(df['RSI'].iloc[-1], 2)}"

    if "price" in q:
        return f"Last Close Price: {round(df['Close'].iloc[-1], 2)}"

    return "You can ask about trend, RSI, or price."

# =========================================
# MAIN DASHBOARD
# =========================================

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📈 Live Market Dashboard")

    try:
        df = fetch_data(symbol_input, timeframe)
        df = add_indicators(df)

        price_chart(df)

        st.markdown("### 📌 Strategy Overview")
        st.write("Selected Strategies:", strategy_selected)
        st.write("Current Signal:", generate_signal(df))

        st.markdown("### 📊 Key Metrics")
        st.metric("Last Price", round(df["Close"].iloc[-1], 2))
        st.metric("RSI", round(df["RSI"].iloc[-1], 2))
        st.metric(
            "Annualized Volatility (%)",
            round(df["Return"].std() * np.sqrt(252) * 100, 2),
        )

    except Exception as e:
        st.error("Error loading data. Check symbol or timeframe.")

with col2:
    st.subheader("🔥 Top 5 Trade Ideas")
    top5 = top5_assets(asset_type)
    st.dataframe(top5, use_container_width=True)

# -----------------------------------------
# CHATBOT SECTION
# -----------------------------------------
st.markdown("---")
st.subheader("💬 Trading Assistant")

user_question = st.text_input("Ask: trend / RSI / price")

if user_question:
    st.success(chatbot_response(user_question, df))

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown(
    f"""
    ---
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
    ✅ Single-file Streamlit trading dashboard  
    ✅ Strategy hooks ready — logic can be added incrementally
    """
)
``
