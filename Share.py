import streamlit as st
import requests
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
    page_title="Predict Stocks – India",
    page_icon="📈",
    layout="wide"
)

st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)

# -------------------------------------------------
# OPEN INDIAN STOCK SEARCH API
# -------------------------------------------------
@st.cache_data(ttl=300)
def search_indian_stocks(query):
    url = "http://65.0.104.9/search"
    params = {"q": query}
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        return data.get("results", [])
    except:
        return []

# -------------------------------------------------
# LOAD STOCK DATA
# -------------------------------------------------
@st.cache_data(ttl=300)
def load_stock_data(symbol, years=5):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.download(symbol, start=start, end=end)
    df.dropna(inplace=True)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# -------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------
st.sidebar.title("📊 Predict Stocks (India)")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Search & Select Stock",
        "Technical Analysis",
        "Screener",
        "Pattern Signal",
        "Next‑Day Forecast"
    ]
)

# -------------------------------------------------
# HOME
# -------------------------------------------------
if page == "Home":
    st.title("📈 Indian Stock Market Screener & Prediction")

    st.markdown("""
An **all‑in‑one Indian stock analysis tool** for NSE stocks.

✅ Live market data  
✅ Company name search  
✅ Technical indicators  
✅ Screeners  
✅ ML‑based next‑day prediction  

**No API keys required.**
""")

# -------------------------------------------------
# SEARCH & SELECT STOCK
# -------------------------------------------------
elif page == "Search & Select Stock":
    st.title("🔎 Search Indian Stocks (Live API)")

    query = st.text_input("Type company name (e.g. Reliance, Tata, HDFC)")

    if len(query) >= 2:
        results = search_indian_stocks(query)

        if results:
            options = {
                f"{r['company_name']} ({r['symbol']}.NS)": r["symbol"] + ".NS"
                for r in results
            }

            selected = st.selectbox("Select Stock", list(options.keys()))
            symbol = options[selected]

            st.success(f"Selected Stock: {symbol}")

            df = load_stock_data(symbol)
            st.line_chart(df["Close"])
        else:
            st.warning("No matching stocks found.")

# -------------------------------------------------
# TECHNICAL ANALYSIS
# -------------------------------------------------
elif page == "Technical Analysis":
    st.title("📈 Technical Analysis")

    symbol = st.text_input("Enter NSE Symbol", "RELIANCE.NS")
    df = load_stock_data(symbol)

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = calculate_rsi(df["Close"])

    fig = go.Figure()
    fig.add_candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    )
    fig.add_scatter(x=df.index, y=df["EMA20"], name="EMA 20")
    fig.add_scatter(x=df.index, y=df["EMA50"], name="EMA 50")

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Latest RSI", round(df["RSI"].iloc[-1], 2))

# -------------------------------------------------
# SCREENER
# -------------------------------------------------
elif page == "Screener":
    st.title("🧮 Stock Screener")

    symbol = st.text_input("Enter NSE Symbol", "INFY.NS")
    df = load_stock_data(symbol)

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    trend = "Bullish" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "Bearish"
    breakout = "YES" if df["Close"].iloc[-1] > df["Close"].rolling(20).max().iloc[-2] else "NO"

    col1, col2 = st.columns(2)
    col1.metric("Trend", trend)
    col2.metric("Breakout", breakout)

# -------------------------------------------------
# PATTERN SIGNAL (RULE‑BASED)
# -------------------------------------------------
elif page == "Pattern Signal":
    st.title("🕯️ Candlestick Signal")

    symbol = st.text_input("Enter NSE Symbol", "HDFCBANK.NS")
    df = load_stock_data(symbol, 1)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = "Neutral"
    if last["Close"] > last["Open"] and prev["Close"] < prev["Open"]:
        signal = "Bullish Engulfing"
    elif last["Close"] < last["Open"] and prev["Close"] > prev["Open"]:
        signal = "Bearish Engulfing"

    st.metric("Detected Signal", signal)

# -------------------------------------------------
# NEXT‑DAY FORECAST
# -------------------------------------------------
elif page == "Next‑Day Forecast":
    st.title("🤖 Next‑Day Price Forecast (ML)")

    symbol = st.text_input("Enter NSE Symbol", "TRIDENT.NS")
    df = load_stock_data(symbol)

    df["t"] = np.arange(len(df))
    X = df[["t"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict([[len(df)]])[0]

    st.metric("Predicted Next Close", f"₹ {round(prediction,2)}")
    st.caption("Baseline ML model (deploy‑safe).")
``
