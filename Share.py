
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Optional NSE option chain support
try:
    from pnsea import NSE
except Exception:
    NSE = None

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Predict Stocks - India",
    page_icon="📈",
    layout="wide"
)

st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)

st.sidebar.caption(
    "⚠️ Data shown is for informational purposes only. "
    "Not investment advice."
)

# ---------------- HELPERS ----------------

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

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return None

# ---------------- SIDEBAR ----------------

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Technical Analysis",
        "Screener",
        "Commodities",
        "Options",
        "Forecast"
    ]
)

# ---------------- HOME ----------------

if page == "Home":
    st.title("Indian Stock Market Dashboard")

    st.markdown("""
✅ Live NSE equity data  
✅ Technical indicators  
✅ Screener signals  
✅ Commodity tracking  
✅ Options snapshot (best‑effort)  
✅ Safe ML forecast  

Built to **never crash** when data is missing.
""")

# ---------------- TECHNICAL ANALYSIS ----------------

elif page == "Technical Analysis":
    st.title("📈 Technical Analysis")

    symbol = st.text_input("Enter NSE Symbol", "RELIANCE.NS")
    df = load_stock(symbol)

    if df.empty:
        st.error("No data available for this symbol.")
    else:
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        df["RSI"] = 100 - (100 / (1 + rs))

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
        last_rsi = safe_float(df["RSI"].iloc[-1])

        if last_close is not None:
            st.metric("Last Close", f"{last_close:.2f}")
        if last_rsi is not None:
            st.metric("RSI", f"{last_rsi:.2f}")

# ---------------- SCREENER ----------------

elif page == "Screener":
    st.title("🧮 Screener")

    symbol = st.text_input("Enter NSE Symbol", "INFY.NS")
    df = load_stock(symbol)

    if df.empty or len(df) < 20:
        st.error("Insufficient data for screener.")
    else:
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        trend = "Bullish" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "Bearish"
        recent_high = safe_float(df["Close"].iloc[-20:].max())
        last_close = safe_float(df["Close"].iloc[-1])

        breakout = "YES" if recent_high and last_close and last_close >= recent_high else "NO"

        col1, col2 = st.columns(2)
        col1.metric("Trend", trend)
        col2.metric("20‑Day Breakout", breakout)

# ---------------- COMMODITIES ----------------

elif page == "Commodities":
    st.title("🌐 Commodities (Best‑Effort)")

    commodities = {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil": "CL=F"
    }

    for name, ticker in commodities.items():
        df = load_stock(ticker, years=1)
        if df.empty:
            st.warning(f"{name}: data unavailable")
        else:
            price = safe_float(df["Close"].iloc[-1])
            if price:
                st.metric(name, f"{price:.2f}")

# ---------------- OPTIONS ----------------

elif page == "Options":
    st.title("📊 Index Options")

    idx = st.selectbox("Index", ["NIFTY", "BANKNIFTY"])

    if NSE is None:
        st.warning("Options API not available on this environment.")
    else:
        try:
            nse = NSE()
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={idx}"
            data = nse.endpoint_tester(url).json()
            records = data.get("records", {})

            spot = records.get("underlyingValue")
            expiries = records.get("expiryDates", [])

            if not expiries:
                st.warning("No option data returned.")
            else:
                st.metric("Spot", spot)
                st.write("Nearest Expiry:", expiries[0])
        except Exception:
            st.warning("Options data blocked by NSE.")

# ---------------- FORECAST ----------------

elif page == "Forecast":
    st.title("🤖 Next‑Day Forecast")

    symbol = st.text_input("Enter NSE Symbol", "RELIANCE.NS")
    df = load_stock(symbol)

    if df.empty or len(df) < 5:
        st.error("Not enough data for forecast.")
    else:
        y = df["Close"].astype(float).values
        X = np.arange(len(y)).reshape(-1, 1)

        try:
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict([[len(y)]])[0]
            st.metric("Predicted Next Close", f"{pred:.2f}")
        except Exception:
            st.error("Forecast model failed safely.")
