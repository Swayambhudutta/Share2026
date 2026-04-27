import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
# Lazy import for heavy modules (NSE API wrapper)
try:
    from pnsea import NSE
except ImportError:
    NSE = None

# Page config
st.set_page_config(page_title="Predict Stocks - India", page_icon="📈", layout="wide")
st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)  # Hide default Streamlit footer

# --- Cached Data Functions ---

@st.cache_data(ttl=300)
def search_indian_stocks(query: str):
    """
    Search Indian stock symbols by company name using an open no-key API.
    Returns a list of {'company_name': ..., 'symbol': ...} results.
    """
    url = "http://65.0.104.9/search"  # Open API for stock search (NSE/BSE)
    try:
        resp = requests.get(url, params={"q": query}, timeout=5)
        data = resp.json()
        return data.get("results", [])
    except Exception:
        # Fallback to Yahoo Finance search if primary fails
        try:
            alt_url = "https://query2.finance.yahoo.com/v1/finance/search"
            resp = requests.get(alt_url, params={"q": query, "region": "IN"}, timeout=5)
            quotes = resp.json().get("quotes", [])
            results = []
            for item in quotes:
                if "symbol" in item and "shortname" in item:
                    results.append({"company_name": item["shortname"], "symbol": item["symbol"]})
            return results
        except Exception:
            return []

@st.cache_data(ttl=300)
def load_stock_data(symbol: str, years: int = 5):
    """
    Load historical OHLC data for a stock (default last 5 years, daily) via Yahoo Finance.
    """
    try:
        end = datetime.today()
        start = end - timedelta(days=365 * years)
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df is None:
            return pd.DataFrame()
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_price_and_change(symbol: str):
    """
    Get current price and daily % change for a given symbol via Yahoo Finance.
    Returns (price, change_percent) or None on failure.
    """
    try:
        df = yf.download(symbol, period="2d", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        last_close = float(df["Close"].iloc[-1])
        if len(df) > 1:
            prev_close = float(df["Close"].iloc[-2])
            change_pct = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else None
        else:
            change_pct = None
        return last_close, (round(change_pct, 2) if change_pct is not None else None)
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_option_chain(index_symbol: str):
    """
    Fetch option chain data for an index (NIFTY or BANKNIFTY) from NSE's public API.
    Uses pnsea library for stealth if available. Returns JSON data or None.
    """
    if NSE is None:
        return None
    try:
        nse = NSE()  # init NSE session with stealth
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={index_symbol}"
        resp = nse.endpoint_tester(url)
        return resp.json()
    except Exception:
        return None

# --- UI: Sidebar and Pages ---

st.sidebar.title("Predict Stocks (India)")
page = st.sidebar.radio("Navigate", ["Home", "Search Stock", "Technical Analysis", "Screener", "Commodities", "Options", "Forecast"])
st.sidebar.caption("**Disclaimer:** Data is for informational purposes only and not intended for trading or investment advice.")

# Home Page
if page == "Home":
    st.title("Indian Stock Market Dashboard")
    st.write("Welcome to the **Indian Stock Market Screener & Prediction** app. This app provides live market data and basic analysis for Indian stocks and indices.")
    st.markdown("""
- **Stock Search:** Look up companies by name or ticker and view their latest price.
- **Technical Analysis:** See historical price charts with technical indicators (EMA, RSI).
- **Screener:** Get simple trend and breakout signals.
- **Commodities:** Track key commodity prices (Gold, Silver, Crude Oil).
- **Options:** View a snapshot of NIFTY/BANKNIFTY option chain data.
- **Forecast:** A baseline next-day price prediction (for demonstration).
""")
    st.info("Data is sourced from free public APIs (Yahoo Finance, NSE). No login or API key is required. Note that data may be delayed or unavailable at times due to source limitations.")

# Search Stock Page
elif page == "Search Stock":
    st.title("🔎 Search Stocks")
    query = st.text_input("Type a company name or stock symbol:")
    if query:
        results = search_indian_stocks(query)
        if results:
            options = {f"{res['company_name']} ({res['symbol']})": res["symbol"] for res in results if res.get('symbol') and res.get('company_name')}
            choice = st.selectbox("Select a stock from results:", list(options.keys()))
            if choice:
                symbol = options[choice]
                st.write(f"**Selected:** {choice}")
                # Price metric
                info = get_price_and_change(symbol)
                if info:
                    price, change = info
                    delta_str = f"{change:.2f}%" if change is not None else None
                    st.metric(f"{symbol} Price", f"{price:.2f}", delta_str)
                else:
                    st.write("Current price data not available.")
                # 6-month closing price chart
                try:
                    df_line = yf.download(symbol, period="6mo", interval="1d", progress=False)
                except Exception:
                    df_line = pd.DataFrame()
                if df_line is not None and not df_line.empty:
                    st.line_chart(df_line["Close"], height=250)
                else:
                    st.write("Historical data unavailable for chart.")
        else:
            st.warning("No matching stocks found. Please try a different query.")

# Technical Analysis Page
elif page == "Technical Analysis":
    st.title("📈 Technical Analysis")
    symbol = st.text_input("Enter stock symbol (NSE):", value="RELIANCE.NS")
    if symbol:
        df = load_stock_data(symbol)
        if df.empty:
            st.error("Unable to load data. Please check the symbol.")
        else:
            # Compute indicators
            df["EMA20"] = df["Close"].ewm(span=20).mean()
            df["EMA50"] = df["Close"].ewm(span=50).mean()
            # RSI calculation
            delta = df["Close"].diff()
            up = delta.clip(lower=0); down = -delta.clip(upper=0)
            avg_gain = up.rolling(14).mean(); avg_loss = down.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))
            # Candlestick chart with EMA
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"),
                go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA 20"),
                go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA 50")
            ])
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            # Latest values
            latest_close = df["Close"].iloc[-1]
            latest_rsi = df["RSI"].iloc[-1]
            st.write(f"**Last Close:** {latest_close:.2f}")
            st.write(f"**Latest RSI:** {latest_rsi:.2f}")

# Screener Page
elif page == "Screener":
    st.title("🧮 Screener Signals")
    symbol = st.text_input("Enter stock symbol:", value="INFY.NS")
    if symbol:
        df = load_stock_data(symbol)
        if df.empty or len(df) < 20:
            st.error("Insufficient data for screening signals.")
        else:
            df["EMA20"] = df["Close"].ewm(span=20).mean()
            df["EMA50"] = df["Close"].ewm(span=50).mean()
            trend = "Bullish" if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] else "Bearish"
            recent_high = df["Close"].iloc[-20:].max()
            breakout = "YES" if df["Close"].iloc[-1] >= recent_high else "NO"
            col1, col2 = st.columns(2)
            col1.metric("Trend (EMA20 vs EMA50)", trend)
            col2.metric("20-day Breakout", breakout)

# Commodities Page
elif page == "Commodities":
    st.title("🌐 Commodity Prices")
    commodities = {
        "Gold (USD)": "GC=F",      # Gold Futures price (USD)
        "Silver (USD)": "SI=F",    # Silver Futures price (USD)
        "Crude Oil (USD)": "CL=F"  # WTI Crude Oil Futures (USD)
    }
    for name, ticker in commodities.items():
        info = get_price_and_change(ticker)
        if info:
            price, change = info
            delta_str = f"{change:.2f}%" if change is not None else None
            st.metric(name, f"{price:.2f}", delta_str)
        else:
            st.write(f"{name}: data not available.")

# Options Page
elif page == "Options":
    st.title("📊 NIFTY/BANKNIFTY Option Chain")
    index = st.selectbox("Select Index:", ["NIFTY", "BANKNIFTY"])
    if NSE is None:
        st.error("Option chain data unavailable (NSE API library not installed).")
    else:
        data = fetch_option_chain(index)
        if data is None:
            st.error("Could not retrieve option chain data. Please try later.")
        else:
            records = data.get("records", {})
            underlying = records.get("underlyingValue")
            expiry_dates = records.get("expiryDates", [])
            if underlying:
                st.write(f"**{index} Spot Price:** {underlying}")
            if expiry_dates:
                st.write(f"**Nearest Expiry:** {expiry_dates[0]}")
            # Filter option data for nearest expiry
            chain_data = records.get("data", [])
            if expiry_dates:
                chain_data = [entry for entry in chain_data if entry.get("expiryDate") == expiry_dates[0]]
            if not chain_data:
                st.write("No option data available.")
            else:
                # Determine ATM strike
                atm_strike = None
                if underlying:
                    try:
                        atm_strike = min(chain_data, key=lambda x: abs(x.get("strikePrice", 0) - underlying)).get("strikePrice")
                    except Exception:
                        atm_strike = None
                # Choose strikes: ATM and one step above/below
                strikes = sorted({item.get("strikePrice") for item in chain_data})
                display_strikes = []
                if atm_strike and atm_strike in strikes:
                    i = strikes.index(atm_strike)
                    display_strikes = strikes[max(0, i-1): i+2]  # ATM, one below and one above
                else:
                    display_strikes = strikes[:3]
                # Prepare table data
                rows = []
                for strike in display_strikes:
                    entry = next((item for item in chain_data if item.get("strikePrice") == strike), {})
                    ce = entry.get("CE", {}); pe = entry.get("PE", {})
                    rows.append({
                        "Strike": strike,
                        "Call LTP": ce.get("lastPrice", "-"),
                        "Call OI": ce.get("openInterest", "-"),
                        "Put LTP": pe.get("lastPrice", "-"),
                        "Put OI": pe.get("openInterest", "-")
                    })
                st.dataframe(pd.DataFrame(rows).set_index("Strike"))
                st.caption("Option chain snapshot (nearest expiry, around ATM strikes).")

# Forecast Page
elif page == "Forecast":
    st.title("🤖 Next-Day Price Forecast")
    symbol = st.text_input("Enter stock symbol for prediction:", value="RELIANCE.NS")
    if symbol:
        df = load_stock_data(symbol)
        if df.empty or len(df) < 2:
            st.error("Not enough data to make a prediction.")
        else:
            df_clean = df.dropna(subset=["Close"])
            if len(df_clean) < 2:
                st.error("Not enough valid data points for forecasting.")
            else:
                X = np.arange(len(df_clean)).reshape(-1, 1)
                y = df_clean["Close"].values
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    next_pred = model.predict([[len(df_clean)]])[0]
                    last_price = df_clean["Close"].iloc[-1]
                    st.write(f"**Last Close:** {last_price:.2f}")
                    st.write(f"**Predicted Next Close:** {next_pred:.2f}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
