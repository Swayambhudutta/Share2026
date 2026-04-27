streamlit as st

st.set_page_config(
    page_title="Predict Stocks - India",
    page_icon="📈",
    layout="wide"
)

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.title("Indian Stock Dashboard")

symbol = st.text_input("NSE Symbol", "RELIANCE.NS")

@st.cache_data
def load(symbol):
    end = datetime.today()
    start = end - timedelta(days=365)
    df = yf.download(symbol, start=start, end=end, progress=False)
    return df.dropna()

df = load(symbol)

if df.empty:
    st.error("No data")
else:
    last = float(df["Close"].iloc[-1])
    st.metric("Last Close", f"{last:.2f}")

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict([[len(df)]])[0]
    st.metric("Next Close (Model)", f"{pred:.2f}")
