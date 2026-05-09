import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import urllib3

# --- 1. CONFIG FROM V4 ---
st.set_page_config(page_title="BTC Multi-Expiry Scanner", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://api.india.delta.exchange/v2"
IST = ZoneInfo("Asia/Kolkata")
EMA_SPAN = 5
CANDLE_SECONDS = 30 * 60
MAX_CANDLES = 48

# Exact list from your v4 script
ALL_EXPIRIES_DDMMYY = ["100526", "110526", "120526", "150526", "220526", "290526"]

def active_expiries():
    today = datetime.now(IST).date()
    return [code for code in ALL_EXPIRIES_DDMMYY if datetime.strptime(code, "%d%m%y").date() >= today]

# --- 2. DATA ENGINE (MATCHES V4 LOGIC) ---
def fetch_expiry_data(exp_code, spot, tickers):
    try:
        # Match specific BTC options using the exact regex from your v4 file
        opts = []
        for t in tickers:
            m = re.search(r'^(C|P)-BTC-(\d+)-(\d{6})$', t['symbol'].upper())
            if m and m.group(3) == exp_code:
                opts.append({"type": m.group(1), "strike": float(m.group(2))})

        if not opts: return None

        strikes = sorted(set(o["strike"] for o in opts))
        atm = min(strikes, key=lambda x: abs(x - spot))

        now_ts = int(time.time())
        start, end = now_ts - 12 * 3600, now_ts

        def _get_candles(sym):
            url = f"{BASE_URL}/history/candles?symbol=MARK:{sym}&resolution=30m&start={start}&end={end}"
            return requests.get(url, verify=False, timeout=12).json().get("result", [])

        # Fetch Call and Put candles
        c_raw = _get_candles(f"C-BTC-{int(atm)}-{exp_code}")
        p_raw = _get_candles(f"P-BTC-{int(atm)}-{exp_code}")

        if not c_raw or not p_raw: return None

        # Process dataframes
        df_c = pd.DataFrame(c_raw)[['time', 'close']].rename(columns={'close': 'c'})
        df_p = pd.DataFrame(p_raw)[['time', 'close']].rename(columns={'close': 'p'})
        
        df = pd.merge(df_c, df_p, on="time")
        df["premium"] = df["c"].astype(float) + df["p"].astype(float)
        df["ema5"] = df["premium"].ewm(span=EMA_SPAN, adjust=False).mean()
        df["datetime"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST)
        
        return {"df": df, "atm": atm}
    except: return None

# --- 3. UI LAYOUT ---
st.title("🚀 BTC Multi-Expiry Straddle Scanner")

try:
    res = requests.get(f"{BASE_URL}/tickers", timeout=12, verify=False).json()
    tickers = res.get("result", [])
    spot = float(next(t for t in tickers if t["symbol"] == "BTCUSD")["mark_price"])
    st.metric("BTC Spot Price", f"${spot:,.2f}")
except:
    st.error("Could not connect to Delta Exchange.")
    st.stop()

# Build the grid
active = active_expiries()
cols = st.columns(2)

for i, exp in enumerate(active):
    with cols[i % 2]:
        data = fetch_expiry_data(exp, spot, tickers)
        if data:
            df = data["df"]
            curr_p, curr_e = df["premium"].iloc[-1], df["ema5"].iloc[-1]
            color = "green" if curr_p > curr_e else "red"
            
            st.subheader(f"📅 {exp}")
            st.markdown(f"**ATM:** {int(data['atm'])} | **Prem:** :{color}[{curr_p:.2f}] | **EMA:** {curr_e:.2f}")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_facecolor("#0f1419"); fig.patch.set_facecolor("#0b0c10")
            ax.plot(df["datetime"], df["premium"], color="#ff9800", label="Premium", marker='o', markersize=3)
            ax.plot(df["datetime"], df["ema5"], color="#2ecc71", label="5-EMA", linewidth=2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=IST))
            plt.xticks(rotation=30, color="#888"); plt.yticks(color="#888")
            st.pyplot(fig)
        else:
            st.info(f"⏳ {exp}: Fetching data or waiting for market history...")

# Auto-refresh
time.sleep(300)
st.rerun()
