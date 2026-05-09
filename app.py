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

# --- 1. SETTINGS ---
st.set_page_config(page_title="BTC Multi-Expiry Scanner", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://api.india.delta.exchange/v2"
IST = ZoneInfo("Asia/Kolkata")
# Specific expiries from your screenshot
ALL_EXPIRIES = ["100526", "110526", "120526", "150526", "220526", "290526"]

# --- 2. DATA FUNCTIONS ---
def get_active_expiries():
    today = datetime.now(IST).date()
    return [c for c in ALL_EXPIRIES if datetime.strptime(c, "%d%m%y").date() >= today]

def fetch_data(exp_code, spot, tickers):
    try:
        # Filter for ATM Strike
        opts = [float(re.search(r'-(\d+)-', t['symbol']).group(1)) 
                for t in tickers if exp_code in t['symbol'] and 'BTC' in t['symbol']]
        if not opts: return None
        atm = min(set(opts), key=lambda x: abs(x - spot))
        
        # Get MARK Price Candles
        start = int(time.time()) - 12 * 3600
        def _get_candles(sym):
            url = f"{BASE_URL}/history/candles?symbol=MARK:{sym}&resolution=30m&start={start}"
            return requests.get(url, verify=False).json().get('result', [])

        df_c = pd.DataFrame(_get_candles(f"C-BTC-{int(atm)}-{exp_code}"))
        df_p = pd.DataFrame(_get_candles(f"P-BTC-{int(atm)}-{exp_code}"))
        
        if df_c.empty or df_p.empty: return None
        
        df = pd.merge(df_c[['time', 'close']], df_p[['time', 'close']], on='time', suffixes=('_c', '_p'))
        df['premium'] = df['close_c'].astype(float) + df['close_p'].astype(float)
        df['ema5'] = df['premium'].ewm(span=5, adjust=False).mean()
        df["datetime"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST)
        return {"df": df, "atm": atm}
    except: return None

# --- 3. UI LAYOUT ---
st.title("🚀 BTC Multi-Expiry Straddle Scanner")
st.caption(f"Last Update: {datetime.now(IST).strftime('%H:%M:%S')} IST")

# Fetch Tickers & Spot
res = requests.get(f"{BASE_URL}/tickers", verify=False).json()
tickers = res.get('result', [])
spot = float(next(t for t in tickers if t['symbol'] == "BTCUSD")['mark_price'])

st.metric("BTC Spot", f"${spot:,.2f}")

# Display Expiries in a Grid
active = get_active_expiries()
cols = st.columns(2)

for i, exp in enumerate(active):
    data = fetch_data(exp, spot, tickers)
    with cols[i % 2]:
        if data:
            df = data['df']
            curr_p = df['premium'].iloc[-1]
            curr_e = df['ema5'].iloc[-1]
            
            # Header with Score-like logic
            status_color = "green" if curr_p > curr_e else "red"
            st.markdown(f"### Expiry: {exp} (ATM: {int(data['atm'])})")
            st.markdown(f":{status_color}[Current Premium: {curr_p:.2f} | EMA5: {curr_e:.2f}]")
            
            # Chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_facecolor("#0f1419")
            fig.patch.set_facecolor("#0b0c10")
            ax.plot(df['datetime'], df['premium'], color="#ff9800", label="Premium", marker='o', markersize=4)
            ax.plot(df['datetime'], df['ema5'], color="#2ecc71", label="5-EMA", linewidth=2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=IST))
            plt.xticks(rotation=30, color="gray")
            plt.yticks(color="gray")
            ax.grid(alpha=0.2)
            st.pyplot(fig)
        else:
            st.error(f"No data for expiry {exp}")

# --- 4. AUTO-REFRESH ---
# This tells the browser to refresh the page every 5 minutes
time.sleep(300)
st.rerun()