import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib3

# --- 1. SETTINGS ---
st.set_page_config(page_title="BTC Multi-Expiry Scanner", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://api.india.delta.exchange/v2"
IST = ZoneInfo("Asia/Kolkata")
ALL_EXPIRIES = ["100526", "110526", "120526", "150526", "220526", "290526"]

# --- 2. DATA FUNCTIONS ---
def get_active_expiries():
    today = datetime.now(IST).date()
    return [c for c in ALL_EXPIRIES if datetime.strptime(c, "%d%m%y").date() >= today]

def fetch_data(exp_code, spot, tickers):
    try:
        # Improved Ticker Search Logic
        opts = []
        for t in tickers:
            sym = t['symbol'].upper()
            # Look for BTC, the specific expiry, and ensure it's an option (Starts with C- or P-)
            if "BTC" in sym and exp_code in sym and (sym.startswith("C-") or sym.startswith("P-")):
                try:
                    # Extract strike price using a simpler split
                    strike = float(sym.split('-')[2])
                    opts.append(strike)
                except:
                    continue
        
        if not opts: return None
        
        # Find ATM strike
        atm = min(set(opts), key=lambda x: abs(x - spot))
        
        # Get MARK Price Candles
        start = int(time.time()) - 12 * 3600
        def _get_candles(s_type):
            symbol = f"{s_type}-BTC-{int(atm)}-{exp_code}"
            url = f"{BASE_URL}/history/candles?symbol=MARK:{symbol}&resolution=30m&start={start}"
            r = requests.get(url, verify=False, timeout=10).json()
            return r.get('result', [])

        df_c = pd.DataFrame(_get_candles("C"))
        df_p = pd.DataFrame(_get_candles("P"))
        
        if df_c.empty or df_p.empty: return None
        
        df = pd.merge(df_c[['time', 'close']], df_p[['time', 'close']], on='time', suffixes=('_c', '_p'))
        df['premium'] = df['close_c'].astype(float) + df['close_p'].astype(float)
        df['ema5'] = df['premium'].ewm(span=5, adjust=False).mean()
        df["datetime"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST)
        return {"df": df, "atm": atm}
    except Exception as e:
        return None

# --- 3. UI LAYOUT ---
st.title("🚀 BTC Multi-Expiry Straddle Scanner")
st.caption(f"Last Sync: {datetime.now(IST).strftime('%H:%M:%S')} IST")

# Fetch Tickers
try:
    res = requests.get(f"{BASE_URL}/tickers", verify=False, timeout=10).json()
    tickers = res.get('result', [])
    spot = float(next(t for t in tickers if t['symbol'] == "BTCUSD")['mark_price'])
    st.metric("BTC Spot Price", f"${spot:,.2f}")
except Exception as e:
    st.error("Failed to connect to Delta Exchange API.")
    st.stop()

# Grid Layout
active = get_active_expiries()
cols = st.columns(2)

for i, exp in enumerate(active):
    with cols[i % 2]:
        data = fetch_data(exp, spot, tickers)
        if data:
            df = data['df']
            curr_p = df['premium'].iloc[-1]
            curr_e = df['ema5'].iloc[-1]
            
            st.subheader(f"📅 Expiry: {exp}")
            color = "green" if curr_p > curr_e else "red"
            st.markdown(f"**ATM Strike:** {int(data['atm'])} | **Premium:** :{color}[{curr_p:.2f}] | **EMA5:** {curr_e:.2f}")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_facecolor("#0f1419")
            fig.patch.set_facecolor("#0b0c10")
            ax.plot(df['datetime'], df['premium'], color="#ff9800", label="Premium", marker='o', markersize=4)
            ax.plot(df['datetime'], df['ema5'], color="#2ecc71", label="5-EMA", linewidth=2.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=IST))
            plt.xticks(rotation=30, color="#888")
            plt.yticks(color="#888")
            ax.grid(alpha=0.1)
            st.pyplot(fig)
        else:
            st.warning(f"⚠️ No data found for {exp}. Checking ATM strikes...")

# --- 4. AUTO-REFRESH ---
time.sleep(300)
st.rerun()
