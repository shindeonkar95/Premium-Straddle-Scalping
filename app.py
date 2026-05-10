import streamlit as st
import os, requests, pandas as pd, numpy as np, time, json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib3
from dotenv import load_dotenv

# --- CONFIGURATION & SETUP ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

st.set_page_config(page_title="Premium Straddle Scalper", layout="wide")

# API & Constants
BASE_URL = "https://api.india.delta.exchange/v2"
IST = ZoneInfo("Asia/Kolkata")
EMA_SPAN = 5
LOOKBACK_HOURS = 12
CANDLE_SECONDS = 30 * 60
ALL_EXPIRIES_DDMMYY = ["100526", "110526", "120526", "150526", "220526", "290526"]

# Colors
COLORS = {
    "premium": "#ff9800", "ema": "#2ecc71", "atm_chg": "#9b59b6",
    "bg": "#0b0c10", "surface": "#161d27", "case_a": "#27ae60",
    "rolling": "#f39c12", "wait": "#e74c3c", "no_spike": "#7f8c8d"
}

# --- SESSION STATE FOR HISTORY ---
if 'iv_history' not in st.session_state:
    st.session_state.iv_history = {}

# --- CORE FUNCTIONS ---

def get_active_expiries():
    today = datetime.now(IST).date()
    return [c for c in ALL_EXPIRIES_DDMMYY if datetime.strptime(c, "%d%m%y").date() >= today]

def fetch_spot_and_tickers():
    try:
        r = requests.get(f"{BASE_URL}/tickers", timeout=12, verify=False).json()
        tics = r.get("result", [])
        spot = float(next(t for t in tics if t["symbol"] == "BTCUSD")["mark_price"])
        return spot, tics
    except: return 0.0, []

def _candles(s, st_ts, en_ts):
    try:
        r = requests.get(f"{BASE_URL}/history/candles", 
                         params={"symbol": s, "resolution": "30m", "start": st_ts, "end": en_ts}, 
                         verify=False).json()
        return r.get("result", [])
    except: return []

def align_and_dedupe(raw, col):
    if not raw: return pd.DataFrame(columns=["time", col])
    df = pd.DataFrame(raw)[["time", "close"]].rename(columns={"close": col})
    df["time"] = (df["time"].astype(int)//CANDLE_SECONDS)*CANDLE_SECONDS
    return df.drop_duplicates("time")

def compute_30d_rv():
    try:
        now = int(time.time())
        start = now - (30 * 24 * 3600)
        r = requests.get(f"{BASE_URL}/history/candles", 
                         params={"symbol": "BTCUSD", "resolution": "1d", "start": start, "end": now}, 
                         verify=False).json()
        df = pd.DataFrame(r.get("result", []))
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        return round(rets.std() * np.sqrt(365) * 100, 2)
    except: return 20.0

def fetch_expiry_data(exp_code, spot, tickers):
    now_ts = int(time.time())
    start = now_ts - LOOKBACK_HOURS * 3600
    df_spot = align_and_dedupe(_candles("BTCUSD", start, now_ts), "btc_price")
    if df_spot.empty: return None
    
    strikes = sorted({float(t['symbol'].split('-')[2]) for t in tickers if t['symbol'].endswith(exp_code)})
    if not strikes: return None
    s_arr = np.array(strikes)
    df_spot["atm"] = s_arr[np.abs(s_arr[:, None] - df_spot["btc_price"].to_numpy()[None, :]).argmin(axis=0)]
    
    rows = []
    for atm_s in df_spot["atm"].unique():
        df_c = align_and_dedupe(_candles(f"MARK:C-BTC-{int(atm_s)}-{exp_code}", start, now_ts), "c")
        df_p = align_and_dedupe(_candles(f"MARK:P-BTC-{int(atm_s)}-{exp_code}", start, now_ts), "p")
        if not df_c.empty and not df_p.empty:
            m = pd.merge(df_c, df_p, on="time")
            for _, r in m.iterrows():
                rows.append({"time": r["time"], "premium": r["c"] + r["p"], "atm": atm_s})
    
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df["ema5"] = df["premium"].ewm(span=EMA_SPAN, adjust=False).mean()
    df["datetime"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST)
    
    # Live IV extraction
    atm_now = s_arr[np.abs(s_arr - spot).argmin()]
    c_s, p_s = f"C-BTC-{int(atm_now)}-{exp_code}", f"P-BTC-{int(atm_now)}-{exp_code}"
    ivs = [float(t.get("mark_iv", 0)) for t in tickers if t["symbol"] in (c_s, p_s) and t.get("mark_iv")]
    live_iv = (sum(ivs)/len(ivs))*100 if ivs else 20.0
    
    return {"df": df, "iv": live_iv, "atm": atm_now}

# --- STREAMLIT UI COMPONENTS ---

def create_plot(exp_code, data):
    df = data["df"]
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('dark_background')
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["surface"])
    
    ax.plot(df["datetime"], df["premium"], color=COLORS["premium"], marker="o", markersize=3, label="Straddle")
    ax.plot(df["datetime"], df["ema5"], color=COLORS["ema"], linewidth=1.5, label="EMA-5")
    
    ax.set_title(f"Expiry: {exp_code}", color="white")
    ax.grid(color="#2b323b", linestyle="--", alpha=0.3)
    return fig

# --- MAIN LOOP ---

st.title("🦅 Premium Straddle Scalper")
placeholder = st.empty()

# Sidebar Settings
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh Interval (sec)", 10, 300, 60)

while True:
    spot, tickers = fetch_spot_and_tickers()
    rv_30d = compute_30d_rv()
    expiries = get_active_expiries()
    
    with placeholder.container():
        # Top Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("BTC Spot", f"${spot:,.2f}")
        col2.metric("30D RV", f"{rv_30d}%")
        col3.metric("Last Update", datetime.now(IST).strftime("%H:%M:%S"))
        
        # Grid for Expiries
        for i in range(0, len(expiries), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(expiries):
                    exp = expiries[i+j]
                    data = fetch_expiry_data(exp, spot, tickers)
                    if data:
                        with cols[j]:
                            st.subheader(f"BTC-{exp}")
                            # Score Cards (simplified for Streamlit)
                            iv_val = data["iv"]
                            spread = iv_val - rv_30d
                            
                            m1, m2 = st.columns(2)
                            m1.metric("IV", f"{iv_val:.1f}%", f"{spread:+.1f} spread")
                            m2.metric("ATM Strike", int(data["atm"]))
                            
                            fig = create_plot(exp, data)
                            st.pyplot(fig)
                            plt.close(fig)
    
    time.sleep(refresh_rate)
