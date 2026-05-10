import streamlit as st
import os, requests, pandas as pd, numpy as np, time, json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib3
from dotenv import load_dotenv

# --- INITIALIZATION ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()
st.set_page_config(page_title="Premium Straddle Scalper v7.16", layout="wide")

# API & Constants
BASE_URL = "https://api.india.delta.exchange/v2"
IST = ZoneInfo("Asia/Kolkata")
EMA_SPAN = 5
CANDLE_SECONDS = 30 * 60
ALL_EXPIRIES_DDMMYY = ["100526", "110526", "120526", "150526", "220526", "290526"]

COLORS = {
    "premium": "#ff9800", "ema": "#2ecc71", "atm_chg": "#9b59b6",
    "bg": "#0b0c10", "surface": "#0f1419", "score_bg": "#161d27",
    "card_bdr": "#252e3b", "case_a": "#27ae60", "wait": "#e74c3c",
    "stale": "#7f8c8d", "no_spike": "#e74c3c"
}

# --- FUNCTION DEFINITIONS (Must be defined before the loop) ---

def get_active_expiries():
    """Identifies expiries that have not yet passed."""
    today = datetime.now(IST).date()
    return [c for c in ALL_EXPIRIES_DDMMYY if datetime.strptime(c, "%d%m%y").date() >= today]

def fetch_spot_and_tickers():
    """Fetches the current BTC price and all available market tickers."""
    try:
        r = requests.get(f"{BASE_URL}/tickers", timeout=12, verify=False).json()
        tics = r.get("result", [])
        spot = float(next(t for t in tics if t["symbol"] == "BTCUSD")["mark_price"])
        return spot, tics
    except: return 0.0, []

def _candles(s, st_ts, en_ts):
    """Internal helper to fetch OHLC candle data."""
    try:
        r = requests.get(f"{BASE_URL}/history/candles", 
                         params={"symbol": s, "resolution": "30m", "start": st_ts, "end": en_ts}, 
                         verify=False).json()
        return r.get("result", [])
    except: return []

def align_and_dedupe(raw, col):
    """Standardizes time buckets and removes duplicate timestamps."""
    if not raw: return pd.DataFrame(columns=["time", col])
    df = pd.DataFrame(raw)[["time", "close"]].rename(columns={"close": col})
    df["time"] = (df["time"].astype(int)//CANDLE_SECONDS)*CANDLE_SECONDS
    return df.drop_duplicates("time")

def compute_30d_rv():
    """Calculates 30-day Realized Volatility for the BTC spot price."""
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
    """Compiles premium and EMA data for a specific expiry date."""
    now_ts = int(time.time()); start = now_ts - 12 * 3600
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
    df["atm_changed"] = df["atm"].ne(df["atm"].shift())
    
    # Live IV extraction
    atm_now = s_arr[np.abs(s_arr - spot).argmin()]
    c_s, p_s = f"C-BTC-{int(atm_now)}-{exp_code}", f"P-BTC-{int(atm_now)}-{exp_code}"
    ivs = [float(t.get("mark_iv", 0)) for t in tickers if t["symbol"] in (c_s, p_s) and t.get("mark_iv")]
    live_iv = (sum(ivs)/len(ivs))*100 if ivs else 20.0
    
    return {"df": df, "iv": live_iv, "atm": atm_now}

def draw_full_panel(exp_code, data, rv_30d):
    """Renders the Production v7.16 visual panel with charts and score cards."""
    df = data["df"]
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.1)
    
    # Main Chart
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(COLORS["surface"])
    ax.plot(df["datetime"], df["premium"], color=COLORS["premium"], marker="o", markersize=3, label="Premium")
    ax.plot(df["datetime"], df["ema5"], color=COLORS["ema"], linewidth=2, label="EMA-5")
    
    # Annotate Strikes
    y_min, y_max = df["premium"].min() * 0.95, df["premium"].max() * 1.05
    for _, row in df[df["atm_changed"]].iterrows():
        ax.axvline(x=row["datetime"], color=COLORS["atm_chg"], linestyle="--", alpha=0.4)
        ax.text(row["datetime"], y_max, f" {int(row['atm'])}", color=COLORS["atm_chg"], rotation=90, fontsize=8, va='top')

    # Price Tags
    cp, ce = df["premium"].iloc[-1], df["ema5"].iloc[-1]
    ax.yaxis.tick_right()
    ax.annotate(f"{cp:.1f}", xy=(1, cp), xycoords=("axes fraction", "data"), xytext=(15, 0), 
                textcoords="offset points", color="white", weight="bold", 
                bbox=dict(facecolor=COLORS["premium"], edgecolor="none"))
    ax.annotate(f"{ce:.1f}", xy=(1, ce), xycoords=("axes fraction", "data"), xytext=(15, -15), 
                textcoords="offset points", color="black", weight="bold", 
                bbox=dict(facecolor=COLORS["ema"], edgecolor="none"))

    ax.set_title(f"BTC {exp_code} | Terminal View", loc="left", color="white", weight="bold")
    ax.grid(color="#2b323b", alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=IST))
    
    # Score Cards
    ax_card = fig.add_subplot(gs[1])
    ax_card.set_facecolor(COLORS["bg"])
    ax_card.axis("off")
    
    cards = [
        {"label": "Premium dir", "val": "SCANNING", "sub": "v7.16", "clr": COLORS["premium"]},
        {"label": "IV/RV spread", "val": f"{data['iv'] - rv_30d:+.1f}", "sub": "SPREAD", "clr": COLORS["ema"]},
        {"label": "ATM Strike", "val": f"{int(data['atm'])}", "sub": "CURRENT", "clr": "white"},
        {"label": "Status", "val": "LIVE", "sub": "CONNECTED", "clr": COLORS["case_a"]}
    ]
    
    for i, c in enumerate(cards):
        x = 0.05 + (i * 0.24)
        ax_card.add_patch(plt.Rectangle((x, 0.1), 0.22, 0.8, transform=ax_card.transAxes, 
                             facecolor=COLORS["score_bg"], edgecolor=COLORS["card_bdr"], lw=1))
        ax_card.text(x+0.11, 0.75, c["label"], ha='center', color="#999", fontsize=8, transform=ax_card.transAxes)
        ax_card.text(x+0.11, 0.45, c["val"], ha='center', color=c["clr"], fontsize=11, weight='bold', transform=ax_card.transAxes)
        ax_card.text(x+0.11, 0.20, c["sub"], ha='center', color="#666", fontsize=7, transform=ax_card.transAxes)

    return fig

# --- MAIN DASHBOARD LOOP ---

# 1. Custom CSS for background and metrics
st.markdown("<style>.block-container {padding-top: 1rem; background-color: #0b0c10;}</style>", unsafe_allow_html=True)

st.title("🦅 Premium Straddle Scalper")

# 2. Sidebar Controls
with st.sidebar:
    st.header("Settings")
    refresh_rate = st.slider("Refresh Interval (sec)", 10, 300, 60)
    st.divider()
    st.info("Scanner v7.16 Production Edition")

# 3. Dynamic Placeholder
placeholder = st.empty()

while True:
    # Logic is executed inside the loop, referencing functions defined above
    spot, tickers = fetch_spot_and_tickers()
    rv_30d = compute_30d_rv()
    expiries = get_active_expiries()
    
    with placeholder.container():
        # Global Metrics Row
        h1, h2, h3 = st.columns(3)
        h1.metric("BTC SPOT", f"${spot:,.2f}")
        h2.metric("30D RV", f"{rv_30d}%")
        h3.metric("TIME (IST)", datetime.now(IST).strftime("%H:%M:%S"))
        
        st.divider()

        # Grid for Expiry Panels
        for i in range(0, len(expiries), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(expiries):
                    exp = expiries[i+j]
                    data = fetch_expiry_data(exp, spot, tickers)
                    if data:
                        with cols[j]:
                            fig = draw_full_panel(exp, data, rv_30d)
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
    
    time.sleep(refresh_rate)
