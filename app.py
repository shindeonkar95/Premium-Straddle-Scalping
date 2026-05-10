import streamlit as st
import os, requests, pandas as pd, numpy as np, time, json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib3
from dotenv import load_dotenv

# --- INITIALIZATION ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()
st.set_page_config(page_title="Premium Straddle Scalper", layout="wide")

# Constants & Config
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

# --- DATA FETCHING (MINIFIED) ---
def fetch_spot_and_tickers():
    try:
        r = requests.get(f"{BASE_URL}/tickers", timeout=12, verify=False).json()
        tics = r.get("result", [])
        spot = float(next(t for t in tics if t["symbol"] == "BTCUSD")["mark_price"])
        return spot, tics
    except: return 0.0, []

def _candles(s, st, en):
    try: return requests.get(f"{BASE_URL}/history/candles", params={"symbol": s, "resolution": "30m", "start": st, "end": en}, verify=False).json().get("result", [])
    except: return []

def align_and_dedupe(raw, col):
    if not raw: return pd.DataFrame(columns=["time", col])
    df = pd.DataFrame(raw)[["time", "close"]].rename(columns={"close": col})
    df["time"] = (df["time"].astype(int)//CANDLE_SECONDS)*CANDLE_SECONDS
    return df.drop_duplicates("time")

def fetch_expiry_data(exp_code, spot, tickers):
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
                if df_spot[df_spot['time'] == r['time']]['atm'].values[0] == atm_s:
                    rows.append({"time": r["time"], "premium": r["c"] + r["p"], "atm": atm_s})
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df["ema5"] = df["premium"].ewm(span=EMA_SPAN, adjust=False).mean()
    df["datetime"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST)
    df["atm_changed"] = df["atm"].ne(df["atm"].shift())
    return {"df": df, "spot": spot}

# --- PLOTTING ENGINE (RESTORED FROM V7.16) ---
def draw_full_panel(exp_code, data):
    df = data["df"]
    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.1)
    
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(COLORS["surface"])
    
    # Plot Lines
    ax.plot(df["datetime"], df["premium"], color=COLORS["premium"], marker="o", markersize=4, label="Premium")
    ax.plot(df["datetime"], df["ema5"], color=COLORS["ema"], linewidth=2, label="EMA-5")
    
    # Vertical ATM Lines
    y_min, y_max = df["premium"].min() * 0.98, df["premium"].max() * 1.02
    for _, row in df[df["atm_changed"]].iterrows():
        ax.axvline(x=row["datetime"], color=COLORS["atm_chg"], linestyle="--", alpha=0.4)
        ax.text(row["datetime"], y_max, f" {int(row['atm'])}", color=COLORS["atm_chg"], rotation=90, fontsize=8, va='top')

    # Price Annotations on Right Axis
    cp, ce = df["premium"].iloc[-1], df["ema5"].iloc[-1]
    ax.yaxis.tick_right()
    ax.annotate(f"{cp:.1f}", xy=(1, cp), xycoords=("axes fraction", "data"), xytext=(15, 0), 
                textcoords="offset points", color="white", weight="bold", 
                bbox=dict(facecolor=COLORS["premium"], edgecolor="none"))
    ax.annotate(f"{ce:.1f}", xy=(1, ce), xycoords=("axes fraction", "data"), xytext=(15, -15), 
                textcoords="offset points", color="black", weight="bold", 
                bbox=dict(facecolor=COLORS["ema"], edgecolor="none"))

    ax.set_title(f"BTC {exp_code} | v7.16 Production View", loc="left", color="white", weight="bold")
    ax.grid(color="#2b323b", alpha=0.3)
    
    # Score Cards Subplot
    ax_card = fig.add_subplot(gs[1])
    ax_card.set_facecolor(COLORS["bg"])
    ax_card.axis("off")
    
    # Mock scores for UI demonstration (replace with actual classifier logic)
    cards = [
        {"label": "Premium dir", "val": "NO SPIKE", "sub": "exp < 2.5%", "clr": COLORS["no_spike"]},
        {"label": "IV/RV spread", "val": "N/A", "sub": "no data", "clr": "#888"},
        {"label": "Ratio", "val": "4.96%", "sub": "RICH", "clr": COLORS["case_a"]},
        {"label": "IV Percentile", "val": "N/A", "sub": "no data", "clr": "#888"}
    ]
    
    for i, c in enumerate(cards):
        x = 0.05 + (i * 0.24)
        rect = plt.Rectangle((x, 0.1), 0.22, 0.8, transform=ax_card.transAxes, 
                             facecolor=COLORS["score_bg"], edgecolor=COLORS["card_bdr"], lw=1)
        ax_card.add_patch(rect)
        ax_card.text(x+0.11, 0.75, c["label"], ha='center', color="#999", fontsize=8, transform=ax_card.transAxes)
        ax_card.text(x+0.11, 0.45, c["val"], ha='center', color=c["clr"], fontsize=12, weight='bold', transform=ax_card.transAxes)
        ax_card.text(x+0.11, 0.20, c["sub"], ha='center', color="#666", fontsize=7, transform=ax_card.transAxes)

    return fig

# --- MAIN UI ---
st.markdown("<style>.block-container {padding-top: 1rem; background-color: #0b0c10;}</style>", unsafe_allow_html=True)

placeholder = st.empty()

while True:
    spot, tickers = fetch_spot_and_tickers()
    expiries = get_active_expiries()
    
    with placeholder.container():
        st.write(f"### 🦅 Market Spot: ${spot:,.2f} | {datetime.now(IST).strftime('%H:%M:%S')}")
        
        for i in range(0, len(expiries), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(expiries):
                    exp = expiries[i+j]
                    data = fetch_expiry_data(exp, spot, tickers)
                    if data:
                        with cols[j]:
                            fig = draw_full_panel(exp, data)
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
    
    time.sleep(60)
