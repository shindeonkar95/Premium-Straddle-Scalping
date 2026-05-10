import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# CONFIG
# =====================================================

BASE_URL = "https://api.india.delta.exchange/v2"

IST = ZoneInfo("Asia/Kolkata")

LOOKBACK_HOURS = 12
EMA_SPAN = 5

ALL_EXPIRIES_DDMMYY = [
    "100526",
    "110526",
    "120526",
    "150526",
    "220526",
    "290526"
]

# =====================================================
# STREAMLIT PAGE
# =====================================================

st.set_page_config(
    page_title="BTC Premium Scanner",
    layout="wide"
)

st.title("BTC Premium Straddle Scanner")

# =====================================================
# HELPERS
# =====================================================

def expiry_label(code):
    d = datetime.strptime(code, "%d%m%y")
    return f"{d.day} {d.strftime('%b')}"

def active_expiries():
    today = datetime.now(IST).date()

    return [
        c for c in ALL_EXPIRIES_DDMMYY
        if datetime.strptime(c, "%d%m%y").date() >= today
    ]

def fetch_spot_and_tickers():

    try:

        r = requests.get(
            f"{BASE_URL}/tickers",
            timeout=10,
            verify=False
        ).json()

        tickers = r.get("result", [])

        spot = float(
            next(
                t for t in tickers
                if t["symbol"] == "BTCUSD"
            )["mark_price"]
        )

        return spot, tickers

    except Exception as e:

        st.error(f"Ticker Error: {e}")

        return 0.0, []

def candles(symbol, start, end):

    try:

        r = requests.get(
            f"{BASE_URL}/history/candles",
            params={
                "symbol": symbol,
                "resolution": "30m",
                "start": start,
                "end": end
            },
            timeout=10,
            verify=False
        ).json()

        return r.get("result", [])

    except:

        return []

def align(raw, col):

    if not raw:
        return pd.DataFrame(columns=["time", col])

    df = pd.DataFrame(raw)[["time", "close"]]

    df = df.rename(columns={"close": col})

    df["time"] = df["time"].astype(int)

    return df.drop_duplicates("time")

# =====================================================
# FETCH EXPIRY DATA
# =====================================================

def fetch_expiry_data(exp_code, spot, tickers):

    now_ts = int(datetime.now().timestamp())

    start = now_ts - LOOKBACK_HOURS * 3600

    df_spot = align(
        candles("BTCUSD", start, now_ts),
        "btc_price"
    )

    if df_spot.empty:
        return None

    strikes = sorted({

        float(t['symbol'].split('-')[2])

        for t in tickers

        if t['symbol'].endswith(exp_code)

    })

    if not strikes:
        return None

    s_arr = np.array(strikes)

    df_spot["atm"] = s_arr[
        np.abs(
            s_arr[:, None]
            -
            df_spot["btc_price"].to_numpy()[None, :]
        ).argmin(axis=0)
    ]

    atm_map = df_spot.set_index("time")["atm"].to_dict()

    rows = []

    for atm_s in df_spot["atm"].unique():

        call_symbol = f"MARK:C-BTC-{int(atm_s)}-{exp_code}"
        put_symbol = f"MARK:P-BTC-{int(atm_s)}-{exp_code}"

        df_c = align(
            candles(call_symbol, start, now_ts),
            "c"
        )

        df_p = align(
            candles(put_symbol, start, now_ts),
            "p"
        )

        if not df_c.empty and not df_p.empty:

            merged = pd.merge(df_c, df_p, on="time")

            for _, row in merged.iterrows():

                if atm_map.get(row["time"]) == atm_s:

                    rows.append({

                        "time": row["time"],
                        "premium": row["c"] + row["p"],
                        "atm": atm_s

                    })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    df = df.sort_values("time")

    df["ema5"] = df["premium"].ewm(
        span=EMA_SPAN,
        adjust=False
    ).mean()

    df["datetime"] = pd.to_datetime(
        df["time"],
        unit="s"
    )

    return df

# =====================================================
# MAIN
# =====================================================

spot, tickers = fetch_spot_and_tickers()

if spot > 0:

    st.metric(
        "BTC Spot Price",
        f"{spot:,.0f}"
    )

    expiries = active_expiries()

    for exp in expiries:

        st.subheader(f"Expiry: {expiry_label(exp)}")

        df = fetch_expiry_data(
            exp,
            spot,
            tickers
        )

        if df is None:

            st.warning("No data found")

            continue

        # ==========================================
        # CREATE FIGURE
        # ==========================================

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        # ==========================================
        # PLOT LINES
        # ==========================================

        ax.plot(
            df["datetime"],
            df["premium"],
            label="Premium",
            linewidth=2
        )

        ax.plot(
            df["datetime"],
            df["ema5"],
            label="EMA5",
            linewidth=2
        )

        # ==========================================
        # LATEST VALUES
        # ==========================================

        latest_premium = df["premium"].iloc[-1]
        latest_ema = df["ema5"].iloc[-1]

        # ==========================================
        # PREVENT LABEL OVERLAP
        # ==========================================

        y_range = (
            max(
                df["premium"].max(),
                df["ema5"].max()
            )
            -
            min(
                df["premium"].min(),
                df["ema5"].min()
            )
        )

        offset = y_range * 0.03

        premium_offset = 0
        ema_offset = 0

        if abs(latest_premium - latest_ema) < offset:

            premium_offset = 12
            ema_offset = -12

        # ==========================================
        # PREMIUM LABEL
        # ==========================================

        ax.annotate(
            f"{latest_premium:.1f}",
            xy=(1, latest_premium),
            xycoords=("axes fraction", "data"),
            xytext=(8, premium_offset),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(
                facecolor="blue",
                edgecolor="none",
                pad=3
            ),
            clip_on=False
        )

        # ==========================================
        # EMA LABEL
        # ==========================================

        ax.annotate(
            f"{latest_ema:.1f}",
            xy=(1, latest_ema),
            xycoords=("axes fraction", "data"),
            xytext=(8, ema_offset),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(
                facecolor="orange",
                edgecolor="none",
                pad=3
            ),
            clip_on=False
        )

        # ==========================================
        # STYLING
        # ==========================================

        ax.grid(True)

        ax.legend()

        ax.set_title(
            f"BTC {expiry_label(exp)}",
            fontsize=14,
            fontweight="bold"
        )

        plt.subplots_adjust(right=0.88)

        st.pyplot(fig)
