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

    # EMA

    df["ema5"] = df["premium"].ewm(
        span=EMA_SPAN,
        adjust=False
    ).mean()

    # Datetime

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
            figsize=(14, 5),
            facecolor="#0b1220"
        )

        ax.set_facecolor("#0b1220")

        # ==========================================
        # PREMIUM LINE
        # ==========================================

        ax.plot(
            df["datetime"],
            df["premium"],
            color="#f59e0b",
            linewidth=2,
            marker="o",
            markersize=6,
            label="Premium"
        )

        # ==========================================
        # EMA LINE
        # ==========================================

        ax.plot(
            df["datetime"],
            df["ema5"],
            color="#00e396",
            linewidth=2,
            label="EMA5"
        )

        # ==========================================
        # CURRENT VALUES
        # ==========================================

        latest_premium = df["premium"].iloc[-1]
        latest_ema = df["ema5"].iloc[-1]

        # ==========================================
        # PREMIUM PRICE LINE
        # ==========================================

        ax.axhline(
            latest_premium,
            color="#f59e0b",
            linestyle=":",
            linewidth=1
        )

        # ==========================================
        # DYNAMIC LABEL POSITION
        # ==========================================

        difference = abs(
            latest_premium - latest_ema
        )

        premium_y_offset = 0
        ema_y_offset = -18

        if difference < 15:

            premium_y_offset = 14
            ema_y_offset = -14

        # ==========================================
        # PREMIUM LABEL
        # ==========================================

        ax.annotate(
            f"{latest_premium:.2f}",
            xy=(1, latest_premium),
            xycoords=("axes fraction", "data"),
            xytext=(10, premium_y_offset),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                facecolor="#f59e0b",
                edgecolor="none",
                pad=3
            ),
            clip_on=False
        )

        # ==========================================
        # EMA LABEL
        # ==========================================

        ax.annotate(
            f"{latest_ema:.2f}",
            xy=(1, latest_ema),
            xycoords=("axes fraction", "data"),
            xytext=(10, ema_y_offset),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                facecolor="#00e396",
                edgecolor="none",
                pad=3
            ),
            clip_on=False
        )

        # ==========================================
        # GRID
        # ==========================================

        ax.grid(
            color="#1f2937",
            linestyle="-",
            linewidth=0.5,
            alpha=0.7
        )

        # ==========================================
        # AXIS COLORS
        # ==========================================

        ax.tick_params(
            colors="#9ca3af",
            labelsize=9
        )

        for spine in ax.spines.values():

            spine.set_color("#1f2937")

        # ==========================================
        # TITLE
        # ==========================================

        ax.set_title(
            f"BTC {expiry_label(exp)}",
            color="white",
            fontsize=14,
            fontweight="bold",
            loc="left"
        )

        # ==========================================
        # LEGEND
        # ==========================================

        legend = ax.legend(
            facecolor="#111827",
            edgecolor="#1f2937"
        )

        for text in legend.get_texts():

            text.set_color("white")

        # ==========================================
        # RIGHT SIDE Y AXIS
        # ==========================================

        ax.yaxis.tick_right()

        ax.yaxis.set_label_position("right")

        # ==========================================
        # EXTRA SPACE FOR LABELS
        # ==========================================

        plt.subplots_adjust(
            right=0.88
        )

        # ==========================================
        # SHOW CHART
        # ==========================================

        st.pyplot(fig)
