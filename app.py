"""
=============================================================================
  BTC MULTI-EXPIRY STRADDLE SCANNER  —  Streamlit Edition
  Tracks: 10-May, 11-May, 12-May, 15-May, 22-May, 29-May (rolls automatically)
  Layout: 2-column grid, one panel per active expiry
  Run   : streamlit run combined_scanner_v4_streamlit.py
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)

import streamlit as st
import requests
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ══════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="BTC Multi-Expiry Straddle Scanner",
    layout="wide",
    page_icon="₿",
)

BASE_URL       = "https://api.india.delta.exchange/v2"
DELTA_CDN_BASE = "https://cdn.india.deltaex.org/v2"
IST            = ZoneInfo("Asia/Kolkata")

REFRESH_SECONDS     = 5 * 60          # 5 min
CANDLE_SECONDS      = 30 * 60
MAX_CANDLES         = 48              # 24 h of 30-min candles
LOOKBACK_HOURS      = 12

EMA_SPAN       = 5
EMA_BUFFER     = 1.0
MIN_DOTS_ABOVE = 3
GO_THRESHOLD   = 7.0
WARN_THRESHOLD = 5.0

TELEGRAM_TOKEN   = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

ALL_EXPIRIES_DDMMYY = [
    "100526",   # 10-May-2026
    "110526",   # 11-May-2026
    "120526",   # 12-May-2026
    "150526",   # 15-May-2026
    "220526",   # 22-May-2026
    "290526",   # 29-May-2026
]

COLORS = {
    "premium": "#ff9800",
    "ema":     "#2ecc71",
    "grid":    "#2b323b",
    "bg":      "#0b0c10",
    "surface": "#0f1419",
    "go":      "#27ae60",
    "watch":   "#f39c12",
    "wait":    "#e74c3c",
}

# ══════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════
def active_expiries() -> list[str]:
    """Return expiry codes that have NOT yet expired (expiry date >= today IST)."""
    today = datetime.now(IST).date()
    return [
        code for code in ALL_EXPIRIES_DDMMYY
        if datetime.strptime(code, "%d%m%y").date() >= today
    ]

def expiry_label(code: str) -> str:
    """'100526' → '10 May'"""
    d = datetime.strptime(code, "%d%m%y")
    return f"{d.day} {d.strftime('%b')}"

def send_telegram(msg: str):
    try:
        requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            params={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=8,
        )
    except Exception:
        pass

# ══════════════════════════════════════════
# 3. DATA FETCH
# ══════════════════════════════════════════
def align_and_dedupe(raw: list, col: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["time", col])
    df = pd.DataFrame(raw)[["time", "close"]].copy()
    df["time"] = (df["time"].astype(int) // CANDLE_SECONDS) * CANDLE_SECONDS
    return (
        df.rename(columns={"close": col})
        .drop_duplicates("time")
        .sort_values("time")
        .tail(MAX_CANDLES)
        .reset_index(drop=True)
    )

def _candles(symbol: str, start: int, end: int) -> list:
    try:
        r = requests.get(
            f"{BASE_URL}/history/candles",
            params={"symbol": symbol, "resolution": "30m", "start": start, "end": end},
            verify=False, timeout=12,
        )
        return r.json().get("result", [])
    except Exception:
        return []

def fetch_spot_and_tickers() -> tuple[float, list]:
    try:
        res     = requests.get(f"{BASE_URL}/tickers", timeout=12, verify=False).json()
        tickers = res.get("result", [])
        spot    = float(next(t for t in tickers if t["symbol"] == "BTCUSD")["mark_price"])
        return spot, tickers
    except Exception:
        return 0.0, []

def fetch_expiry_data(exp_code: str, spot: float, tickers: list) -> dict | None:
    """Fetch 12-h premium history for a specific BTC expiry."""
    try:
        opts = []
        for t in tickers:
            m = re.search(r'^(C|P)-BTC-(\d+)-(\d{6})$', t['symbol'].upper())
            if m and m.group(3) == exp_code:
                opts.append({"type": m.group(1), "strike": float(m.group(2))})

        if not opts:
            return None

        strikes = sorted(set(o["strike"] for o in opts))
        atm     = min(strikes, key=lambda x: abs(x - spot))

        now_ts = int(time.time())
        start  = now_ts - LOOKBACK_HOURS * 3600
        end    = now_ts

        ce_sym = f"C-BTC-{int(atm)}-{exp_code}"
        pe_sym = f"P-BTC-{int(atm)}-{exp_code}"

        df_c = align_and_dedupe(_candles(f"MARK:{ce_sym}", start, end), "c")
        df_p = align_and_dedupe(_candles(f"MARK:{pe_sym}", start, end), "p")

        if df_c.empty or df_p.empty:
            return None

        df = pd.merge(df_c, df_p, on="time")
        df["premium"] = df["c"].astype(float) + df["p"].astype(float)
        df["ema5"]    = df["premium"].ewm(span=EMA_SPAN, adjust=False).mean()

        # BTC spot history for RV
        df_spot = align_and_dedupe(_candles("BTCUSD", start, end), "btc_price")
        df      = pd.merge(df, df_spot, on="time", how="left")

        df["datetime"] = (
            pd.to_datetime(df["time"], unit="s")
            .dt.tz_localize("UTC")
            .dt.tz_convert(IST)
        )

        # IV from CDN
        iv = None
        try:
            iv_res = requests.get(
                f"{DELTA_CDN_BASE}/options_iv",
                params={
                    "asset_symbol": "BTC", "maturity": "daily",
                    "resolution": "30m", "start_time": start, "end_time": end,
                },
                verify=False, timeout=10,
            ).json()
            ivs = [x for x in iv_res.get("result", {}).get("atm_iv", []) if x is not None]
            if ivs:
                iv = float(ivs[-1])
        except Exception:
            pass

        df["iv"] = iv

        return {
            "df": df, "atm": atm,
            "ce_sym": ce_sym, "pe_sym": pe_sym,
            "iv": iv, "spot": spot,
        }
    except Exception:
        return None

# ══════════════════════════════════════════
# 4. SCORING
# ══════════════════════════════════════════
def score_premium_direction(df: pd.DataFrame):
    if len(df) < 2:
        return 0.0, 0, "WAIT"
    recent     = df.tail(16)
    dots_above = int((recent["premium"] > (recent["ema5"] + EMA_BUFFER)).sum())
    if dots_above < MIN_DOTS_ABOVE:
        return 0.0, dots_above, f"{dots_above} dots"
    ratio = min((dots_above - MIN_DOTS_ABOVE) / (16 - MIN_DOTS_ABOVE), 1.0)
    return round(ratio * 3.5, 2), dots_above, f"{dots_above}/16 above"

def compute_rv(df: pd.DataFrame):
    prices = df["btc_price"].dropna()
    if len(prices) < 2:
        return None
    return round(np.log(prices / prices.shift(1)).dropna().std() * np.sqrt(48 * 365) * 100, 2)

def score_iv_rv(iv, rv):
    if iv is None or rv is None:
        return 1.75, "N/A"
    s = iv - rv
    if s < -10: return 3.5, f"CHEAP Δ{s:.0f}"
    if s <   0: return 2.5, f"CHEAP Δ{s:.0f}"
    return 1.0, f"EXP Δ{s:.0f}"

def score_ratio(premium, spot):
    if not premium or not spot:
        return 0.75, "N/A"
    r = (premium / spot) * 100
    return 0.75, f"{r:.3f}%"

def score_iv_pct(iv):
    return (0.75, f"{iv:.1f}%") if iv else (0.0, "N/A")

def total_score(s1, s2, s3, s4):
    t = round(s1 + s2 + s3 + s4, 2)
    if t >= GO_THRESHOLD:   return t, "GO",    COLORS["go"]
    if t >= WARN_THRESHOLD: return t, "WATCH", COLORS["watch"]
    return t, "WAIT", COLORS["wait"]

# ══════════════════════════════════════════
# 5. PER-PANEL CHART (returns fig, score info)
# ══════════════════════════════════════════
def build_panel_figure(exp_code: str, data: dict):
    """
    Build and return a matplotlib figure for one expiry.
    Caller is responsible for plt.close(fig) after st.pyplot().
    """
    df  = data["df"]
    atm = data["atm"]
    iv  = data["iv"]
    rv  = compute_rv(df)
    now = datetime.now(IST)

    cp = df["premium"].iloc[-1]
    ce = df["ema5"].iloc[-1]

    # ── scores ──────────────────────────────
    s1, dots, l1 = score_premium_direction(df)
    s2, l2       = score_iv_rv(iv, rv)
    s3, l3       = score_ratio(cp, data["spot"])
    s4, l4       = score_iv_pct(iv)
    tot, verd, vcol = total_score(s1, s2, s3, s4)

    # ── figure ──────────────────────────────
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(COLORS["surface"])
    fig.patch.set_facecolor(COLORS["bg"])

    ax.plot(df["datetime"], df["premium"],
            color=COLORS["premium"], marker="o", markersize=5,
            linewidth=1.8, label="Premium", zorder=4)
    ax.plot(df["datetime"], df["ema5"],
            color=COLORS["ema"], linewidth=2.4,
            label=f"EMA-{EMA_SPAN}", zorder=3)
    ax.axhline(y=cp, color=COLORS["premium"], linestyle=":", alpha=0.45, linewidth=1)

    # Y-axis right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    y_vals = pd.concat([df["premium"], df["ema5"]]).dropna()
    if not y_vals.empty:
        ylo, yhi = y_vals.min(), y_vals.max()
        pad = max((yhi - ylo) * 0.25, 10)
        ax.set_ylim(ylo - pad, yhi + pad)

    # ── label overlap fix (from test.py) ────
    ylim = ax.get_ylim()
    min_sep = (ylim[1] - ylim[0]) * 0.06
    p_label_y, e_label_y = cp, ce
    if abs(p_label_y - e_label_y) < min_sep:
        if p_label_y >= e_label_y:
            p_label_y += min_sep / 2
            e_label_y -= min_sep / 2
        else:
            p_label_y -= min_sep / 2
            e_label_y += min_sep / 2

    ax.annotate(
        f"{cp:.2f}",
        xy=(1, p_label_y), xycoords=("axes fraction", "data"),
        xytext=(8, 0), textcoords="offset points",
        ha="left", va="center", fontsize=9, weight="bold", color="white",
        bbox=dict(facecolor=COLORS["premium"], edgecolor="none", pad=3),
        annotation_clip=False,
    )
    ax.annotate(
        f"{ce:.2f}",
        xy=(1, e_label_y), xycoords=("axes fraction", "data"),
        xytext=(8, 0), textcoords="offset points",
        ha="left", va="center", fontsize=9, weight="bold", color="#0b0c10",
        bbox=dict(facecolor=COLORS["ema"], edgecolor="none", pad=3),
        annotation_clip=False,
    )

    # X-axis time
    ax.set_xlim(now - timedelta(hours=12.5), now + timedelta(minutes=45))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3, tz=IST))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=IST))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7, color="#888")
    plt.setp(ax.yaxis.get_majorticklabels(), color="#888")
    ax.grid(color=COLORS["grid"], alpha=0.35, linewidth=0.6)

    # Days-to-expiry badge
    exp_date = datetime.strptime(exp_code, "%d%m%y").date()
    dte      = (exp_date - now.date()).days
    dte_str  = f"{dte}d to exp" if dte > 0 else "EXPIRY TODAY"

    # Title + verdict badge
    ax.set_title(
        f"BTC  {expiry_label(exp_code)}   {dte_str}   ATM {int(atm):,}   {now.strftime('%H:%M:%S')}",
        loc="left", fontsize=10, color="white", pad=8, weight="bold",
    )
    ax.text(
        0.99, 1.02, f"Score {tot:.1f}  {verd}",
        transform=ax.transAxes, ha="right", va="bottom",
        color="white", weight="bold", fontsize=10,
        bbox=dict(facecolor=vcol, edgecolor="none", pad=4, alpha=0.9),
    )

    ax.legend(loc="lower left", fontsize=8, framealpha=0.25, ncol=2)

    return fig, {
        "tot": tot, "verd": verd, "vcol": vcol,
        "cp": cp, "ce": ce, "iv": iv, "rv": rv,
        "l1": l1, "l2": l2, "l3": l3, "l4": l4,
        "s1": s1, "s2": s2, "s3": s3, "s4": s4,
        "dots": dots,
    }

# ══════════════════════════════════════════
# 6. MAIN STREAMLIT APP
# ══════════════════════════════════════════

# ── Telegram alert state (persists across reruns via st.session_state) ──
if "alerted" not in st.session_state:
    st.session_state.alerted = {}

st.title("₿ BTC Multi-Expiry Straddle Scanner")
st.caption(f"Auto-refreshes every {REFRESH_SECONDS // 60} minutes · Data: Delta Exchange India · IST")

# ── Fetch spot + tickers ──────────────────────────────────────────────
with st.spinner("Fetching market data…"):
    spot, tickers = fetch_spot_and_tickers()

if spot <= 0:
    st.error("⚠️ Could not connect to Delta Exchange. Check your internet connection.")
    st.stop()

# ── Header row ───────────────────────────────────────────────────────
col_spot, col_time, col_refresh = st.columns([1, 1, 1])
with col_spot:
    st.metric("BTC Spot (Mark)", f"${spot:,.2f}")
with col_time:
    st.metric("Last Updated", datetime.now(IST).strftime("%H:%M:%S IST"))
with col_refresh:
    st.metric("Next Refresh", f"in {REFRESH_SECONDS // 60} min")

st.divider()

# ── Active expiries ───────────────────────────────────────────────────
active = active_expiries()

if not active:
    st.warning("No active BTC expiries found. All May 2026 expiries have passed.")
    st.stop()

# ── 2-column panel grid ───────────────────────────────────────────────
cols = st.columns(2)

for i, exp_code in enumerate(active):
    with cols[i % 2]:
        label = expiry_label(exp_code)

        with st.spinner(f"Loading {label}…"):
            data = fetch_expiry_data(exp_code, spot, tickers)

        if data is None or data["df"] is None or len(data["df"]) < 2:
            st.info(f"⏳ {label}: Awaiting data or no options found for this expiry.")
            continue

        # Build figure
        fig, scores = build_panel_figure(exp_code, data)
        st.pyplot(fig)
        plt.close(fig)   # prevent memory warning

        # ── Score card below chart ───────────────────────────────────
        tot   = scores["tot"]
        verd  = scores["verd"]
        vcol  = scores["vcol"]

        verdict_emoji = "🟢" if verd == "GO" else ("🟡" if verd == "WATCH" else "🔴")
        st.markdown(
            f"**{verdict_emoji} {verd}** &nbsp; Score: `{tot:.1f}` &nbsp;|&nbsp; "
            f"Direction: `{scores['l1']}` &nbsp;|&nbsp; "
            f"IV/RV: `{scores['l2']}` &nbsp;|&nbsp; "
            f"Ratio: `{scores['l3']}` &nbsp;|&nbsp; "
            f"IV: `{scores['l4']}`"
        )

        # ── Telegram alert (fire once per GO per expiry per session) ─
        alert_key = f"{exp_code}_{verd}"
        if verd == "GO" and not st.session_state.alerted.get(alert_key):
            cp = scores["cp"]
            msg = (
                f"🚨 BTC Straddle ALERT\n"
                f"Expiry : {label}\n"
                f"Score  : {tot:.1f} — {verd}\n"
                f"Premium: ${cp:.2f}\n"
                f"IV/RV  : {scores['l2']}\n"
                f"Time   : {datetime.now(IST).strftime('%H:%M:%S IST')}"
            )
            send_telegram(msg)
            st.session_state.alerted[alert_key] = True
            st.toast(f"🚨 Telegram alert sent for {label}!", icon="📲")

st.divider()
st.caption(
    f"Scores — Direction (/3.5) · IV/RV (/3.5) · Ratio (/0.75) · IV% (/0.75) · "
    f"GO ≥ {GO_THRESHOLD} · WATCH ≥ {WARN_THRESHOLD}"
)

# ══════════════════════════════════════════
# 7. AUTO-REFRESH
# ══════════════════════════════════════════
time.sleep(REFRESH_SECONDS)
st.rerun()
