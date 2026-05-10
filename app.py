"""
=============================================================================
  Premium Straddle Scalping — v7.16 (Standalone Production Build)
=============================================================================
"""
import warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)

print("Starting Premium Straddle Scalping (v7.16 Production Edition)...")
import matplotlib
matplotlib.use('TkAgg')

import os, requests, pandas as pd, numpy as np, re, time, json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from rich.console import Console
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# ══════════════════════════════════════════
# 1. API & PERSISTENCE CONFIG
# ══════════════════════════════════════════
BASE_URL       = "https://api.india.delta.exchange/v2"
DELTA_CDN_BASE = "https://cdn.india.deltaex.org/v2"
IST            = ZoneInfo("Asia/Kolkata")
HISTORY_FILE   = "iv_history.json"
console        = Console()

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

REFRESH_INTERVAL_MS = 5 * 60 * 1000   
CANDLE_SECONDS      = 30 * 60
MAX_CANDLES         = 48               
LOOKBACK_HOURS      = 12
EMA_SPAN            = 5

ALL_EXPIRIES_DDMMYY = ["100526", "110526", "120526", "150526", "220526", "290526"]

# --- TUNING THRESHOLDS ---
T_STALE_DROP_PCT = 25.0       
T_SPIKE_MIN_EXPANSION = 2.5   
T_CONSEC_UP_SPIKING  = 3      
T_CONSEC_DOWN_CASE_A = 3      
T_CONSEC_DOWN_ROLLING= 2      
T_EMA_SLOPE_CASE_A  = -0.2    
T_EMA_SLOPE_ROLLING = 0.0     
T_IV_RV_RICH  = 5.0           
T_RATIO_RICH  = 2.2           
T_RATIO_AVG   = 1.2           
MAX_IV_HISTORY = 10000        
SAVE_COOLDOWN_SEC = 3600      

# ══════════════════════════════════════════
# 2. UI COLORS & STATE HELPERS
# ══════════════════════════════════════════
COLORS = {
    "premium":  "#ff9800", "ema": "#2ecc71", "atm_chg": "#9b59b6",
    "grid": "#2b323b", "bg": "#0b0c10", "surface": "#0f1419",
    "go": "#27ae60", "watch": "#f39c12", "wait": "#e74c3c",
    "score_bg": "#161d27", "card_bdr": "#252e3b",
    "spiking": "#e74c3c", "cooling": "#e67e22", "rolling": "#f39c12",
    "case_a": "#27ae60", "no_spike": "#e74c3c", "stale": "#7f8c8d",
}

_exp_state: dict[str, dict] = {}
_fig_state: dict = {"expiries": [], "fig": None, "axes": {}}
_last_save_time = 0

def load_iv_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f: return json.load(f)
        except Exception as e: print(f"History Load Error: {e}")
    return {}

def save_iv_history(history_dict, force=False):
    global _last_save_time
    now = time.time()
    if force or (now - _last_save_time > SAVE_COOLDOWN_SEC):
        try:
            with open(HISTORY_FILE, "w") as f: json.dump(history_dict, f)
            _last_save_time = now
            print(f"[{datetime.now().strftime('%H:%M:%S')}] IV History saved to disk.")
        except Exception as e: print(f"History Save Error: {e}")

_persistent_history = load_iv_history()

def ensure_state(exp: str):
    if exp not in _exp_state:
        hist = _persistent_history.get(exp, [])
        _exp_state[exp] = dict(iv_history=hist, last_score=0.0, last_verdict="WAIT")

def active_expiries():
    today = datetime.now(IST).date()
    return [c for c in ALL_EXPIRIES_DDMMYY if datetime.strptime(c, "%d%m%y").date() >= today]

def expiry_label(code: str):
    d = datetime.strptime(code, "%d%m%y")
    return f"{d.day} {d.strftime('%b')}"

# ══════════════════════════════════════════
# 3. SCORING & CLASSIFIER
# ══════════════════════════════════════════
def score_iv_rv(iv, rv):
    if rv is None: rv = 20.0
    if iv is None: return 1.75, "N/A", "no data", "#888"
    spread = iv - rv
    rel = "IV>RV" if spread >= 0 else "IV<RV"
    delta = f"Δ {spread:+.1f}"
    if spread > T_IV_RV_RICH: return 3.5, rel, f"RICH {delta}", COLORS["case_a"]
    if spread >= 0: return 2.5, rel, f"FAIR {delta}", COLORS["rolling"]
    return 1.0, rel, f"RISK {delta}", COLORS["wait"]

def score_ratio(premium, spot):
    if not premium or not spot: return 0.75, "N/A", "no data", "#888"
    r = (premium / spot) * 100
    if r > T_RATIO_RICH: return 1.50, f"{r:.2f}%", "RICH", COLORS["case_a"]
    if r > T_RATIO_AVG: return 1.00, f"{r:.2f}%", "AVG", COLORS["watch"]
    return 0.50, f"{r:.2f}%", "CHEAP", COLORS["wait"]

def score_iv_percentile(current_iv, exp_code):
    history = _exp_state[exp_code]["iv_history"]
    if current_iv is None: return 0.75, "N/A", "no data", "#888"
    if len(history) < 5:
        res = (0.75, "N/A", "collecting...", "#888")
    else:
        inclusive_count = sum(1 for h_iv in history if h_iv <= current_iv)
        ivp = (inclusive_count / len(history)) * 100
        if ivp > 80: res = (1.50, f"{ivp:.0f}%", "HIGH", "#27ae60")
        elif ivp > 40: res = (1.00, f"{ivp:.0f}%", "NORMAL", "#f39c12")
        else: res = (0.50, f"{ivp:.0f}%", "LOW", "#e74c3c")
    history.append(current_iv)
    if len(history) > MAX_IV_HISTORY: history.pop(0)
    _persistent_history[exp_code] = history
    save_iv_history(_persistent_history)
    return res

def classify_premium_state(df):
    if len(df) < 5: return dict(state="NO SPIKE", sublabel="...", score=0.5, color=COLORS["no_spike"])
    prem = df["premium"].to_numpy(); ema = df["ema5"].to_numpy()
    cp = float(prem[-1]); ce = float(ema[-1])
    win = prem[-16:] if len(prem) >= 16 else prem
    pv = float(np.max(win)); pe = (pv - float(np.mean(win))) / (float(np.mean(win)) + 1e-9) * 100
    pfp = (pv - cp) / (pv + 1e-9) * 100
    es = ((ce - ema[-4]) / ema[-4] * 100) if len(ema) >= 4 else 0.0
    diffs = np.diff(prem[-9:]); cd, cu = 0, 0
    for d in reversed(diffs):
        if d < 0:
            if cu > 0: break
            cd += 1
        elif d > 0:
            if cd > 0: break
            cu += 1
        else: break
    if pe < T_SPIKE_MIN_EXPANSION: return dict(state="NO SPIKE", sublabel=f"exp < {T_SPIKE_MIN_EXPANSION}%", score=0.5, color=COLORS["no_spike"])
    if pfp > T_STALE_DROP_PCT: return dict(state="STALE", sublabel=f"−{pfp:.0f}% drop", score=0.0, color=COLORS["stale"])
    if cu >= T_CONSEC_UP_SPIKING and es > 0.0 and cp > ce: return dict(state="SPIKING", sublabel=f"{cu}↑ consec · Prem>EMA", score=0.5, color=COLORS["spiking"])
    if cd >= T_CONSEC_DOWN_CASE_A and es < T_EMA_SLOPE_CASE_A and cp < ce: return dict(state="CASE A", sublabel=f"{cd}↓ drops · Prem<EMA", score=3.5, color=COLORS["case_a"])
    if cd >= T_CONSEC_DOWN_ROLLING and es < T_EMA_SLOPE_ROLLING and cp < ce: return dict(state="ROLLING", sublabel=f"{cd}↓ drops · Prem<EMA", score=2.5, color=COLORS["rolling"])
    return dict(state="COOLING", sublabel="watch", score=1.0, color=COLORS["cooling"])

# ══════════════════════════════════════════
# 4. DATA FETCH & PERSISTENT RV
# ══════════════════════════════════════════
def compute_30d_rv():
    try:
        now = int(time.time()); start = now - (30 * 24 * 3600)
        r = requests.get(f"{BASE_URL}/history/candles", params={"symbol": "BTCUSD", "resolution": "1d", "start": start, "end": now}, verify=False, timeout=12).json()
        df = pd.DataFrame(r.get("result", []))
        if len(df) < 2: return 20.0
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        return round(rets.std() * np.sqrt(365) * 100, 2)
    except Exception as e: print(f"RV Error: {e}"); return 20.0

def extract_live_atm_iv(tickers, exp_code, atm_s):
    c_s, p_s = f"C-BTC-{int(atm_s)}-{exp_code}", f"P-BTC-{int(atm_s)}-{exp_code}"
    ivs = []
    for t in tickers:
        if t["symbol"] in (c_s, p_s):
            v = t.get("mark_iv") or t.get("implied_volatility") or t.get("iv")
            if v is not None:
                try: ivs.append(float(v))
                except Exception as e: print(f"IV Conversion Error: {e}"); continue
    if not ivs: return None
    avg = sum(ivs) / len(ivs)
    return avg * 100 if avg < 2.0 else avg

def fetch_expiry_data(exp_code, spot, tickers):
    try:
        now_ts = int(time.time()); start = now_ts - LOOKBACK_HOURS * 3600
        df_spot = align_and_dedupe(_candles("BTCUSD", start, now_ts), "btc_price")
        if df_spot.empty: return None
        strikes = sorted({float(t['symbol'].split('-')[2]) for t in tickers if t['symbol'].endswith(exp_code)})
        if not strikes: return None
        s_arr = np.array(strikes); df_spot["atm"] = s_arr[np.abs(s_arr[:, None] - df_spot["btc_price"].to_numpy()[None, :]).argmin(axis=0)]
        atm_map = df_spot.set_index("time")["atm"].to_dict()
        spot_map = df_spot.set_index("time")["btc_price"].to_dict()
        rows = []
        for atm_s in df_spot["atm"].unique():
            df_c = align_and_dedupe(_candles(f"MARK:C-BTC-{int(atm_s)}-{exp_code}", start, now_ts), "c")
            df_p = align_and_dedupe(_candles(f"MARK:P-BTC-{int(atm_s)}-{exp_code}", start, now_ts), "p")
            if not df_c.empty and not df_p.empty:
                m = pd.merge(df_c, df_p, on="time")
                for _, r in m.iterrows():
                    if atm_map.get(r["time"]) == atm_s:
                        rows.append({"time": r["time"], "premium": r["c"] + r["p"], "btc_price": spot_map[r["time"]], "atm": atm_s})
        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        df["premium"] = df["premium"].rolling(window=2).mean().fillna(df["premium"]) 
        df["ema5"] = df["premium"].ewm(span=EMA_SPAN, adjust=False).mean()
        df["datetime"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("UTC").dt.tz_convert(IST)
        df["atm_changed"] = df["atm"].ne(df["atm"].shift()); df.iloc[0, df.columns.get_loc("atm_changed")] = False
        live_iv = extract_live_atm_iv(tickers, exp_code, s_arr[np.abs(s_arr - spot).argmin()])
        return {"df": df, "atm": s_arr[np.abs(s_arr - spot).argmin()], "spot": spot, "iv": live_iv}
    except Exception as e: print(f"Fetch Error: {e}"); return None

# ══════════════════════════════════════════
# 5. UI FRAMEWORK
# ══════════════════════════════════════════
_CARD_NUMS = ["①", "②", "③", "④"]

def build_figure(expiries):
    n = len(expiries); cols = 2; rows = (n + 1) // 2
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 6.2 * rows))
    fig.patch.set_facecolor(COLORS["bg"])
    outer = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.2)
    axes = {}
    for i, exp in enumerate(expiries):
        r, c = divmod(i, cols)
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, c], height_ratios=[5, 1], hspace=0.08)
        ax_m = fig.add_subplot(inner[0]); ax_c = fig.add_subplot(inner[1])
        ax_m.set_facecolor(COLORS["surface"]); ax_c.axis("off")
        axes[exp] = {"main": ax_m, "cards": ax_c}
    _fig_state.update(expiries=list(expiries), fig=fig, axes=axes)
    return fig

def draw_panel(exp_code, data, btc_rv):
    axd = _fig_state["axes"][exp_code]; df = data["df"]; iv = data["iv"]; ax = axd["main"]; ax.clear()
    cp, ce = df["premium"].iloc[-1], df["ema5"].iloc[-1]
    st = classify_premium_state(df); s2, v2, sub2, c2 = score_iv_rv(iv, btc_rv)
    s3, v3, sub3, c3 = score_ratio(cp, data["spot"]); s4, v4, sub4, c4 = score_iv_percentile(iv, exp_code)
    tot = st["score"] + s2 + s3 + s4
    y_vals = pd.concat([df["premium"], df["ema5"]]).dropna()
    ylo, yhi = y_vals.min() - max((y_vals.max() - y_vals.min()) * 0.20, 5), y_vals.max() + max((y_vals.max() - y_vals.min()) * 0.20, 5)
    ax.set_ylim(ylo, yhi)
    for _, sr in df[df["atm_changed"]].iterrows():
        ax.axvline(x=sr["datetime"], color=COLORS["atm_chg"], linestyle="--", alpha=0.5)
        ax.text(sr["datetime"], yhi, f" {int(sr['atm']):,}", color=COLORS["atm_chg"], fontsize=7, rotation=90, va="top")
    ax.plot(df["datetime"], df["premium"], color=COLORS["premium"], marker="o", markersize=4, label="Premium")
    ax.plot(df["datetime"], df["ema5"], color=COLORS["ema"], linewidth=2, label="EMA-5")
    ax.axhline(y=cp, color=COLORS["premium"], linestyle=":", alpha=0.4); ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    y_range = yhi - ylo; p_off, e_off = (12, -12) if cp >= ce else (-12, 12) if abs(cp-ce) < y_range*0.12 else (0,0)
    ax.annotate(f"{cp:.1f}", xy=(1, cp), xycoords=("axes fraction", "data"), xytext=(6, p_off), textcoords="offset points", ha="left", va="center", fontsize=9, weight="bold", color="white", bbox=dict(facecolor=COLORS["premium"], edgecolor="none", pad=3), annotation_clip=False)
    ax.annotate(f"{ce:.1f}", xy=(1, ce), xycoords=("axes fraction", "data"), xytext=(6, e_off), textcoords="offset points", ha="left", va="center", fontsize=9, weight="bold", color="#0b0c10", bbox=dict(facecolor=COLORS["ema"], edgecolor="none", pad=3), annotation_clip=False)
    ax.set_title(f"BTC {expiry_label(exp_code)} | Score: {tot:.1f}", loc="left", color="white", weight="bold")
    axd["cards"].clear(); _draw_score_cards(axd["cards"], [dict(label="Premium dir", value=st["state"], sublabel=st["sublabel"], value_color=st["color"]), dict(label="IV/RV spread", value=v2, sublabel=sub2, value_color=c2), dict(label="Ratio", value=v3, sublabel=sub3, value_color=c3), dict(label="IV Percentile", value=v4, sublabel=sub4, value_color=c4)])

def _draw_score_cards(ax, cards):
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(cards); pad, gap = 0.012, 0.01; total_w = 1.0 - 2*pad - (n-1)*gap; cw = total_w/n
    for i, c in enumerate(cards):
        x0 = pad + i*(cw+gap); cx = x0+cw/2
        ax.add_patch(FancyBboxPatch((x0, 0.04), cw, 0.92, boxstyle="round,pad=0.015", facecolor=COLORS["score_bg"], edgecolor=COLORS["card_bdr"], linewidth=0.8, transform=ax.transAxes, clip_on=False))
        ax.text(cx, 0.88, _CARD_NUMS[i], ha="center", va="center", fontsize=8, color="#666", transform=ax.transAxes)
        ax.text(cx, 0.72, c["label"], ha="center", va="center", fontsize=7.5, color="#999", transform=ax.transAxes)
        ax.text(cx, 0.46, c["value"], ha="center", va="center", fontsize=12, weight="bold", color=c["value_color"], transform=ax.transAxes)
        ax.text(cx, 0.18, f"{c['sublabel']}", ha="center", va="center", fontsize=7, color="#888", transform=ax.transAxes)

# ══════════════════════════════════════════
# 6. HELPERS & STARTUP
# ══════════════════════════════════════════
def _candles(s, st, en):
    try: return requests.get(f"{BASE_URL}/history/candles", params={"symbol": s, "resolution": "30m", "start": st, "end": en}, verify=False, timeout=12).json().get("result", [])
    except Exception as e: print(f"API Error ({s}): {e}"); return []

def fetch_spot_and_tickers():
    try:
        r = requests.get(f"{BASE_URL}/tickers", timeout=12, verify=False).json(); tics = r.get("result", [])
        spot = float(next(t for t in tics if t["symbol"] == "BTCUSD")["mark_price"])
        return spot, tics
    except Exception as e: print(f"Ticker Error: {e}"); return 0.0, []

def align_and_dedupe(raw, col):
    if not raw: return pd.DataFrame(columns=["time", col])
    df = pd.DataFrame(raw)[["time", "close"]].rename(columns={"close": col})
    df["time"] = (df["time"].astype(int)//CANDLE_SECONDS)*CANDLE_SECONDS
    return df.drop_duplicates("time")

def update(frame):
    exps = active_expiries(); spot, tics = fetch_spot_and_tickers()
    if spot <= 0: return
    rv_30d = compute_30d_rv()
    for exp in exps:
        ensure_state(exp); data = fetch_expiry_data(exp, spot, tics)
        if data: draw_panel(exp, data, rv_30d)

if __name__ == "__main__":
    init_exps = active_expiries()
    if init_exps:
        fig = build_figure(init_exps)
        ani = FuncAnimation(fig, update, interval=REFRESH_INTERVAL_MS, cache_frame_data=False)
        plt.show()
