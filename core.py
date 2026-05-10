"""
core.py — Pure business logic (no UI, no matplotlib, no streamlit).
All data fetching, scoring, classification, and state management lives here.
Imported by both app.py (Streamlit) and any future CLI/test harness.
"""

import os, json, time, warnings
import requests, numpy as np, pandas as pd
import urllib3
from datetime import datetime
from zoneinfo import ZoneInfo

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ══════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════
BASE_URL        = "https://api.india.delta.exchange/v2"
IST             = ZoneInfo("Asia/Kolkata")
HISTORY_FILE    = "iv_history.json"

CANDLE_SECONDS  = 30 * 60
LOOKBACK_HOURS  = 12
EMA_SPAN        = 5
MAX_EXPIRIES    = 6
MAX_IV_HISTORY  = 10000
SAVE_COOLDOWN_SEC = 3600

# Tuning thresholds
T_STALE_DROP_PCT      = 25.0
T_SPIKE_MIN_EXPANSION = 2.5
T_CONSEC_UP_SPIKING   = 3
T_CONSEC_DOWN_CASE_A  = 3
T_CONSEC_DOWN_ROLLING = 2
T_EMA_SLOPE_CASE_A    = -0.2
T_EMA_SLOPE_ROLLING   = 0.0
T_IV_RV_RICH          = 5.0
T_RATIO_RICH          = 2.2
T_RATIO_AVG           = 1.2

COLORS = {
    "premium": "#ff9800", "ema": "#2ecc71", "atm_chg": "#9b59b6",
    "bg": "#0b0c10", "surface": "#0f1419",
    "go": "#27ae60", "watch": "#f39c12", "wait": "#e74c3c",
    "score_bg": "#161d27", "card_bdr": "#252e3b",
    "spiking": "#e74c3c", "cooling": "#e67e22", "rolling": "#f39c12",
    "case_a": "#27ae60", "no_spike": "#e74c3c", "stale": "#7f8c8d",
}

# ══════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════
_last_save_time = 0

def load_iv_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"History Load Error: {e}")
    return {}

def save_iv_history(history_dict: dict, force: bool = False):
    global _last_save_time
    now = time.time()
    if force or (now - _last_save_time > SAVE_COOLDOWN_SEC):
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(history_dict, f)
            _last_save_time = now
        except Exception as e:
            print(f"History Save Error: {e}")

# ══════════════════════════════════════════
# STATE
# ══════════════════════════════════════════
_exp_state: dict[str, dict] = {}
_persistent_history: dict = load_iv_history()

def ensure_state(exp: str):
    if exp not in _exp_state:
        hist = _persistent_history.get(exp, [])
        _exp_state[exp] = dict(iv_history=hist, last_score=0.0, last_verdict="WAIT")

# ══════════════════════════════════════════
# EXPIRY HELPERS
# ══════════════════════════════════════════
def fetch_available_expiries(tickers: list) -> list[str]:
    """
    Extract all unique BTC option expiry codes from live Delta tickers.
    Symbol format: C-BTC-<strike>-<DDMMYY>  or  P-BTC-<strike>-<DDMMYY>
    Returns a sorted list of DDMMYY strings for future (>= today) expiries only.
    """
    today = datetime.now(IST).date()
    seen: set[str] = set()
    for t in tickers:
        sym = t.get("symbol", "")
        parts = sym.split("-")
        if len(parts) == 4 and parts[0] in ("C", "P") and parts[1] == "BTC":
            exp_code = parts[3]
            if len(exp_code) == 6 and exp_code.isdigit():
                try:
                    exp_date = datetime.strptime(exp_code, "%d%m%y").date()
                    if exp_date >= today:
                        seen.add(exp_code)
                except ValueError:
                    continue
    sorted_exps = sorted(seen, key=lambda c: datetime.strptime(c, "%d%m%y").date())
    return sorted_exps[:MAX_EXPIRIES]

def expiry_label(code: str) -> str:
    d = datetime.strptime(code, "%d%m%y")
    return f"{d.day} {d.strftime('%b')}"

# ══════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════
def score_iv_rv(iv, rv):
    if rv is None: rv = 20.0
    if iv is None: return 1.75, "N/A", "no data", "#888"
    spread = iv - rv
    rel = "IV>RV" if spread >= 0 else "IV<RV"
    delta = f"Δ {spread:+.1f}"
    if spread > T_IV_RV_RICH: return 3.5, rel, f"RICH {delta}", COLORS["case_a"]
    if spread >= 0:            return 2.5, rel, f"FAIR {delta}", COLORS["rolling"]
    return 1.0, rel, f"RISK {delta}", COLORS["wait"]

def score_ratio(premium, spot):
    if not premium or not spot: return 0.75, "N/A", "no data", "#888"
    r = (premium / spot) * 100
    if r > T_RATIO_RICH: return 1.50, f"{r:.2f}%", "RICH", COLORS["case_a"]
    if r > T_RATIO_AVG:  return 1.00, f"{r:.2f}%", "AVG",  COLORS["watch"]
    return 0.50, f"{r:.2f}%", "CHEAP", COLORS["wait"]

def score_iv_percentile(current_iv, exp_code: str):
    history = _exp_state[exp_code]["iv_history"]
    if current_iv is None: return 0.75, "N/A", "no data", "#888"
    if len(history) < 5:
        res = (0.75, "N/A", "collecting...", "#888")
    else:
        inclusive_count = sum(1 for h_iv in history if h_iv <= current_iv)
        ivp = (inclusive_count / len(history)) * 100
        if ivp > 80:   res = (1.50, f"{ivp:.0f}%", "HIGH",   "#27ae60")
        elif ivp > 40: res = (1.00, f"{ivp:.0f}%", "NORMAL", "#f39c12")
        else:          res = (0.50, f"{ivp:.0f}%", "LOW",    "#e74c3c")
    history.append(current_iv)
    if len(history) > MAX_IV_HISTORY:
        history.pop(0)
    _persistent_history[exp_code] = history
    save_iv_history(_persistent_history)
    return res

# ══════════════════════════════════════════
# CLASSIFIER
# ══════════════════════════════════════════
def classify_premium_state(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return dict(state="NO SPIKE", sublabel="...", score=0.5, color=COLORS["no_spike"])
    prem = df["premium"].to_numpy()
    ema  = df["ema5"].to_numpy()
    cp, ce = float(prem[-1]), float(ema[-1])
    win  = prem[-16:] if len(prem) >= 16 else prem
    pv   = float(np.max(win))
    pe   = (pv - float(np.mean(win))) / (float(np.mean(win)) + 1e-9) * 100
    pfp  = (pv - cp) / (pv + 1e-9) * 100
    es   = ((ce - ema[-4]) / ema[-4] * 100) if len(ema) >= 4 else 0.0
    diffs = np.diff(prem[-9:])
    cd, cu = 0, 0
    for d in reversed(diffs):
        if d < 0:
            if cu > 0: break
            cd += 1
        elif d > 0:
            if cd > 0: break
            cu += 1
        else:
            break
    if pe < T_SPIKE_MIN_EXPANSION:
        return dict(state="NO SPIKE", sublabel=f"exp < {T_SPIKE_MIN_EXPANSION}%", score=0.5, color=COLORS["no_spike"])
    if pfp > T_STALE_DROP_PCT:
        return dict(state="STALE", sublabel=f"−{pfp:.0f}% drop", score=0.0, color=COLORS["stale"])
    if cu >= T_CONSEC_UP_SPIKING and es > 0.0 and cp > ce:
        return dict(state="SPIKING", sublabel=f"{cu}↑ consec · Prem>EMA", score=0.5, color=COLORS["spiking"])
    if cd >= T_CONSEC_DOWN_CASE_A and es < T_EMA_SLOPE_CASE_A and cp < ce:
        return dict(state="CASE A", sublabel=f"{cd}↓ drops · Prem<EMA", score=3.5, color=COLORS["case_a"])
    if cd >= T_CONSEC_DOWN_ROLLING and es < T_EMA_SLOPE_ROLLING and cp < ce:
        return dict(state="ROLLING", sublabel=f"{cd}↓ drops · Prem<EMA", score=2.5, color=COLORS["rolling"])
    return dict(state="COOLING", sublabel="watch", score=1.0, color=COLORS["cooling"])

# ══════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════
def _candles(symbol: str, start: int, end: int) -> list:
    try:
        return requests.get(
            f"{BASE_URL}/history/candles",
            params={"symbol": symbol, "resolution": "30m", "start": start, "end": end},
            verify=False, timeout=12,
        ).json().get("result", [])
    except Exception as e:
        print(f"API Error ({symbol}): {e}")
        return []

def align_and_dedupe(raw: list, col: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["time", col])
    df = pd.DataFrame(raw)[["time", "close"]].rename(columns={"close": col})
    df["time"] = (df["time"].astype(int) // CANDLE_SECONDS) * CANDLE_SECONDS
    return df.drop_duplicates("time")

def fetch_spot_and_tickers() -> tuple[float, list]:
    try:
        r    = requests.get(f"{BASE_URL}/tickers", timeout=12, verify=False).json()
        tics = r.get("result", [])
        spot = float(next(t for t in tics if t["symbol"] == "BTCUSD")["mark_price"])
        return spot, tics
    except Exception as e:
        print(f"Ticker Error: {e}")
        return 0.0, []

def compute_30d_rv() -> float:
    try:
        now = int(time.time()); start = now - (30 * 24 * 3600)
        r   = requests.get(
            f"{BASE_URL}/history/candles",
            params={"symbol": "BTCUSD", "resolution": "1d", "start": start, "end": now},
            verify=False, timeout=12,
        ).json()
        df = pd.DataFrame(r.get("result", []))
        if len(df) < 2: return 20.0
        rets = np.log(df["close"] / df["close"].shift(1)).dropna()
        return round(rets.std() * np.sqrt(365) * 100, 2)
    except Exception as e:
        print(f"RV Error: {e}")
        return 20.0

def extract_live_atm_iv(tickers: list, exp_code: str, atm_s: float):
    c_s = f"C-BTC-{int(atm_s)}-{exp_code}"
    p_s = f"P-BTC-{int(atm_s)}-{exp_code}"
    ivs = []
    for t in tickers:
        if t["symbol"] in (c_s, p_s):
            v = t.get("mark_iv") or t.get("implied_volatility") or t.get("iv")
            if v is not None:
                try:
                    ivs.append(float(v))
                except Exception:
                    continue
    if not ivs: return None
    avg = sum(ivs) / len(ivs)
    return avg * 100 if avg < 2.0 else avg

def fetch_expiry_data(exp_code: str, spot: float, tickers: list) -> dict | None:
    try:
        now_ts = int(time.time())
        start  = now_ts - LOOKBACK_HOURS * 3600
        df_spot = align_and_dedupe(_candles("BTCUSD", start, now_ts), "btc_price")
        if df_spot.empty: return None

        strikes = sorted({
            float(t["symbol"].split("-")[2])
            for t in tickers if t["symbol"].endswith(exp_code)
        })
        if not strikes: return None

        s_arr = np.array(strikes)
        df_spot["atm"] = s_arr[
            np.abs(s_arr[:, None] - df_spot["btc_price"].to_numpy()[None, :]).argmin(axis=0)
        ]
        atm_map  = df_spot.set_index("time")["atm"].to_dict()
        spot_map = df_spot.set_index("time")["btc_price"].to_dict()

        rows = []
        for atm_s in df_spot["atm"].unique():
            df_c = align_and_dedupe(_candles(f"MARK:C-BTC-{int(atm_s)}-{exp_code}", start, now_ts), "c")
            df_p = align_and_dedupe(_candles(f"MARK:P-BTC-{int(atm_s)}-{exp_code}", start, now_ts), "p")
            if not df_c.empty and not df_p.empty:
                m = pd.merge(df_c, df_p, on="time")
                for _, r in m.iterrows():
                    if atm_map.get(r["time"]) == atm_s:
                        rows.append({
                            "time":      r["time"],
                            "premium":   r["c"] + r["p"],
                            "btc_price": spot_map[r["time"]],
                            "atm":       atm_s,
                        })

        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        if df.empty: return None
        df["premium"]     = df["premium"].rolling(window=2).mean().fillna(df["premium"])
        df["ema5"]        = df["premium"].ewm(span=EMA_SPAN, adjust=False).mean()
        df["datetime"]    = (
            pd.to_datetime(df["time"], unit="s")
            .dt.tz_localize("UTC")
            .dt.tz_convert(IST)
        )
        df["atm_changed"] = df["atm"].ne(df["atm"].shift())
        df.iloc[0, df.columns.get_loc("atm_changed")] = False

        live_iv = extract_live_atm_iv(tickers, exp_code, s_arr[np.abs(s_arr - spot).argmin()])
        return {
            "df":  df,
            "atm": s_arr[np.abs(s_arr - spot).argmin()],
            "spot": spot,
            "iv":   live_iv,
        }
    except Exception as e:
        print(f"Fetch Error [{exp_code}]: {e}")
        return None

# ══════════════════════════════════════════
# ONE-SHOT FULL REFRESH (called by app.py)
# ══════════════════════════════════════════
def full_refresh() -> dict:
    """
    Fetch everything needed for one dashboard render.
    Returns a dict keyed by expiry code with all scores + dataframe.
    """
    spot, tics = fetch_spot_and_tickers()
    if spot <= 0:
        return {"error": "Could not fetch spot price from Delta Exchange."}

    expiries = fetch_available_expiries(tics)
    if not expiries:
        return {"error": "No BTC option expiries found. Market may be closed."}

    rv_30d   = compute_30d_rv()
    results  = {}

    for exp in expiries:
        ensure_state(exp)
        data = fetch_expiry_data(exp, spot, tics)
        if not data:
            continue
        df = data["df"]
        iv = data["iv"]
        cp = df["premium"].iloc[-1]

        st              = classify_premium_state(df)
        s2, v2, sub2, c2 = score_iv_rv(iv, rv_30d)
        s3, v3, sub3, c3 = score_ratio(cp, data["spot"])
        s4, v4, sub4, c4 = score_iv_percentile(iv, exp)
        total_score      = st["score"] + s2 + s3 + s4

        results[exp] = {
            "label":       expiry_label(exp),
            "df":          df,
            "spot":        data["spot"],
            "atm":         data["atm"],
            "iv":          iv,
            "rv":          rv_30d,
            "total_score": total_score,
            "state":       st,
            "iv_rv":       {"score": s2, "value": v2, "sublabel": sub2, "color": c2},
            "ratio":       {"score": s3, "value": v3, "sublabel": sub3, "color": c3},
            "iv_pct":      {"score": s4, "value": v4, "sublabel": sub4, "color": c4},
        }

    results["__meta__"] = {
        "spot":      spot,
        "rv_30d":    rv_30d,
        "expiries":  expiries,
        "refreshed": datetime.now(IST).strftime("%H:%M:%S IST"),
    }
    return results
