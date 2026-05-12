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
DELTA_CDN       = "https://cdn.india.deltaex.org/v2"
IST             = ZoneInfo("Asia/Kolkata")
HISTORY_FILE    = "iv_history.json"

CANDLE_SECONDS  = 30 * 60
LOOKBACK_HOURS  = 12
EMA_SPAN        = 5
MAX_EXPIRIES    = 6
MAX_IV_HISTORY  = 10000
SAVE_COOLDOWN_SEC = 3600

# ── Market-wide IV/RV and IVP config (from BTC_IV_RV_Monitor) ────────────────
CDN_ASSET         = "BTC"
CDN_MATURITY      = "daily"
CDN_LIVE_DAYS     = 7      # lookback for live 5m CDN IV/RV
IVP_LOOKBACK_DAYS = 90     # days of history for IVP baseline
IVP_CHEAP         = 25     # IVP < 25  → CHEAP
IVP_RICH          = 75     # IVP > 75  → RICH
CDN_HEADERS       = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko)"
    )
}

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
    # ── NO DATA — too few dots to classify (new expiry just started) ────────
    if len(df) < 6:
        n = len(df)
        return dict(state="NO DATA", sublabel=f"only {n} dot{'s' if n != 1 else ''} — need ≥6", score=0.5, color="#888")

    prem = df["premium"].to_numpy()
    ema  = df["ema5"].to_numpy()
    cp, ce = float(prem[-1]), float(ema[-1])

    # Window of last 16 dots for spike / stale analysis
    win_prem = prem[-16:] if len(prem) >= 16 else prem
    win_ema  = ema[-16:]  if len(ema)  >= 16 else ema

    # ── NO SPIKE — no dot in the last 16 had premium ≥ EMA + $1 ────────────
    spike_indices = [
        i for i, (p, e) in enumerate(zip(win_prem, win_ema))
        if float(p) >= float(e) + 1.0
    ]
    if not spike_indices:
        return dict(state="NO SPIKE", sublabel=f"no dot ≥ EMA+$1 in last {len(win_prem)}", score=0.5, color=COLORS["no_spike"])

    # ── SPIKING — current dot still above EMA + $1 → wait for rollover ─────
    if cp >= ce + 1.0:
        return dict(state="SPIKING", sublabel="Prem > EMA+$1 · wait for rollover", score=0.5, color=COLORS["spiking"])

    # ── STALE — most recent spike dot was ≥ 6 dots ago → window closed ──────
    last_spike_idx    = spike_indices[-1]
    dots_since_spike  = (len(win_prem) - 1) - last_spike_idx
    if dots_since_spike >= 6:
        return dict(state="STALE", sublabel=f"peak {dots_since_spike} dots ago · window closed", score=0.0, color=COLORS["stale"])

    # ── Post-peak diffs — premium changes from the spike dot onward ─────────
    # spike dot is at win_prem[last_spike_idx]; post-peak dots follow it
    post_peak_prem = win_prem[last_spike_idx:]   # includes the spike dot itself
    if len(post_peak_prem) < 2:
        # Only the spike dot itself — no diffs yet
        return dict(state="COOLING", sublabel="only 1 dot since peak — need ≥2", score=1.0, color=COLORS["cooling"])

    post_diffs = np.diff(post_peak_prem).tolist()   # diffs after the spike dot
    n_post     = len(post_diffs)                     # number of post-peak moves

    # ── COOLING — fewer than 3 post-peak diffs → too early to judge ─────────
    if n_post < 3:
        diff_str = ", ".join(f"{d:+.2f}" for d in post_diffs)
        return dict(
            state="COOLING",
            sublabel=f"only {n_post} dot{'s' if n_post != 1 else ''} since peak — need ≥3",
            score=1.0,
            color=COLORS["cooling"],
        )

    # ── CASE A — last 3-4 diffs all negative, EMA slope falling, Prem < EMA ─
    last_diffs = post_diffs[-4:] if len(post_diffs) >= 4 else post_diffs[-3:]
    es = ((ce - float(ema[-4])) / float(ema[-4]) * 100) if len(ema) >= 4 else 0.0
    if all(d < 0 for d in last_diffs) and es < T_EMA_SLOPE_CASE_A and cp < ce:
        diff_str = ", ".join(f"{d:+.2f}" for d in post_diffs[-4:])
        return dict(
            state="CASE A",
            sublabel=f"post-peak diffs: {diff_str}",
            score=3.5,
            color=COLORS["case_a"],
        )

    # ── ROLLING — 2 of last 3 post-peak diffs are negative ──────────────────
    last3 = post_diffs[-3:]
    n_down = sum(1 for d in last3 if d < 0)
    if n_down >= 2:
        diff_str = ", ".join(f"{d:+.2f}" for d in post_diffs[-3:])
        return dict(
            state="ROLLING",
            sublabel=f"post-peak diffs: {diff_str}",
            score=2.5,
            color=COLORS["rolling"],
        )

    # ── COOLING — spike ended but not enough downward pressure yet ───────────
    diff_str = ", ".join(f"{d:+.2f}" for d in post_diffs[-3:])
    return dict(
        state="COOLING",
        sublabel=f"only {n_post} dot(s) since peak — need ≥3",
        score=1.0,
        color=COLORS["cooling"],
    )

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
        # NOTE: rolling(2) mean was removed — it caused systematic upward bias
        # (in a declining premium series each bar averaged with the prior higher bar)
        # Raw call+put sum is used directly, matching TradeSteady's approach.
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
# MARKET-WIDE IV / RV  (Delta CDN 5m)
# ══════════════════════════════════════════
def fetch_market_iv_rv() -> tuple[float | None, float | None]:
    """
    Fetch live market-wide ATM IV and RV from Delta CDN options_iv / options_rv.
    Uses 5m resolution over the past 7 days and returns the latest non-null value.
    Returns (rv, iv) both as annualised % floats, or (None, None) on failure.
    This is the same method used in BTC_IV_RV_Monitor.py.
    """
    end_t   = int(time.time())
    start_t = end_t - CDN_LIVE_DAYS * 24 * 3600

    def _cdn(endpoint: str) -> str:
        return (
            f"{DELTA_CDN}/{endpoint}"
            f"?start_time={start_t}&end_time={end_t}"
            f"&asset_symbol={CDN_ASSET}&maturity={CDN_MATURITY}&resolution=5m"
        )

    rv = iv = None
    try:
        r = requests.get(_cdn("options_rv"), headers=CDN_HEADERS, timeout=10)
        if r.status_code == 200:
            vals = [x for x in r.json().get("result", {}).get("rv", []) if x is not None]
            if vals:
                rv = round(float(vals[-1]), 2)
    except Exception:
        pass
    try:
        r = requests.get(_cdn("options_iv"), headers=CDN_HEADERS, timeout=10)
        if r.status_code == 200:
            vals = [x for x in r.json().get("result", {}).get("atm_iv", []) if x is not None]
            if vals:
                iv = round(float(vals[-1]), 2)
    except Exception:
        pass
    return rv, iv


def fetch_delta_iv_history(days: int = IVP_LOOKBACK_DAYS) -> list[float]:
    """
    Fetch `days` of ATM IV history from Delta CDN on the SAME scale as the
    live market IV (options_iv endpoint).  Used to calculate IVP.

    Why not resolution=1D: Delta CDN silently returns [] for 1D — unusable.
    Why not Deribit DVOL: DVOL runs 50-80; Delta CDN atm_iv runs 25-45 —
      mixing them makes IVP = 0% because all history >> live IV.

    Strategy: fetch 1h bars (same endpoint, same scale), downsample to one
    value per UTC calendar day.  Fallback to 5m bars if 1h returns < 10.
    """
    end_t   = int(time.time())
    start_t = end_t - days * 24 * 3600

    def _downsample(iv_list: list, ts_list: list, stride: int) -> list[float]:
        if ts_list and len(ts_list) == len(iv_list):
            daily: dict = {}
            for ts, iv in zip(ts_list, iv_list):
                if iv is None or float(iv) <= 0:
                    continue
                key = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                daily[key] = round(float(iv), 2)   # last value of day wins
            return list(daily.values())
        valid = [round(float(v), 2) for v in iv_list if v is not None and float(v) > 0]
        return valid[::stride] if valid else []

    for resolution, stride in [("1h", 24), ("5m", 288)]:
        try:
            url = (
                f"{DELTA_CDN}/options_iv"
                f"?start_time={start_t}&end_time={end_t}"
                f"&asset_symbol={CDN_ASSET}&maturity={CDN_MATURITY}"
                f"&resolution={resolution}"
            )
            r = requests.get(url, headers=CDN_HEADERS, timeout=20)
            if r.status_code != 200:
                continue
            result  = r.json().get("result", {})
            iv_list = result.get("atm_iv", [])
            ts_list = result.get("time", [])
            if len(iv_list) < 10:
                continue
            daily = _downsample(iv_list, ts_list, stride)
            if len(daily) >= 10:
                return daily
        except Exception:
            continue
    return []


def calculate_ivp(iv_history: list[float], current_iv: float | None) -> float | None:
    """
    IV Percentile: % of historical days where ATM IV < today's IV.
    Returns 0-100 float, or None if history is too short (< 10 points).
    """
    if len(iv_history) < 10 or current_iv is None:
        return None
    return round(sum(1 for v in iv_history if v < current_iv) / len(iv_history) * 100, 1)


def market_iv_rv_signal(iv: float | None, rv: float | None) -> dict:
    """
    Build the IV/RV card dict using market-wide CDN values (not per-expiry IV).
    Returns keys: value, sublabel, color, score — same shape as score_iv_rv().
    """
    if iv is None or rv is None:
        return {"score": 1.75, "value": "N/A", "sublabel": "no data", "color": "#888"}
    spread = round(iv - rv, 2)
    rel    = "IV>RV" if spread >= 0 else "IV<RV"
    delta  = f"Δ {spread:+.1f}"
    if spread > T_IV_RV_RICH:
        return {"score": 3.5, "value": rel, "sublabel": f"RICH {delta}", "color": COLORS["case_a"]}
    if spread >= 0:
        return {"score": 2.5, "value": rel, "sublabel": f"FAIR {delta}", "color": COLORS["rolling"]}
    return    {"score": 1.0, "value": rel, "sublabel": f"RISK {delta}", "color": COLORS["wait"]}


def market_ivp_signal(ivp: float | None, iv: float | None) -> dict:
    """
    Build the IV Percentile card dict from IVP calculation.
    Thresholds: CHEAP < 25%, FAIR 25-75%, RICH > 75%.
    Returns keys: value, sublabel, color, score — same shape as score_iv_percentile().
    """
    if ivp is None:
        lbl = "no data" if iv is None else "collecting..."
        return {"score": 0.75, "value": "N/A", "sublabel": lbl, "color": "#888"}
    if ivp < IVP_CHEAP:
        return {"score": 0.50, "value": f"{ivp:.0f}%", "sublabel": "CHEAP", "color": COLORS["case_a"]}
    if ivp > IVP_RICH:
        return {"score": 1.50, "value": f"{ivp:.0f}%", "sublabel": "RICH",  "color": COLORS["wait"]}
    return     {"score": 1.00, "value": f"{ivp:.0f}%", "sublabel": "FAIR",  "color": COLORS["rolling"]}


# ══════════════════════════════════════════
# ONE-SHOT FULL REFRESH (called by app.py)
# ══════════════════════════════════════════
def full_refresh() -> dict:
    """
    Fetch everything needed for one dashboard render.
    Returns a dict keyed by expiry code with all scores + dataframe.

    IV/RV card  — uses market-wide Delta CDN options_iv / options_rv (5m live).
                  Same source and scale as BTC_IV_RV_Monitor.py.
    IV Pct card — uses 90-day Delta CDN options_iv history (1h→daily downsample)
                  compared against the same live market IV.  Guarantees apples-
                  to-apples percentile (no Deribit DVOL scale mismatch).
    """
    spot, tics = fetch_spot_and_tickers()
    if spot <= 0:
        return {"error": "Could not fetch spot price from Delta Exchange."}

    expiries = fetch_available_expiries(tics)
    if not expiries:
        return {"error": "No BTC option expiries found. Market may be closed."}

    rv_30d = compute_30d_rv()

    # ── Market-wide IV/RV + IVP (fetched once, shared across all expiry rows) ─
    mkt_rv, mkt_iv   = fetch_market_iv_rv()
    iv_history        = fetch_delta_iv_history(IVP_LOOKBACK_DAYS)
    ivp_value         = calculate_ivp(iv_history, mkt_iv)
    mkt_iv_rv_card    = market_iv_rv_signal(mkt_iv, mkt_rv)
    mkt_ivp_card      = market_ivp_signal(ivp_value, mkt_iv)

    results = {}
    for exp in expiries:
        ensure_state(exp)
        data = fetch_expiry_data(exp, spot, tics)
        if not data:
            continue
        df = data["df"]
        iv = data["iv"]
        cp = df["premium"].iloc[-1]

        st_             = classify_premium_state(df)
        s3, v3, sub3, c3 = score_ratio(cp, data["spot"])

        # IV/RV and IVP come from market-wide CDN data — not per-expiry option IV
        total_score = st_["score"] + mkt_iv_rv_card["score"] + s3 + mkt_ivp_card["score"]

        results[exp] = {
            "label":       expiry_label(exp),
            "df":          df,
            "spot":        data["spot"],
            "atm":         data["atm"],
            "iv":          iv,          # per-expiry ATM IV (for chart title)
            "mkt_iv":      mkt_iv,      # market-wide CDN IV
            "mkt_rv":      mkt_rv,      # market-wide CDN RV
            "ivp":         ivp_value,   # 90-day percentile
            "rv":          rv_30d,
            "total_score": total_score,
            "state":       st_,
            "iv_rv":       mkt_iv_rv_card,   # ← market-wide, correct IV>RV / IV<RV
            "ratio":       {"score": s3, "value": v3, "sublabel": sub3, "color": c3},
            "iv_pct":      mkt_ivp_card,     # ← 90-day IVP on same scale as live IV
        }

    results["__meta__"] = {
        "spot":       spot,
        "rv_30d":     rv_30d,
        "mkt_iv":     mkt_iv,
        "mkt_rv":     mkt_rv,
        "ivp":        ivp_value,
        "iv_history_days": len(iv_history),
        "expiries":   expiries,
        "refreshed":  datetime.now(IST).strftime("%H:%M:%S IST"),
    }
    return results
