# 📊 BTC Premium Straddle Scanner

Live dashboard for scanning BTC straddle premiums across all upcoming option expiries on **Delta Exchange India**.  
Auto-rolls to new expiries (June, July, …) — no manual configuration needed.

## Features

- **Auto-expiry discovery** — fetches all live BTC option symbols from Delta, extracts expiry dates, drops expired ones automatically
- **Premium chart** with EMA-5 overlay per expiry
- **4-factor scoring**: Premium direction classifier · IV/RV spread · Premium/Spot ratio · IV Percentile
- **States detected**: CASE A · ROLLING · SPIKING · COOLING · STALE · NO SPIKE
- **Auto-refresh** every 5 minutes (configurable in sidebar)
- Persistent IV history across sessions (`iv_history.json`)

## Project Structure

```
straddle_scanner/
├── app.py            # Streamlit UI — charts, layout, auto-refresh
├── core.py           # Business logic — all scoring, data fetch, state
├── requirements.txt
└── .gitignore
```

`core.py` has zero UI dependencies — you can import it in scripts, notebooks, or tests independently.

## Local Setup

```bash
git clone https://github.com/<your-username>/straddle-scanner.git
cd straddle-scanner
pip install -r requirements.txt
streamlit run app.py
```

Optional — create a `.env` file for Telegram alerts (not wired into Streamlit UI yet):
```
TELEGRAM_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo · branch · set **Main file path** = `app.py`
4. Click **Deploy** — no secrets needed for basic use

> **Note**: Streamlit Cloud's ephemeral filesystem means `iv_history.json` resets on each deploy/restart.  
> For persistent IV history in production, swap `load_iv_history` / `save_iv_history` in `core.py` to use a database (e.g. Supabase, Redis, or Streamlit's built-in `st.session_state`).

## Scoring Reference

| Score | Meaning |
|-------|---------|
| **CASE A** | 3+ consecutive drops, EMA slope < −0.2, Premium < EMA — best entry signal |
| **ROLLING** | 2+ consecutive drops, EMA slope falling, Premium < EMA |
| **COOLING** | Premium declining but thresholds not fully met |
| **SPIKING** | Premium rising above EMA — wait |
| **STALE**   | Premium has dropped >25% from recent peak |
| **NO SPIKE**| No meaningful expansion yet |
