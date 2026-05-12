"""
app.py — Streamlit dashboard for Premium Straddle Scalping (v7.17)
Replaces matplotlib TkAgg + FuncAnimation with Streamlit auto-refresh.
All scoring / data-fetch logic lives in core.py — unchanged.
"""

import time
import streamlit as st
import matplotlib
matplotlib.use("Agg")          # headless backend — no Tk needed on Streamlit Cloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from core import (
    full_refresh,
    COLORS,
    EMA_SPAN,
    expiry_label,
    IVP_CHEAP,
    IVP_RICH,
)

# ══════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════
st.set_page_config(
    page_title="Premium Straddle Scalping",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark theme CSS injection
st.markdown("""
<style>
    body, .stApp { background-color: #0b0c10; color: #e0e0e0; }
    .block-container { padding-top: 1rem; }
    .metric-card {
        background: #0f1419;
        border: 1px solid #252e3b;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .score-badge {
        font-size: 2rem;
        font-weight: 800;
    }
    .state-label {
        font-size: 1.1rem;
        font-weight: 700;
    }
    .sublabel {
        font-size: 0.75rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

REFRESH_SECONDS = 180   # 3-minute refresh
_CARD_NUMS = ["①", "②", "③", "④"]

# ══════════════════════════════════════════
# CHART BUILDER (matplotlib → st.pyplot)
# ══════════════════════════════════════════
def build_chart(exp_code: str, result: dict) -> plt.Figure:
    df  = result["df"]
    iv  = result["iv"]
    cp  = df["premium"].iloc[-1]
    ce  = df["ema5"].iloc[-1]
    st_ = result["state"]

    fig = plt.figure(figsize=(10, 5), facecolor=COLORS["bg"], constrained_layout=True)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1], hspace=0.08, figure=fig)
    ax  = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])

    ax.set_facecolor(COLORS["surface"])
    ax_c.axis("off")

    y_vals = pd.concat([df["premium"], df["ema5"]]).dropna()
    pad    = max((y_vals.max() - y_vals.min()) * 0.20, 5)
    ylo, yhi = y_vals.min() - pad, y_vals.max() + pad
    ax.set_ylim(ylo, yhi)

    # ATM change lines
    for _, sr in df[df["atm_changed"]].iterrows():
        ax.axvline(x=sr["datetime"], color=COLORS["atm_chg"], linestyle="--", alpha=0.5)
        ax.text(sr["datetime"], yhi, f" {int(sr['atm']):,}",
                color=COLORS["atm_chg"], fontsize=7, rotation=90, va="top")

    ax.plot(df["datetime"], df["premium"],
            color=COLORS["premium"], marker="o", markersize=4, label="Premium")
    ax.plot(df["datetime"], df["ema5"],
            color=COLORS["ema"], linewidth=2, label="EMA-5")
    ax.axhline(y=cp, color=COLORS["premium"], linestyle=":", alpha=0.4)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(colors="#888", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#252e3b")

    y_range = yhi - ylo
    p_off, e_off = (12, -12) if cp >= ce else ((-12, 12) if abs(cp - ce) < y_range * 0.12 else (0, 0))

    ax.annotate(f"{cp:.1f}", xy=(1, cp), xycoords=("axes fraction", "data"),
                xytext=(6, p_off), textcoords="offset points",
                ha="left", va="center", fontsize=9, weight="bold", color="white",
                bbox=dict(facecolor=COLORS["premium"], edgecolor="none", pad=3),
                annotation_clip=False)
    ax.annotate(f"{ce:.1f}", xy=(1, ce), xycoords=("axes fraction", "data"),
                xytext=(6, e_off), textcoords="offset points",
                ha="left", va="center", fontsize=9, weight="bold", color="#0b0c10",
                bbox=dict(facecolor=COLORS["ema"], edgecolor="none", pad=3),
                annotation_clip=False)

    total = result["total_score"]
    ax.set_title(
        f"BTC {result['label']} | Score: {total:.1f}",
        loc="left", color="white", weight="bold", fontsize=11,
    )
    ax.legend(fontsize=8, labelcolor="#ccc", facecolor=COLORS["surface"],
              edgecolor="#252e3b", loc="upper left")

    # Score cards strip
    _draw_score_cards(ax_c, [
        dict(label="Premium dir", value=st_["state"],          sublabel=st_["sublabel"],         value_color=st_["color"]),
        dict(label="IV/RV spread", value=result["iv_rv"]["value"], sublabel=result["iv_rv"]["sublabel"], value_color=result["iv_rv"]["color"]),
        dict(label="Ratio",        value=result["ratio"]["value"],  sublabel=result["ratio"]["sublabel"],  value_color=result["ratio"]["color"]),
        dict(label="IV Percentile",value=result["iv_pct"]["value"], sublabel=result["iv_pct"]["sublabel"], value_color=result["iv_pct"]["color"]),
    ])

    return fig

def _draw_score_cards(ax, cards):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(cards); pad, gap = 0.012, 0.01
    total_w = 1.0 - 2 * pad - (n - 1) * gap
    cw = total_w / n
    for i, c in enumerate(cards):
        x0 = pad + i * (cw + gap); cx = x0 + cw / 2
        ax.add_patch(FancyBboxPatch(
            (x0, 0.04), cw, 0.92,
            boxstyle="round,pad=0.015",
            facecolor=COLORS["score_bg"], edgecolor=COLORS["card_bdr"],
            linewidth=0.8, transform=ax.transAxes, clip_on=False,
        ))
        ax.text(cx, 0.88, _CARD_NUMS[i],  ha="center", va="center", fontsize=8,    color="#666",             transform=ax.transAxes)
        ax.text(cx, 0.72, c["label"],      ha="center", va="center", fontsize=7.5,  color="#999",             transform=ax.transAxes)
        ax.text(cx, 0.46, c["value"],      ha="center", va="center", fontsize=12,   color=c["value_color"],   transform=ax.transAxes, weight="bold")
        ax.text(cx, 0.18, c["sublabel"],   ha="center", va="center", fontsize=7,    color="#888",             transform=ax.transAxes)

# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    refresh_interval = st.slider("Refresh interval (sec)", 60, 600, REFRESH_SECONDS, step=30)
    st.markdown("---")
    st.markdown("**Thresholds** (display only)")
    st.caption("Edit `core.py` to change scoring thresholds.")
    from core import (
        T_IV_RV_RICH, T_RATIO_RICH, T_RATIO_AVG,
        T_STALE_DROP_PCT, T_SPIKE_MIN_EXPANSION,
    )
    st.json({
        "IV-RV rich >":    T_IV_RV_RICH,
        "Ratio rich >":    T_RATIO_RICH,
        "Ratio avg >":     T_RATIO_AVG,
        "Stale drop %":    T_STALE_DROP_PCT,
        "Spike min exp %": T_SPIKE_MIN_EXPANSION,
    })

# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
st.markdown("## 📊 Premium Straddle Scalping")
st.caption("Live data · Delta Exchange India · Auto-rolls to new expiries")

status_bar = st.empty()
refresh_btn = st.button("🔄 Refresh Now")

# ══════════════════════════════════════════
# MAIN RENDER LOOP
# ══════════════════════════════════════════
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def cached_refresh():
    return full_refresh()

def render_dashboard():
    with st.spinner("Fetching live data from Delta Exchange..."):
        # Use button press to bust cache
        if refresh_btn:
            st.cache_data.clear()
        results = cached_refresh()

    if "error" in results:
        st.error(f"❌ {results['error']}")
        return

    meta = results.get("__meta__", {})
    expiries = meta.get("expiries", [])

    mkt_iv  = meta.get("mkt_iv")
    mkt_rv  = meta.get("mkt_rv")
    ivp     = meta.get("ivp")
    iv_days = meta.get("iv_history_days", 0)

    iv_str  = f"{mkt_iv:.2f}%" if mkt_iv is not None else "N/A"
    rv_str  = f"{mkt_rv:.2f}%" if mkt_rv is not None else "N/A"
    ivp_str = f"{ivp:.0f}%" if ivp is not None else "N/A"

    if mkt_iv is not None and mkt_rv is not None:
        spread = round(mkt_iv - mkt_rv, 2)
        iv_rv_signal = "IV&gt;RV ▲" if spread >= 0 else "IV&lt;RV ▼"
        iv_rv_color  = "#e74c3c" if spread >= 0 else "#27ae60"
        iv_rv_html   = f'<span style="color:{iv_rv_color};font-weight:700">{iv_rv_signal}</span>'
    else:
        iv_rv_html = '<span style="color:#888">N/A</span>'

    if ivp is not None:
        ivp_color = "#27ae60" if ivp < 25 else ("#e74c3c" if ivp > 75 else "#f39c12")
        ivp_label = "CHEAP" if ivp < 25 else ("RICH" if ivp > 75 else "FAIR")
        ivp_html  = f'<span style="color:{ivp_color};font-weight:700">{ivp_str} {ivp_label}</span>'
    else:
        ivp_html = '<span style="color:#888">N/A</span>'

    status_bar.markdown(
        f"**BTC Spot:** ${meta.get('spot', 0):,.0f} &nbsp;|&nbsp; "
        f"**Market IV:** {iv_str} &nbsp;|&nbsp; "
        f"**Market RV (30d):** {rv_str} &nbsp;|&nbsp; "
        f"**IV/RV:** {iv_rv_html} &nbsp;|&nbsp; "
        f"**IVP (90d / {iv_days}d data):** {ivp_html} &nbsp;|&nbsp; "
        f"**Last refresh:** {meta.get('refreshed', '—')}",
        unsafe_allow_html=True,
    )

    if not expiries:
        st.warning("No upcoming BTC option expiries found.")
        return

    # Render 2-column grid
    exp_results = {k: v for k, v in results.items() if k != "__meta__"}

    cols = st.columns(2)
    for idx, (exp_code, result) in enumerate(exp_results.items()):
        with cols[idx % 2]:
            fig = build_chart(exp_code, result)
            st.pyplot(fig, width='stretch')
            plt.close(fig)

    # Summary table
    st.markdown("---")
    st.markdown("### 📋 Summary")

    # Market-wide IV/RV + IVP info banner
    if mkt_iv is not None and mkt_rv is not None:
        spread = round(mkt_iv - mkt_rv, 2)
        sig_color = "#e74c3c" if spread >= 0 else "#27ae60"
        sig_text  = f"IV {'>' if spread >= 0 else '<'} RV  (Δ {spread:+.2f}%)"
        st.markdown(
            f'<div style="background:#0f1419;border:1px solid #252e3b;border-radius:8px;'
            f'padding:10px 18px;margin-bottom:10px;font-size:0.9rem">'
            f'📡 <b>Market-wide (Delta CDN 5m)</b> &nbsp;—&nbsp; '
            f'ATM IV: <b>{mkt_iv:.2f}%</b> &nbsp;|&nbsp; '
            f'RV (30d): <b>{mkt_rv:.2f}%</b> &nbsp;|&nbsp; '
            f'Signal: <span style="color:{sig_color};font-weight:700">{sig_text}</span>'
            f'&nbsp;|&nbsp; IVP (90d): {ivp_html}'
            f'&nbsp; <span style="color:#555;font-size:0.8rem">'
            f'({iv_days} daily readings — same scale as live IV)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    rows = []
    for exp_code, r in exp_results.items():
        ivp_val  = r["ivp"]
        ivp_disp = f"{ivp_val:.0f}%" if ivp_val is not None else "N/A"
        ivp_lbl  = r["iv_pct"]["sublabel"]   # CHEAP / FAIR / RICH / collecting...
        rows.append({
            "Expiry":       r["label"],
            "Score":        f"{r['total_score']:.1f}",
            "State":        r["state"]["state"],
            "IV/RV":        r["iv_rv"]["value"],      # IV>RV or IV<RV
            "IV/RV signal": r["iv_rv"]["sublabel"],   # RICH Δ+X / FAIR Δ+X / RISK Δ-X
            "Ratio":        r["ratio"]["sublabel"],
            "IVP (90d)":    ivp_disp,
            "IVP Label":    ivp_lbl,
            "Mkt IV":       f"{r['mkt_iv']:.2f}%" if r.get("mkt_iv") else "N/A",
            "Mkt RV":       f"{r['mkt_rv']:.2f}%" if r.get("mkt_rv") else "N/A",
            "ATM Strike":   f"{int(r['atm']):,}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

render_dashboard()

# ── Auto-refresh countdown ──────────────────
st.markdown("---")
countdown = st.empty()
for remaining in range(refresh_interval, 0, -1):
    countdown.caption(f"⏱ Next auto-refresh in {remaining}s")
    time.sleep(1)
st.rerun()
