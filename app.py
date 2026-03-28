"""
VECTOR VXP2 — Predictive Health Dashboard
Real-time turbofan engine RUL monitoring with physics-aware safety alerts.
"""

import os
import sys
import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Ensure the project root is on sys.path so we can import engine
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.engine import VectorInference

# ────────────────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VECTOR VXP2 — Engine Health Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark-mode polish, alert styling
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ──────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800;900&display=swap');
    
    html, body, [class*="st-"] { 
        font-family: 'Montserrat', sans-serif; 
        background-color: #050505; 
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* ── Header ──────────────────────────────────────────────────────── */
    .header-container {
        background: #050505;
        border: 1px solid #222222;
        border-radius: 0px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        text-transform: uppercase;
        border-top: 4px solid #FF0000;
    }
    .header-container h1 {
        color: #FFFFFF;
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: 0.05em;
    }
    .header-container p {
        color: #A0A0A0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 600;
        letter-spacing: 0.1em;
    }

    /* ── Metric cards ────────────────────────────────────────────────── */
    .metric-card {
        background: #0A0A0A;
        border: 1px solid #222222;
        border-radius: 0px;
        padding: 1.5rem;
        text-align: center;
        border-bottom: 2px solid #D4AF37;
    }
    .metric-card .label { color: #888888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 600; }
    .metric-card .value { color: #FFFFFF; font-size: 2.2rem; font-weight: 800; margin-top: 0.5rem; letter-spacing: 0.05em; }

    /* ── Alert Center ────────────────────────────────────────────────── */
    .alert-box {
        border-radius: 0px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.5rem;
        font-weight: 800;
        font-size: 1.1rem;
        line-height: 1.4;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-left: 4px solid;
    }
    .alert-safe {
        background: #0A0A0A;
        border: 1px solid #222222;
        border-left-color: #FFFFFF;
        color: #E0E0E0;
    }
    .alert-warning {
        background: #14110A;
        border: 1px solid #332B1A;
        border-left-color: #D4AF37;
        color: #D4AF37;
    }
    .alert-critical {
        background: #1A0505;
        border: 1px solid #400000;
        border-left-color: #FF0000;
        color: #FF3C3C;
    }

    /* ── Sidebar branding ────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #050505;
        border-right: 1px solid #222222;
    }
    .sidebar-title {
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #333333;
        padding-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
logo_path = os.path.join(_PROJECT_ROOT, "media", "Logo-1 PNG.png")
if os.path.exists(logo_path):
    logo_b64 = get_base64_image(logo_path)
    # Increasing max-height further as requested to ensure proper visibility
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="max-height: 250px; margin-bottom: 0px;" alt="ORION SPACETECH">'
else:
    logo_html = "<h1>ORION SPACETECH</h1>"

st.markdown(
    f"""
    <div class="header-container">
        {logo_html}
        <p>VXP2 Predictive Maintenance Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# Sidebar — dataset upload
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">TELEMETRY STREAM</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload NASA CMAPSS dataset (.csv / .txt)",
        type=["csv", "txt"],
        help="Upload a CSV/TXT file from the NASA CMAPSS Turbofan Engine Degradation dataset.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sidebar-title">SIMULATION CONTROLS</p>', unsafe_allow_html=True)
    sim_speed = st.slider("Playback speed (sec/cycle)", 0.05, 1.0, 0.15, 0.05)
    buffer_size = st.slider("Buffer Size", min_value=10, max_value=200, value=50, step=10)
    noise_level = st.slider("Noise Level (%)", min_value=0, max_value=10, value=0, step=1)
    auto_run = st.button("ENGAGE SIMULATION", use_container_width=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#A0A0A0; font-size:0.75rem; line-height:1.6; text-transform:uppercase; font-weight:600; letter-spacing:1px; border-top:1px solid #222; padding-top:1rem;">
            <strong>VECTOR VXP2</strong><br>
            Orion Spacetech Prototype<br><br>
            <span style="color:#D4AF37;">Powered by Aerospace AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────────────────
# Helpers — CMAPSS column names
# ────────────────────────────────────────────────────────────────────────────
CMAPSS_COLS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]          # operational settings
    + [f"s{i}" for i in range(1, 22)]           # 21 sensor readings
)

# ────────────────────────────────────────────────────────────────────────────
# Initialise VectorInference engine (cached so it loads once)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine() -> VectorInference | None:
    """Load the Random Forest model; returns None if the .pkl is missing."""
    try:
        return VectorInference()          # default path: models/rf_v1.pkl
    except FileNotFoundError:
        return None


engine = load_engine()


def load_cmapss(file) -> pd.DataFrame:
    """Parse an uploaded NASA CMAPSS file into a clean DataFrame."""
    try:
        df = pd.read_csv(file)
        if "s1" in df.columns or "s11" in df.columns:
            return df
    except Exception:
        pass

    if hasattr(file, "seek"):
        file.seek(0)
    df = pd.read_csv(file, sep=r"\s+", header=None)
    if df.shape[1] == len(CMAPSS_COLS):
        df.columns = CMAPSS_COLS
    elif df.shape[1] == len(CMAPSS_COLS) + 2:
        # Some files include two trailing NaN columns
        df = df.iloc[:, : len(CMAPSS_COLS)]
        df.columns = CMAPSS_COLS
    else:
        # Fall back — just use numeric indices
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df


def build_gauge(rul_value: float) -> go.Figure:
    """Create a Plotly gauge for Remaining Useful Life (0–200 cycles)."""
    rul_clamped = max(0, min(200, rul_value))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=rul_clamped,
            number={"suffix": " CYCLES", "font": {"size": 28, "color": "#FFFFFF", "family": "Montserrat"}},
            title={"text": "REMAINING USEFUL LIFE", "font": {"size": 14, "color": "#A0A0A0", "family": "Montserrat"}},
            gauge={
                "axis": {"range": [0, 200], "tickcolor": "#555555", "tickwidth": 1},
                "bar": {"color": "#FFFFFF", "thickness": 0.25},
                "bgcolor": "#050505",
                "borderwidth": 1,
                "bordercolor": "#333333",
                "steps": [
                    {"range": [0, 30], "color": "rgba(255, 0, 0, 0.15)"},
                    {"range": [30, 80], "color": "rgba(212, 175, 55, 0.15)"},
                    {"range": [80, 200], "color": "rgba(255, 255, 255, 0.05)"},
                ],
                "threshold": {
                    "line": {"color": "#FF0000", "width": 3},
                    "thickness": 0.8,
                    "value": 30,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin=dict(l=30, r=30, t=50, b=10),
        font={"family": "Montserrat, sans-serif"}
    )
    return fig


def build_s11_chart(cycles: list, s11_values: list) -> go.Figure:
    """Create a real-time line chart for sensor S11 (Turbine Temp)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=s11_values,
            mode="lines+markers",
            line=dict(color="#D4AF37", width=2),
            marker=dict(size=4, color="#FFFFFF", line=dict(width=1, color="#D4AF37")),
            fill="tozeroy",
            fillcolor="rgba(212, 175, 55, 0.08)",
            name="S11",
        )
    )
    fig.update_layout(
        title=dict(text="S11 — TURBINE OUTLET TEMPERATURE", font=dict(size=14, color="#A0A0A0", family="Montserrat")),
        xaxis=dict(
            title="CYCLE",
            color="#A0A0A0",
            gridcolor="#222222",
            zeroline=False,
        ),
        yaxis=dict(
            title="TEMPERATURE",
            color="#A0A0A0",
            gridcolor="#222222",
            zeroline=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=50, r=20, t=45, b=40),
        showlegend=False,
        font={"family": "Montserrat, sans-serif"}
    )
    return fig


def _row_to_sensor_dict(row: pd.Series) -> dict:
    """Convert a CMAPSS DataFrame row to the dict expected by VectorInference.

    Maps CMAPSS column names to the engine's expected keys:
      s21 → P30  (HPC outlet pressure)
      s9  → T30  (HPC outlet temperature)
    All other columns are passed through as-is.
    """
    d = row.to_dict()
    
    # Ensure all 21 sensors are captured with a default nominal value
    for i in range(1, 22):
        key = f"s{i}"
        if key not in d:
            d[key] = 1500.0

    # Map canonical CMAPSS sensor names → engine convention
    if "s21" in d:
        d["P30"] = d["s21"]
    if "s9" in d:
        d["T30"] = d["s9"]
    return d


# ────────────────────────────────────────────────────────────────────────────
# Main dashboard body
# ────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    df = load_cmapss(uploaded_file)

    with st.sidebar:
        units = sorted(df["unit"].unique()) if "unit" in df.columns else [1]
        selected_unit = st.selectbox("Select Engine Unit", units)

    unit_df = df[df["unit"] == selected_unit].reset_index(drop=True) if "unit" in df.columns else df.copy()

    # ── Top metrics row ──────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    total_cycles = int(unit_df["cycle"].max()) if "cycle" in unit_df.columns else len(unit_df)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="label">Total Cycles</div>'
            f'<div class="value">{total_cycles}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Sensors Active</div>'
            f'<div class="value">21</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="label">Engine Unit</div>'
            f'<div class="value">#{selected_unit}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Layout: Gauge + S11 chart ────────────────────────────────────
    col_gauge, col_chart = st.columns([1, 2])

    with col_gauge:
        gauge_placeholder = st.empty()
    with col_chart:
        chart_placeholder = st.empty()

    # ── Alert Center ─────────────────────────────────────────────────
    st.markdown("### ALERT CENTER")
    alert_placeholder = st.empty()

    # Default state (before simulation)
    alert_placeholder.markdown(
        '<div class="alert-box alert-safe"><span style="color:#FFFFFF;">[ STANDBY ]</span> AWAITING TELEMETRY STREAM...</div>',
        unsafe_allow_html=True,
    )
    gauge_placeholder.plotly_chart(build_gauge(total_cycles), use_container_width=True, key="gauge_init")

    # ── Simulation loop ──────────────────────────────────────────────
    if auto_run:
        if engine is None:
            st.error(
                "⚠️ Random Forest model not found at `models/rf_v1.pkl`.  "
                "Place the trained model file in the `/models` directory and restart."
            )
            st.stop()

        cycles_hist: list[int] = []
        s11_hist: list[float] = []
        violation_detected = False

        progress_bar = st.progress(0)

        for idx in range(len(unit_df)):
            row = unit_df.iloc[idx].copy()
            
            # --- Apply Noise ---
            if noise_level > 0:
                noise_factor = noise_level / 100.0
                for col in row.index:
                    if str(col).startswith('s') or col in ["op1", "op2", "op3", "P30", "T30"]:
                        try:
                            val = float(row[col])
                            row[col] = val * (1.0 + np.random.normal(0, noise_factor))
                        except (ValueError, TypeError):
                            pass

            cycle_num = int(row.get("cycle", idx + 1))
            s11_val = float(row.get("s11", 0))

            # --- Build sensor dict & run the VectorInference engine ---
            row_dict = _row_to_sensor_dict(row)
            rul = engine.predict_rul(row_dict)
            is_valid, msg = engine.physics_guardrail(row_dict)

            # --- Determine safety status ---
            rul_ok = rul > 30
            if rul_ok and is_valid:
                safety = "SAFE"
            elif rul_ok or is_valid:
                safety = "WARNING"
            else:
                safety = "CRITICAL"

            cycles_hist.append(cycle_num)
            s11_hist.append(s11_val)

            # Update gauge
            gauge_placeholder.plotly_chart(
                build_gauge(rul), use_container_width=True, key=f"gauge_{idx}"
            )

            # Update S11 chart
            plot_cycles = cycles_hist[-buffer_size:]
            plot_s11 = s11_hist[-buffer_size:]
            chart_placeholder.plotly_chart(
                build_s11_chart(plot_cycles, plot_s11), use_container_width=True, key=f"s11_{idx}"
            )

            # Update Alert Center (driven by engine's safety status)
            if safety == "CRITICAL":
                violation_detected = True
                alert_placeholder.markdown(
                    f'<div class="alert-box alert-critical">'
                    f"<span style='color:#FF0000;'>[ CRITICAL ]</span> CYCLE {cycle_num}<br>"
                    f"<div style='font-size:0.85rem; color:#888888; margin-top:0.3rem;'>RUL: {rul:.0f} CYCLES | {msg}</div></div>",
                    unsafe_allow_html=True,
                )
            elif safety == "WARNING":
                violation_detected = violation_detected or not is_valid
                alert_placeholder.markdown(
                    f'<div class="alert-box alert-warning">'
                    f"<span style='color:#D4AF37;'>[ WARNING ]</span> CYCLE {cycle_num}<br>"
                    f"<div style='font-size:0.85rem; color:#888888; margin-top:0.3rem;'>RUL: {rul:.0f} CYCLES | {msg}</div></div>",
                    unsafe_allow_html=True,
                )
            elif not violation_detected:
                alert_placeholder.markdown(
                    f'<div class="alert-box alert-safe">'
                    f"<span style='color:#FFFFFF;'>[ NOMINAL ]</span> CYCLE {cycle_num} &nbsp;&mdash;&nbsp; ALL READINGS IN TOLERANCE<br>"
                    f"<div style='font-size:0.85rem; color:#888888; margin-top:0.3rem;'>RUL: {rul:.0f} CYCLES | STATUS: {safety}</div></div>",
                    unsafe_allow_html=True,
                )

            progress_bar.progress((idx + 1) / len(unit_df))
            time.sleep(sim_speed)

        progress_bar.empty()
        st.success("✅ Simulation complete.")

else:
    # ── Empty state ──────────────────────────────────────────────────
    st.markdown("")
    empty_left, empty_mid, empty_right = st.columns([1, 2, 1])
    with empty_mid:
        st.markdown(
            """
            <div style="text-align:center; padding:4rem 0;">
                <h3 style="color:#A0A0A0; font-weight:800; letter-spacing:0.05em; text-transform:uppercase;">VXP2 OFFLINE</h3>
                <p style="color:#555555; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                    UPLOAD TELEMETRY LOG TO INITIALIZE DIAGNOSTICS
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
