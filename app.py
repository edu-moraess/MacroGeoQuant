import sys
import os

# =========================
# PATH FIX (Streamlit Cloud safe)
# =========================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
sys.path.append(PARENT_DIR)

import streamlit as st

from utils.constants import TICKERS
from utils.config import GUERRA_START, SIMS, STEPS, MC_SEED

from services.yahoo_service import download_market_data

from models.volatility.garch_engine import fit_garch_x
from models.montecarlo.simulator import run_monte_carlo


# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="MacroGeoQuant",
    layout="wide"
)

st.title("📊 Macro Geopolitical Quant Platform")
st.markdown("---")


# =========================
# DATA
# =========================
with st.spinner("Downloading market data..."):
    prices = download_market_data(TICKERS, GUERRA_START)

if prices is None or prices.empty:
    st.error("Failed to load market data.")
    st.stop()

st.subheader("Market Prices")
st.dataframe(prices.tail())


# =========================
# RETURNS
# =========================
returns = prices.pct_change().dropna()

if "oil" not in returns.columns:
    st.error("Column 'oil' not found in dataset.")
    st.stop()


# =========================
# GARCH VOLATILITY
# =========================
with st.spinner("Fitting GARCH model..."):
    vol = fit_garch_x(returns["oil"])

if vol is None or len(vol) == 0:
    st.error("GARCH model failed to compute volatility.")
    st.stop()

st.subheader("WTI Conditional Volatility")
st.line_chart(vol)


# =========================
# MONTE CARLO
# =========================
if "oil" not in prices.columns:
    st.error("Oil price column missing.")
    st.stop()

with st.spinner("Running Monte Carlo simulation..."):
    mc = run_monte_carlo(
        spot=float(prices["oil"].iloc[-1]),
        vol=float(vol.iloc[-1]),
        sims=SIMS,
        steps=STEPS,
        seed=MC_SEED
    )

fan = mc["fan"]

st.subheader("Monte Carlo Fan Chart")

mc_df = {
    "P5": fan[5],
    "P50": fan[50],
    "P95": fan[95]
}

st.line_chart(mc_df)


# =========================
# METRICS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric(
    "WTI Spot",
    f"${prices['oil'].iloc[-1]:.2f}"
)

col2.metric(
    "Volatility",
    f"{vol.iloc[-1] * 100:.2f}%"
)

col3.metric(
    "Monte Carlo P50",
    f"${fan[50][-1]:.2f}"
)

st.markdown("---")

st.success("MacroGeoQuant loaded successfully 🚀")
