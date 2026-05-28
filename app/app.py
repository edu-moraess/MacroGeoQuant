import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "models"))
sys.path.append(str(ROOT_DIR / "services"))
sys.path.append(str(ROOT_DIR / "utils"))

import streamlit as st

from constants import TICKERS

from config import (
    GUERRA_START
)

from yahoo_service import (
    download_market_data
)

from volatility.garch_engine import (
    fit_garch_x
)

from montecarlo.simulator import (
    run_monte_carlo
)

st.set_page_config(
    page_title="Macro GeoQuant",
    layout="wide"
)

st.title(
    "Macro Geopolitical Quant Platform"
)

st.markdown("---")

# =========================
# DOWNLOAD MARKET DATA
# =========================

prices = download_market_data(
    TICKERS,
    GUERRA_START
)

st.subheader("Market Prices")

st.dataframe(
    prices.tail()
)

# =========================
# RETURNS
# =========================

returns = (
    prices
    .pct_change()
    .dropna()
)

# =========================
# GARCH VOLATILITY
# =========================

vol = fit_garch_x(
    returns["oil"]
)

st.subheader(
    "WTI Conditional Volatility"
)

st.line_chart(vol)

# =========================
# MONTE CARLO
# =========================

mc = run_monte_carlo(
    spot=float(
        prices["oil"].iloc[-1]
    ),
    vol=float(
        vol.iloc[-1]
    )
)

fan = mc["fan"]

st.subheader(
    "Monte Carlo Fan Chart"
)

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

st.success(
    "MacroGeoQuant successfully loaded."
)
