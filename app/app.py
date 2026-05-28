import streamlit as st

from utils.constants import TICKERS

from utils.config import (
    GUERRA_START
)

from services.yahoo_service import (
    download_market_data
)

from models.volatility.garch_engine import (
    fit_garch_x
)

from models.montecarlo.simulator import (
    run_monte_carlo
)

st.set_page_config(
    page_title="Macro GeoQuant",
    layout="wide"
)

st.title(
    "Macro Geopolitical Quant Platform"
)

prices = download_market_data(
    TICKERS,
    GUERRA_START
)

returns = (
    prices
    .pct_change()
    .dropna()
)

vol = fit_garch_x(
    returns["oil"]
)

mc = run_monte_carlo(
    spot=float(
        prices["oil"].iloc[-1]
    ),
    vol=float(
        vol.iloc[-1]
    )
)

st.write(mc["fan"])
