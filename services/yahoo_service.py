import yfinance as yf


def download_market_data(
    tickers,
    start
):

    prices = yf.download(
        list(tickers.values()),
        start=start,
        progress=False
    )["Close"]

    prices.columns = list(
        tickers.keys()
    )

    prices = (
        prices
        .ffill()
        .dropna()
    )

    return prices
