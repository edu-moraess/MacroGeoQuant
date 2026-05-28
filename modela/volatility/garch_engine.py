from arch import arch_model

import pandas as pd


def fit_garch_x(series):

    model = arch_model(
        series * 100,
        vol="GARCH",
        p=1,
        q=1
    )

    result = model.fit(
        disp="off"
    )

    vol = (
        result
        .conditional_volatility / 100
    )

    return pd.Series(
        vol,
        index=series.index
    )
