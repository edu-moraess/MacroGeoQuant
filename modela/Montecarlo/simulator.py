import numpy as np


def run_monte_carlo(
    spot,
    vol,
    sims=5000,
    steps=30,
    seed=42
):

    np.random.seed(seed)

    dt = 1 / 252

    paths = np.zeros(
        (steps + 1, sims)
    )

    paths[0] = spot

    for t in range(
        1,
        steps + 1
    ):

        shock = np.random.normal(
            0,
            1,
            sims
        )

        paths[t] = (
            paths[t - 1]
            * np.exp(
                -0.5 * vol**2 * dt
                + vol * np.sqrt(dt) * shock
            )
        )

    fan = {

        5: np.percentile(
            paths,
            5,
            axis=1
        ),

        50: np.percentile(
            paths,
            50,
            axis=1
        ),

        95: np.percentile(
            paths,
            95,
            axis=1
        )
    }

    return {
        "paths": paths,
        "fan": fan
    }
