# File: computing.py
# Purpose: Fitting Models for the Recording Data

# Packages
from typing import Any

import numpy as np
# NEMS Packages
from nems.tools.signal import RasterizedSignal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Computing
from computing import population_spike_rate
# Tools
from tools import (
    load_state,
    save_state,
    load_datafile,
    splitting_recording,
    RecordingData,
    save_results,
    prepare_stimuli,
)


# TODO: Create Linear Non-Linear model


def simple_linear_model(
        stim_signal: RasterizedSignal,
        resp_signal: RasterizedSignal,
        m: int,
        d: int,
        save: bool = True,
        display: bool = True,
        **kwargs
) -> Any:
    # First 27 seconds is data validation
    interval = (27, stim_signal.shape[1] / 100)

    X = prepare_stimuli(stim_signal, interval, m, d)
    if "function" in kwargs:
        y = kwargs["function"](resp_signal, interval)
    else:
        y = population_spike_rate(resp_signal, interval)

    model = LinearRegression()
    model.fit(X, y)
    coefficients = model.coef_
    intercepts = np.array([model.intercept_])
    r2_score = model.score(X, y)
    mae = mean_absolute_error(y, model.predict(X))
    mse = mean_squared_error(y, model.predict(X))

    if save:
        results = RecordingData(
            coefficients,
            intercepts,
            d,
            m,
            r2_score,
            mae,
            mse,
            population_spike_rate.__name__,
        )
        save_results(results)

    return model


if __name__ == "__main__":
    tgz_file: str = "A1_NAT4_ozgf.fs100.ch18.tgz"

    state_file = "state.pkl"
    state = load_state(state_file)
    if state is None:
        rec = load_datafile(tgz_file, True)
        stim, resp = splitting_recording(rec, True)
        save_state(state_file, stim, resp)
    else:
        stim, resp = load_state(state_file)

    b = simple_linear_model(stim, resp, 18, 5, True, False)
