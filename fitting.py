# File: computing.py
# Purpose: Fitting Models for the Recording Data

# Packages
from typing import Any

import joblib
import numpy as np
# NEMS Packages
from nems.tools.signal import RasterizedSignal
from sklearn.linear_model import Ridge
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
    prepare_response,
)


# TODO: Create Linear Non-Linear model


def simple_linear_model(
        stim_signal: RasterizedSignal,
        resp_signal: RasterizedSignal,
        m: int,
        d: int,
        save: bool = True,
        display: bool = True,
        **kwargs,
) -> Any:
    # First 27 seconds is data validation
    interval = (27, stim_signal.shape[1] / 100)
    # interval = (27, 27.5)

    print("Preparing data")
    print("Preparing x")
    stim_state_file = f"stim_m{m}_d{d}_t{interval}.pkl"
    stim_state = load_state(stim_state_file)
    if stim_state is None:
        X = prepare_stimuli(stim_signal, interval, m, d)
        save_state(stim_state_file, X)
    else:
        X = load_state(stim_state_file)

    print(X.shape)
    print("Preparing y")
    resp_state_file = f"resp_m{m}_t{interval}.pkl"
    resp_state = load_state(resp_state_file)
    if resp_state is None:
        y = prepare_response(resp_signal, interval, d)
        save_state(resp_state_file, y)
    else:
        y = load_state(resp_state_file)
    print(y.shape)

    print("Using function")
    if "function" in kwargs:
        y = kwargs["function"](y, interval)
    else:
        y = np.mean(y, axis=0)

    print("Fitting Model")
    alpha = 0.00001
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    joblib.dump(model, "nr_linear_model.pkl")
    coefficients = model.coef_
    intercepts = np.array([model.intercept_])
    print("Getting Statistics")
    r2_score = model.score(X, y)
    mae = mean_absolute_error(y, model.predict(X))
    mse = mean_squared_error(y, model.predict(X))

    if save:
        results = RecordingData(
            coefficients,
            intercepts,
            d,
            m,
            "L2",
            alpha,
            r2_score,
            mae,
            mse,
            interval,
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
        save_state(state_file, (stim, resp))
    else:
        stim, resp = load_state(state_file)

    b = simple_linear_model(stim, resp, 18, 20, True, False)
