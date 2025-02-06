# File: computing.py
# Purpose: Fitting Models for the Recording Data

# Packages
from typing import Any
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# NEMS Packages
from tools.signal import RasterizedSignal
from scipy.optimize import least_squares

# Computing
from tools.computing import population_spike_rate

# Tools
from tools.utils import (
    load_state,
    save_state,
    load_datafile,
    splitting_recording,
    RecordingData,
    save_results_txt,
    prepare_stimuli,
    prepare_response,
)


def simple_linear_model(
        stim_signal: RasterizedSignal,
        resp_signal: RasterizedSignal,
        m: int,
        d: int,
        save: bool = True,
        **kwargs,
) -> Any:
    """
    Create a simple linear regression model given stimuli and response signal.
    Then, save results to txt file.

    :param stim_signal: Stimuli signal
    :param resp_signal: Response signal
    :param m: Number of channels
    :param d: Number of previous stimuli
    :param save: Save results to txt file
    :param kwargs: Function to compute the response data. Default: population spike rate.
    :return: Any
    """

    # First 27 seconds is data validation
    interval = (27, stim_signal.shape[1] / 100)

    print("Preparing data")
    print("Preparing x")

    # Save prepared stimuli for later use.
    stim_state_file = f"stim_m{m}_d{d}_t{interval}.pkl"
    stim_state = load_state(stim_state_file)
    if stim_state is None:
        X = prepare_stimuli(stim_signal, interval, m, d)
        save_state(stim_state_file, X)
    else:
        X = load_state(stim_state_file)

    print(X.shape)
    print("Preparing y")

    # Save prepared response for later use.
    resp_state_file = f"resp_m{m}_t{interval}.pkl"
    resp_state = load_state(resp_state_file)
    if resp_state is None:
        y = prepare_response(resp_signal, interval, d)
        save_state(resp_state_file, y)
    else:
        y = load_state(resp_state_file)
    print(y.shape)

    # Calculating population spike rate of response.
    print("Using function")
    if "function" in kwargs:
        y = kwargs["function"](y, interval)
    else:
        y = np.mean(y, axis=0)

    print("Fitting Model")
    # Alpha for models with regularization
    alpha = 0.00001
    # Simple Linear Regression model.
    model = LinearRegression()
    # Simple Linear Regression model with regularization.
    # model = Ridge(alpha=alpha)
    # Fit model.
    model.fit(X, y)
    # Save model to a pkl file.
    joblib.dump(model, "../models/nr_linear_model.pkl")
    coefficients = model.coef_
    intercepts = np.array([model.intercept_])

    # Getting basic statistics of model performance.
    print("Getting Statistics")
    r2_score = model.score(X, y)
    mae = mean_absolute_error(y, model.predict(X))
    mse = mean_squared_error(y, model.predict(X))

    # Save coefficients and results of linear regression model to txt file.
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
        save_results_txt(results)

    return model


def sigmoid(params, res) -> float:
    """
    Sigmoid Transformation

    :param params:
    :param res:
    :return: float
    """
    a, b, c = params
    return a / (1 + np.exp(b * (c - res)))


def fl(params, res) -> float:
    """
    Double Exponential Transformation

    :param params:
    :param res:
    :return: float
    """
    a, b, c, s = params
    return b + a * np.exp(-np.exp(c * (res - s)))


def hyperbolic_tan(param, res) -> float:
    """
    Hyperbolic Tangential Transformation.

    :param param:
    :param res:
    :return: float
    """
    a, b, c = param
    result = a * np.tanh(b * (res - c))
    return result * (result > 0)


def residuals(params, x, r) -> float:
    """
    Calculate residuals of given r and x.

    :param params:
    :param x:
    :param r:
    :return: float
    """
    return r - hyperbolic_tan(params, x)


def non_linear(model, s, r):
    """
    Create a linear-nonlinear regression model given stimuli and response signal.

    :param model:
    :param s:
    :param r:
    :return:
    """
    # Calculating population spiking rate of the response data.
    r = np.mean(r, axis=0)

    # Choosing initial values for trainable parameters.
    theta = [np.max(r), 0.1, 0.1]
    # theta = [0.5, 0.5, 0.5, 0.5]

    # Lineal predictions of the model.
    predictions = model.predict(s)

    # Train model.
    nl_model = least_squares(residuals, theta, args=(predictions, r))
    import matplotlib.pyplot as plt

    print(nl_model.x)
    y = hyperbolic_tan(nl_model.x, predictions)
    # Print linear predictions against new model predictions.
    plt.scatter(predictions, y)
    plt.show()

    # Print basic statistics of new model against lineal results.
    print(nl_model)
    print(r2_score(r, predictions))
    print(r2_score(r, y))


if __name__ == "__main__":
    tgz_file: str = "../A1_NAT4_ozgf.fs100.ch18.tgz"

    state_file = "../../state.pkl"
    state = load_state(state_file)
    if state is None:
        rec = load_datafile(tgz_file, True)
        stim, resp = splitting_recording(rec, True)
        save_state(state_file, (stim, resp))
    else:
        stim, resp = load_state(state_file)

    interval = (27, stim.shape[1] / 100)
    m, d = 18, 20

    stim_state_file = f"stim_m{m}_d{d}_t{interval}.pkl"
    stim_state = load_state(stim_state_file)
    X = load_state(stim_state_file)

    resp_state_file = f"resp_m{m}_t{interval}.pkl"
    resp_state = load_state(resp_state_file)
    y = load_state(resp_state_file)

    b = simple_linear_model(stim, resp, 18, 20, False)
    # non_linear(joblib.load("nr_linear_model.pkl"), X, y)
