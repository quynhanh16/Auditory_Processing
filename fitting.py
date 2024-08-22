# File: computing.py
# Purpose: Fitting Models for the Recording Data

# Packages
from typing import Tuple

import numpy as np
# NEMS Packages
from nems.tools.signal import RasterizedSignal

# Computing
from computing import population_evoked_firing_rate, population_spike_rate
# Tools
from tools import load_state, save_state, load_datafile, splitting_recording


def simple_linear_model(
        stim_signal: RasterizedSignal, resp_signal: RasterizedSignal, d: int
) -> Tuple[float, float]:
    # b = (x'x)^-1(x'y)
    # y: response
    # x: stimuli
    print(stim_signal.shape, resp_signal.shape)
    interval = (1, stim_signal.shape[1] / 100)

    # [0.00161091 0.07375191]

    y = np.array(population_spike_rate(resp_signal, interval)).reshape(-1, 1)
    x = np.array(population_evoked_firing_rate(stim_signal, interval)).reshape(-1, 1)
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    print(x.shape, y.shape)

    x_t = x.T
    x_tx = np.matmul(x_t, x)
    inv = np.linalg.pinv(x_tx)
    x_ty = np.matmul(x_t, y)
    b = np.matmul(inv, x_ty)

    return b.flatten()


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

    b = simple_linear_model(stim, resp, 0)
    print(b)
