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
    # First 27 seconds is data validation
    interval = (27, stim_signal.shape[1] / 100)

    y = np.array(population_spike_rate(resp_signal, interval))
    x = np.array(population_evoked_firing_rate(stim_signal, interval))
    print(y)


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
