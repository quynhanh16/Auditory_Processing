# File: computing.py
# Purpose: Calculations of the recordings

# Packages
import math
from typing import Tuple

import numpy as np
# NEMS Packages
from nems.tools.signal import RasterizedSignal

# Tools
from tools import load_state, save_state, load_datafile, splitting_recording


# TODO: Write unit tests


def unit_mean_baseline_activity(unit) -> float:
    result = 0
    n = 0

    for i in range(0, len(unit), 150):
        for j in range(20):
            result += unit[j + i]
            n += 1

        for j in range(20):
            result += unit[j + i + 129]
            n += 1

    return result / n


def unit_std_baseline_activity(unit, mean) -> float:
    result = 0

    for i in range(0, len(unit), 150):
        for j in range(20):
            result += (unit[j + i] - mean) ** 2

        for j in range(20):
            result += (unit[j + i + 129] - mean) ** 2

    n = len(unit) / 150 * 40

    return math.sqrt(result / (n - 1))


# Stimulus-evoked firing rates per unit
def evoked_firing_rate(data: np.array, t: float, **kwargs) -> float | np.array(float):
    if "mean" in kwargs:
        baseline_activity_mean = kwargs["mean"]
    else:
        baseline_activity_mean = unit_mean_baseline_activity(data)

    if "std" in kwargs:
        baseline_activity_std = kwargs["std"]
    else:
        baseline_activity_std = unit_std_baseline_activity(data, baseline_activity_mean)

    result = (data[t] - baseline_activity_mean) / baseline_activity_std

    return result


def population_evoked_firing_rate(
        stim_signal: RasterizedSignal, t: float | Tuple[float, float]
) -> float | np.array(float) | None:
    if isinstance(t, float):
        if t * 100 < 0 or t * 100 > stim_signal.shape[1]:
            raise IndexError(
                "Invalid t: {}. Size of data: {}".format(t, stim_signal.shape[1])
            )

        result = 0

        for unit in range(0, stim_signal.shape[0]):
            data = stim_signal[unit]
            mean = unit_mean_baseline_activity(data)
            std = unit_std_baseline_activity(data, mean)
            result += evoked_firing_rate(data, t * 100, mean=mean, std=std)

        result /= stim_signal.shape[0]
        return result
    elif isinstance(t, tuple):
        a = int(t[0] * 100)
        b = int(t[1] * 100)

        if a < 0 or b > stim_signal.shape[1]:
            raise IndexError(
                "Invalid t: {}. Size of data: {}".format(t, stim_signal.shape[1])
            )

        result = np.zeros(b - a)

        for unit in range(0, stim_signal.shape[0]):
            data = stim_signal[unit]
            mean = unit_mean_baseline_activity(data)
            std = unit_std_baseline_activity(data, mean)
            for idx in range(len(result)):
                result[idx] += evoked_firing_rate(data, idx + a, mean=mean, std=std)

        result /= stim_signal.shape[0]

        return result

    return None


def population_spike_rate(
        resp_signal: RasterizedSignal, t: float | Tuple[float, float]
) -> float | np.array(float) | None:
    if isinstance(t, float):
        if t * 100 < 0 or t * 100 > resp_signal.shape[1]:
            raise IndexError(
                "Invalid t: {}. Size of data: {}".format(t, resp_signal.shape[1])
            )

        result = 0

        for unit in range(0, resp_signal.shape[0]):
            result += resp_signal[unit][t * 100]

        result /= resp_signal.shape[0]

        return result
    elif isinstance(t, tuple):
        a = int(t[0] * 100)
        b = int(t[1] * 100)

        if a < 0 or b > resp_signal.shape[1]:
            raise IndexError(
                "Invalid t: {}. Size of data: {}".format(t, resp_signal.shape[1])
            )

        result = np.zeros(b - a)

        for unit in range(0, resp_signal.shape[0]):
            result += resp_signal[unit][a:b]

        result /= resp_signal.shape[0]

        return result

    return None


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

    # ab = population_spike_rate(resp, (0, 1.5))
    # ba = population_evoked_firing_rate(resp, (0, 1.5))
