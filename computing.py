# File: computing.py
# Purpose: Calculations of the recordings

# Packages
from functools import reduce
from typing import List, Tuple

import numpy as np
# NEMS Packages
from nems.tools.signal import RasterizedSignal

# Tools
from tools import load_state, save_state, load_datafile, splitting_recording


# QUESTION: Return list of floats or a numpy array


def unit_mean_baseline_activity(unit) -> float:
    # NOTE: Faster than combining the first 20 sec and last 20 sec and adding them at the end
    intervals = [unit[i: i + 150] for i in range(0, len(unit), 150)]

    # 0.01628902941296821
    unit_mean: float = reduce(
        lambda x, y: x + sum(y[:21]) + sum(y[130:]), intervals, 0
    ) / len(intervals)

    return unit_mean


# QUESTION: Are stimuli independent of each other?
def unit_std_baseline_activity(unit, mean) -> float:
    intervals = [
        np.hstack((unit[i: i + 20], unit[(i + 150 - 20): i + 150]))
        for i in range(0, len(unit), 150)
    ]

    unit_std: float = reduce(lambda x, y: x + sum(y), intervals, 0) / len(intervals)

    return 1


# QUESTION: Is this function needed?
def units_mean_baseline_activity(data) -> float:
    units_mean = (
            reduce(lambda x, y: x + unit_mean_baseline_activity(y), data, 0) / data.shape[0]
    )

    return units_mean


# QUESTION: Are units independent of each other?
#           Is this function needed?
def units_std_baseline_activity(data, mean) -> float:
    pass


# Stimulus-evoked firing rates per unit
def evoked_firing_rate(data: np.array, t: float, **kwargs) -> float | list[float]:
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


# QUESTION: Make this parallel?
def population_evoked_firing_rate(
        stim_signal: RasterizedSignal, t: float | Tuple[float, float]
) -> float | List[float] | None:
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

        return result.tolist()

    return None


# QUESTION: Make this parallel?
# FIXME: Not returning the last 100 records
def population_spike_rate(
        resp_signal: RasterizedSignal, t: float | Tuple[float, float]
) -> float | List[float] | None:
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

        return result.tolist()

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

    ab = population_spike_rate(resp, (0, 1.5))
    ba = population_evoked_firing_rate(stim, (0, 1.5))
