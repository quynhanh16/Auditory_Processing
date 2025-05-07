from typing import Callable

import numpy as np


# Data was recorded 10Khz
# Each data point is 10 ms apart.

def _firing_rate_window(spike_data: np.ndarray, window_size: int,
                        kernel: Callable[[np.ndarray], np.ndarray] | None) -> np.ndarray:
    """
    Computes firing rate in fixed non-overlapping windows over spike data.

    The spike data is divided into consecutive windows of size `window_size`, and the number of spikes
    in each window is normalized by time (assuming a sampling rate of 10,000 Hz). An optional kernel can
    be applied to the resulting firing rate array.

    :param spike_data: 1D NumPy array of binary spike data.
    :param window_size: Number of samples per window for rate calculation.
    :param kernel: Optional function to transform the output firing rate (e.g., smoothing).
    :return: NumPy array of firing rates computed over each window.
    """

    final = []

    for i in range(0, len(spike_data), window_size):
        interval = spike_data[i:i + window_size]
        final.append(sum(interval) / (len(interval) / 10000))

    final = np.array(final)
    return kernel(final) if kernel is not None else final


def _firing_rate_sliding_window(spike_data: np.ndarray, window_size: int, gap: int,
                                kernel: Callable[[np.ndarray], np.ndarray] | None) -> np.ndarray:
    """
    Computes firing rate using a sliding window over the spike data.

    The function moves a fixed-size window across the spike train in steps of `gap`, calculating the number
    of spikes per window and normalizing by time (assuming 10,000 samples per second). An optional kernel
    can be applied to the resulting rate series.

    :param spike_data: 1D NumPy array of binary spike data.
    :param window_size: Number of samples in each window.
    :param gap: Step size in samples to slide the window.
    :param kernel: Optional function to transform the firing rate output (e.g., smoothing).
    :return: NumPy array of firing rates computed at each sliding window step.
    """

    final = []

    for i in range(0, len(spike_data) - window_size, gap):
        interval = spike_data[i:i + window_size]
        final.append(sum(interval) / (len(interval) / 10000))

    final = np.array(final)
    return kernel(final) if kernel is not None else final


def _population_firing_rate(spike_data: dict[str, np.ndarray], window_size: int, sliding_window: bool,
                            gap: int | None, kernel: Callable[[np.ndarray], np.ndarray] | None) -> np.ndarray:
    """
    Computes the average firing rate across a population of neurons.

    For each neuron in the spike_data dictionary, the function calculates the firing rate using either a fixed or
    sliding window method. Optionally, a kernel can be applied to each spike train before computation.

    :param spike_data: Dictionary mapping neuron names to 1D NumPy arrays of binary spike data.
    :param window_size: Number of time points in each window for firing rate calculation.
    :param sliding_window: If True, uses a sliding window with step size defined by `gap`.
    :param gap: Number of time points to move the sliding window. Required if `sliding_window` is True.
    :param kernel: Optional function to apply a transformation (e.g., smoothing) to the spike data.
    :return: A NumPy array representing the mean firing rate across all neurons over time.
    """

    final = []
    for data in spike_data.values():
        if sliding_window:
            final.append(_firing_rate_sliding_window(data, window_size, gap, kernel))
        else:
            final.append(_firing_rate_window(data, window_size, kernel))

    return np.mean(np.array(final), axis=0)


def firing_rate(spike_data: dict[str, np.ndarray] | np.ndarray, window_size: int, sliding_window: bool = False,
                gap: int | None = None, population: bool = False,
                kernel: Callable[[np.ndarray], np.ndarray] | None = None):
    """
    Calculates the firing rate of neuron(s) based on spike data over a fixed or sliding time window.

    This function supports both single-neuron and population-level firing rate calculations. For a single neuron,
    the spike data is provided as a NumPy array. For population analysis, a dictionary mapping neuron names to
    spike arrays is used. The firing rate is computed as the number of spikes in a given window divided by the
    window size (in samples). Optionally, a kernel can be applied to smooth or filter the spike signal.
    If `sliding_window` is enabled, the function computes firing rate over overlapping windows, advancing by `gap`
    steps each time.

    :param spike_data: Either a NumPy array of spike data (1D array of binary spikes) for a single neuron, or a
                       dictionary mapping neuron names to such arrays for population-level analysis.
    :param window_size: The number of time points over which to compute the firing rate.
    :param sliding_window: If True, computes the firing rate using a sliding window approach.
    :param gap: Number of time points to shift the window in the sliding window approach. Required if
                `sliding_window` is True.
    :param population: If True, treats `spike_data` as a dictionary and computes the average firing rate across
                       all neurons in the population.
    :param kernel: Optional function to apply a kernel (e.g., Gaussian or smoothing filter) to each neuron's
                   spike data before computing firing rate. The kernel should take and return a NumPy array.
    :return: A NumPy array representing the firing rate(s) over time. For single-neuron input, returns the firing
             rate for that neuron. For population input, returns the mean firing rate across neurons.
    """

    if population:
        if isinstance(spike_data, dict):
            return _population_firing_rate(spike_data, window_size, sliding_window, gap, kernel)
        else:
            raise Exception("Data type not supported.")

    if sliding_window:
        return _firing_rate_sliding_window(spike_data, window_size, gap, kernel)
    else:
        return _firing_rate_window(spike_data, window_size, kernel)
