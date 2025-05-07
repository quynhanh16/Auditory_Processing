import numpy as np

# Data was recorded 10Khz
# Each data point is 10 ms apart.

def _firing_rate_window(spike_data, window_size, kernel):
    final = []

    for i in range(0, len(spike_data), window_size):
        interval = spike_data[i:i + window_size]
        final.append(sum(interval) / (len(interval) / 10000))

    return kernel(final) if kernel is not None else final


def _firing_rate_sliding_window(spike_data, window_size, gap, kernel):
    final = []

    for i in range(0, len(spike_data) - window_size, gap):
        interval = spike_data[i:i + window_size]
        final.append(sum(interval) / (len(interval) / 10000))

    return kernel(final) if kernel is not None else final


def _population_firing_rate(spike_data, window_size, sliding_window, gap, kernel):
    final = []
    for data in spike_data.values():
        if sliding_window:
            final.append(_firing_rate_sliding_window(data, window_size, gap, kernel))
        else:
            final.append(_firing_rate_window(data, window_size, kernel))

    return np.mean(np.array(final), axis=0)


def firing_rate(spike_data, window_size, sliding_window=False, gap=None, population=False, kernel=None):
    """
    Calculate the firing rate of a neuron.

    Parameters:
    spike_times (list): List of spike times in seconds.
    start_time (float): Start time of the interval in seconds.
    end_time (float): End time of the interval in seconds.

    Returns:
    float: Firing rate in Hz.
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