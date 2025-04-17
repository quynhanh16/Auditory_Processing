import numpy as np

def _firing_rate_window(spike_data, window_size):
    final = []

    for i in range(0, len(spike_data), window_size):
        interval = spike_data[i:i + window_size]
        final.append(sum(interval) / len(interval))

    return final


def _firing_rate_sliding_window(spike_data, window_size, gap):
    final = []

    for i in range(0, len(spike_data) - window_size, gap):
        interval = spike_data[i:i + window_size]
        final.append(sum(interval) / len(interval))

    return final


def _population_firing_rate(spike_data, window_size, sliding_window, gap):
    final = []
    for data in spike_data.values():
        if sliding_window:
            final.append(_firing_rate_sliding_window(data, window_size, gap))
        else:
            final.append(_firing_rate_window(data, window_size))

    print("Single:", len(final[0]))
    return np.sum(final, axis=0).tolist()


def firing_rate(spike_data, window_size, sliding_window=False, gap=None, population=False):
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
            return _population_firing_rate(spike_data, window_size, sliding_window, gap)
        else:
            raise Exception("Data type not supported.")

    if sliding_window:
        return _firing_rate_sliding_window(spike_data, window_size, gap)
    else:
        return _firing_rate_window(spike_data, window_size)