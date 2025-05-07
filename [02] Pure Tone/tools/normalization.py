import numpy as np


def min_normalized_response_per_neuron(response_data):
    final = {}

    for neuron, data in response_data.items():
        final[neuron] = data - np.min(data)

    return final


def min_normalized_response_per_trigger(response_data, n_stim=360):
    final = {}

    for neuron, data in response_data.items():
        normalized_data = []
        window_size = data.shape[0] / n_stim

        for i in range(window_size):
            normalized_data.extend(data[i:i + window_size] - np.min(data[i:i + window_size]))

        final[neuron] = np.array(normalized_data)

    return final


def standardize_response(response_data):
    """
    Standardize te response data by subtracting the mean and dividing by the standard deviation.
    The mean and standard deviation are calculated for the entirety of each neuron.
    :return:
    """
    final = {}

    for neuron, data in response_data.items():
        final[neuron] = (data - np.mean(data)) / np.std(data)

    return final
