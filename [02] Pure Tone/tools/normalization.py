import numpy as np


def min_normalized_response_per_neuron(response_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Normalizes response data for each neuron by subtracting the minimum value.

    For each neuron in the input dictionary, the function subtracts the minimum value of its response array,
    shifting the data so the smallest value becomes zero.

    :param response_data: Dictionary mapping neuron names to 1D NumPy arrays of response data.
    :return: Dictionary with the same structure, containing min-normalized response arrays for each neuron.
    """
    final = {}

    for neuron, data in response_data.items():
        final[neuron] = data - np.min(data)

    return final


def min_normalized_response_per_trigger(response_data: dict[str, np.ndarray], n_stim: int = 360) -> dict[str, np.ndarray]:
    """
    Performs min-normalization of response data per stimulus trigger for each neuron.

    For each neuron, the response array is divided into `n_stim` intervals (one per stimulus).
    Within each interval, the minimum value is subtracted from the response segment,
    effectively shifting each stimulus-aligned response window to start at zero.

    :param response_data: Dictionary mapping neuron names to 1D NumPy arrays of concatenated response data.
    :param n_stim: Number of stimulus presentations in the dataset, used to determine interval size.
    :return: Dictionary with the same structure, containing min-normalized response arrays per stimulus trigger.
    """

    final = {}

    for neuron, data in response_data.items():
        normalized_data = []
        window_size = int(data.shape[0] / n_stim)

        for i in range(0, len(data), window_size):
            interval = data[i:i + window_size]
            normalized_data.extend(interval - np.min(interval))

        final[neuron] = np.array(normalized_data)

    return final


def standardize_response(response_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Standardizes the response data for each neuron by applying z-score normalization.

    For each neuron in the input dictionary, the function subtracts the mean and divides by the standard deviation
    of that neuron's full response array. This transforms the data to have zero mean and unit variance.

    :param response_data: Dictionary mapping neuron names to 1D NumPy arrays of response data.
    :return: A dictionary with the same structure, containing standardized response arrays for each neuron.
    """

    final = {}

    for neuron, data in response_data.items():
        final[neuron] = (data - np.mean(data)) / np.std(data)

    return final
