import pickle
from typing import Any

import h5py
import numpy as np
import os
from scipy.io import loadmat


def load_response_information(path: str, spike: int | None) -> (np.ndarray, np.ndarray):
    # Loading trigger and response data for raw data.
    if spike is None:
        file = os.path.join(path, os.listdir(path)[0])
        mat_file = loadmat(file)
        triggers = mat_file["trigger"][0]
        response = mat_file["response_signal"].T[0]
    # Loading trigger and response data for spike data.
    else:
        file = ""
        for f in os.listdir(path):
            if f"spike{spike}" in f:
                file = os.path.join(path, f)
                break
        mat_file = loadmat(file)
        triggers = mat_file["trigger"][0]
        if "binarySpike" in mat_file:
            response = mat_file["binarySpike"].T[0]
        else:
            raise Exception("File does not contain binary spikes")

    return triggers, response


def load_data(path: str = "./data/pure_tones", stim_path: str = "./data/Stimulus/PureToneParameters.mat",
              stim_window: int = 3000, spike: int | None = None) -> (dict[str, np.ndarray], np.ndarray):
    """
    Output: Dictionary: Neuron -> Response Data

    :param path:
    :param stim_path:
    :param stim_window:
    :param spike:
    :return:
    """
    # Loading stimulus file
    stim = loadmat(stim_path)
    order = stim["stimulus_presentation_order"][:, 2:4]
    stim_size = order.shape
    final = {}

    # New: Neuron (date): Response data
    for neuron in os.listdir(path):
        neuron_data = []

        # Loading neuron data
        files = os.path.join(path, neuron)
        triggers, response = load_response_information(files, spike)

        # Collecting all response data (from the start to the stimuli to the next 3000 data points).
        for idx in range(stim_size[0]):
            int_data = response[triggers[idx]: triggers[idx] + stim_window].tolist()
            neuron_data.extend(int_data)

        final[neuron] = np.array(neuron_data)

    return final, order


def load_state(filename: str) -> Any:
    """
    Load an object from a pickle file with name *filename*.

    :param filename:
    :return: Any
    """
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def process_data(filename: str, process_data_function):
    """
    Load a pickle file given a *filename*. If the file does not exist, run a given function to process data and save it
    to a file with the given *filename*. Return the processed data.

    :param filename:
    :param process_data_function:
    :return:
    """
    dataset = load_state(filename)
    if dataset is None:
        dataset = process_data_function()
        save_state(filename, dataset)

    return dataset


def save_recording_data_to_h5file(h5file: h5py.File, recording_data: dict[Any, dict]) -> None:
    """
    The function takes a dictionary with the structures of frequencies -> amps -> 2D array [neurons x time data].
    For the data used in this project, the structure is 45 frequencies -> 8 amps -> [66 neurons x 3000 time data].

    :param h5file:
    :param recording_data:
    :return:
    """
    for freq in recording_data.keys():
        h5file.create_group(str(freq))

    for freq, amps in recording_data.items():
        for amp, data in amps.items():
            h5file.create_dataset(f"{freq}/{amp}", data=data)


def save_state(filename: str, data: Any) -> None:
    """
    Save an object as a pickle file with name *filename*.

    :param filename:
    :param data:
    :return: None
    """
    with open(filename, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def response_data_by_order(data, order, n_order: int = 360):
    """
    Output: Convert a dictionary of Neuron -> Response Data to a dictionary of Neuron -> Frequency -> Amplitude ->
    Response Data.

    :param data:
    :param order:
    :param n_order:
    :return:
    """
    final = {}

    for neuron, data in data.items():
        j = 0
        window_size = data.shape / n_order
        final[neuron] = {}

        for freq, amp in order:
            if freq not in final[neuron]:
                final[neuron][freq] = {}

            if amp not in final[neuron][freq]:
                final[neuron][freq][amp] = []

            final[neuron][freq][amp] = data[j:j + window_size]

            j += window_size

    return final
