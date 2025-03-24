import pickle
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


def get_response_data_all_triggers(file_path: str, trigger_length: int = 3000) -> np.ndarray:
    """
    Given a path to a .mat file, return the nth trigger of specified length.
    For this dataset, the trigger length is set to 3000.

    :param file_path:
    :param trigger_length:
    :return: np.ndarray
    """
    file = loadmat(file_path)
    response_signals = file['response_signal']
    triggers = file['trigger'][0]

    result = response_signals[triggers[0]: triggers[0] + trigger_length]
    for trigger in triggers[1:]:
        result = np.concatenate((result, response_signals[trigger: trigger + trigger_length]))

    return result


def get_response_data_by_trigger(file_path: str, n_trigger: int, trigger_length: int = 3000) -> np.ndarray:
    """
    Given a path to a .mat file, return the nth trigger of specified length.
    For this dataset, the trigger length is set to 3000.

    :param file_path:
    :param n_trigger:
    :param trigger_length:
    :return: np.ndarray
    """
    file = loadmat(file_path)
    response_signals = file['response_signal']
    triggers = file['trigger'][0]

    trigger = triggers[n_trigger - 1]

    return response_signals[trigger: trigger + trigger_length].T[0]


def get_response_data_by_trigger_spikes(file_path: str, n_trigger: int, trigger_length: int = 3000) -> np.ndarray:
    """
    Given a path to a .mat file, return the nth trigger of specified length.
    For this dataset, the trigger length is set to 3000.

    :param file_path:
    :param n_trigger:
    :param trigger_length:
    :return: np.ndarray
    """
    file = loadmat(file_path)
    response_signals = file['binarySpike']
    triggers = file['trigger'][0]

    trigger = triggers[n_trigger - 1]

    return response_signals[trigger: trigger + trigger_length].T[0]


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
