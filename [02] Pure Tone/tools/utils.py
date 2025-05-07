import pickle
from typing import Any

import h5py
import numpy as np
import os
from scipy.io import loadmat


def load_response_information(path: str, spike: int | None) -> (np.ndarray, np.ndarray):
    """
    Loads trigger and response data from .mat files located in a specified directory.

    This function accesses a directory containing .mat files with either raw or spike-based neuronal data.
    If the `spike` parameter is None, it assumes the directory contains raw data and loads the first available file.
    If a specific `spike` index is provided, it searches for a file whose name includes the substring "spike{spike}"
    and loads the corresponding binary spike data.

    :param path: Path to the directory containing the .mat files for either raw or spike data.
    :param spike: Integer specifying the spike index to load. If None, raw data from the first file is loaded.
                  If an integer, the function searches for a file containing "spike{spike}" in its name and
                  retrieves the corresponding binary spike signal.
    :return: A tuple containing two NumPy arrays:
             (1) triggers - A 1D array of trigger signals.
             (2) response - A 1D array of either raw response signals or binary spike signals,
                         depending on the `spike` parameter.
    """

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
    Loads neural response data for each neuron in a given directory, aligned to stimulus trigger events.

    This function reads stimulus presentation metadata and loops over each neuron subdirectory to extract 
    either raw signal data or binary spike data depending on the `spike` parameter. For each stimulus presentation, 
    it collects the response window starting from the trigger point and extending for `stim_window` samples. 
    The function builds and returns a dictionary mapping each neuron to its full concatenated response data,
    along with the stimulus order (frequency, amplitude) used in the experiment.

    :param path: Path to the main directory containing subdirectories for each neuron. Each subdirectory
                 should contain .mat files with trigger and response data.
    :param stim_path: Path to the stimulus metadata .mat file containing stimulus presentation order.
    :param stim_window: Number of data points to extract from the response signal starting at each trigger.
    :param spike: If None, raw data is extracted. If an integer is provided, the function searches for
                  spike-specific files and loads binary spike data accordingly.
    :return: A tuple containing a dictionary and NumPy array:
            (1) A dictionary mapping neuron names (directory names) to a NumPy array of concatenated response data.
            (2) A 2D NumPy array containing the stimulus order metadata, where each row is (frequency, amplitude).
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


def response_data_by_order(response_data: dict[str, np.ndarray], order: np.ndarray) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Converts flat neuron response data into a structured format organized by frequency and amplitude.

    This function restructures a dictionary mapping each neuron to its full response signal into a nested
    dictionary format: Neuron → Frequency → Amplitude → Response Data. It assumes that the original
    response array for each neuron is a concatenation of equally sized windows corresponding to stimulus
    presentations. Each stimulus presentation is described by a frequency and amplitude tuple in the `order` array.

    :param response_data: Dictionary mapping neuron names to their 1D response arrays (concatenated across stimuli).
    :param order: A 2D NumPy array of shape (N, 2), where each row contains the frequency and amplitude of a stimulus presentation.
    :return: A nested dictionary where the structure is:
             {
                 neuron_name: {
                     frequency: {
                         amplitude: response_array
                     }
                 }
             }
    """
    final = {}

    for neuron, data in response_data.items():
        j = 0
        window_size = data.shape[0] / order.shape[0]
        final[neuron] = {}

        for freq, amp in order:
            if freq not in final[neuron]:
                final[neuron][freq] = {}

            if amp not in final[neuron][freq]:
                final[neuron][freq][amp] = []

            final[neuron][freq][amp] = data[j:j + window_size]

            j += window_size

    return final
