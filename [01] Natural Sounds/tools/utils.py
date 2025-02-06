# File: tools.py
# Purpose: Recording help tools

# Packages
import math
import os
import pickle
import re
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List, Tuple, Dict, Any

import h5py
import numpy as np
from tqdm import tqdm

# NEMS Packages
from tools import epoch
from tools.recording import load_recording, Recording
from tools.signal import RasterizedSignal, SignalBase


# TODO: Add function that gives a summary about a Signal (RasterizedSignal? Recording?)
# TODO: Add descriptions to all the tool functions
# TODO: Add function to read the results of models


@dataclass
class RecordingData:
    """
    Used to format model results.
    """
    coefficients: np.array
    intercepts: np.array
    d: int
    m: int
    regularization: str
    alpha: float
    r2: float
    mae: float
    mse: float
    interval: Tuple[float, float]
    function: str


def save_results(filename: str, label: str, info: Any) -> None:
    """
    Save object as h5 file.

    :param filename:
    :param label:
    :param info:
    :return:
    """
    with h5py.File(filename, "w") as f:
        f.create_dataset(label, data=info)


def read_results(filename: str, label: str):
    """
    Read object from h5 file.

    :param filename:
    :param label:
    :return: Any
    """
    with h5py.File(filename, "r") as f:
        data = f[label][:]
        return data


def prepare_stimulus(
        idx, data, d
) -> np.ndarray:
    """
    Given an array of numbers and d, ignore the first d elements of the array every 150
    time points, and return sets of arrays of length d stacked vertically.

    :param idx: Index of channel
    :param data: Time data of channel
    :param d: Previous stimuli
    :return:
    """
    length_stim = data.shape[0]
    matrix = np.empty((0, d + 1))

    with tqdm(total=100, desc=f"Preparing Stimuli Channel {idx + 1}", leave=False) as chan_pbar:
        for j in range(length_stim):
            bar_increment = round(100 / length_stim, 5)

            if chan_pbar.n + bar_increment < 100:
                chan_pbar.update(bar_increment)
            else:
                chan_pbar.update(100 - chan_pbar.n)

            if (j % 150) >= d:
                matrix = np.vstack((matrix, data[j - d: j + 1]))

    return matrix


def prepare_stimuli(
        stim_signal: RasterizedSignal, interval: Tuple[float, float], m: int, d: int
) -> np.ndarray:
    """
    Prepare the stimuli recording by ignoring the first d data points per interval.
    In this case, the interval is 150, therefore, the function will return only
    data points between 0.2 - 1.5 seconds of the recording.
    Return a matrix of prepared stimuli. The shape of the matrix is the number of
    datatime points minus the first d elements every 150 data points by d + 1 elements
    times the number of channels (m).

    :param stim_signal: Stimulation signal
    :param interval: Time interval
    :param m: Number of channels
    :param d: N number of previous stimuli
    :return: np.ndarray
    """
    stim_data = stim_signal.extract_epoch(np.array([list(interval)]))[0, :m]
    stim_data = [(idx, arr, d) for idx, arr in enumerate(stim_data)]

    with Pool() as pool:
        stim_matrix = pool.starmap(prepare_stimulus, stim_data)
        stim_matrix = np.hstack(stim_matrix)

    return stim_matrix


# TODO: Adding multiprocessing?
def prepare_response(
        resp_signal: RasterizedSignal, interval: Tuple[float, float], d: int
) -> np.array:
    """
    Prepare the response recording by ignoring the first d data points per interval.
    In this case, the interval is 150, therefore, the function will return only
    data points between 0.2 - 1.5 seconds of the recording.

    :param resp_signal:
    :param interval:
    :param d: N number of previous stimuli
    :type d: int
    :return: np.array
    """
    resp_data = resp_signal.extract_epoch(np.array([list(interval)]))[0]
    resp_matrix = np.array([])

    with tqdm(total=100, desc="Preparing Response", leave=False) as pbar:
        for i in range(resp_data.shape[0]):
            bar_increment = round(100 / resp_data.shape[0], 5)
            if pbar.n + bar_increment < 100:
                pbar.update(bar_increment)
            else:
                pbar.update(100 - pbar.n)

            matrix = []

            for idx, ele in enumerate(resp_data[i]):
                if (idx % 150) >= d:
                    matrix.append(ele)

            if resp_matrix.size == 0:
                resp_matrix = np.array(matrix)
            else:
                resp_matrix = np.vstack((resp_matrix, matrix))

    return resp_matrix


# Development Tools
def save_state(filename: str, data: Any) -> None:
    """
    Save an object as a pickle file with name *filename*.

    :param filename:
    :param data:
    :return: None
    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def prepare_stimuli_model(stimuli, m: int, d: int):
    new_x = np.empty((stimuli.shape[0], d + 1, m))
    for i in range(stimuli.shape[0]):
        for j in range(int(stimuli.shape[1] / m)):
            for k in range(int(stimuli.shape[1] / (d + 1))):
                new_x[i, j, k] = stimuli[i, (m * j) + k]
    return new_x


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


# General Tools
def print_step(text: str, end=False) -> None:
    """
    Simple function to print a step.

    :param text:
    :param end:
    :return:
    """
    print("-" * 30)
    print(f"\n{text:^30}\n")
    if end:
        print("-" * 30)


def print_data(data: RasterizedSignal, name="Data") -> None:
    """
    Print RasterizedSignal data with its shape.

    :param data:
    :param name:
    :return: None
    """
    print("-" * 20)
    print(name)
    print(data.as_continuous())
    print(data.as_continuous().shape)
    print()


def signal_channels(signal: RasterizedSignal, display: bool = False) -> List[str]:
    """
    Return the signal channels of Rasterized Signal. If display is True, also print channels.

    :param signal:
    :param display:
    :return: List[str]
    """
    channels = signal.chans

    if display:
        print(channels)

    return channels


def load_datafile(path: str, display=False) -> Recording:
    """
    Return the recording data from a tgz file.

    :param path:
    :param display:
    :return: Recording
    """
    if display:
        print_step("LOADING FILES")

    signals_dir: str = "file://"
    datafile = os.path.join(signals_dir, path)
    recordings: Recording = load_recording(datafile)

    return recordings


def load_single_sites(dir_path: str, display=False) -> Dict[str, Recording]:
    """
    Return the recording data from single sites data.

    :param dir_path:
    :param display:
    :return: Dict[str, Recording]
    """
    single_site_recordings: Dict[str, Recording] = {}
    single_site_paths = os.listdir(dir_path)
    single_site_names = list(simplify_site_names(single_site_paths).keys())

    for i in range(len(single_site_paths)):
        path = os.path.join(dir_path, single_site_paths[i])
        single_site_recordings[single_site_names[i]] = load_datafile(path)

    print(single_site_recordings)
    print(single_site_recordings["ARM031a"]["resp"].shape)
    return single_site_recordings


def simplify_site_names(names: List[str]) -> Dict[str, int]:
    """
    Simplify single site recordings names and return number of occurrences of recording name.

    :param names:
    :return: Dict[str, int]
    """
    pattern = r"A1_(.*?)_"
    site_names: Dict[str, int] = {}

    for name in names:
        match = re.search(pattern, name)
        if match:
            if match.group(1) in site_names:
                site_names[match.group(1)] += 1
            else:
                site_names[match.group(1)] = 1

    return site_names


# Loading recordings
def splitting_recording(
        recording: Recording, display=False
) -> Tuple[RasterizedSignal, RasterizedSignal]:
    """
    Divides tgz files into stimuli and response data.

    Current recordings are divided into resp, stim, and mask_est

    :param recording:
    :param display:
    :return: Tuple[RasterizedSignal, RasterizedSignal]
    """
    if display:
        print_step("SPLITTING RECORDING")

    stimuli: RasterizedSignal = recording["stim"]
    response: RasterizedSignal = recording["resp"]

    return stimuli, response


def time_to_stimuli(
        signal: SignalBase, interval: Tuple[float, float]
) -> (List[str], Tuple[float, float]):
    """
    Takes an interval, in seconds, and returns the stimulus in that interval

    :param signal:
    :param interval:
    :return: (List[str], Tuple[float, float])
    """
    if interval[0] < 0:
        raise ValueError("Start Index out of range")

    index: (int, int) = math.floor(interval[0] / 1.5), math.floor(interval[1] / 1.5)
    val_epochs = epoch.epoch_names_matching(signal.epochs, "^STIM_00")

    if index[1] == len(val_epochs):
        return val_epochs[index[0]:], index
    elif index[1] > len(val_epochs) - 1:
        raise ValueError("End Index out of range")
    elif index[1] != 0 and interval[1] % 1.5 == 0.0:
        return val_epochs[index[0]: index[1]], index

    return val_epochs[index[0]: index[1] + 1], index


def single_site_similar_stim(site: str, top_n: int = 0) -> None:
    """
    Check if single site recording contain similar stimuli and print the top n.

    :param site:
    :param top_n:
    :return: None
    """
    pattern = r"rec\d+_(.*?)_excerpt"
    path = os.path.join("A1_single_sites", site)
    recording = load_datafile(path)
    resp = recording["resp"]
    stim = {}

    val_epochs = epoch.epoch_names_matching(resp.epochs, "^STIM_00")
    for val in val_epochs:
        match = re.search(pattern, val)
        if match:
            text = match.group(1)
            if text in stim:
                stim[text] += 1
            else:
                stim[text] = 1
        else:
            print("Not found")

    sorted_stim = sorted(stim.items(), key=lambda x: x[1], reverse=True)

    if 0 < top_n < len(sorted_stim):
        sorted_stim = sorted_stim[:top_n]
    else:
        raise IndexError(f"Invalid N. Length of the stimuli list: {len(sorted_stim)}")

    for stimulus, occ in sorted_stim:
        print(f"({stimulus}: {occ})", end=" | ")
    print()


def multiple_site_similar_stim(sites: List[str], top_n: int) -> None:
    """
    Check and print if any of the recordings contain similar stimuli and print the top n.

    :param sites:
    :param top_n:
    :return: None
    """
    pattern = r"rec\d+_(.*?)_excerpt"

    for site in sites:
        print(site[:-4] + ":")
        site_stim = {}
        path = os.path.join("A1_single_sites", site)
        recording = load_datafile(path)
        resp = recording["resp"].rasterize()

        val_epochs = epoch.epoch_names_matching(resp.epochs, "^STIM_00")
        for val in val_epochs:
            match = re.search(pattern, val)
            if match:
                text = match.group(1)
                if text in site_stim:
                    site_stim[text] += 1
                else:
                    site_stim[text] = 1

        sorted_stim = sorted(site_stim.items(), key=lambda x: x[1], reverse=True)

        if 0 < top_n < len(sorted_stim):
            sorted_stim = sorted_stim[:top_n]

        print("|", end=" ")
        for stimulus, occ in sorted_stim:
            print(f"({stimulus}: {occ})", end=" | ")

        print("\n")


def open_file(file_path: str) -> str:
    """
    Open file and return content.

    :param file_path:
    :return: str
    """
    try:
        file = open(file_path, "r+")
    except FileNotFoundError:
        file = open(file_path, "w+")

    content = file.read()
    file.close()
    return content


def result_text(results: RecordingData, n: int) -> str:
    """
    Return formatted string with results of linear models.

    :param results:
    :param n:
    :return: str
    """
    coefficients = "["
    intercepts = "["

    coefficients += ", ".join([str(x) for x in results.coefficients]) + "]"
    intercepts += ", ".join([str(x) for x in results.intercepts]) + "]"

    if len(results.regularization) == 0:
        regularization = "No regularization"
    else:
        regularization = results.regularization

    display = f"""Trial: n={n}, m={results.m}, d={results.d}, interval={results.interval}, {regularization}, alpha={results.alpha}
Results:
\tCoefficients: {coefficients}
\tIntercepts: {intercepts}
\tR2: {results.r2}
\tMSE: {results.mse}
\tMAE: {results.mae}
\tFunction: {results.function}\n"""

    return display


def save_results_txt(results: RecordingData, file_path: str = "results.txt") -> None:
    """
    Save results of linear models in txt format.

    :param results:
    :param file_path:
    :return: None
    """
    n_trial = 0
    content = open_file(file_path)

    if len(content) != 0:
        prev_text = content.split("\n")
        n_trial = re.search(r"\s*(\d+)", prev_text[0])
        if n_trial is not None:
            n_trial = int(n_trial.group(1))
        else:
            n_trial = 0
        with open(file_path, "r+") as file:
            lines = file.readlines()
            lines[0] = f"N: {n_trial + 1}\n"

            file.seek(0)
            file.writelines(lines)
            file.truncate()
            file.close()

    with open(file_path, "a+") as file:
        n_trial += 1
        new_trial = result_text(results, n_trial)

        if n_trial == 1:
            file.write(f"N: {n_trial}\n")

        file.write(new_trial)


def coefficient_channels(coefficient: List[float], m: int, d: int) -> np.array:
    """
    Returned rasterized matrix of channels.

    :param coefficient:
    :param m:
    :param d:
    :return: np.array
    """
    result = []

    row = 0
    for i in range(m * (d + 1)):
        if i % (d + 1) == 0:
            result.append([coefficient[i]])
            row += 0 if i == 0 else 1
        else:
            result[row].append(coefficient[i])

    return np.array(result)


if __name__ == "__main__":
    tgz_file: str = "../A1_NAT4_ozgf.fs100.ch18.tgz"

    state_file = "../../state.pkl"
    state = load_state(state_file)
    if state is None:
        rec = load_datafile(tgz_file, True)
        stim, resp = splitting_recording(rec, True)
        save_state(state_file, (stim, resp))
    else:
        stim, resp = load_state(state_file)
