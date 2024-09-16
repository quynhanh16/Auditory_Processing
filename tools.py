# File: tools.py
# Purpose: Recording help tools

# Packages
import math
import os
import pickle
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
# NEMS Packages
from nems.tools import epoch
from nems.tools.recording import load_recording, Recording
from nems.tools.signal import RasterizedSignal, SignalBase


# TODO: Add function that gives a summary about a Signal (RasterizedSignal? Recording?)
# TODO: Add descriptions to all the tool functions
# TODO: Add function to read the results of models


@dataclass
class RecordingData:
    coefficients: np.array
    intercepts: np.array
    d: int
    m: int
    regularization: str
    alpha: float
    r2: float
    mae: float
    mse: float
    interval: (float, float)
    function: str


def prepare_stimuli(
        stim_signal: RasterizedSignal, interval: Tuple[float, float], m: int, d: int
) -> np.ndarray:
    stim_data = stim_signal.extract_epoch(np.array([list(interval)]))[0, :m]

    # Used for buffering
    # if d > 0:
    #     buffer = np.zeros((m, d))
    #     stim_data = np.hstack((buffer, stim_data))

    length_stim = stim_data.shape[1]
    # stim_matrix = np.array([stim_data[0][d:length_stim]]).T
    stim_matrix = np.array([])
    # print(stim_data)

    for i in range(m):
        matrix = np.empty((0, d + 1))
        for j in range(length_stim):
            if (j % 150) >= d:
                data = stim_data[i][j - d: j + 1]
                matrix = np.vstack((matrix, data))
        if stim_matrix.size == 0:
            stim_matrix = matrix
        else:
            stim_matrix = np.hstack((stim_matrix, matrix))

    # print(stim_matrix)
    return stim_matrix


def prepare_response(
        resp_signal: RasterizedSignal, interval: Tuple[float, float], d: int
) -> np.array:
    resp_data = resp_signal.extract_epoch(np.array([list(interval)]))[0]
    resp_matrix = np.array([])

    for i in range(resp_data.shape[0]):
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
def save_state(filename: str, data) -> None:
    with open(filename, "wb") as f:
        pickle.dump((data), f)


def load_state(filename: str):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


# General Tools
def print_step(text: str, end=False) -> None:
    print("-" * 30)
    print(f"\n{text:^30}\n")
    if end:
        print("-" * 30)


def print_data(data: RasterizedSignal, name="Data"):
    print("-" * 20)
    print(name)
    print(data.as_continuous())
    print(data.as_continuous().shape)
    print()


def signal_channels(signal: RasterizedSignal, display: bool = False) -> List[str]:
    channels = signal.chans

    if display:
        print(channels)

    return channels


def load_datafile(path: str, display=False) -> Recording:
    if display:
        print_step("LOADING FILES")

    signals_dir: str = "file://"
    datafile = os.path.join(signals_dir, path)
    recordings: Recording = load_recording(datafile)

    return recordings


def load_single_sites(dir_path: str, display=False) -> Dict[str, Recording]:
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
# Current recordings are divided into resp, stim, and mask_est
# mark_est does not contain anything
def splitting_recording(
        recording: Recording, display=False
) -> Tuple[RasterizedSignal, RasterizedSignal]:
    if display:
        print_step("SPLITTING RECORDING")

    stimuli: RasterizedSignal = recording["stim"]
    response: RasterizedSignal = recording["resp"]

    return stimuli, response


# Time to Stimuli
# Takes an interval, in seconds, and returns the stimulus in that interval
def time_to_stimuli(
        signal: SignalBase, interval: Tuple[float, float]
) -> (List[str], Tuple[float, float]):
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
    try:
        file = open(file_path, "r+")
    except FileNotFoundError:
        file = open(file_path, "w+")

    content = file.read()
    file.close()
    return content


def result_text(results: RecordingData, n: int) -> str:
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


def save_results(results: RecordingData, file_path: str = "results.txt") -> None:
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

    t = prepare_stimuli(stim, (0, 0.2), 18, 2)
    print(t.shape)

    r = prepare_response(resp, (0, 0.2), 2)
    print(r.shape)
