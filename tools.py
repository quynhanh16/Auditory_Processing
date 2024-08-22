# File: tools.py
# Purpose: Recording help tools

# Packages
import math
import os
import pickle
import re
from typing import List, Tuple

# NEMS Packages
from nems.tools import epoch
from nems.tools.recording import load_recording, Recording
from nems.tools.signal import RasterizedSignal, SignalBase


# TODO: Create a function that gives a summary about a Signal (RasterizedSignal? Recording?)
# TODO: Add descriptions to all the tool functions
# TODO: Add compatability to


# Development Tools
def save_state(filename: str, stim, resp) -> None:
    with open(filename, "wb") as f:
        pickle.dump((stim, resp), f)


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


def signal_channels(signal: SignalBase, display: bool = False) -> List[str]:
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
    resp = recording["resp"].rasterize()
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
