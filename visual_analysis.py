# File: visual_analysis.py
# Purpose: Visual analysis of the recordings.
import math
import os

import matplotlib.pyplot as plt
# Packages
import numpy as np
# NEMS Packages
# from nems import visualization
from nems.tools import epoch
from nems.tools.recording import load_recording, Recording
from nems.tools.signal import RasterizedSignal, SignalBase

global all_cellids


def print_step(text: str, end=False) -> None:
    print('-' * 30)
    print(f"\n{text:^30}\n")
    if end:
        print('-' * 30)


def load_datafile(path: str) -> Recording:
    print_step("LOADING FILES")

    signals_dir: str = "file://"
    datafile = os.path.join(signals_dir, path)
    recordings: Recording = load_recording(datafile)

    return recordings


# Loading recordings
# Current recordings are divided into resp, stim, and mask_est
# mark_est does not contain anything
def splitting_recording(recording: Recording) -> (RasterizedSignal, RasterizedSignal):
    print_step('SPLITTING RECORDING')
    stimuli: RasterizedSignal = recording['stim']
    response: RasterizedSignal = recording['resp']

    return stimuli, response


def print_data(data: RasterizedSignal, name="Data"):
    print('-' * 20)
    print(name)
    print(data.as_continuous())
    print(data.as_continuous().shape)
    print()


# Time to Stimuli
# Takes an interval, in seconds, and returns the stimulus in that interval
def time_to_stimuli(signal: SignalBase, interval: (float, float)) -> ([str], (float, float)):
    if interval[0] < 0:
        raise ValueError('Start Index out of range')

    index: (int, int) = math.floor(interval[0] / 1.5), math.floor(interval[1] / 1.5)
    val_epochs = epoch.epoch_names_matching(signal.epochs, '^STIM_00')

    if index[1] == len(val_epochs):
        return val_epochs[index[0]:], index
    elif index[1] > len(val_epochs) - 1:
        raise ValueError('End Index out of range')
    elif index[1] != 0 and interval[1] % 1.5 == 0.0:
        return val_epochs[index[0]:index[1]], index

    return val_epochs[index[0]:index[1] + 1], index


def print_channels(signal: SignalBase) -> None:
    print(signal.chans)


# Spike Count Plot
# Plots the spike count located in the response data.
# TODO: Add space or only display (start, first_multiple) and (last_multiple, end)?
# TODO: Make the plot size dynamic
def resp_spike_count_plot(signal: RasterizedSignal, sec_interval: (float, float), cells: [str],
                          hist: bool = False) -> None:
    data = signal.extract_channels(cells)
    stimuli, index = time_to_stimuli(signal, sec_interval)
    n_cells = len(cells)
    n_stimuli = len(stimuli)
    f, ax = plt.subplots(n_stimuli, n_cells)

    for i in range(n_cells):
        for j in range(n_stimuli):
            t0 = (index[0] * 150) + j * 150
            t1 = t0 + 150
            x = np.arange(t0, t1, 1) / 100
            x = x.tolist()
            if j == 0:
                frame_data = data.extract_epoch(stimuli[j])[0, 0, int(sec_interval[0] * 10 / 15):].tolist()
                a = [0 for i in range(150 - len(frame_data))]
                frame_data = a + frame_data
            elif j == n_stimuli - 1:
                frame_data = data.extract_epoch(stimuli[j])[0, 0, :int(sec_interval[1] * 100 - t0)].tolist()
                frame_data += [0 for i in range(150 - len(frame_data))]
            else:
                frame_data = data.extract_epoch(stimuli[j])[0, 0, :]

            if hist:
                ax[j, i].bar(x, frame_data, edgecolor='black', width=0.01)
            else:
                ax[j, i].plot(x, frame_data)
            ax[j, i].set_title(f'Channel: {cells[i]}\nStim: {stimuli[j][7:-4]}')
            ax[j, i].set_ylabel('Spikes/s')

    f.suptitle('Spike Count Plot', weight='bold')
    plt.tight_layout()
    plt.show()


# TODO: Raster Plot
def resp_raster_plot(signal: RasterizedSignal) -> None:
    raster_signal = signal.rasterize()
    found = False
    for i in range(len(all_cellids)):
        single_cell_resp = raster_signal.extract_channels([all_cellids[i]])
        val_epochs = epoch.epoch_names_matching(raster_signal.epochs, '^STIM_00')
        for j in range(len(val_epochs)):
            r = single_cell_resp.extract_epoch(val_epochs[j])
            if r.shape[0] > 1:
                found = True
                print(r)
                print(r.shape)
                break


# TODO: Stimuli Heatmap
def stim_heatmap(signal: SignalBase, interval: (float, float)) -> [str]:
    pass


def exploring(signal: RasterizedSignal) -> None:
    val_epochs = epoch.epoch_names_matching(signal.epochs, '^STIM_00')
    print(all_cellids)

    for j in range(len(all_cellids)):
        for i in range(len(val_epochs)):
            data = signal.extract_channels([all_cellids[j]]).rasterize()
            r = data.extract_epoch(val_epochs[i])
            if r.shape[0] > 1 or r.shape[1] > 1:
                print(r)
                print(r.shape)
                print(i)


if __name__ == '__main__':
    tgz_file: str = 'A1_NAT4_ozgf.fs100.ch18.tgz'
    rec = load_datafile(tgz_file)
    stim, resp = splitting_recording(rec)
    all_cellids = resp.chans
    print(len(all_cellids))
    # print_data(stim, "Stimuli")
    # print_data(resp, "Response")
    resp_spike_count_plot(resp, (15, 19.5), [all_cellids[i] for i in [300, 600, 830]], hist=False)
    # resp_raster_plot(resp)
    # time_to_stimuli = time_to_stimuli(resp, (0, 3.0))
    # print(time_to_stimuli)
    # print_step('END OF PROGRAM', True)
    # exploring(resp)
