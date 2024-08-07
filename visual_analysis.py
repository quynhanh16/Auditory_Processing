# File: visual_analysis.py
# Purpose: Visual analysis of the recordings.

import os

import matplotlib.pyplot as plt
# Packages
import numpy as np

# NEMS Packages
from nems import visualization
from nems.tools import epoch
from nems.tools.recording import load_recording, Recording
from nems.tools.signal import RasterizedSignal

cellids = ["DRX006b-128-2", "ARM030a-40-2"]


def load_datafile(path: str) -> Recording:
    print('-' * 20)
    print("LOADING FILES\n")

    signals_dir: str = "file://"
    datafile = os.path.join(signals_dir, path)
    recordings: Recording = load_recording(datafile)

    return recordings


# Loading recordings
# Current recordings are divided into resp, stim, and mask_est
# mark_est does not contain anything
def splitting_recording(recording: Recording) -> (RasterizedSignal, RasterizedSignal):
    print('-' * 20)
    print("Splitting recording\n")
    stim: RasterizedSignal = recording['stim']
    resp: RasterizedSignal = recording['resp']

    return stim, resp


def print_data(data: RasterizedSignal, name="Data"):
    print('-' * 20)
    print(name)
    print(data._data)
    print(data._data.shape)
    print()


# Plotting
# TODO: Create Spike Count Histogram
# TODO: Create Raster Plot
def data_visualization(data: RasterizedSignal, name="Data"):
    print('-' * 20)
    print(f'Visualizing {name}\n')

    visualization.plot_data(data._data[0:5, ], "response data")
    plt.show()

    # Raster Plot
    resp_rasterized = resp.rasterize()
    single_cell_resp = resp_rasterized.extract_channels(cellids)
    val_epochs = epoch.epoch_names_matching(resp.epochs, "^stim_00")
    r = single_cell_resp.extract_epoch(val_epochs[0])

    y, _, x = np.where(r)
    x_sec = x / 1000 - 0.25
    plt.figure()
    plt.scatter(x_sec, y, s=1, color='black')
    plt.title(f"raster, cell: pending stim: {val_epochs[0]}")
    plt.xlabel('time from stimulus onset (s)')
    plt.ylabel('trial')
    plt.show()


# Testing
def signal_viz(signal: RasterizedSignal):
    data = signal._data
    t0, t1 = 0, 450
    x = np.arange(t0, t1, 1) / 100

    plt.plot(x, data[0, t0:t1])
    plt.show()


if __name__ == '__main__':
    tgz_file: str = 'A1_NAT4_ozgf.fs100.ch18.tgz'
    recording = load_datafile(tgz_file)
    stim, resp = splitting_recording(recording)
    print_data(stim, "Stimuli")
    print_data(resp, "Response")
    # data_visualization(resp, "Response")
    signal_viz(resp)
