# File: visual_analysis.py
# Purpose: Visual analysis of the recordings.

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
# Packages
import seaborn as sns
# NEMS Packages
from nems.tools.signal import RasterizedSignal

# Tools
from tools import load_datafile, splitting_recording, time_to_stimuli, save_state, load_state

# from main import file_list

global all_cellids


# TODO: How to deal with different trials in the plots?

# Response: Multiple Spike Count Plot
# Plots the spike count located in the response data.
# TODO: Make the plot size dynamic
def resp_spike_rate_plot(signal: RasterizedSignal, sec_interval: (float, float), cells: [str],
                         hist: bool = False) -> None:
    stimuli, index = time_to_stimuli(signal, sec_interval)
    n_cells = len(cells)
    n_stimuli = len(stimuli)
    f, ax = plt.subplots(n_stimuli, n_cells)

    for i in range(n_cells):
        data = signal.extract_channels([cells[i]])
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
            ax[j, i].set_ylabel('Spikes/ms')

    f.suptitle('Spike Rate Plot', weight='bold')
    plt.tight_layout()
    plt.show()


# TODO: Use start and end in seconds when extracting epochs
def resp_raster_plot(signal: RasterizedSignal, interval: (float, float), cells: str | List[str]) -> None:
    stimuli, index = time_to_stimuli(signal, interval)
    raster_signal = signal.rasterize()
    n_cells = len(cells)
    n_stimuli = len(stimuli)

    if isinstance(cells, str):
        cells = [cells]

    single_cell_resp = raster_signal.extract_channels(cells)

    plt.figure()
    for j in range(n_stimuli):
        r = single_cell_resp.extract_epoch(stimuli[j])

        if j == n_stimuli - 1 and interval[1] % 1.5 != 0:
            r = r[:, :, :int(interval[1] * 100 + 1 - j * 150)]

        y, _, x = np.where(r)
        x_sec = (x + (150 * j)) / 100

        if j == 0:
            x_sec = []
            for xt in x:
                if xt >= interval[0] * 100:
                    x_sec.append((xt / 100))
            y = y[:len(x_sec)]

        plt.scatter(x_sec, y, s=1, color='black')

    plt.xticks(np.arange(interval[0], interval[1] + .5, .5))
    plt.suptitle("Raster Plot")
    plt.title(f"Cell: {", ".join(cells)}")
    plt.xlabel('Time (s)')
    plt.ylabel('Trial')
    plt.tight_layout()
    plt.show()


# Stimuli Heatmap
# TODO: How to plot different trials?
def stim_heatmap(signal: RasterizedSignal, interval: Tuple[float, float]) -> None:
    r = signal.extract_epoch(np.array([[interval[0], interval[1]]]))
    y = np.linspace(np.log10(200), np.log10(20000), 18, endpoint=True)
    y = [round(i, 2) for i in y]
    r = r[0, :, :]

    plt.figure()
    ax = sns.heatmap(r, cmap='viridis', fmt='d')
    ax.invert_yaxis()
    plt.suptitle('Stimuli Heatmap')
    plt.title(f'Stimulation Heatmap')
    plt.ylabel('Sample (Hz)')
    plt.xlabel('Time from stimulus onset (s)')
    ax.set_xticks(np.arange(interval[0] * 100, interval[1] * 100 + 1, 50))
    ax.set_xticklabels(np.arange(interval[0], interval[1] + .5, .5), rotation=0)
    plt.yticks(np.arange(1, 19, 1), labels=y, rotation=0)
    plt.tight_layout()
    plt.show()


def population_spike_rate(signal: RasterizedSignal, interval: Tuple[float, float]) -> None:
    data = signal.extract_epoch(np.array([[interval[0], interval[1]]]))
    y = data[0, :, :]
    # Adding all the data mong the 849 units
    y = np.sum(y, axis=0)
    # Divided by the 849 units
    y /= data.shape[1]
    x = np.arange(interval[0], len(y), 1)

    plt.figure()
    plt.plot(x, y)
    plt.xticks(np.arange(interval[0] * 100, interval[1] * 100 + 1, 50),
               labels=np.arange(interval[0], interval[1] + .5, .5))
    plt.title(f'Population Spike Rate')
    plt.ylabel('Rate (ms)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def exploring(signal: RasterizedSignal) -> None:
    print(signal.as_continuous())
    print(signal.as_continuous().shape)


if __name__ == '__main__':
    tgz_file: str = 'A1_NAT4_ozgf.fs100.ch18.tgz'

    state_file = "state.pkl"
    state = load_state(state_file)
    if state is None:
        rec = load_datafile(tgz_file, True)
        stim, resp = splitting_recording(rec, True)
        save_state(state_file, stim, resp)
    else:
        stim, resp = load_state(state_file)

    all_cellids = resp.chans
    # resp_spike_rate_plot(resp, (0, 3.5), [all_cellids[i] for i in [0, 1]], hist=True)
    # resp_raster_plot(resp, (1, 4.0), all_cellids[0])
    # stim_heatmap(stim, (0, 3.0))
    # population_spike_rate(resp, (0, 8.0))
    # exploring(stim)
