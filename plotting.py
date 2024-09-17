# File: plotting.py
# Purpose: Visual analysis of the recordings.

from typing import List, Tuple

import joblib  # Use pickle
import matplotlib.pyplot as plt
import numpy as np

# Packages
import seaborn as sns
from matplotlib.gridspec import GridSpec

# NEMS Packages
from nems.tools.signal import RasterizedSignal

# Computing
from computing import population_spike_rate, population_evoked_firing_rate
from fitting import simple_linear_model

# Tools
from tools import (
    load_datafile,
    splitting_recording,
    time_to_stimuli,
    save_state,
    load_state,
    prepare_stimuli,
    prepare_response,
    coeff_channels,
)

global all_cellids


# Response: Multiple Spike Count Plot
# Plots the spike count located in the response data.
# TODO: Make the plot size dynamic
#       Make a simpler version of this function
def resp_spike_rate_plot(
    signal: RasterizedSignal,
    sec_interval: (float, float),
    cells: [str],
    hist: bool = False,
) -> None:
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
                frame_data = data.extract_epoch(stimuli[j])[
                    0, 0, int(sec_interval[0] * 10 / 15) :
                ].tolist()
                a = [0 for i in range(150 - len(frame_data))]
                frame_data = a + frame_data
            elif j == n_stimuli - 1:
                frame_data = data.extract_epoch(stimuli[j])[
                    0, 0, : int(sec_interval[1] * 100 - t0)
                ].tolist()
                frame_data += [0 for i in range(150 - len(frame_data))]
            else:
                frame_data = data.extract_epoch(stimuli[j])[0, 0, :]

            if hist:
                ax[j, i].bar(x, frame_data, edgecolor="black", width=0.01)
            else:
                ax[j, i].plot(x, frame_data)
            ax[j, i].set_title(f"Channel: {cells[i]}\nStim: {stimuli[j][7:-4]}")
            ax[j, i].set_ylabel("Spikes/ms")

    f.suptitle("Spike Rate Plot", weight="bold")
    plt.tight_layout()
    plt.show()


def resp_raster_plot(
    signal: RasterizedSignal, interval: (float, float), cells: str | List[str]
) -> None:
    raster_signal = signal.rasterize()

    if isinstance(cells, str):
        cells = [cells]

    single_cell_resp = raster_signal.extract_channels(cells)

    plt.figure()
    r = single_cell_resp.extract_epoch(np.array([list(interval)]))
    y, _, x = np.where(r)
    x_sec = (x + (interval[0] * 100)) / 100
    plt.scatter(x_sec, y, s=1, color="black")
    plt.xticks(np.arange(interval[0], interval[1] + 0.5, 0.5))
    plt.suptitle("Raster Plot")
    plt.title(f"Cell: {", ".join(cells)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Trial")
    plt.tight_layout()
    plt.show()


# Stimuli Heatmap
def stim_heatmap(
    signal: RasterizedSignal,
    interval: Tuple[float, float],
    display: bool = True,
    **kwargs,
) -> None:
    r = signal.extract_epoch(np.array([list(interval)]))
    y = np.linspace(np.log10(200), np.log10(20000), 18, endpoint=True)
    y = [round(i, 2) for i in y]
    r = r[0, :, :]

    if "ax" in kwargs.keys():
        ax = kwargs["ax"]
    else:
        f, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(r, cmap="viridis", fmt="d", ax=ax)
    ax.invert_yaxis()
    ax.set_title(f"Stimulation Heatmap")
    ax.set_xticks(np.arange(0, (interval[1] - interval[0]) * 100 + 1, 50))
    ax.set_xticklabels(np.arange(interval[0], interval[1] + 0.5, 0.5), rotation=0)
    ax.set_xlabel("Time (s)")
    ax.set_yticks(np.arange(1, 19, 1), labels=y, rotation=0)
    ax.set_ylabel("Sample (Hz)")

    if display:
        plt.tight_layout()
        plt.show()


def population_spike_rate_plot(
    resp_signal: RasterizedSignal,
    interval: Tuple[float, float],
    display: bool = True,
    **kwargs,
) -> None:
    y = population_spike_rate(resp_signal, interval) * 100
    x = np.arange(interval[0], interval[1], 0.01)

    if "ax" in kwargs.keys():
        ax = kwargs["ax"]
    else:
        f, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y, label="Actual")
    ax.set_title(f"Population Spike Rate")
    ax.set_xticks(
        np.arange(interval[0], interval[1] + 0.5, 0.5),
        labels=np.arange(interval[0], interval[1] + 0.5, 0.5),
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (Hz)")

    if display:
        plt.tight_layout()
        plt.show()


# Question: Are we still doing evoked-spike firing rate?
def linear_model_plot(
    stim_signal: RasterizedSignal,
    resp_signal: RasterizedSignal,
    interval: Tuple[float, float],
) -> None:
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.25])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Population evoked firing rate
    y_resp = population_evoked_firing_rate(resp_signal, interval)
    model = simple_linear_model(
        stim_signal,
        resp_signal,
        18,
        20,
        False,
        False,
        function=population_evoked_firing_rate,
    )
    X = prepare_stimuli(stim_signal, interval, 18, 20)
    y_pred = model.predict(X) * 100
    ax1.plot(y_resp * 100, label="Actual")
    ax1.plot(y_pred, label="Predicted")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Rate (Hz)")
    ax1.set_xticks(np.arange(0, (interval[1] - interval[0]) * 100 + 50, 50))
    ax1.set_xticklabels(np.arange(interval[0], interval[1] + 0.5, 0.5), rotation=0)
    ax1.legend()

    # Population spike rate
    population_spike_rate_plot(resp_signal, interval, False, ax=ax2)
    model = simple_linear_model(stim_signal, resp_signal, 18, 20, False, False)
    X = prepare_stimuli(stim_signal, interval, 18, 20)
    y_pred = model.predict(X) * 100
    x = np.arange(interval[0], interval[1], 0.01)
    ax2.plot(x, y_pred, label="Predicted")
    ax2.legend()

    # Heatmap
    stim_heatmap(stim_signal, interval, False, ax=ax3)

    plt.tight_layout()
    plt.show()


def actual_predicted_plot(
    stim_signal: RasterizedSignal,
    resp_signal: RasterizedSignal,
    interval: Tuple[float, float],
    model,
) -> None:
    X = prepare_stimuli(stim_signal, interval, 18, 20)
    y = prepare_response(resp_signal, interval, 20) * 100
    predictions = model.predict(X) * 100
    print(y.shape, predictions.shape)
    plt.figure()
    plt.scatter(np.mean(y, axis=0), predictions, s=5, alpha=0.5)
    plt.title("Actual vs. Predicted (Validation Data)")
    plt.xlabel("Actual (Hz)")
    plt.ylabel("Predicted (Hz)")
    plt.show()


def coefficient_heatmap(coefficients: List[float], m: int, d: int) -> None:
    coeff_matrix = coeff_channels(coefficients, m, d)
    plt.figure()
    plt.suptitle("Coefficient Heatmap")
    ax = sns.heatmap(coeff_matrix, cmap="flare", fmt="d")
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Channels")
    ax.set_xticklabels(np.arange(1, d + 2, 1))
    ax.set_yticklabels(np.arange(1, m + 1, 1))
    plt.show()


if __name__ == "__main__":
    tgz_file: str = "A1_NAT4_ozgf.fs100.ch18.tgz"

    state_file = "state.pkl"
    state = load_state(state_file)
    if state is None:
        rec = load_datafile(tgz_file, True)
        stim, resp = splitting_recording(rec, True)
        save_state(state_file, (stim, resp))
    else:
        stim, resp = load_state(state_file)

    all_cellids = resp.chans
    # resp_spike_rate_plot(resp, (0, 3.5), [all_cellids[i] for i in [0, 1]], hist=True)
    # resp_raster_plot(resp, (1.4, 3.8), all_cellids[0])
    # stim_heatmap(stim, (27, 30))
    # population_spike_rate_plot(resp, (3, 6))
    # linear_model_plot(stim, resp, (1.5, 3.0))
    # actual_predicted_plot(stim, resp, (0, 27), joblib.load("nr_linear_model.pkl"))
    coefficient_heatmap(joblib.load("nr_linear_model.pkl").coef_, 18, 20)
