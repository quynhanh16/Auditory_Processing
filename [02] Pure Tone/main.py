from functools import partial

from numpy.lib.stride_tricks import sliding_window_view

import seaborn as sns

import librosa
from scipy.ndimage import gaussian_filter

from tools import *
from scipy.io import loadmat, wavfile
import numpy as np
import random
import joblib
import pickle
import pandas as pd

# TODO: Verify that the matrices are shaped correctly
np.random.seed(42)
random.seed(42)

def response():
    data, order = load_data("./data/pure_tones_spikes/", spike=10)
    kernel = partial(gaussian_filter, sigma=1)
    fr = firing_rate(data, 10, population=True, kernel=kernel)
    print(fr.shape)
    plot_firing_rate(fr, (0, 2000))


def load_stimuli(path: str = "./data/Stimulus/PureToneSound.wav"):
    sample_rate, data = wavfile.read(path)

    trial1 = data[480000:11328000, :]
    trial2 = trial1[0:10368000, :]

    final = np.empty((360, 45, 300))

    signal = trial2[:, 0].reshape(360, 28800)

    for i in range(signal.shape[0]):
        a = signal[i, :]
        b = gammagram(a, fs=96000, window_time=0.001, hop_time=0.001, channels=45, f_min=4000, f_max=40000)
        C = np.abs(np.transpose(b))
        D = librosa.power_to_db(C, ref=1)
        D = np.flipud(D)
        final[i, :, :] = D

    return final

def stimuli_heatmap(stim_data, response_data, order, interval):
    freq = np.sort(np.unique(order[:,0].T))
    data = np.hstack(stim_data)[:, interval[0]:interval[1]]

    kernel = partial(gaussian_filter, sigma=1)
    fr = firing_rate(response_data, 10, population=True, kernel=kernel)[interval[0]:interval[1]]

    fig, ax = plt.subplots(2, 1)
    sns.heatmap(data, cmap="viridis", vmin=-50, vmax=50, ax=ax[0])

    x_index = [i for i in range(0, interval[1] - interval[0], 300)]
    y_index = np.linspace(0, 44, 11, dtype=int)

    ax[0].set_xticks(x_index)
    ax[0].set_xticklabels([str(n) for n in x_index], rotation=90)
    ax[0].set_yticks(np.linspace(0, 44, 11))
    ax[0].set_yticklabels(labels=np.flipud(freq[y_index]).round(0), rotation=0)

    ax[1].plot(fr)
    ax[1].set_xticks(x_index)
    ax[1].set_xticklabels([str(n) for n in x_index], rotation=90)

    plt.tight_layout()
    plt.show()




def prepare_stimuli(stim_data: np.ndarray, previous_n: int = 200) -> np.ndarray:
    """
    Prepares a time-windowed stimulus representation with historical context for each time point.

    Given a 2D NumPy array of shape (n_frequencies, n_timepoints), this function pads the time dimension
    with `previous_n` zeros at the beginning, then constructs a sliding window of size `previous_n + 1`
    for each time step. Each window contains the current and previous stimulus values for all frequencies.

    The resulting data is reshaped into a 2D array of shape (n_timepoints, n_frequencies * (previous_n + 1)),
    suitable for use as input to machine learning models.

    :param stim_data: A NumPy array of shape (n_frequencies, n_timepoints) representing the stimulus signal.
    :param previous_n: The number of previous time steps to include in each window (default is 200).
    :return: A NumPy array of shape (n_timepoints, n_frequencies * (previous_n + 1)) containing flattened
             historical stimulus windows for each time point.
    """
    padded = np.pad(stim_data, ((0, 0), (previous_n, 0)), mode="constant")
    windows = sliding_window_view(padded, window_shape=previous_n + 1, axis=1)
    final = windows.transpose(1, 0, 2).reshape(108000, -1)
    return final

def run_linear_model():
    # Response
    df, order = load_data("./data/pure_tones/", spike=55)
    kernel = partial(gaussian_filter, sigma=1)
    response_data = firing_rate(df, 10, population=True, kernel=kernel)
    # plot_firing_rate(response_data, (0, 2000))

    # Stimuli
    stim_data = load_stimuli()
    # stimuli_heatmap(stim_data, response_data, order, (0, 2000))
    stim_data = np.hstack(stim_data)
    stim_data = prepare_stimuli(stim_data)

    print("Firing rate shape:", response_data.shape, "Stimulus shape:", stim_data.shape)
    linear_model(stim_data, response_data)

def save_coefficient_matrix(path_model):
    loaded_model = joblib.load(path_model)
    lasso_model = loaded_model.named_steps["model"]
    coefficients = lasso_model.coef_
    intercept = lasso_model.intercept_
    coefficients_df = pd.DataFrame(coefficients)
    coefficients_df.columns = [f'feature_{i}' for i in range(coefficients_df.shape[1])]
    coefficients_df["intercept"] = intercept
    coefficients_df.to_csv("model_coefficients.csv", index=True)

def save_input_output_actual(path_model):
    df, order = load_data("./data/pure_tones_spikes/", spike=55)
    loaded_model = joblib.load(path_model)
    lasso_model = loaded_model.named_steps["model"]
    X = prepare_stimuli(np.hstack(load_stimuli()))
    pred = lasso_model.predict(X)
    kernel = partial(gaussian_filter, sigma=1)
    response_data = firing_rate(df, 10, population=True, kernel=kernel)

    stim_df = pd.DataFrame(X)
    pred_df = pd.DataFrame(pred)
    actual_df = pd.DataFrame(response_data)

    stim_df.to_csv("./stimuli.csv", index=True)
    pred_df.to_csv("./predictions.csv", index=True)
    actual_df.to_csv("./actual.csv", index=True)


if __name__ == '__main__':
    df, order = load_data("./data/pure_tones/", spike=55)
    kernel = partial(gaussian_filter, sigma=1)
    response_data = firing_rate(df, 10, population=True, kernel=kernel)

    stim_data = load_stimuli()
    stim_data = np.hstack(stim_data)
    stim_data = prepare_stimuli(stim_data)

    print("Firing rate shape:", response_data.shape, "Stimulus shape:", stim_data.shape)
    # nonlinear_pipeline(stim_data, response_data)
    # cnn_model(stim_data, response_data)
    evaluate_cnn_model("cnn_model.pt", stim_data, response_data)
