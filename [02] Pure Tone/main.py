from functools import partial

import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns

import librosa
from scipy.ndimage import gaussian_filter

from tools import *
from scipy.io import loadmat, wavfile
import numpy as np
import random

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


def linear_model(X: np.ndarray, y: np.ndarray):
    """
    Fitting a simple linear regression model with k-folds as cross-validation.

    :param X:
    :param y:
    :return:
    """
    pipeline = Pipeline([
        # ('poly', PolynomialFeatures(include_bias=False)),
        # ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    param_grid = {
        # 'poly__degree': [1, 2],
        # 'model__alpha': [0.001, 0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=3, scoring='neg_mean_squared_error', return_train_score=True)
    grid.fit(X, y)

    print("\nCross-Validation Results:")
    for mean_train, mean_test, params in zip(
            grid.cv_results_['mean_train_score'],
            grid.cv_results_['mean_test_score'],
            grid.cv_results_['params']
    ):
        print(f"{params}: Train MSE = {-mean_train:.4f}, Test MSE = {-mean_test:.4f}")

    model = grid.best_estimator_
    # degree = grid.best_params_['poly__degree']
    y_pred = model.predict(X)

    final_mse = mean_squared_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)

    # print(f"\nSelected Polynomial Degree: {degree}")
    print(f"Final MSE: {final_mse:.4f}")
    print(f"Final R²: {final_r2:.4f}")

    plt.plot(y[:2000], color="black", label="Actual")
    plt.plot(y_pred[:2000], color="red", label="Prediction", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.scatter(y_pred, y, color="black", s=1, alpha=0.5)
    plt.ylabel("Actual")
    plt.ylabel("predicted")
    plt.tight_layout()
    plt.show()

    # Coefficient heatmap
    coefs = model.named_steps['model'].coef_
    print("Coefficient", coefs.shape)

    coef_df = np.abs(coefs.reshape(1, -1))  # reshape to 2D for heatmap
    sns.heatmap(coef_df, cmap="coolwarm", yticklabels=[], cbar_kws={'label': 'Coefficient Magnitude'})
    plt.title("Coefficient Heatmap")
    plt.xticks(rotation=90)
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


if __name__ == '__main__':
    # Response
    df, order = load_data("./data/pure_tones_spikes/", spike=55)
    kernel = partial(gaussian_filter, sigma=1)
    response_data = firing_rate(df, 10, population=True, kernel=kernel)
    # plot_firing_rate(response_data, (0, 2000))

    # Stimuli
    data = load_stimuli()
    stim_data = np.hstack(data)
    stim_data = prepare_stimuli(stim_data)
    stimuli_heatmap(stim_data, response_data, order, (0, 2000))

    print("Firing rate shape:", response_data.shape, "Stimulus shape:", stim_data.shape)
    linear_model(stim_data, response_data)