from functools import partial
import seaborn as sns

import librosa
from scipy.ndimage import gaussian_filter

from tools import *
from scipy.io import loadmat, wavfile
import numpy as np

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
        if i == 3:
            break
        a = signal[i, :]
        b = gammagram(a, fs=96000, window_time=0.001, hop_time=0.001, channels=45, f_min=4000, f_max=40000)
        C = np.abs(np.transpose(b))
        D = librosa.power_to_db(C, ref=1)
        final[i, :, :] = D

    return final

def stimuli_heatmap(data, interval):
    data = np.hstack(data)[:, interval[0]:interval[1]]
    print(data.shape)
    sns.heatmap(data, cmap="viridis")
    plt.show()

if __name__ == '__main__':
    # response()
    data = load_stimuli()
    stimuli_heatmap(data, (0, 2000))