import sys
import pickle

from tools import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


def load_spike(spike = 10):
    # TODO: Still need to implement details
    path = "./data/pure_tones_spikes"

    stim_file = "./data/Stimulus/PureToneParameters.mat"
    stim = loadmat(stim_file)
    order = [(freq, amp) for _, _, freq, amp in stim["stimulus_presentation_order"]]

    final = {}

    for neuron in os.listdir(path):
        final[neuron] = {}
        files = os.path.join(path, neuron)
        file = ""
        for f in os.listdir(files):
            if f"spike{spike}" in f:
                file = os.path.join(files, f)
                break
        mat_file = loadmat(file)
        triggers = mat_file["trigger"][0]
        response = mat_file["binarySpike"].T[0]

        for idx, (freq, amp) in enumerate(order):
            if freq not in final[neuron]:
                final[neuron][freq] = {}
            data = response[triggers[idx]: triggers[idx] + 3000].tolist()

            final[neuron][freq][amp] = data

    return final

if __name__ == '__main__':
    # a = load_data()
    a = load_data_by_neuron(55)
    df = a['2017-08-15_cell1']
    print(np.unique(np.array(df)))
    b = firing_rate(a, 100, population=True)
    print(len(b))
    plot_firing_rate(b, (0,1000))
    # b = standardize_response_100ms(a)

    # fra_plot_with_spikes(b, load_spike(10), n_neuron=66, n_freq=(0, 45))
