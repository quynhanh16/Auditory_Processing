import os

import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import mean_shift

from .utils import get_response_data_by_trigger, save_recording_data_to_h5file, get_response_data_all_triggers


def min_normalized_response_per_neuron(data):
    pass


def min_normalized_response_per_trigger(data):
    for neuron, freqs in data.items():
        for freq, amps in freqs.items():
            for amp, response in amps.items():
                min_value = min(response)
                amps[amp] = [n - min_value for n in response]

    return data


def standardize_response_100ms(data):
    """
    Standardize te response data by subtracting the mean and dividing by the standard deviation.
    The mean and standard deviation are calculated for the entirety of each neuron.
    :return:
    """
    for neuron, freqs in data.items():
        all_data = []
        for freq, amps in freqs.items():
            for amp, response in amps.items():
                all_data.extend(response)

        exp_value = 0
        exp_value2 = 0

        for n in all_data:
            exp_value += n
            exp_value2 += n ** 2

        mean = exp_value / len(all_data)
        std = (exp_value2 / len(all_data) - mean ** 2) ** 0.5

        for freq, amps in freqs.items():
            for amp, response in amps.items():
                data[neuron][freq][amp] = [(n - mean) / std for n in response]

    return data
