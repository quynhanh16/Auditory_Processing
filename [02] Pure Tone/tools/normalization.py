import os

import h5py
import numpy as np
from scipy.io import loadmat

from .utils import get_response_data_by_trigger, save_recording_data_to_h5file, get_response_data_all_triggers


def raw_response(save_h5file: bool = False):
    stimulus = loadmat('./data/Stimulus/PureToneParameters.mat')
    # params = stimulus['params']
    # frequency = stimulus['frequency']
    # attenuation = stimulus['attenuation']

    # Stimuli Duration | Total Time | Frequency | Amplitude
    stimulus_presentation_order = stimulus['stimulus_presentation_order']

    # First element in params. Sampling Frequency of the stimulus
    # Attenuation: how loud the sounds are? There are 8 level. 0-70 in decibels.
    # Frequency: how to frequent? 4000-40,000 hz. 45.
    print("Stim Ord", stimulus_presentation_order.shape)

    # 45 Frequencies -> 8 Amplitudes -> num_files x 3000
    normal_response = {}

    for idx, order in enumerate(stimulus_presentation_order):
        print(f"Working on {idx + 1} / {stimulus_presentation_order.shape[0]}...")
        freq = order[2]
        ampl = order[3]

        if freq not in normal_response:
            normal_response[freq] = {}

        if ampl not in normal_response[freq]:
            normal_response[freq][ampl] = np.empty((66, 3000))

        path = "./data/pure_tones"

        for idx2, tone in enumerate(os.listdir(path)):
            mat_file = os.listdir(os.path.join(path, tone))[0]
            trig_data = get_response_data_by_trigger(os.path.join(path, tone, mat_file), idx, 3000)
            normal_response[freq][ampl][idx2] = trig_data

    if save_h5file:
        with h5py.File("./raw_data.h5", "w") as h5file:
            save_recording_data_to_h5file(h5file, normal_response)

    return normal_response


def min_normalized_response_per_neuron():
    stimulus = loadmat('./data/Stimulus/PureToneParameters.mat')
    stimulus_presentation_order = stimulus['stimulus_presentation_order']
    normalized_response = {}

    for idx, order in enumerate(stimulus_presentation_order):
        print(f"Working on {idx + 1} / {stimulus_presentation_order.shape[0]}...")
        freq = order[2]
        ampl = order[3]

        if freq not in normalized_response:
            normalized_response[freq] = {}
        if ampl not in normalized_response[freq]:
            normalized_response[freq][ampl] = np.empty((66, 3000))

        path = "./data/pure_tones"

        for idx2, tone in enumerate(os.listdir(path)):
            mat_file = os.listdir(os.path.join(path, tone))[0]
            idk_man = get_response_data_by_trigger(os.path.join(path, tone, mat_file), idx, -1)
            trig_data = get_response_data_by_trigger(os.path.join(path, tone, mat_file), idx, 3000)
            min_value = np.min(idk_man)
            normalized_response[freq][ampl][idx2] = trig_data - min_value

    return normalized_response


def min_normalized_response_per_trigger():
    stimulus = loadmat('./data/Stimulus/PureToneParameters.mat')

    stimulus_presentation_order = stimulus['stimulus_presentation_order']

    normalized_response = {}

    for idx, order in enumerate(stimulus_presentation_order):
        print(f"Working on {idx + 1} / {stimulus_presentation_order.shape[0]}...")
        freq = order[2]
        ampl = order[3]

        if freq not in normalized_response:
            normalized_response[freq] = {}

        if ampl not in normalized_response[freq]:
            normalized_response[freq][ampl] = np.empty((66, 3000))

        path = "./data/pure_tones"

        for idx2, tone in enumerate(os.listdir(path)):
            mat_file = os.listdir(os.path.join(path, tone))[0]
            trig_data = get_response_data_by_trigger(os.path.join(path, tone, mat_file), idx, 3000)
            min_value = np.min(trig_data)
            normalized_response[freq][ampl][idx2] = trig_data - min_value

    return normalized_response


def standardize_response_100ms():
    stimulus = loadmat('./data/Stimulus/PureToneParameters.mat')

    stimulus_presentation_order = stimulus['stimulus_presentation_order']

    standardize_response = {}

    for idx, order in enumerate(stimulus_presentation_order):
        print(f"Working on {idx + 1} / {stimulus_presentation_order.shape[0]}...")
        freq = order[2]
        ampl = order[3]

        if freq not in standardize_response:
            standardize_response[freq] = {}

        if ampl not in standardize_response[freq]:
            standardize_response[freq][ampl] = np.empty((66, 3000))

        path = "./data/pure_tones"

        for idx2, tone in enumerate(os.listdir(path)):
            mat_file = os.listdir(os.path.join(path, tone))[0]
            all_data = get_response_data_all_triggers(os.path.join(path, tone, mat_file))
            trig_data = get_response_data_by_trigger(os.path.join(path, tone, mat_file), idx, 3000)
            data = []

            for n, obs in enumerate(all_data):
                if n % 3000 > 1000:
                    data.append(obs)

            stats = np.array(data)
            mean = np.mean(stats)
            std = np.std(stats)

            standardize_response[freq][ampl][idx2] = (trig_data - mean) / std

    return standardize_response


def pool_function(idx: int, path: str):
    """
    Pool function: Given a freq and amp open

    :param idx:
    :param path:
    :return:
    """
    mat_file = os.listdir(path)[0]
    trig_data = get_response_data_by_trigger(os.path.join(path, mat_file), idx, 3000)
    result = trig_data

    return idx, result


# TODO: Function does not fasten process dramatically. Another option is to task each processor to a recording and
#  let the processor find the data for each of the Combinations of frequency and amplitude.
def loading_data_faster():
    from multiprocessing import Pool
    stimulus = loadmat('./data/Stimulus/PureToneParameters.mat')
    stimulus_presentation_order = stimulus['stimulus_presentation_order']
    path = "./data/pure_tones"
    response = {}

    for idx, order in enumerate(stimulus_presentation_order):
        print(f"Working on {idx + 1} / {stimulus_presentation_order.shape[0]}...")
        freq = order[2]
        ampl = order[3]

        if freq not in response:
            response[freq] = {}

        neurons = [(n_file, os.path.join(path, neuron_file)) for n_file, neuron_file in enumerate(os.listdir(path))]
        with Pool(5) as pool:
            data_across_neurons = pool.starmap(pool_function, neurons)
            sorted_data_across_neurons = sorted(data_across_neurons, key=lambda x: x[0])
            data = [sorted_data_across_neurons[0][1]]

            for neuron in sorted_data_across_neurons[1:]:
                data = np.vstack((data, neuron[1]))

            response[freq][ampl] = data

    return response
