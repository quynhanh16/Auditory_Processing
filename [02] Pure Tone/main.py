import h5py
from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tools import utils


def trigger_data(file_path: str, n_trigger: int, trigger_length: int = 3000) -> np.ndarray:
    """
    Given a path to a .mat file, return the nth trigger of specified length.
    For this dataset, the trigger length is set to 3000.

    :param file_path:
    :param n_trigger:
    :param trigger_length:
    :return: np.ndarray
    """
    file = loadmat(file_path)
    response_signals = file['response_signal']
    triggers = file['trigger'][0]

    trigger = triggers[n_trigger - 1]

    if trigger_length == -1:
        return response_signals.T[0]

    return response_signals[trigger: trigger + trigger_length].T[0]


def save_dict(h5file, dictionary) -> None:
    """

    :param h5file:
    :param dictionary:
    :return:
    """
    for freq in dictionary.keys():
        h5file.create_group(str(freq))

    for freq, amps in dictionary.items():
        for amp, data in amps.items():
            h5file.create_dataset(f"{freq}/{amp}", data=data)


def raw_x_normalized_z_data():
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

    # TODO: Frequency to Amplitude
    # 45 Frequencies -> 8 Amplitudes -> num_files x 3000
    normal_response = {}
    normalized_response = {}

    for idx, order in enumerate(stimulus_presentation_order):
        print(f"Working on {idx + 1} / {stimulus_presentation_order.shape[0]}...")
        freq = order[2]
        ampl = order[3]

        if freq not in normal_response:
            normal_response[freq] = {}

        if freq not in normalized_response:
            normalized_response[freq] = {}

        if ampl not in normal_response[freq]:
            normal_response[freq][ampl] = np.empty((66, 3000))

        if ampl not in normalized_response[freq]:
            normalized_response[freq][ampl] = np.empty((66, 3000))

        path = "./data/pure_tones"

        for idx2, tone in enumerate(os.listdir(path)):
            mat_file = os.listdir(os.path.join(path, tone))[0]
            trig_data = trigger_data(os.path.join(path, tone, mat_file), idx, 3000)
            normal_response[freq][ampl][idx2] = trig_data
            min_value = np.min(trig_data)
            normalized_response[freq][ampl][idx2] = trig_data - min_value

    with h5py.File("./raw_data.h5", "w") as h5file:
        save_dict(h5file, normal_response)

    with h5py.File("./normalized_data.h5", "w") as h5file:
        save_dict(h5file, normalized_response)

    return normal_response, normalized_response


def graphing(data):
    # Frequencies by Amplitudes
    freq_n = 5
    amp_n = 5
    f, ax = plt.subplots(freq_n, amp_n)
    n = 0

    for _, amps in data.items():
        if n >= freq_n:
            break
        m = 0
        for amp, data in amps.items():
            if m >= amp_n:
                break
            for rec in data:
                ax[n, m].plot(rec, alpha=0.5, color="red")
            ax[n, m].plot(np.mean(data, axis=0), color="blue")
            m += 1
        n += 1

    # Baseline: 150 ms - 300 ms -> 1500 - 3000
    # z-score using the sum of the 360 units.

    plt.show()

def frequency_response_area(data1, data2):
    # 66 x 360
    a = np.empty((66, 8, 45))
    b = np.empty((66, 8, 45))

    n = 0
    sort_items = sorted(data1.items(), key=lambda item: int(item[0]))
    for freq, amps in sort_items:
        m = 0
        sort_amps = sorted(amps.items(), key=lambda item:int(item[0]))
        if m > 0:
            break
        for amp, value in sort_amps:
            for idx, rec in enumerate(value):
                max_value = np.max(rec[:500])
                a[idx, m, n] = max_value
            m += 1
        n += 1

    n = 0
    sort_items = sorted(data2.items(), key=lambda item: int(item[0]))
    for freq, amps in sort_items:
        m = 0
        sort_amps = sorted(amps.items(), key=lambda item:int(item[0]))
        if m > 0:
            break
        for amp, value in sort_amps:
            for idx, rec in enumerate(value):
                max_value = np.max(rec[:500])
                b[idx, m, n] = max_value
            m += 1
        n += 1


    for i in range(66):
        f, ax = plt.subplots(1, 2)
        plt.title(f"Neuron {i + 1}")
        sns.heatmap(a[i,:,:], ax=ax[0])
        ax[0].set_title("Raw")
        sns.heatmap(b[i,:,:], ax=ax[1])
        ax[1].set_title("Normalized")
        plt.gca().invert_yaxis()
        plt.show()

def normalization(data):
    new_data = {}
    big_m = [np.array([])] * 66
    # print(f"({len(big_m)}, {big_m[0].shape})")

    for _, amps in data.items():
        for amp, value in amps.items():
            for idx, rec in enumerate(value):
                big_m[idx] = np.append(big_m[idx], rec[1500:])


    # print(f"({len(big_m)}, {big_m[0].shape})")
    average = [np.mean(a) for a in big_m]
    std = [np.std(a) for a in big_m]

    # print(average)
    # print(std)

    for freq, amps in data.items():
        new_data[freq] = {}
        for amp, value in amps.items():
            new_data[freq][amp] = np.empty((66, 3000))
            for idx, rec in enumerate(value):
                new_data[freq][amp][idx] = (rec - average[idx]) / std[idx]

    return new_data

# List of TODOs
# - Time plot of the neurons
def fra_plot(data1, data2):
    # 66 x 360
    # Frequency, Amplitude, Time Series
    a = np.empty((66, 8, 45 * 500))
    b = np.empty((66, 8, 45 * 500))

    n = 0
    sort_items = sorted(data1.items(), key=lambda item: int(item[0]))
    for freq, amps in sort_items:
        m = 0
        sort_amps = sorted(amps.items(), key=lambda item:int(item[0]))
        if m > 0:
            break
        for amp, value in sort_amps:
            for idx, rec in enumerate(value):
                for ids, obs in enumerate(rec[:500]):
                    a[idx, m, n * 500 + ids] = obs
            m += 1
        n += 1

    n = 0 # N
    sort_items = sorted(data2.items(), key=lambda item: int(item[0]))
    for freq, amps in sort_items:
        m = 0 # Amplitudes
        sort_amps = sorted(amps.items(), key=lambda item:int(item[0]))
        for amp, value in sort_amps:
            print(freq, amp)
            for idx, rec in enumerate(value):
                for ids, obs in enumerate(rec[:500]):
                    b[idx, m, n * 500 + ids] = obs
            m += 1
        n += 1


    # for i in range(66):
    #     f, ax = plt.subplots(1, 2)
    #     plt.title(f"Neuron {i + 1}")
    #     for j in range(0, 71, 10):
    #         ax[0].plot(a[i,int(j/10),:] + j)
    #     ax[0].set_title("Raw")
    #     for j in range(0, 71, 10):
    #         ax[1].plot(b[i,int(j/10),:] + j)
    #     ax[1].set_title("Normalized")
    #     plt.show()

def new_fra_plot(data1):
    # 66 x 360
    # Frequency, Amplitude, Time Series
    a = np.empty((66, 8, 2 * 500))

    n = 0 # Frequencies
    sort_items = sorted(data1.items(), key=lambda item: int(item[0]))
    for freq, amps in sort_items:
        m = 0 # Amplitudes
        sort_amps = sorted(amps.items(), key=lambda item:int(item[0]))
        if n > 1:
            continue
        for amp, value in sort_amps:
            print(freq, amp)
            for idx, rec in enumerate(value):
                for ids, obs in enumerate(rec[:500]):
                    a[idx, m, n * 500 + ids] = obs
            m += 1
        n += 1

    for i in range(66):
        plt.title(f"Neuron {i + 1}")
        for j in range(0, 71, 10):
            plt.plot(a[i,int(j/10),:] + j)
        plt.show()

def normalization_whole_thing():
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
            idk_man = trigger_data(os.path.join(path, tone, mat_file), idx, -1)
            trig_data = trigger_data(os.path.join(path, tone, mat_file), idx, 3000)
            min_value = np.min(idk_man)
            normalized_response[freq][ampl][idx2] = trig_data - min_value

    return normalized_response

def normalization_base_line():
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
            trig_data = trigger_data(os.path.join(path, tone, mat_file), idx, 3000)
            mean = np.mean(trig_data[1000:])
            std = np.std(trig_data[1000:])
            normalized_response[freq][ampl][idx2] = (trig_data - mean) / std

    return normalized_response

def pool_function(path, idx):
    mat_file = os.listdir(path)[0]
    trig_data = trigger_data(os.path.join(path, mat_file), idx, 3000)
    mean = np.mean(trig_data[1000:])
    std = np.std(trig_data[1000:])


def loading_data_faster():
    from multiprocessing import Pool
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

        neurons = os.listdir(path)
        with Pool() as pool:
            pass

        for idx2, tone in enumerate(os.listdir(path)):
            mat_file = os.listdir(os.path.join(path, tone))[0]
            idk_man = trigger_data(os.path.join(path, tone, mat_file), idx, -1)
            trig_data = trigger_data(os.path.join(path, tone, mat_file), idx, 3000)
            normalized_response[freq][ampl][idx2] = (trig_data - mean) / std

    return normalized_response

if __name__ == '__main__':
    # trigger_data('./data/pure_tones/2017-12-26/171226001_raw.mat', 1, 3000)
    # response()
    # normal_data = utils.load_state("./data/states/raw_data.pkl")
    # normalized_data = utils.load_state("./data/states/normalized_data.pkl")
    #
    # if normal_data is None or normalized_data is None:
    #     normal_data, normalized_data = raw_x_normalized_z_data()
    #     utils.save_state("./data/states/raw_data.pkl", normal_data)
    #     utils.save_state("./data/states/normalized_data.pkl", normalized_data)

    min_normalized_across_all_stimuli = utils.load_state("./data/states/min_normalized_data_across_all_stimuli.h5")
    if min_normalized_across_all_stimuli is None:
        min_normalized_across_all_stimuli = normalization_whole_thing()
        utils.save_state("./data/states/min_normalized_across_all_stimuli.pkl", min_normalized_across_all_stimuli)

    normalized_from_100ms_300ms_per_stimuli = utils.load_state("./data/states/normalized_from_100ms_300ms_per_stimuli")
    if normalized_from_100ms_300ms_per_stimuli is None:
        normalized_from_100ms_300ms_per_stimuli = normalization_base_line()
        utils.save_state("./data/states/normalized_from_100ms_300ms_per_stimuli.pkl", normalized_from_100ms_300ms_per_stimuli)

    # graphing(normal_data)
    # frequency_response_area(normal_data, d)
    # fra_plot(normal_data, d)
    new_fra_plot(min_normalized_across_all_stimuli)
    # a = list(list(d.items())[0][1].items())[0][1][0]
    # a = list(list(normal_data.items())[2][1].items())[0][1][0]
    # plt.plot(a)
    # plt.show()
