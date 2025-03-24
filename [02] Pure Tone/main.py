import sys

from tools import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

def spikes_response(spike, save_h5file: bool = False):
    stimulus = loadmat('./data/Stimulus/PureToneParameters.mat')
    stimulus_presentation_order = stimulus['stimulus_presentation_order']
    # 45 Frequencies -> 8 Amplitudes -> num_files x 3000
    spike_file_name = f"spike{spike}"
    normal_response = {}
    for idx, order in enumerate(stimulus_presentation_order):
        print(f"Working on {idx + 1} / {stimulus_presentation_order.shape[0]}...")
        freq = order[2]
        ampl = order[3]

        if freq not in normal_response:
            normal_response[freq] = {}

        if ampl not in normal_response[freq]:
            normal_response[freq][ampl] = np.empty((66, 3000))

        path = "./data/pure_tones_spikes"

        for idx2, neuron in enumerate(os.listdir(path)):
            mat_file = os.listdir(os.path.join(path, neuron))
            mat_file = [file for file in mat_file if spike_file_name in file][0]
            trig_data = get_response_data_by_trigger_spikes(os.path.join(path, neuron, mat_file), idx + 1, 3000)
            normal_response[freq][ampl][idx2] = trig_data

    if save_h5file:
        with h5py.File("./raw_spike10_data.pkl", "w") as h5file:
            save_recording_data_to_h5file(h5file, normal_response)

    return normal_response


def fra_plot2(original, data, n_neuron, n_freq=45):
    if n_neuron > 66:
        print("Too many neurons. Max: 66")
        return

    if n_freq > 45:
        print("Too many frequencies. Max: 45")
        return

    # 66 x 360
    # Frequency, Amplitude, Time Series
    org = np.empty((66, 8, n_freq * 500))
    df = np.empty((66, 8, n_freq * 500))

    n = 0  # Frequencies
    org_sort = sorted(original.items(), key=lambda item: int(item[0]))
    for freq, amps in org_sort:
        m = 0  # Amplitudes
        sort_amps = sorted(amps.items(), key=lambda item: int(item[0]))
        if n > n_freq - 1:
            continue
        for amp, value in sort_amps:
            for idx, rec in enumerate(value):
                for ids, obs in enumerate(rec[:500]):
                    org[idx, m, n * 500 + ids] = obs
            m += 1
        n += 1


    n = 0  # Frequencies
    sort_items = sorted(data.items(), key=lambda item: int(item[0]))
    for freq, amps in sort_items:
        m = 0  # Amplitudes
        sort_amps = sorted(amps.items(), key=lambda item: int(item[0]))
        if n > n_freq - 1:
            continue
        for amp, value in sort_amps:
            for idx, rec in enumerate(value):
                for ids, obs in enumerate(rec[:500]):
                    df[idx, m, n * 500 + ids] = obs
            m += 1
        n += 1

    for i in range(n_neuron):
        # if i + 1 != n_neuron:
        #     continue
        fig, axs = plt.subplots(8, 1)

        for nax, ax in enumerate(axs.flatten()):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
            if nax != 7:
                ax.set_xticks([])
            else:
                keys = [item[0] for item in sort_items]
                f = [str(int(key / 1000)) for key in keys[:n_freq]]
                g = [(interval * 500) for interval in range(len(f))]
                if len(f) > 5:
                    f = [f[j] for j in range(0, len(f), int(len(f) / 5))]
                    g = [g[j] for j in range(0, len(g), int(len(g) / 5))]
                ax.set_xticks(g, labels=f)

        for j in range(8):
            axs[j].plot(df[i, (7 - j), :], color="red", linewidth=0.5)
            # axs[j].plot(org[i, (7 - j), :], color="black", linewidth=0.5)
            axs[j].set_ylabel((7 - j) * 10, fontsize=9, rotation="horizontal")

        fig.suptitle(f"Neuron {i + 1}")
        fig.text(0.03, 0.5, 'Amplitude', va='center', ha='center', rotation='vertical', fontsize=9)
        fig.text(0.5, 0.03, 'Frequency (Hz)', va='center', ha='center', rotation='horizontal', fontsize=9)
        plt.tight_layout(rect=(0.03, 0.03, 1, 1))
        plt.show()
        _ = input()

def testing():
    d = loadmat("./data/pure_tones/2017-09-01/17901001_raw.mat")
    df = loadmat("./data/pure_tones_spikes/2017-09-01/17901001_spike10.mat")
    df2 = loadmat("./data/pure_tones_spikes/2017-09-01/17901001_raw.mat")
    print(df['trigger'][0])
    d_resp = d['response_signal'].T[0]
    df_resp = df['binarySpike'].T[0]
    df2_resp = df2['response_signal'].T[0]
    print(np.where(df_resp == 1))
    new_resp = []
    for n in range(d_resp.shape[0]):
        if n == 0:
            continue
        if d_resp[n] - d_resp[n-1] > 10:
            new_resp.append(1)
        else:
            new_resp.append(0)

    new_resp = np.array(new_resp)
    spikes = np.where(new_resp == 1)
    print(spikes)
    i = spikes[0][3]
    print(df2_resp[i], df2_resp[i-1])
    plt.plot(d_resp[i-20:i+20])
    plt.show()
    # 96083 or 103151



if __name__ == '__main__':
    org = load_state("./data/states/raw_data.pkl")
    df = load_state("./data/states/raw_spike10_data.pkl")
    # fra_plot2(org, df, 66, 45)
    path = "./data/pure_tones_spikes"
    n = 0  # Frequencies
    # sort_items = sorted(df.items(), key=lambda item: int(item[0]))
    # df = np.empty((66, 8, 45 * 500))
    # for freq, amps in sort_items:
    #     m = 0  # Amplitudes
    #     sort_amps = sorted(amps.items(), key=lambda item: int(item[0]))
    #     for amp, value in sort_amps:
    #         for idx, rec in enumerate(value):
    #             for ids, obs in enumerate(rec[:500]):
    #                 df[idx, m, n * 500 + ids] = obs
    #         m += 1
    #     n += 1

    order = loadmat("./data/Stimulus/PureToneParameters.mat")['stimulus_presentation_order']
    for idx, neuron in enumerate(os.listdir(path)):
        t = None
        for resp in os.listdir(os.path.join(path, neuron)):
            if "spike10" in resp:
                t = loadmat(os.path.join(path, neuron, resp))
                first = get_response_data_by_trigger_spikes(os.path.join(path, neuron, resp), 1, 3000)
                print(np.where(first == 1))

    print(df.keys())
    print(np.where(df[4000.0][0] == 1))
    sys.exit()

    for idx, neuron in enumerate(os.listdir(path)):
        t = None
        for resp in os.listdir(os.path.join(path, neuron)):
            if "spike10" in resp:
                t = loadmat(os.path.join(path, neuron, resp))
        plt.plot(t['binarySpike'].T[0], color="red")
        plt.title(f"Neuron {idx + 1}")

        print(np.where(t['binarySpike'].T[0] == 1))

        _ = input("Next:")

    # d = loadmat("./data/pure_tones_spikes/2017-05-27/17527017_raw.mat")
    # f = loadmat("./data/pure_tones_spikes/2017-05-27/17527017_spike10.mat")
    # plt.plot(d['response_signal'].T[0], color="black")
    # plt.plot(f['binarySpike'].T[0], color="red")
    # plt.show()
    # df = load_state("./data/states/raw_spike10_data.pkl")
    # fra_plot2(org, df, 66, 45)
    # print(df[4000][0])
    # print(np.where(df[4000][0][46] == 1))
    # i = 205
    # plt.plot(org[4000][0][46][i-20:i+20])
    # plt.show()
    # spikes_response(10)
    # testing()
