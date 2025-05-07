import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# FRA Plot
def fra_plot(data, n_neuron=66, n_freq=(0, 45)):
    if n_neuron > 66 or (n_freq[0] < 0 and n_freq[1] > 45):
        print("Invalid input")
        return

    for idx, (n, v) in enumerate(data.items()):
        if idx == n_neuron:
            break
        neuron = sorted(list(v.items()), key=lambda a: a[0])
        new = [[] for _ in range(8)]

        for freq, amps in neuron[n_freq[0]:n_freq[1]]:
            ordered_amps = sorted(list(amps.items()), key=lambda a: a[0])
            for i, (amp, data) in enumerate(ordered_amps):
                new[i].extend(data)

        fig, ax = plt.subplots(8, 1)
        for j in range(len(new)):
            i = 7 - j
            ax[i].plot(new[j], color = "black", linewidth = 0.5)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['left'].set_visible(False)
            ax[i].set_xticks([])
            if i == 7:
                ax[i].set_xticks([0], labels=["40"])
                ticks = [t for t in range(0, len(new[0]), 3000)]
                labels = [f"{neuron[i][0] / 1000:.2f}" for i in range(n_freq[0], n_freq[1])]
                if len(ticks) > 5:
                    ticks = [ticks[i] for i in range(0, len(ticks), int(len(ticks) / 5))]
                    labels = [labels[i] for i in range(0, len(labels), int(len(labels) / 5))]
                ax[i].set_xticks(ticks, labels=labels)
            ax[i].set_yticks([])
            ax[i].set_ylabel(int(i * 10), fontsize=9, rotation="horizontal")

        fig.suptitle(f"Neuron {idx + 1}: {n}", fontsize=9)
        fig.text(0.03, 0.5, 'Amplitude', va='center', ha='center', rotation='vertical', fontsize=9)
        fig.text(0.5, 0.03, 'Frequency (Hz)', va='center', ha='center', rotation='horizontal', fontsize=9)
        plt.tight_layout(rect=(0.03, 0.03, 1, 1))
        plt.show()


def fra_plot_with_spikes(data, spike_data, n_neuron=66, n_freq=(0, 45)):
    if n_neuron > 66 or (n_freq[0] < 0 and n_freq[1] > 45):
        print("Invalid input")
        return

    for idx, ((n, v), (sn, sv)) in enumerate(zip(data.items(), spike_data.items())):
        if idx == n_neuron:
            break

        neuron = sorted(list(v.items()), key=lambda a: a[0])
        spike_neuron = sorted(list(sv.items()), key=lambda a: a[0])
        new = [[] for _ in range(8)]
        spike_new = [[] for _ in range(8)]

        for freq, amps in neuron[n_freq[0]:n_freq[1]]:
            ordered_amps = sorted(list(amps.items()), key=lambda a: a[0])
            for i, (amp, data) in enumerate(ordered_amps):
                new[i].extend(data)

        for freq, amps in spike_neuron[n_freq[0]:n_freq[1]]:
            ordered_amps = sorted(list(amps.items()), key=lambda a: a[0])
            for i, (amp, data) in enumerate(ordered_amps):
                spike_new[i].extend(data)

        fig, ax = plt.subplots(8, 1)
        for j in range(len(new)):
            i = 7 - j
            ax[i].plot(new[j], color = "black", linewidth = 0.5)
            for x, y in enumerate(spike_new[j]):
                if y == 1:
                    ax[i].axvline(x, color = "red", linewidth = 0.5)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['left'].set_visible(False)
            ax[i].set_xticks([])
            if i == 7:
                ax[i].set_xticks([0], labels=["40"])
                ticks = [t for t in range(0, len(new[0]), 3000)]
                labels = [f"{neuron[i][0] / 1000:.2f}" for i in range(n_freq[0], n_freq[1])]
                if len(ticks) > 5:
                    ticks = [ticks[i] for i in range(0, len(ticks), int(len(ticks) / 5))]
                    labels = [labels[i] for i in range(0, len(labels), int(len(labels) / 5))]
                ax[i].set_xticks(ticks, labels=labels)
            ax[i].set_yticks([])
            ax[i].set_ylabel(int(i * 10), fontsize=9, rotation="horizontal")

        fig.suptitle(f"Neuron {idx + 1}: {n}", fontsize=9)
        fig.text(0.03, 0.5, 'Amplitude', va='center', ha='center', rotation='vertical', fontsize=9)
        fig.text(0.5, 0.03, 'Frequency (Hz)', va='center', ha='center', rotation='horizontal', fontsize=9)
        plt.tight_layout(rect=(0.03, 0.03, 1, 1))
        plt.show()

def plot_firing_rate(firing_rate, interval=None):
    if interval is not None:
        firing_rate = firing_rate[interval[0]:interval[1]]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.step([i for i in range(len(firing_rate))], firing_rate, where='pre', color="black")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def new_fra_plot(data, n_neuron=66, n_freq=(0, 45)):
    # n -> data in 30
    # n -> freq x ampl -> 30
    if n_neuron > 66 or (n_freq[0] < 0 and n_freq[1] > 45):
        print("Invalid input")
        return

    for idx, (n, v) in enumerate(data.items()):
        if idx == n_neuron:
            break
        neuron = sorted(list(v.items()), key=lambda a: a[0])
        new = [[] for _ in range(8)]

        for freq, amps in neuron[n_freq[0]:n_freq[1]]:
            ordered_amps = sorted(list(amps.items()), key=lambda a: a[0])
            for i, (amp, data) in enumerate(ordered_amps):
                new[i].extend(data)

        print(len(new))
        print(len(new[0]))
        y_labels = ["0", "10", "20", "30", "40", "50", "60", "70"]
        ticks = [t for t in range(0, len(new[0]), 30)]
        labels = [f"{neuron[i][0] / 1000:.2f}" for i in range(n_freq[0], n_freq[1])]
        if len(ticks) > 5:
            ticks = [ticks[i] for i in range(0, len(ticks), int(len(ticks) / 5))]
            labels = [labels[i] for i in range(0, len(labels), int(len(labels) / 5))]
        sns.heatmap(new, yticklabels=y_labels)

        plt.gca().invert_yaxis()
        plt.suptitle(f"Neuron {idx + 1}: {n}", fontsize=9)
        plt.ylabel("Amplitude", fontsize=9)
        plt.xlabel("Frequency", fontsize=9)
        plt.tight_layout(rect=(0.03, 0.03, 1, 1))
        plt.show()

def time_plot(data, n_neuron=66, n_freq=(0, 45)):
    if n_neuron > 66 or (n_freq[0] < 0 and n_freq[1] > 45):
        print("Invalid input")
        return

    for idx, (n, v) in enumerate(data.items()):
        if idx == n_neuron:
            break
        neuron = sorted(list(v.items()), key=lambda a: a[0])
        new = []

        for freq, amps in neuron[n_freq[0]:n_freq[1]]:
            ordered_amps = sorted(list(amps.items()), key=lambda a: a[0])
            for i, (amp, data) in enumerate(ordered_amps):
                new[i].extend(data)

        fig, ax = plt.subplots(8, 1)
        for j in range(len(new)):
            i = 7 - j
            ax[i].plot(new[j], color = "black", linewidth = 0.5)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['left'].set_visible(False)
            ax[i].set_xticks([])
            if i == 7:
                ax[i].set_xticks([0], labels=["40"])
                ticks = [t for t in range(0, len(new[0]), 3000)]
                labels = [f"{neuron[i][0] / 1000:.2f}" for i in range(n_freq[0], n_freq[1])]
                if len(ticks) > 5:
                    ticks = [ticks[i] for i in range(0, len(ticks), int(len(ticks) / 5))]
                    labels = [labels[i] for i in range(0, len(labels), int(len(labels) / 5))]
                ax[i].set_xticks(ticks, labels=labels)
            ax[i].set_yticks([])
            ax[i].set_ylabel(int(i * 10), fontsize=9, rotation="horizontal")

        fig.suptitle(f"Neuron {idx + 1}: {n}", fontsize=9)
        fig.text(0.03, 0.5, 'Amplitude', va='center', ha='center', rotation='vertical', fontsize=9)
        fig.text(0.5, 0.03, 'Frequency (Hz)', va='center', ha='center', rotation='horizontal', fontsize=9)
        plt.tight_layout(rect=(0.03, 0.03, 1, 1))
        plt.show()
