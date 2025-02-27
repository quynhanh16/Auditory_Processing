import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# FRA Plot
def fra_plot(data, n_neurons=66, n_freq=6):
    if n_neurons > 66:
        print("Too many neurons. Max: 66")
        return

    if n_freq > 45:
        print("Too many frequencies. Max: 45")
        return

    # 66 x 360
    # Frequency, Amplitude, Time Series
    df = np.empty((66, 8, n_freq * 500))

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

    for i in range(n_neurons):
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
                    f = [f[i] for i in range(0, len(f), int(len(f) / 5))]
                    g = [g[i] for i in range(0, len(g), int(len(g) / 5))]
                ax.set_xticks(g, labels=f)

        for j in range(8):
            axs[j].plot(df[i, (7 - j), :], color="black", linewidth=0.5)
            axs[j].set_ylabel((7 - j) * 10, fontsize=9, rotation="horizontal")

        fig.suptitle(f"Neuron {i + 1}")
        fig.text(0.03, 0.5, 'Amplitude', va='center', ha='center', rotation='vertical', fontsize=9)
        fig.text(0.5, 0.03, 'Frequency (Hz)', va='center', ha='center', rotation='horizontal', fontsize=9)
        plt.tight_layout(rect=(0.03, 0.03, 1, 1))
        plt.show()


def frequency_response_area(data1, data2):
    # 66 x 360
    a = np.empty((66, 8, 45))
    b = np.empty((66, 8, 45))

    n = 0
    sort_items = sorted(data1.items(), key=lambda item: int(item[0]))
    for freq, amps in sort_items:
        m = 0
        sort_amps = sorted(amps.items(), key=lambda item: int(item[0]))
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
        sort_amps = sorted(amps.items(), key=lambda item: int(item[0]))
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
        sns.heatmap(a[i, :, :], ax=ax[0])
        ax[0].set_title("Raw")
        sns.heatmap(b[i, :, :], ax=ax[1])
        ax[1].set_title("Normalized")
        plt.gca().invert_yaxis()
        plt.show()


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
