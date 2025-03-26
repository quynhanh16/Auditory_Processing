import sys

from tools import *
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


def m():
    a = load_state("./data/states/min_normalized.pkl")
    fra_plot(a, 1, 5)


def main():
    path = "./data/pure_tones"
    neuron_files = [os.path.join(path, f) for f in os.listdir("./data/pure_tones")]
    print(neuron_files)
    neuron = [neuron_files + f for f in os.listdir(neuron_files)[0]]
    print(neuron)
    mat_file = loadmat(neuron)
    print(mat_file.keys())


if __name__ == '__main__':
    main()
