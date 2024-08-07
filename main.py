# File: main.py
# Purpose: This file is a place to test nems tools.
# the final version will abstract each of the steps
# into different files.

import os

import matplotlib.pyplot as plt
# Packages
import numpy as np

from nems import Model
from nems.layers import WeightChannels, FiniteImpulseResponse, DoubleExponential
from nems.tools.recording import load_recording, Recording
from nems.tools.signal import RasterizedSignal

options = {'options': {'maxiter': 10, 'ftol': 1e-4}}

# READING TGZ DATA
print('-' * 20)
print("STARTING PROGRAM")

signals_dir: str = "file://"
tgz_file: str = 'A1._NAT4_ozgf.fs100.ch18tgz'
datafile = os.path.join(signals_dir, tgz_file)

# Loading recordings
# Current recordings are divided into resp, stim, and mask_est
# mark_est does not contain anything
print('-' * 20)
print("LOADING FILES")

recordings: Recording = load_recording(datafile)
stim: RasterizedSignal = recordings['stim']
resp: RasterizedSignal = recordings['resp']
stim_data = np.transpose(stim)
resp_data = np.transpose(resp.data)

print('-' * 20)
print('Stimuli Data')
print(stim_data.shape)

print('-' * 20)
print('Response Data')
print(resp_data.shape)

# Creating ModeResponse Data
# TODO: Replicate the Paper's Model Architecture
print('-' * 20)
print('Model')
model = Model(name="Rank1LNSTRF")
model.add_layers(
    WeightChannels(shape=(18, 1)),  # 18 spectral channels->1 composite channels
    FiniteImpulseResponse(shape=(15, 1)),  # 15 taps, 1 spectral channel
    DoubleExponential(shape=(849,))  # static nonlinearity, 1 output
)

# TODO: Fitting Model using GPU
# fitted_model = model.fit(stim_data, resp_data, fitter_options=options, backend='scipy')
# fitted_model.plot(stim_data, target=resp_data)

model.plot(stim_data, target=resp_data)
plt.show()
