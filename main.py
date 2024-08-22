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

file_list = [
    'A1_TNC015a_abf3608afd9b5c83bf8ed17a5ac61040e9c83261.tgz',
    'A1_TNC020a_2f0624a284401ad0e40680435fc57de3873e0a2c.tgz',
    'A1_ARM030a_9b0586bd962b60ab7b02a21771c98695134b8be8.tgz',
    'A1_TNC008a_2eb34ed73bb45f087a72652b5fcea810a9f2e9c9.tgz',
    'A1_ARM029a_aefada7c479857e55fdafe3e00e94df189250651.tgz',
    'A1_TNC021a_89668bbfb0d89473b8efff14b7b6625bb58060dd.tgz',
    'A1_TNC017a_2d5a6c489bf573e2f762eb2e17ca5478271fcc64.tgz',
    'A1_TNC012a_c677f8485dfd121765cd31de50230196a8a4bcc1.tgz',
    'A1_TNC010a_2f50864a77d8ec2f1786ebd8b0161114a9e905b1.tgz',
    'A1_TNC014a_a0ebe478cdce9612658c29d6766859fd66f1a3b6.tgz',
    'A1_TNC016a_13decf694dbf004e611c8e362e23b6fa7852ee4b.tgz',
    'A1_DRX006b_adb77e6b989c6b08bf7fce0c19ab1a94ba124399.tgz',
    'A1_DRX007a_5409c3cb81c2745cec8af5f9a1402ed24ea7ec7d.tgz',
    'A1_DRX008b_240d4f51148b270f6dc099eaccd4f316ce3021f5.tgz',
    'A1_ARM032a_d28371c918efb9917c560f41e51ff15efdf516c5.tgz',
    'A1_CRD017c_07010f7f2781fc649e4f1f90679212e60a7ff3b4.tgz',
    'A1_CRD016d_f9cf97eab58415d6187e5f104e9df6bf06a9fd99.tgz',
    'A1_TNC013a_cc40dccc6e141410c8b0c16e403b763ad368b170.tgz',
    'A1_ARM031a_e73a3420ba4e26d680d9a8adc5bef1c32f6d9617.tgz',
    'A1_ARM033a_8bc7cdda34517574d7973a1b6352d52d873bad7b.tgz',
    'A1_TNC009a_91819235d1188908cee2787e5769d3613fbd756f.tgz',
    'A1_TNC018a_2d5a31aeb27af29f52739c37061da96fcb058e96.tgz'
]

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
