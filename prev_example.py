import os

import matplotlib.pyplot as plt
import numpy as np

from nems.tools import epoch
from nems.tools.recording import load_recording

# READING TGZ DATA
print("STARTING PROGRAM")
signals_dir = "file://"
cellids = ["DRX006b-128-2", "ARM030a-40-2"]
filebase = 'A1_DRX006b_adb77e6b989c6b08bf7fce0c19ab1a94ba124399.tgz'
tgz_file = 'A1_NAT4_ozgf.fs100.ch18.tgz'
cellid = "DRX006b-128-2"

datafile = os.path.join(signals_dir, 'A1_single_sites', filebase)
# datafile = os.path.join(signals_dir, tgz_file)

print("LOADING FILES")
recordings = load_recording(datafile)
resp = recordings['resp'].rasterize()
single_cell_resp = resp.extract_channels([cellid])
val_epochs = epoch.epoch_names_matching(resp.epochs, "^STIM_00")
r = single_cell_resp.extract_epoch(val_epochs[0])

# print("-" * 20)
# print("Recording: \n", recordings)
# print("-" * 20)
# print("Response: \n", resp)

# Raster Plot
y, _, x = np.where(r)
x_sec = x / 1000 - 0.25
plt.figure()
plt.scatter(x_sec, y, s=1, color='black')
plt.title(f"Raster, cell: {cellid} stim: {val_epochs[0]}")
plt.xlabel('time from stimulus onset (s)')
plt.ylabel('trial')
plt.show()
