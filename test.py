import numpy as np

from tools.utils import load_state, prepare_stimuli, prepare_response, save_state
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys

#
# from tools.utils import load_state, save_state
#
# X = load_state("./stim_m18_d20_t(27, 889.5).pkl")
#

# save_state("x_model.pkl", new_x)

model = tf.keras.models.load("ours.keras")
