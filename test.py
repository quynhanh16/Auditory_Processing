import numpy as np

from tools.utils import load_state, prepare_stimuli
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

state_file = "state.pkl"
stim, resp = load_state(state_file)

from tools.utils import prepare_response

X = prepare_stimuli(stim, (0, 27), 18, 20)

new_x = np.empty((X.shape[0], 21, 18))
for i in range(X.shape[0]):
    for j in range(int(X.shape[1] / 18)):
        for k in range(int(X.shape[1] / 21)):
            new_x[i, j, k] = X[i, (18 * j) + k]

y = prepare_response(resp, (0, 27), 20)
m = tf.keras.models.load_model("ours.keras")
pred = m.predict(new_x)

print(pred.shape, y.shape)
print("Mean Squared Error:", mean_squared_error(y.T, pred))

plt.plot(y[600:900], label="Actual")
plt.plot(pred[600:900], label="Predicted")
plt.xlabel("Time")
plt.ylabel("Response")
plt.legend()
plt.show()

