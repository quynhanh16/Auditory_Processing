from tools import load_state, load_datafile, splitting_recording, save_state, prepare_stimuli, prepare_response

import matplotlib.pyplot as plt

# TensorFlow Package
import tensorflow as tf
from tensorflow.keras import layers, models


def run(stimuli, response):
    print("Preparing x")
    stim_state_file = f"stim_m{m}_d{d}_t{interval}.pkl"
    stim_state = load_state(stim_state_file)
    if stim_state is None:
        X = prepare_stimuli(stimuli, interval, m, d)
        save_state(stim_state_file, X)
    else:
        X = load_state(stim_state_file)

    print("Preparing y")
    resp_state_file = f"resp_m{m}_t{interval}.pkl"
    resp_state = load_state(resp_state_file)
    if resp_state is None:
        y = prepare_response(response, interval, d)
        save_state(resp_state_file, y)
    else:
        y = load_state(resp_state_file)

    model = models.Sequential()

    model.add(layers.Conv2D(1, (3, 3), activation="relu", input_shape=(21, 18, 74750)))

    model.add(layers.MaxPool2D((3, 3)))

    model.add(layers.Flatten())

    model.add(layers.Dense(1, activation="relu"))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(X, y, epochs=20)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


if __name__ == "__main__":
    tgz_file: str = "A1_NAT4_ozgf.fs100.ch18.tgz"

    state_file = "state.pkl"
    state = load_state(state_file)
    if state is None:
        rec = load_datafile(tgz_file, True)
        stim, resp = splitting_recording(rec, True)
        save_state(state_file, (stim, resp))
    else:
        stim, resp = load_state(state_file)

    interval = (27, stim.shape[1] / 100)
    m, d = 18, 20
    run(stim, resp)

