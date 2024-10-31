# Packages
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Tools
from tools.utils import (
    load_datafile,
    splitting_recording,
    save_state,
    prepare_stimuli,
    prepare_response,
    load_state,
)


# TODO: Make a function
# TODO: Fix matrix
# TODO: Return r2 score, and mean squared error.
def reshape_stim(stimuli):
    # 74750 * 378
    stimuli = stimuli.reshape(74750, 21, 18)
    return stimuli


# I give up
def run(stimuli, response):
    print("Preparing x")
    stim_state_file = f"stim_m{m}_d{d}_t{interval}.pkl"
    stim_state = load_state(stim_state_file)
    if stim_state is None:
        X = prepare_stimuli(stimuli, interval, m, d)
        save_state(stim_state_file, X)
    else:
        X = load_state(stim_state_file)

    X = reshape_stim(X)

    print("Preparing y")
    resp_state_file = f"resp_m{m}_t{interval}.pkl"
    resp_state = load_state(resp_state_file)
    if resp_state is None:
        y = prepare_response(response, interval, d)
        save_state(resp_state_file, y)
    else:
        y = load_state(resp_state_file)

    print(X.shape, y.shape)

    tf.random.set_seed(42)
    model = models.Sequential()

    model.add(layers.Conv2D(1, (3, 3), activation="relu", input_shape=(21, 18, 1)))

    model.add(layers.MaxPool2D((3, 3)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation="relu"))

    model.add(layers.Dense(1))

    model.compile(
        optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(X, y.T, epochs=20)
    pred = model.predict(X)
    plt.plot(history.history["accuracy"], label="accuracy")
    # plt.plot(y[:300], label="Actual")
    # plt.plot(pred[:300], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.legend()
    plt.show()


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
