# Packages
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import numpy as np

# Tools
from tools.utils import (
    load_datafile,
    splitting_recording,
    save_state,
    prepare_response,
    load_state, prepare_stimuli,
)


# TODO: Make a function
# TODO: Fix matrix
# TODO: Return r2 score, and mean squared error.
def test(model_path):
    stim, resp = load_state("state.pkl")
    model = tf.keras.models.load_model(model_path)
    X = prepare_stimuli(stim, (0, 27), 18, 20)
    y = prepare_response(resp, (0, 27), 20).T

    print("Stimuli shape:", X.shape, "Response shape:", y.shape)

    new_x = np.empty((X.shape[0], 21, 18))
    for i in range(X.shape[0]):
        for j in range(int(X.shape[1] / 18)):
            for k in range(int(X.shape[1] / 21)):
                new_x[i, j, k] = X[i, (18 * j) + k]

    pred = model.predict(new_x)

    actual = np.mean(y, axis=1)
    print(pred.shape, actual.shape)
    # prediction = np.mean(pred, axis=1)

    # print("Mean Squared Error Individually:", mean_squared_error(pred, y))
    # print("Mean Squared Error of Average:", mean_squared_error(actual, prediction))

    plt.plot(actual[750:900], label="Actual", color="blue", alpha=0.8)
    plt.plot(pred[750:900], label="Predicted", alpha=0.5, color="red")
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

# I give up
def run(stimuli, response):
    print("Preparing x")
    stim_state_file = f"x_model.pkl"
    X = load_state(stim_state_file)

    print(X.shape)
    # X = reshape_stim(X)

    print("Preparing y")
    resp_state_file = f"resp_m{m}_t{interval}.pkl"
    resp_state = load_state(resp_state_file)
    if resp_state is None:
        y = prepare_response(response, interval, d)
        save_state(resp_state_file, y)
    else:
        y = load_state(resp_state_file)

    print(X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    tf.random.set_seed(42)
    model = models.Sequential()

    model.add(layers.Conv2D(70, (3, 3), activation="relu", input_shape=(21, 18, 1)))

    model.add(layers.Conv2D(80, (3, 3), activation="relu"))

    model.add(layers.MaxPool2D((3, 3)))

    model.add(layers.Flatten())

    model.add(layers.Dense(100, activation="relu"))

    model.add(layers.Dense(1))

    model.compile(
        optimizer="adam", loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.R2Score(),
        ]
    )

    model.summary()

    # history = model.fit(X_train, y_train.T, epochs=20, validation_data=(X_val, y_val.T))
    model = tf.keras.models.load_model("ours2D-70-80.keras")
    pred = model.predict(X)

    plt.plot(y[900:1200], label="Actual")
    plt.plot(pred[900:1200], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

    model.save('ours2D-70-80.keras')


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
    # test("ours.keras")
