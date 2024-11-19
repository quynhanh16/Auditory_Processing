# Packages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Tools
from tools.utils import (
    load_datafile,
    splitting_recording,
    save_state,
    prepare_response,
    load_state, prepare_stimuli, prepare_stimuli_model,
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
def run():
    X = load_state("x_model.pkl")
    y = load_state("y_model.pkl").T
    y = np.mean(y, axis=1)
    print(X.shape, y.shape)
    print(np.mean(X), np.mean(y))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    norm = tf.keras.layers.Normalization(axis=None)
    norm.adapt(X)
    X_train = norm(X_train)
    return
    X_val = norm(X_val)

    tf.random.set_seed(42)
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(21, 18, 1)))

    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.MaxPool2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1))

    model.compile(
        optimizer="adam", loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.R2Score(),
        ]
    )

    model.summary()

    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
    # model = tf.keras.models.load_model("ours2D-70-80.keras")
    pred = model.predict(X)

    plt.plot(y[900:1200], label="Actual")
    plt.plot(pred[900:1200], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

    # plt.plot(history.history['r2_score'])
    # plt.xlabel('Epoch')
    # plt.ylabel('R2 Score')
    # plt.show()

    model.save('final_model.keras')


def validate_test():
    stim, resp = load_state("state.pkl")
    model = tf.keras.models.load_model("final_model.keras")
    model.summary()

    X = prepare_stimuli(stim, (0, 27), 18, 20)
    y = prepare_response(resp, (0, 27), 20).T
    y = np.mean(y, axis=1)

    print(X.shape, y.shape)

    X = prepare_stimuli_model(X, 18, 20)
    predictions = model.predict(X)
    print("Mean Squared Error:", mean_squared_error(y, predictions))
    print("R2 Score:", r2_score(y, predictions))
    plt.plot(y[600:1200])
    plt.plot(predictions[600:1200])
    plt.title("Population Spiking Rate 2Dx3 (Normalized)")
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
    run()
    # validate_test()
    # test("ours.keras")
