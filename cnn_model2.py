import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from tools.utils import (
    prepare_response,
    load_state, prepare_stimuli,
)


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
    prediction = np.mean(pred, axis=1)

    print("Mean Squared Error Individually:", mean_squared_error(pred, y))
    print("Mean Squared Error of Average:", mean_squared_error(actual, prediction))

    plt.plot(actual[750:900], label="Actual", color="blue", alpha=0.8)
    plt.plot(prediction[750:900], label="Predicted", alpha=0.5, color="red")
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.legend()
    plt.show()


def graph(model_path):
    stim = load_state("x_model.pkl")
    resp = load_state("y_model.pkl").T
    model = tf.keras.models.load_model(model_path)

    print(stim.shape, resp.shape)
    pred = model.predict(stim)

    print(pred.shape)
    actual = np.mean(resp, axis=1)
    prediction = np.mean(pred, axis=1)
    print(actual.shape, prediction.shape)

    print("Mean Squared Error:", mean_squared_error(actual, prediction))

    plt.plot(actual[600:750], label="Actual")
    plt.plot(prediction[600:750], label="Predicted", alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Response")
    # plt.legend()
    plt.show()


def run():
    stim = load_state("x_model.pkl")
    resp = load_state("y_model.pkl").T

    print("Stimuli shape:", stim.shape, "Response shape:", resp.shape)
    X_train, X_test, y_train, y_test = train_test_split(stim, resp, test_size=0.2)

    tf.random.set_seed(42)

    model = models.Sequential()

    model.add(layers.Conv2D(70, (3, 3), activation="relu", input_shape=(21, 18, 1)))

    model.add(layers.Conv2D(80, (3, 3), activation="relu"))

    model.add(layers.MaxPool2D((3, 3)))

    model.add(layers.Flatten())

    model.add(layers.Dense(80, activation="relu"))

    model.add(layers.Dense(849))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.R2Score(),
        ]
    )

    model.summary()

    history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

    pred = model.predict(X_test)
    print("Before:", pred.shape, y_test.shape)
    actual = np.mean(resp, axis=1)
    prediction = np.mean(pred, axis=1)
    print("After:", actual.shape, prediction.shape)

    plt.plot(history.history['r2_score'])
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.show()

    model.save("model_average.keras")


if __name__ == "__main__":
    # run()
    # test("model_average.keras")
    graph("model_average.keras")
