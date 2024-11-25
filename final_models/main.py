import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models

from tools import load_state, prepare_stimuli, prepare_stimuli_model, prepare_response


# Normalized stimuli, 2 convolutional layers, drouput (0.25)
def first_model():
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Training data
    X_train = load_state("../x_model.pkl")
    y = load_state("../y_model.pkl").T
    y = np.mean(y, axis=1)
    y_train = (y - 0.077222307) / 0.29057816

    # Validation data
    stim, resp = load_state("../state.pkl")
    X_test = prepare_stimuli(stim, (0, 27), 18, 20)
    X_test = prepare_stimuli_model(X_test, 18, 20)
    y_test = np.mean(prepare_response(resp, (0, 27), 20), axis=0).T
    y_test = (y_test - 0.077222307) / 0.29057816

    # Architecture
    model = models.Sequential([
        layers.Conv2D(24, (3, 3), activation="relu", input_shape=(21, 18, 1)),
        layers.MaxPool2D((3, 3)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.R2Score(),
        ]
    )
    model.summary()

    # Training
    _ = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
    model.save("2Dx1.keras")
    predictions = model.predict(X_test)

    plt.plot(y_test[0:600])
    plt.plot(predictions[0:600])
    plt.title("Time plot")
    plt.show()


# Normalized stimuli, 1 convolutional layers, drouput (0.25)
def second_model():
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Training data
    X_train = load_state("../x_model.pkl")
    y = load_state("../y_model.pkl").T
    y = np.mean(y, axis=1)
    y_train = (y - 0.077222307) / 0.29057816

    # Validation data
    stim, resp = load_state("../state.pkl")
    X_test = prepare_stimuli(stim, (0, 27), 18, 20)
    X_test = prepare_stimuli_model(X_test, 18, 20)
    y_test = np.mean(prepare_response(resp, (0, 27), 20), axis=0).T
    y_test = (y_test - 0.077222307) / 0.29057816

    # Architecture
    model = models.Sequential([
        layers.Conv2D(12, (3, 3), activation="relu", input_shape=(21, 18, 1)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPool2D((3, 3)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.R2Score(),
        ]
    )
    model.summary()

    _ = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
    model.save("2Dx2.keras")
    predictions = model.predict(X_test)

    plt.plot(y_test[600:1200])
    plt.plot(predictions[600:1200])
    plt.title("Time plot")
    plt.show()


def test():
    model = tf.keras.models.load_model("2Dx2.keras")
    stim, resp = load_state("../state.pkl")
    # 2Dx2 val_r2_score: 0.5706
    X = prepare_stimuli(stim, (0, 27), 18, 20)
    X = prepare_stimuli_model(X, 18, 20)
    y = np.mean(prepare_response(resp, (0, 27), 20), axis=0)
    y = (y - 0.077222307) / 0.29057816
    predictions = model.predict(X)

    print("Mean Squared Error:", mean_squared_error(y, predictions))
    print("R2 Score:", r2_score(y, predictions))

    plt.plot(y[600:1200])
    plt.plot(predictions[600:1200])
    plt.title("Time plot")
    plt.show()


def samples():
    model = tf.keras.models.load_model("2Dx2.keras")
    stim, resp = load_state("../state.pkl")
    X = prepare_stimuli(stim, (0, 27), 18, 20)
    X = prepare_stimuli_model(X, 18, 20)
    y = np.mean(prepare_response(resp, (0, 27), 20), axis=0)
    y = (y - 0.077222307) / 0.29057816
    pred = model.predict(X)[:, 0]

    set_y = np.empty((130, 18), dtype=np.float64)
    set_pred = np.empty((130, 18), dtype=np.float64)

    for i in range(130):
        for j in range(int(y.shape[0] / 130)):
            set_y[i, j] = y[j * 130 + i]
            set_pred[i, j] = pred[j * 130 + i]

    r2_scores = np.empty(130, dtype=np.float64)
    for i in range(130):
        r2_scores[i] = r2_score(set_y[i], set_pred[i])
        plt.figure(figsize=(10, 6))
        plt.plot(set_y[i], label="Actual")
        plt.plot(set_pred[i], label="Predicted")
        plt.text(0.95, 0.95, f'$train: r^2 = {r2_scores[i]:.4f}$', fontsize=16, ha='right', va='top',
                 transform=plt.gca().transAxes,  # Position based on axes coordinates (0 to 1 scale)
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # first_model()
    # second_model()
    # test()
    samples()
