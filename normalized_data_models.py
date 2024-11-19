import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from tools.utils import load_state, prepare_stimuli, prepare_response, prepare_stimuli_model


def normalized_run():
    X = load_state("x_model.pkl")
    # y = load_state("y_model.pkl").T
    y = load_state("resp_m18_t(27, 889.5).pkl")

    X_std = np.std(X)
    X_mean = np.mean(X)
    print(X_mean, X_std)
    return

    X = (X - X_mean) / X_std

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # (59800, 21, 18)(59800, 849)
    print(X_train.shape, y_train.shape)

    tf.random.set_seed(42)
    model = models.Sequential()

    model.add(layers.Conv2D(30, (3, 3), activation="relu", input_shape=(21, 18, 1)))

    model.add(layers.Conv2D(50, (3, 3), activation="relu"))

    model.add(layers.MaxPool2D((3, 3)))

    model.add(layers.Flatten())

    model.add(layers.Dense(100, activation="relu"))

    model.add(layers.Dense(1))

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

    model.save("normalized_data_model2.keras")

    plt.plot(history.history['r2_score'])
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.show()


def min_max_run():
    X = load_state("x_model.pkl")
    # y = load_state("y_model.pkl").T
    y = load_state("resp_m18_t(27, 889.5).pkl")

    X_max = np.max(X)
    X_min = np.min(X)

    y_max = np.max(y)
    y_min = np.min(y)

    X = (X - X_min) / (X_max - X_min)
    y = (y - y_min) / (y_max - y_min)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # (59800, 21, 18)(59800, 849)
    print(X_train.shape, y_train.shape)

    tf.random.set_seed(42)
    model = models.Sequential()

    model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=(21, 18, 1)))

    model.add(layers.Conv2D(20, (3, 3), activation="relu"))

    model.add(layers.MaxPool2D((3, 3)))

    model.add(layers.Flatten())

    model.add(layers.Dense(100, activation="relu"))

    model.add(layers.Dense(1))

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

    model.save("minmax_data_model.keras")

    plt.plot(history.history['r2_score'])
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.show()


def validate_test():
    stim, resp = load_state("state.pkl")
    model = tf.keras.models.load_model("normalized_data_model2.keras")
    model.summary()

    X = prepare_stimuli(stim, (0, 27), 18, 20)
    y = prepare_response(resp, (0, 27), 20).T
    y = np.mean(y, axis=1)

    X = (X - 2.4414) / 6.3967
    print(X.shape, y.shape)

    X = prepare_stimuli_model(X, 18, 20)
    predictions = model.predict(X)
    print("Mean Squared Error:", mean_squared_error(y, predictions))
    print("R2 Score:", r2_score(y, predictions))
    plt.plot(y[600:1200])
    plt.plot(predictions[600:1200])
    plt.show()


if __name__ == '__main__':
    # normalized_run()
    validate_test()