# Packages
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Tools
from tools.utils import (
    load_datafile,
    splitting_recording,
    save_state,
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
        optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["accuracy"]
    )

    history = model.fit(X_train, y_train.T, epochs=20, validation_data=(X_val, y_val.T))
    pred = model.predict(X)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(y[:300], label="Actual")
    plt.plot(pred[:300], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

    model.summary()
    model.save('1Dx2-CNN.keras')


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
