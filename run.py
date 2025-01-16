import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Temporal Fix: TensorFlow on Windows

import numpy as np
import tensorflow as tf
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from tools import load_state, load_datafile, splitting_recording, save_state, stim_heatmap, prepare_stimuli, \
    prepare_stimuli_model, prepare_response, RasterizedSignal, population_spike_rate_plot

# Global variables
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
global y_mean, y_std

# Validation data 0 - 27 seconds
observed_data = (9, 13.5)  # 3 Stimuli in the validation data
val_int = (0, 27)  # Validation data is given by the first 27 seconds of all recordings.
m, d = 18, 20  # 18 Channels. 20 previous stimuli


def preparing_data() -> tuple[RasterizedSignal, RasterizedSignal]:
    global y_mean, y_std
    # Loading files
    tgz_file: str = "./A1_NAT4_ozgf.fs100.ch18.tgz"
    recordings_file = "./data/recordings.pkl"

    with tqdm(total=100, desc="Loading Data") as main_pbar:
        try:
            if load_state(recordings_file) is None:
                main_pbar.update(5)
                rec = load_datafile(tgz_file, False)
                main_pbar.update(5)
                stim, resp = splitting_recording(rec, False)
                main_pbar.update(5)
                save_state(recordings_file, (stim, resp))
                main_pbar.update(5)
            else:
                stim, resp = load_state(recordings_file)
                main_pbar.update(20)
        except Exception as e:
            raise RuntimeError(f"Error: An error occurred when trying to load/save {recordings_file}.\n\t{e}")

        y_mean = np.mean(resp._data)
        y_std = np.std(resp._data)
        train_int = (27, stim.shape[1] / 100)

        # File names
        train_stim_file = "./data/train_stimuli.pkl"
        train_resp_file = "./data/train_response.pkl"
        val_stim_file = "./data/val_stimuli.pkl"
        val_resp_file = "./data/val_response.pkl"

        # Note: Takes the longest to process.
        try:
            if load_state(train_stim_file) is None:
                stimuli = prepare_stimuli(stim, train_int, m, d)
                save_state(train_stim_file, stimuli)
            main_pbar.update(20)
        except Exception as e:
            raise RuntimeError(f"Error: An error occurred when trying to load and prepare the training stimuli.\n\t{e}")

        try:
            train_resp = load_state(train_resp_file)
            if train_resp is None:
                response = prepare_response(resp, train_int, d)
                main_pbar.update(5)
                response = np.mean(response, axis=0)
                main_pbar.update(5)
                # Normalizing the response
                response = (response - y_mean) / y_std
                main_pbar.update(5)
                save_state(train_resp_file, response)
                main_pbar.update(5)
            else:
                main_pbar.update(20)
        except Exception as e:
            raise RuntimeError(
                f"Error: An error occurred when trying to load and prepare the training response.\n\t{e}")

        try:
            if load_state(val_stim_file) is None:
                val_stimuli = prepare_stimuli(stim, val_int, m, d)
                save_state(val_stim_file, val_stimuli)
            main_pbar.update(20)
        except Exception as e:
            raise RuntimeError(f"Error: An error occurred when trying to load and prepare the testing stimuli.\n\t{e}")

        try:
            if load_state(val_resp_file) is None:
                response = prepare_response(resp, val_int, d)
                response = np.mean(response, axis=0)
                response = (response - y_mean) / y_std
                save_state(val_resp_file, response)
            main_pbar.update(20)
        except Exception as e:
            raise RuntimeError(f"Error: An error occurred when trying to load and prepare the testing response.\n\t{e}")

    print("-> FINISHED: Preparing Data\n")
    return stim, resp


def general(stimuli_data: np.ndarray, response_data: np.ndarray) -> None:
    # General
    stimuli_stats = {"channel": [], "mean": [], "std": []}  # Channel | Mean | Std
    for idx, channel in enumerate(stimuli_data):
        channel_mean = np.mean(channel)
        channel_std = np.std(channel)
        stimuli_stats["channel"].append(idx + 1)
        stimuli_stats["mean"].append(channel_mean)
        stimuli_stats["std"].append(channel_std)

    response_stats = {"a1_unit": [], "mean": [], "std": []}  # A1 Unit | Mean | Std
    for idx, a1_unit in enumerate(response_data):
        unit_mean = np.mean(a1_unit)
        unit_std = np.std(a1_unit)
        response_stats["a1_unit"].append(idx + 1)
        response_stats["mean"].append(unit_mean)
        response_stats["std"].append(unit_std)

    stimuli_df = pd.DataFrame(stimuli_stats)
    stimuli_df.to_csv("./results/stimuli_stats.csv", index=False)

    response_df = pd.DataFrame(response_stats)
    response_df.to_csv("./results/response_stats.csv", index=False)

    print("-> FINISHED: Calculating General Statistics\n")


def ln_model() -> None:
    model_file = "./models/lr.pkl"
    model = load_state(model_file)
    if model is not None:
        raise FileExistsError("ERROR: Model already exists. Try running removing file first")

    # Train data
    stim_data = load_state("./data/train_stimuli.pkl")
    resp_data = load_state("./data/train_response.pkl")

    if stim_data is None or resp_data is None:
        raise FileNotFoundError("ERROR: Data not loaded correctly. Try running preparing_data() first")

    # Here you can change the architecture of the model.
    model = LinearRegression()
    model.fit(stim_data, resp_data)
    save_state(model_file, model)

    print("-> FINISHED: Fitting Linear Model\n")


def cnn_1l() -> None:
    try:
        model_file = "./models/2Dx1.keras"
        _ = tf.keras.models.load_model(model_file)
        raise FileExistsError("ERROR: Model already exists. Try running removing file first")
    except Exception:
        pass

    result_path = "./results/2Dx1_history.csv"

    # Training data
    x_train = load_state("./data/train_stimuli.pkl")
    x_train = prepare_stimuli_model(x_train, m, d)
    y_train = load_state("./data/train_response.pkl")

    # Testing data
    x_val = load_state("./data/val_stimuli.pkl")
    x_val = prepare_stimuli_model(x_val, m, d)
    y_val = load_state("./data/val_response.pkl")

    # Architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(d + 1, m, 1)),
        tf.keras.layers.Conv2D(24, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D((3, 3)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
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
    history = model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val))

    # Save History and Model
    df = pd.DataFrame(history.history)
    df.to_csv(result_path)
    model.save("./models/2Dx1.keras")

    print("-> FINISHED: Training 1D CNN Model\n")


def cnn_2l() -> None:
    try:
        model_file = "./models/2Dx2.keras"
        _ = tf.keras.models.load_model(model_file)
        raise FileExistsError("ERROR: Model already exists. Try running removing file first")
    except Exception:
        pass

    result_path = "./results/2Dx2_history.csv"

    # Training data
    x_train = load_state("./data/train_stimuli.pkl")
    x_train = prepare_stimuli_model(x_train, m, d)
    y_train = load_state("./data/train_response.pkl")

    # Testing data
    x_val = load_state("./data/val_stimuli.pkl")
    x_val = prepare_stimuli_model(x_val, m, d)
    y_val = load_state("./data/val_response.pkl")

    # Architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(d + 1, m, 1)),
        tf.keras.layers.Conv2D(12, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D((3, 3)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
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
    history = model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val))

    # Save History and Model
    df = pd.DataFrame(history.history)
    df.to_csv(result_path)
    model.save("./models/2Dx2.keras")

    print("-> FINISHED: Training 2D CNN Model\n")


def results() -> None:
    # Training data
    x_train = load_state("./data/train_stimuli.pkl")
    lr_x_train = x_train
    x_train = prepare_stimuli_model(x_train, m, d)
    y_train = load_state("./data/train_response.pkl")

    # Testing data
    x_val = load_state("./data/val_stimuli.pkl")
    lr_x_val = x_val
    x_val = prepare_stimuli_model(x_val, m, d)
    y_val = load_state("./data/val_response.pkl")

    try:
        lr_model = load_state("./models/lr.pkl")
        cnn1 = tf.keras.models.load_model("./models/2Dx1.keras")
        cnn2 = tf.keras.models.load_model("./models/2Dx2.keras")
    except Exception as e:
        raise FileNotFoundError(e)

    # Linear
    r2s = []
    mses = []
    val_r2s = []
    val_mses = []
    predictions = lr_model.predict(lr_x_train)
    r2 = r2_score(y_train, predictions)
    r2s.append(r2)
    mse = mean_squared_error(y_train, predictions)
    mses.append(mse)
    val_predictions = lr_model.predict(lr_x_val)
    r2 = r2_score(y_val, val_predictions)
    val_r2s.append(r2)
    mse = mean_squared_error(y_val, val_predictions)
    val_mses.append(mse)

    model_results = pd.DataFrame(
        {"actual": np.hstack((y_train, y_val)), "predicted": np.hstack((predictions, val_predictions))})
    model_results.to_csv("./results/ln_model_pred.csv")

    # CNN
    predictions = cnn1.predict(x_train)
    r2 = r2_score(y_train, predictions)
    r2s.append(r2)
    mse = mean_squared_error(y_train, predictions)
    mses.append(mse)
    val_predictions = cnn1.predict(x_val)
    r2 = r2_score(y_val, val_predictions)
    val_r2s.append(r2)
    mse = mean_squared_error(y_val, val_predictions)
    val_mses.append(mse)

    model_results = pd.DataFrame(
        {"actual": np.hstack((y_train, y_val)),
         "predicted": np.hstack((predictions.flatten(), val_predictions.flatten()))})
    model_results.to_csv("./results/2Dx1_pred.csv")

    predictions = cnn2.predict(x_train)
    r2 = r2_score(y_train, predictions)
    r2s.append(r2)
    mse = mean_squared_error(y_train, predictions)
    mses.append(mse)
    val_predictions = cnn2.predict(x_val)
    r2 = r2_score(y_val, val_predictions)
    val_r2s.append(r2)
    mse = mean_squared_error(y_val, val_predictions)
    val_mses.append(mse)

    model_results = pd.DataFrame(
        {"actual": np.hstack((y_train, y_val)),
         "predicted": np.hstack((predictions.flatten(), val_predictions.flatten()))})
    model_results.to_csv("./results/2Dx2_pred.csv")

    model_results = {"model": ["Linear", "CNN-1D", "CNN-2D"], "r2": r2s, "mse": mses, "val_r2": val_r2s,
                     "val_mse": val_mses}

    df = pd.DataFrame(model_results)
    df.to_csv("./results/model_results.csv")

    print("-> FINISHED: Getting results of models\n")


# TODO: Still need to work on graphing
def graph_results(stimuli_data: RasterizedSignal, response_data: RasterizedSignal) -> None:
    # Normalize response
    y_data = np.mean(response_data._data, axis=0)
    y_data = (y_data - y_mean) / y_std

    # Linear Model
    # Stimuli heatmap
    f, axs = plt.subplots(4, 1, figsize=(10, 8))
    stim_heatmap(stimuli_data, observed_data, False, ax=axs[0])

    # Population spike rate plot
    population_spike_rate_plot(response_data, observed_data, False, ax=axs[1])
    for i in range(3):
        a = observed_data[0] + (1.50 * i) + .25
        b = observed_data[0] + (1.50 * (i + 1)) - .25
        axs[1].axvline(x=a, color="orange", ls="--", lw=2)
        axs[1].axvline(x=b, color="red", ls="--", lw=2)

    # Actual vs. Predicted: Linear
    axs[2].plot(y_data[900: 1350], label="Actual")
    model = load_state("./models/lr.pkl")
    x = prepare_stimuli(stimuli_data, observed_data, m, d)
    y = model.predict(x)
    for i in range(3):
        start = 150 * i
        x = np.arange(0, 130, 1) + start + 20
        axs[2].plot(x, y[(i * 130): ((i + 1) * 130)], label="Predicted", color="red", alpha=0.7)
    axs[2].legend(loc="upper left")

    # Actual vs. Predicted
    df = pd.read_csv("./results/ln_model_pred.csv")
    y_actual, y_pred = df["actual"].to_numpy().reshape((-1, 1)), df["predicted"].to_numpy().reshape((-1, 1))
    reg = LinearRegression().fit(y_pred, y_actual)
    reg = reg.predict(y_pred)
    axs[3].scatter(y_actual, y_pred, s=2)
    axs[3].plot(y_pred, reg, color="red")

    plt.suptitle("Linear Model Results")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./results/linear_results.png")
    plt.show()

    # CNN 1 Model
    # Stimuli heatmap
    f, axs = plt.subplots(4, 1, figsize=(10, 8))
    stim_heatmap(stimuli_data, observed_data, False, ax=axs[0])

    # Population spike rate plot
    population_spike_rate_plot(response_data, observed_data, False, ax=axs[1])
    for i in range(3):
        a = observed_data[0] + (1.50 * i) + .25
        b = observed_data[0] + (1.50 * (i + 1)) - .25
        axs[1].axvline(x=a, color="orange", ls="--", lw=2)
        axs[1].axvline(x=b, color="red", ls="--", lw=2)

    # Actual vs. Predicted: Linear
    axs[2].plot(y_data[900: 1350], label="Actual")
    model = tf.keras.models.load_model("./models/2Dx1.keras")
    x = prepare_stimuli(stimuli_data, observed_data, m, d)
    x = prepare_stimuli_model(x, m, d)
    y = model.predict(x)
    for i in range(3):
        start = 150 * i
        x = np.arange(0, 130, 1) + start + 20
        axs[2].plot(x, y[(i * 130): ((i + 1) * 130)], label="Predicted", color="red", alpha=0.7)
    axs[2].legend(loc="upper left")

    # Actual vs. Predicted
    df = pd.read_csv("./results/2Dx1_pred.csv")
    y_actual, y_pred = df["actual"].to_numpy().reshape((-1, 1)), df["predicted"].to_numpy().reshape((-1, 1))
    reg = LinearRegression().fit(y_pred, y_actual)
    reg = reg.predict(y_pred)
    axs[3].scatter(y_actual, y_pred, s=2)
    axs[3].plot(y_pred, reg, color="red")

    plt.suptitle("CNN 2Dx1 Results")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./results/2Dx1_results.png")
    plt.show()

    # CNN 2 Model
    # Stimuli heatmap
    f, axs = plt.subplots(4, 1, figsize=(10, 8))
    stim_heatmap(stimuli_data, observed_data, False, ax=axs[0])

    # Population spike rate plot
    population_spike_rate_plot(response_data, observed_data, False, ax=axs[1])
    for i in range(3):
        a = observed_data[0] + (1.50 * i) + .25
        b = observed_data[0] + (1.50 * (i + 1)) - .25
        axs[1].axvline(x=a, color="orange", ls="--", lw=2)
        axs[1].axvline(x=b, color="red", ls="--", lw=2)

    # Actual vs. Predicted: Linear
    axs[2].plot(y_data[900: 1350], label="Actual")
    model = tf.keras.models.load_model("./models/2Dx2.keras")
    x = prepare_stimuli(stimuli_data, observed_data, m, d)
    x = prepare_stimuli_model(x, m, d)
    y = model.predict(x)
    for i in range(3):
        start = 150 * i
        x = np.arange(0, 130, 1) + start + 20
        axs[2].plot(x, y[(i * 130): ((i + 1) * 130)], label="Predicted", color="red", alpha=0.7)
    axs[2].legend(loc="upper left")

    # Actual vs. Predicted
    df = pd.read_csv("./results/2Dx2_pred.csv")
    y_actual, y_pred = df["actual"].to_numpy().reshape((-1, 1)), df["predicted"].to_numpy().reshape((-1, 1))
    reg = LinearRegression().fit(y_pred, y_actual)
    reg = reg.predict(y_pred)
    axs[3].scatter(y_actual, y_pred, s=2)
    axs[3].plot(y_pred, reg, color="red")

    plt.suptitle("CNN 2Dx2 Results")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("./results/2Dx2_results.png")
    plt.show()

    print("-> FINISHED: Plotting and saving results of models\n")


def run() -> None:
    # Creating main folder
    main_path = Path("./results")
    main_path.mkdir(parents=True, exist_ok=True)

    # Creating data folder
    data_path = Path("./data")
    data_path.mkdir(parents=True, exist_ok=True)

    # Creating model folder
    model_path = Path("./models")
    model_path.mkdir(parents=True, exist_ok=True)

    # Processing and Saving data
    stim, resp = preparing_data()

    # General results
    # general(stim._data, resp._data)

    # Lineal Model
    # ln_model()

    # CNN model
    # cnn_1l()
    # cnn_2l()

    # Getting results
    # results()
    graph_results(stim, resp)


if __name__ == "__main__":
    run()
