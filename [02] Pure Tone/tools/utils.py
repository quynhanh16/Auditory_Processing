import pickle
from typing import Any


def save_state(filename: str, data) -> None:
    """
    Save an object as a pickle file with name *filename*.

    :param filename:
    :param data:
    :return: None
    """
    with open(filename, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def load_state(filename: str) -> Any:
    """
    Load an object from a pickle file with name *filename*.

    :param filename:
    :return: Any
    """
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
