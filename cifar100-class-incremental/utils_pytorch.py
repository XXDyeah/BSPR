import os
import pickle


def savepickle(data, file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    print("pickle into", file_path)
    with open(file_path, "wb") as file_handle:
        pickle.dump(data, file_handle, pickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, "rb") as file_handle:
        return pickle.load(file_handle)
