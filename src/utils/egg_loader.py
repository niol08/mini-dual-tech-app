
import numpy as np
import pandas as pd

def load_signal_from_file(uploaded_file):
    """
    Load raw EGG signal (channels x samples) from txt/csv.
    Assumes whitespace or comma separated values.
    Returns: 1D numpy array (single channel) and sampling freq (Hz)
    """
    if uploaded_file.name.endswith(".csv"):
        arr = pd.read_csv(uploaded_file).values
    elif uploaded_file.name.endswith(".txt"):
        arr = np.loadtxt(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

    if arr.ndim > 1:
        arr = arr[np.argmax(np.var(arr, axis=0))] if arr.shape[1] > 1 else arr[:,0]
    else:
        arr = arr.ravel()
    fs = 2.0  
    return arr, fs
