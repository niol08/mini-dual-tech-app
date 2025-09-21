# utils/erp_hf_model.py
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from keras.models import load_model
from PIL import Image
import mne
from typing import Dict


DEFAULT_REPO = "niol08/EEG_epilepsy_classifier"
DEFAULT_FILE = "m31.h5"
INPUT_SIZE = 128  

@st.cache_resource(show_spinner=False)
def get_erp_model(repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE):
    model_path = hf_hub_download(repo_id=repo, filename=filename)
    model = load_model(model_path)
    return model

def _create_tf_image(uploaded_file) -> np.ndarray:
    """Convert EEG (edf/csv/txt) to a 128x128 RGB image for model input."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        path = tmp.name

    if suffix in [".edf", ".fif", ".set", ".bdf"]:
        raw = mne.io.read_raw(path, preload=True, verbose=False)
        data = raw.get_data()  
        sig = np.mean(data, axis=0)  
    elif suffix in [".csv", ".txt"]:
        import pandas as pd
        df = pd.read_csv(path)
        df = df.select_dtypes(include=[np.number])
        if df.shape[1] > 1:
            sig = df.mean(axis=1).values  
        else:
            sig = df.values.squeeze()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

    N = INPUT_SIZE * INPUT_SIZE
    if len(sig) > N:
        sig = sig[:N]
    else:
        sig = np.pad(sig, (0, N - len(sig)), 'constant')

    img = sig.reshape((INPUT_SIZE, INPUT_SIZE))
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    img = Image.fromarray(img).convert("RGB")
    return np.array(img)


def run_erp_detector(uploaded_file, repo: str = DEFAULT_REPO, filename: str = DEFAULT_FILE) -> Dict:
    model = get_erp_model(repo, filename)
    img_arr = _create_tf_image(uploaded_file)
    img_arr = np.expand_dims(img_arr, axis=0)  

    preds = model.predict(img_arr)
    p = float(preds[0][0])
    prediction = "epilepsy" if p >= 0.5 else "normal"
    confidence = p if p >= 0.5 else 1 - p

    return {
        "model": f"{repo}/{filename}",
        "prediction": prediction,
        "confidence": confidence,
        "n_samples": 1,
        "per_sample": [{"p_normal": float(1-p), "p_epilepsy": p, "predicted_label": int(p>=0.5)}]
    }
