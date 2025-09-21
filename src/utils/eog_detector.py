# utils/eog_detector.py
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List
import numpy as np
import mne
import os
import json
import requests
import torch
import streamlit as st



def _save_uploaded_temp(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".edf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        return tmp.name

def _pick_eog_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    eog_channels = [ch for ch in raw.ch_names if "EOG" in ch.upper() or ch.upper().startswith("EOG")]
    if not eog_channels:
        eog_channels = [ch for ch in raw.ch_names if ch.upper().startswith(("FP","AF","F"))][:2]
    raw.pick(eog_channels)
    return raw

def _preprocess_raw_for_windows(raw: mne.io.BaseRaw, target_sfreq=256.0):
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq)
    raw.filter(0.3, 30.0, fir_design='firwin', verbose='ERROR') 
    return raw

def _segment_windows(data: np.ndarray, sfreq: float, win_s: float = 5.0, step_s: float = 1.0, max_windows=None):
    C, T = data.shape
    win = int(win_s * sfreq)
    step = int(step_s * sfreq)
    starts = list(range(0, max(1, T - win + 1), step))
    if max_windows is not None:
        starts = starts[:max_windows]
    windows = []
    times = []
    for s in starts:
        seg = data[:, s:s+win]
        if seg.shape[1] == win:
            windows.append(seg.astype(np.float32))
            times.append(s/sfreq)
    return windows, times

def call_hf_space_api(space_url: str, edf_path: str, extra: dict = None):
    """
    POST to your HF Space that exposes an inference endpoint.
    Implementation depends on how you write the Space (gradio/fastapi).
    Here we assume the space accepts a multipart upload at /predict or /run.
    You will likely customize this for your space.
    """
    extra = extra or {}
    files = {"file": open(edf_path, "rb")}
    resp = requests.post(space_url, files=files, data={"meta": json.dumps(extra)}, timeout=60)
    resp.raise_for_status()
    return resp.json()

def run_eog_detector(uploaded_file,
                     use_hf_space=False,
                     space_url=None,
                     local_model_path=None,
                     win_s=5.0, step_s=1.0, image_size=None, max_windows=600) -> Dict:
    """
    uploaded_file: Streamlit UploadedFile
    Returns dict with keys: model, prediction, confidence, n_windows, sfreq, channels, per_window
    """
    tmp_edf = _save_uploaded_temp(uploaded_file)
    if use_hf_space and space_url:
        return call_hf_space_api(space_url, tmp_edf)

   
    raw = mne.io.read_raw_edf(tmp_edf, preload=True, verbose='ERROR')
    raw = _pick_eog_raw(raw)
    raw = _preprocess_raw_for_windows(raw, target_sfreq=256.0)
    data = raw.get_data()  
    sfreq = raw.info['sfreq']

    windows, times = _segment_windows(data, sfreq, win_s=win_s, step_s=step_s, max_windows=max_windows)
    if not windows:
        raise RuntimeError("No windows created from EOG.")

    if local_model_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(local_model_path, map_location=device)
        model.eval()
        results = []
        batch_size = 64
        for i in range(0, len(windows), batch_size):
            batch = np.stack(windows[i:i+batch_size])  
            x = torch.from_numpy(batch).to(device)
            with torch.inference_mode():
                logits = model(x).cpu().numpy()
            probs = softmax(logits, axis=1)
            for p in probs:
                results.append({"p_no_disease": float(p[0]), "p_disease": float(p[1])})
    else:
        results = []
        for w in windows:
            var = float(np.mean(w.var(axis=1)))
            p_disease = min(0.9, var / (var + 1.0))
            results.append({"p_no_disease": 1.0 - p_disease, "p_disease": p_disease})

 
    p_disease_mean = float(np.mean([r["p_disease"] for r in results]))
    p_no_mean = float(np.mean([r["p_no_disease"] for r in results]))
    prediction = "disease" if p_disease_mean > p_no_mean else "no_disease"
    confidence = max(p_disease_mean, p_no_mean)

    per_window = [{"start_s": float(t), "p_no_disease": r["p_no_disease"], "p_disease": r["p_disease"]}
                  for t, r in zip(times, results)]

    return {
        "model": local_model_path or "hf_space" if use_hf_space else "baseline",
        "prediction": prediction,
        "confidence": confidence,
        "n_windows": len(per_window),
        "sfreq": float(sfreq),
        "channels": raw.ch_names,
        "per_window": per_window,
    }

def softmax(x, axis=None):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)
