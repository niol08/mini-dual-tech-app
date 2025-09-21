
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from PIL import Image
import torch
import mne
import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from typing import List, Dict


DEFAULT_REPO = "JLB-JLB/ViT_Seizure_Detection"


@st.cache_resource(show_spinner=False)
def get_vit_model(repo: str = DEFAULT_REPO):
    """
    Returns (processor, model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForImageClassification.from_pretrained(repo)
    model.eval()
    model.to(device)
    return processor, model, device

def _raw_from_uploaded(uploaded_file) -> mne.io.BaseRaw:
    suffix = Path(uploaded_file.name).suffix or ".edf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp_path = tmp.name
    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
    return raw


def window_to_channel_time_image(seg: np.ndarray, out_size: int = 224, clip_std: float = 5.0) -> Image.Image:
    """
    seg: (C, T) float array
    Steps:
      - per-channel zscore (channel mean/std)
      - clip to [-clip_std, clip_std], scale to 0..255
      - create image height=C, width=T, resize -> out_size x out_size
      - convert to RGB (3 channels) so ViT processor is happy
    """

    mu = seg.mean(axis=1, keepdims=True)
    sigma = seg.std(axis=1, keepdims=True) + 1e-6
    seg_z = (seg - mu) / sigma
    seg_z = np.clip(seg_z, -clip_std, clip_std)

    seg_scaled = ((seg_z + clip_std) / (2 * clip_std) * 255.0).astype(np.uint8)

    img = Image.fromarray(seg_scaled)  
    img = img.resize((out_size, out_size))
    img = img.convert("RGB")
    return img


def run_vit_seizure_detector(
    uploaded_file,
    repo: str = DEFAULT_REPO,
    win_s: int = 6,
    step_s: int = 1,
    image_size: int = 224,
    batch_size: int = 64,
    max_windows: int | None = None,
) -> Dict:
    """
    uploaded_file: Streamlit UploadedFile (has .name and .getbuffer())
    Returns a dict: similar shape to your earlier outputs:
      { model, prediction, confidence, n_windows, sfreq, channels, per_window: [{start_s, p_no_seiz, p_seiz}, ...], image_size_used }
    Notes:
      - win_s default 6 (JLB ViT training used 6s windows). See model card. :contentReference[oaicite:1]{index=1}
    """
    processor, model, device = get_vit_model(repo)
    raw = _raw_from_uploaded(uploaded_file)


    picks = [ch for ch in raw.ch_names if not ch.upper().startswith("DC")]
    if len(picks) == 0:
        picks = raw.ch_names 
    raw.pick(picks)


    target_sfreq = 256.0
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)

    data = raw.get_data() 
    sfreq = raw.info["sfreq"]
    n_channels, n_samples = data.shape

    win = int(win_s * sfreq)
    step = int(step_s * sfreq)
    if win <= 0 or step <= 0:
        raise ValueError("Invalid window/step sizes.")

    starts = list(range(0, max(1, n_samples - win + 1), step))
    if max_windows is not None:
        starts = starts[:max_windows]

  
    mu_global = data.mean(axis=1, keepdims=True)
    sigma_global = data.std(axis=1, keepdims=True) + 1e-6

    imgs: List[Image.Image] = []
    times = []
    for s in starts:
        seg = data[:, s : s + win]  
        seg = (seg - mu_global) / sigma_global
        img = window_to_channel_time_image(seg, out_size=image_size)
        imgs.append(img)
        times.append(s / sfreq)


    per_window = []
    for i in range(0, len(imgs), batch_size):
        batch_imgs = imgs[i : i + batch_size]
        inputs = processor(images=batch_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)
            logits = out.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()  
        for p in probs:
            if p.shape[0] > 1:
                p_no_seiz = float(p[0])
                p_seiz    = float(p[1])
            else:
                p_no_seiz = float(1.0 - p[0])
                p_seiz    = float(p[0])
            per_window.append({"p_no_seiz": p_no_seiz, "p_seiz": p_seiz})


    n_windows = len(per_window)
    if n_windows == 0:
        raise RuntimeError("No windows produced from the recording.")

    p_seiz_mean = np.mean([w["p_seiz"] for w in per_window])
    p_no_seiz_mean = np.mean([w["p_no_seiz"] for w in per_window])
    if p_seiz_mean > p_no_seiz_mean:
        prediction = "seizure"
        confidence = float(p_seiz_mean)
    else:
        prediction = "no_seizure"
        confidence = float(p_no_seiz_mean)


    per_window_with_times = []
    for t, w in zip(times, per_window):
        per_window_with_times.append({"start_s": float(t), "p_no_seiz": w["p_no_seiz"], "p_seiz": w["p_seiz"]})

    result = {
        "model": repo,
        "prediction": prediction,
        "confidence": confidence,
        "n_windows": n_windows,
        "sfreq": float(sfreq),
        "channels": raw.ch_names,
        "per_window": per_window_with_times,
        "image_size_used": image_size,
    }
    return result
