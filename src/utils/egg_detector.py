
import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch


MODEL_PATH = os.path.join("models", "egg_xgb.pkl")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"EGG model not found at {MODEL_PATH}")

_artifact = joblib.load(MODEL_PATH)
CLF = _artifact["model"]
SCALER = _artifact.get("scaler", None)
FEATURE_NAMES = _artifact["feature_names"]
LABEL_ENCODER = _artifact.get("label_encoder", None)


def cpm_to_hz(cpm): return cpm / 60.0
BRADY = (0.005, cpm_to_hz(2.4))
NORMA = (cpm_to_hz(2.4), cpm_to_hz(3.7))
TACHY = (cpm_to_hz(3.7), 0.2)

def band_power(f, Pxx, band):
    fmin, fmax = band
    mask = (f >= fmin) & (f <= fmax)
    return float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0

def spectral_entropy(Pxx, eps=1e-12):
    p = Pxx / (Pxx.sum() + eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def extract_features_window(seg, fs):
    nperseg = min(len(seg), max(256, int(fs*60)))
    f, Pxx = welch(seg, fs=fs, nperseg=nperseg)
    mask = (f >= 0.005) & (f <= 0.2)
    f_use, P_use = (f[mask], Pxx[mask]) if mask.any() else (f, Pxx)

    idx_dom = np.argmax(P_use)
    f_dom = float(f_use[idx_dom])
    p_dom = float(P_use[idx_dom])
    p_brady = band_power(f_use, P_use, BRADY)
    p_norm = band_power(f_use, P_use, NORMA)
    p_tachy = band_power(f_use, P_use, TACHY)
    total_p = float(np.trapz(P_use, f_use) + 1e-12)

    return {
        "f_dom": f_dom,
        "p_dom": p_dom,
        "frac_brady": float(p_brady / total_p),
        "frac_norm": float(p_norm / total_p),
        "frac_tachy": float(p_tachy / total_p),
        "total_power": total_p,
        "rms": float(np.sqrt(np.mean(seg**2))),
        "var": float(np.var(seg)),
        "spec_entropy": spectral_entropy(P_use)
    }


def windows_from_signal(sig, fs, win_s=180.0, step_s=60.0):
    win = int(win_s * fs)
    step = int(step_s * fs)
    if len(sig) < win:

        return [(0, sig)]
    starts = list(range(0, len(sig) - win + 1, step))
    return [(s, sig[s:s+win]) for s in starts]


def predict_egg_full_signal(sig, fs, win_s=180.0, step_s=60.0):
    """
    Predict EGG state from full signal.
    Returns dict with average prediction and per-window features.
    """

    all_feats = []
    for start, seg in windows_from_signal(sig, fs, win_s, step_s):
        feats = extract_features_window(seg, fs)
        feats["start_s"] = float(start / fs)
        all_feats.append(feats)

    if not all_feats:
        raise ValueError("No windows extracted from signal. Check signal length or sampling rate.")

    features_df = pd.DataFrame(all_feats)


    for col in FEATURE_NAMES:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[FEATURE_NAMES].values
    if SCALER is not None:
        X_scaled = SCALER.transform(X)
    else:
        X_scaled = X

    probs = CLF.predict_proba(X_scaled)  
    avg_probs = probs.mean(axis=0)
    pred_idx = np.argmax(avg_probs)


    if LABEL_ENCODER is not None:
        pred_class = LABEL_ENCODER.inverse_transform([pred_idx])[0]
        class_names = LABEL_ENCODER.classes_
    else:
        pred_class = str(pred_idx)
        class_names = [str(c) for c in CLF.classes_]

    probabilities = {cls: float(avg_probs[i]) for i, cls in enumerate(class_names)}

    return {
        "pred_class": pred_class,
        "probabilities": probabilities,
        "per_window": [
            {"start_s": float(r["start_s"]), **{k: float(r[k]) for k in FEATURE_NAMES}}
            for r in all_feats
        ]
    }
