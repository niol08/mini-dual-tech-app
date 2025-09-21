
import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import scale
import soundfile as sf
from gemini import query_gemini_rest
import librosa
import tempfile


EXPECTED_LEN = 256
STEP = 128

PCG_LABELS = [
    "Normal",
    "Aortic Stenosis",
    "Mitral Stenosis",
    "Mitral Valve Prolapse",
    "Pericardial Murmurs"
]

LABELS_EMG = ["healthy", "myopathy", "neuropathy"]

def load_uploaded_file(file, signal_type="ECG") -> np.ndarray:
    name = file.name.lower()

 
    if signal_type in ("ECG", "EMG"):
        text = file.read().decode("utf-8").strip()
        if "," in text:
            vals = [float(x) for x in text.split(",") if x.strip()]
        else:
            vals = [float(x) for x in text.splitlines() if x.strip()]
        return np.array(vals, dtype=np.float32)

 
    if signal_type == "VAG":
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            features = [
                "rms_amplitude",
                "peak_frequency",
                "spectral_entropy",
                "zero_crossing_rate",
                "mean_frequency",
            ]
            return df[features].iloc[0].values.astype(np.float32)

        elif name.endswith(".npy"):
            return np.load(file)

        elif name.endswith(".wav"):
            data, _ = sf.read(file)
            return data.astype(np.float32)

        raise ValueError("Unsupported VAG file format.")

  
    if signal_type == "PCG" and name.endswith((".wav", ".flac", ".mp3")):
        data, _ = sf.read(file)
        if data.ndim > 1:         
            data = data[:, 0]
        return data.astype(np.float32)

    raise ValueError("Unsupported file format.")


def preprocess_signal(x: np.ndarray) -> np.ndarray:
    if x.size != EXPECTED_LEN:
        x = resample(x, EXPECTED_LEN)
    return scale(x).astype(np.float32)   


def segment_signal(raw: np.ndarray) -> np.ndarray:
    raw = preprocess_signal(raw)        
    seg = raw.reshape(EXPECTED_LEN, 1)    
    return seg[np.newaxis, ...]            



PCG_INPUT_LEN = 995        

def preprocess_pcg_waveform(wave: np.ndarray) -> np.ndarray:

    if wave.ndim > 1:
        wave = wave.mean(axis=1)

  
    if len(wave) < PCG_INPUT_LEN:
        wave = np.pad(wave, (0, PCG_INPUT_LEN - len(wave)))
    else:
        wave = wave[:PCG_INPUT_LEN]

    
    wave = (wave - np.mean(wave)) / (np.std(wave) + 1e-8)
    return wave.astype(np.float32)

def analyze_pcg_signal(file, model, gemini_key=None):
    
    signal, _ = sf.read(file)                      
    signal = preprocess_pcg_waveform(signal)       

    input_data = signal.reshape(1, PCG_INPUT_LEN, 1)  
    preds = model.predict(input_data, verbose=0)[0]

    labels = [
        "Normal",
        "Aortic Stenosis",
        "Mitral Stenosis",
        "Mitral Valve Prolapse",
        "Pericardial Murmurs",
    ]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = labels[idx]

    gem_txt = None
    if gemini_key:
        gem_txt = query_gemini_rest("PCG", label, confidence, gemini_key)

    return label, label, confidence, gem_txt




def pcg_to_features(file_obj, target_sr=16000, n_mels=128, n_frames=112):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name

   
    y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)

    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=256, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max) 
    if logmel.shape[1] < n_frames:
        
        pad_width = n_frames - logmel.shape[1]
        pad = np.zeros((n_mels, pad_width))
        logmel = np.hstack((logmel, pad))
    else:
        logmel = logmel[:, :n_frames]

    
    feat = logmel.flatten().astype(np.float32)

    return feat[np.newaxis, ...]


def analyze_emg_signal(file, model, gemini_key=""):
    raw  = load_uploaded_file(file, signal_type="EMG")    
    
    WINDOW = 1000

    wins = []
    if len(raw) < WINDOW:                                
        pad = np.pad(raw, (0, WINDOW - len(raw)))
        wins.append(((pad - pad.mean()) / (pad.std()+1e-6)).reshape(WINDOW, 1))
    else:                                                 
        for i in range(0, len(raw) - WINDOW + 1, WINDOW):
            win = raw[i:i+WINDOW]
            win = (win - win.mean()) / (win.std() + 1e-6)
            wins.append(win.reshape(WINDOW, 1))
    X = np.array(wins, dtype=np.float32)

    preds = model.predict(X, verbose=0)
    classes = np.argmax(preds, axis=1)
    final   = int(np.bincount(classes).argmax())          
    conf    = float(preds[:, final].mean())               
    human   = LABELS_EMG[final]

    gemini_txt = None
    if gemini_key:
        gemini_txt = query_gemini_rest("EMG", human, conf, gemini_key)

    return human, conf, gemini_txt



FEATURE_COLS = [
    "rms_amplitude",
    "peak_frequency",
    "spectral_entropy",
    "zero_crossing_rate",
    "mean_frequency",
]

def vag_to_features(file_obj) -> np.ndarray:
    df = pd.read_csv(file_obj)
    x = df[FEATURE_COLS].iloc[0].values.astype(np.float32)
    return x.reshape(1, -1)


def predict_vag_from_features(file_obj, model_bundle, gemini_key=""):
    model   = model_bundle["model"]
    scaler  = model_bundle["scaler"]
    encoder = model_bundle["encoder"]

    x   = vag_to_features(file_obj)
    x_s = scaler.transform(x)
    prob = model.predict_proba(x_s)[0]
    idx  = int(np.argmax(prob))
    conf = float(prob[idx])
    label = encoder.inverse_transform([idx])[0].title()

    gem_note = (
        query_gemini_rest("VAG", label, conf, gemini_key)
        if gemini_key else None
    )
    return label, label, conf, gem_note