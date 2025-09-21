
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Dict, List, Tuple, Optional
import importlib.util
import zipfile
import io
import os
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, detrend, resample
from sklearn.ensemble import IsolationForest
import mne


def _read_ppg_from_uploaded(uploaded) -> Tuple[np.ndarray, float]:
    name = (uploaded.name or "").lower()
    raw = uploaded.getbuffer()
    if name.endswith(".csv") or name.endswith(".txt"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw))
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            t_col = None
            for c in num.columns:
                cl = c.lower()
                if cl in {"time", "t", "sec", "seconds", "ms", "millisecond"}:
                    t_col = c; break
            if t_col is not None:
                ppg_col = [c for c in num.columns if c != t_col][0]
                t = num[t_col].astype(float).to_numpy()
                x = num[ppg_col].astype(float).to_numpy()
                dt = np.median(np.diff(t))
                fs = 125.0 if dt <= 0 else (1000.0/dt if t.max() > 1000 else 1.0/dt)
            else:
                x = num.iloc[:, 0].astype(float).to_numpy(); fs = 125.0
        elif num.shape[1] == 1:
            x = num.iloc[:, 0].astype(float).to_numpy(); fs = 125.0
        else:
            raise ValueError("No numeric columns found in uploaded file.")
        return x.astype(np.float32), float(fs)

    
    with NamedTemporaryFile(delete=False, suffix=Path(name).suffix or ".edf") as tmp:
        tmp.write(raw); tmp.flush(); path = tmp.name
    raw_mne = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    picks = [ch for ch in raw_mne.ch_names if not ch.upper().startswith("DC")]
    if not picks: picks = raw_mne.ch_names[:1]
    raw_mne.pick(picks[:1])
    return raw_mne.get_data()[0].astype(np.float32), float(raw_mne.info["sfreq"])

def _bandpass_ppg(x: np.ndarray, fs: float, lo=0.5, hi=12.0, order=3) -> np.ndarray:
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, detrend(x)).astype(np.float32)

def _segment(x: np.ndarray, fs: float, win_s=10.0, step_s=2.0, max_windows=None) -> Tuple[List[np.ndarray], List[float]]:
    win = int(win_s * fs); step = int(step_s * fs)
    starts = list(range(0, max(1, len(x)-win+1), step))
    if max_windows is not None: starts = starts[:max_windows]
    segs, times = [], []
    for s in starts:
        seg = x[s:s+win]
        if len(seg) == win:
            segs.append(seg.copy()); times.append(s/fs)
    return segs, times


def _extract_best_pt_from_zip(zip_path: str, out_dir: str) -> Optional[str]:
    """
    Extract candidate .pt/.pth from zip. Preference:
      - files with 'ppg' or 'finetune' or 'ppgpt' or 'classifier' in name
      - else first .pth/.pt inside zip
    Returns extracted filepath or None.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        candidates = [n for n in names if n.lower().endswith((".pt", ".pth"))]

        pref = [n for n in candidates if any(k in n.lower() for k in ("ppg", "ppgpt", "finetun", "finetune", "classifier", "arrhythm"))]
        chosen = (pref + candidates)[:1]
        if not chosen:
            return None
        chosen = chosen[0]
        out_path = Path(out_dir) / Path(chosen).name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with z.open(chosen) as fh, open(out_path, "wb") as out_f:
            out_f.write(fh.read())
        return str(out_path)

def _resolve_ckpt(ckpt_path: str) -> str:
    """If ckpt_path is a zip, extract best candidate and return file path; else return input."""
    p = Path(ckpt_path)
    if p.is_file() and p.suffix.lower() == ".zip":
        tmp = TemporaryDirectory()
        out = _extract_best_pt_from_zip(str(p), tmp.name)
        if out is None:
            raise FileNotFoundError(f"No .pt/.pth checkpoint found inside zip {ckpt_path}")
        return out
    if p.is_file():
        return str(p)

    if p.exists() and p.is_dir():
        for cand in p.glob("**/*.zip"):
            if "ppg" in cand.name.lower():
                return _resolve_ckpt(str(cand))
    raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")


def _load_heartgpt_module(heartgpt_dir: Optional[str] = None):
    """
    Try to import Heart_PT_finetune.py as a module using HEARTGPT_DIR env or default ./HeartGPT.
    Returns the loaded module object.
    """
    heartgpt_dir = heartgpt_dir or os.environ.get("HEARTGPT_DIR") or "./HeartGPT"
    finetune_py = Path(heartgpt_dir) / "legacy_scripts" / "Heart_PT_finetune.py"
    if not finetune_py.exists():
        raise FileNotFoundError(f"Heart_PT_finetune.py not found in {heartgpt_dir}. Set HEARTGPT_DIR to your clone.")
    spec = importlib.util.spec_from_file_location("heartpt_finetune", str(finetune_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  
    return mod

def _build_heartgpt_model_from_ckpt(ckpt_path: str, device: torch.device, heartgpt_dir: Optional[str] = None):
    """
    - resolve zip (if given),
    - load state_dict,
    - instantiate Heart_GPT_FineTune from Heart_PT_finetune.py
    - load state_dict into model.
    Returns model (nn.Module) ready for inference.
    """
    ckpt_file = _resolve_ckpt(ckpt_path)
    mod = _load_heartgpt_module(heartgpt_dir)
    if not hasattr(mod, "Heart_GPT_FineTune"):
        raise RuntimeError("Heart_GPT_FineTune not found in Heart_PT_finetune.py")
    model = mod.Heart_GPT_FineTune()
    sd = torch.load(ckpt_file, map_location=device)

    if isinstance(sd, dict) and ("state_dict" in sd or "model_state_dict" in sd):
        sd2 = sd.get("state_dict", sd.get("model_state_dict"))
    else:
        sd2 = sd
 
    if isinstance(sd2, dict):

        try:
            model.load_state_dict(sd2)
        except RuntimeError:

            new_sd = {k.replace("module.", ""): v for k, v in sd2.items()}
            model.load_state_dict(new_sd)
    else:
 
        if hasattr(sd, "state_dict"):
            return sd.to(device).eval()
        else:
            raise RuntimeError("Checkpoint format not recognized (neither state_dict nor full module).")
    model.to(device).eval()
    return model


def _try_repo_tokeniser(heartgpt_dir: Optional[str] = None):
    """
    Try to import a tokeniser from the repo tokenise/ folder (if present).
    We'll attempt a few plausible names and return a function `timeseries_to_tokens(sig, target_length)`.
    If not found, return None.
    """
    heartgpt_dir = heartgpt_dir or os.environ.get("HEARTGPT_DIR") or "./HeartGPT"
    tokenise_folder = Path(heartgpt_dir) / "tokenise"
    if not tokenise_folder.exists():
        return None
    for fname in ("tokenise_ppg.py", "ppg_tokeniser.py", "ppg_tokeniser.py", "tokeniser_ppg.py"):
        fpath = tokenise_folder / fname
        if fpath.exists():
            spec = importlib.util.spec_from_file_location("heart_tokeniser", str(fpath))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  
            for attr in ("timeseries_to_tokens", "ppg_to_tokens", "tokenise_ppg", "tokenise"):
                if hasattr(mod, attr):
                    return getattr(mod, attr)
    return None

def _fallback_quantize(seg: np.ndarray, target_len: int, vocab_size: int):
    """
    Simple fallback tokeniser:
     - resample to target_len
     - normalize to 0..1
     - quantize into vocab bins
    """
    seg = (seg - np.nanmin(seg)) / (np.nanmax(seg) - np.nanmin(seg) + 1e-9)
    seg_r = resample(seg.astype(np.float32), target_len)
    toks = np.floor(seg_r * (vocab_size - 1)).astype(np.int64)
    toks = np.clip(toks, 0, vocab_size - 1)
    return toks


@torch.no_grad()
def _embed_windows_with_heartgpt(model: torch.nn.Module, tokens_batch: np.ndarray, device: torch.device):
    """
    tokens_batch: (N, T) int numpy
    returns: embeddings (N, D) float numpy by:
       token_embedding + pos_embedding -> blocks -> ln_f -> mean_pool
    """
    model = model.to(device)
    model.eval()
    idx = torch.tensor(tokens_batch, dtype=torch.long, device=device)
    tok_emb = model.token_embedding_table(idx)           
    T = idx.shape[1]
    pos = model.position_embedding_table(torch.arange(T, device=device))[None, :, :]  
    x = tok_emb + pos
    x = model.blocks(x)   
    x = model.ln_f(x)     
  
    emb = x.mean(dim=1).cpu().numpy()   
    return emb


def run_heartgpt_ppg_detector(uploaded,
                              ckpt_path: str,
                              heartgpt_dir: Optional[str] = None,
                              win_s: float = 10.0,
                              step_s: float = 2.0,
                              max_windows: int | None = 600) -> Dict:
    """
    Read uploaded PPG (CSV/TXT/EDF), run foundation model embeddings (HeartGPT PPG),
    then a simple unsupervised IsolationForest to return `p_issue` per window.
    - ckpt_path: can be a .pth/.pt file or a .zip archive (e.g. PPGPT_500k_iters.zip)
    - heartgpt_dir: optional path to the cloned HeartGPT repo (defaults to ./HeartGPT or $HEARTGPT_DIR)
    """

    x, fs = _read_ppg_from_uploaded(uploaded)
    x = _bandpass_ppg(x, fs, 0.5, 12.0)
    segs, times = _segment(x, fs, win_s=win_s, step_s=step_s, max_windows=max_windows)
    if not segs:
        raise RuntimeError("No windows produced from the PPG signal.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = _build_heartgpt_model_from_ckpt(ckpt_path, device=device, heartgpt_dir=heartgpt_dir)

        mod = _load_heartgpt_module(heartgpt_dir)
        block_size = getattr(mod, "block_size", 500)
        vocab_size = getattr(mod, "vocab_size", 102)
    except Exception as e:
        model = None
        block_size = 500
        vocab_size = 102

        print("Warning: HeartGPT load failed:", e)

    repo_tokeniser = None
    try:
        repo_tokeniser = _try_repo_tokeniser(heartgpt_dir)
    except Exception:
        repo_tokeniser = None


    tokens_list = []
    for seg in segs:

        if repo_tokeniser:
            try:
                toks = repo_tokeniser(seg, target_length=block_size)
            except Exception:
                toks = _fallback_quantize(seg, target_len=block_size, vocab_size=vocab_size)
        else:
            toks = _fallback_quantize(seg, target_len=block_size, vocab_size=vocab_size)
        tokens_list.append(toks)
    tokens_batch = np.stack(tokens_list, axis=0)  
    if model is not None:
        try:
            emb = _embed_windows_with_heartgpt(model, tokens_batch, device=device)  
        except Exception as e:
            print("Warning: embedding with HeartGPT failed:", e)
            emb = np.stack([np.concatenate([np.mean(s), np.std(s), np.max(s), np.min(s)]) for s in segs])
    else:
        emb = np.stack([np.array([np.mean(s), np.std(s), np.max(s), np.min(s)]) for s in segs])

    iso = IsolationForest(n_estimators=200, contamination=0.15, random_state=0)
    iso.fit(emb)
    scores = -iso.score_samples(emb)  
    s_norm = (scores - scores.min()) / (np.ptp(scores) + 1e-12)
    p_issue = s_norm.astype(float)

    per_window = []
    for t, p in zip(times, p_issue):
        per_window.append({
            "start_s": float(t),
            "p_no_disease": float(1.0 - p),
            "p_disease": float(p),
            "p_no_seiz": float(1.0 - p),
            "p_seiz": float(p),
        })

    model_name = Path(ckpt_path).name
    return {
        "model": f"HeartGPT({model_name})",
        "prediction": "abnormal_ppg" if float(np.mean(p_issue)) > 0.5 else "normal_ppg",
        "confidence": float(np.mean(p_issue)) if float(np.mean(p_issue)) > 0.5 else float(1.0 - np.mean(p_issue)),
        "n_windows": len(per_window),
        "sfreq": float(fs),
        "channels": ["PPG"],
        "per_window": per_window,
    }
