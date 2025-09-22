import os
import torch
import soundfile as sf
import librosa
import numpy as np
from pydub import AudioSegment
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch.nn.functional as F
from typing import Dict, Any

MODEL_NAME = "ManyaGupta/wav2vec2-for-dysarthria-detection"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def load_audio(file_path: str):
    """
    Load .wav or .egg audio and return waveform + sample rate.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".wav":
        audio, sr = sf.read(file_path)
    elif ext == ".egg":
        audio_seg = AudioSegment.from_file(file_path)
        audio = np.array(audio_seg.get_array_of_samples()).astype(float)
        sr = audio_seg.frame_rate
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    target_sr = feature_extractor.sampling_rate
    if sr != target_sr:
        audio = librosa.resample(audio.astype(float), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr

def process_speech(file_path: str) -> Dict[str, Any]:
    """Process audio for speech analysis."""
    try:
        audio_input, sr = load_audio(file_path)

        inputs = feature_extractor(
            audio_input,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = F.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probs, dim=-1).item()
        predicted_prob = probs[0, predicted_class_idx].item()

        label_map = model.config.id2label
        prediction = label_map.get(predicted_class_idx, f"LABEL_{predicted_class_idx}")

        return {
            "summary": f"Predicted dysarthria severity: **{prediction}** ({predicted_prob*100:.1f}% confidence)",
            "metadata": {
                "model": MODEL_NAME,
                "predicted_class": prediction,
                "predicted_prob": predicted_prob,
                "all_logits": logits.cpu().tolist(),
                "all_probs": probs.cpu().tolist(),
                "file_type": "speech_analysis",
                "processing_method": "wav2vec2_dysarthria_detection"
            },
            "images": [],
            "figures": [],
            "audio": file_path
        }

    except Exception as e:
        return {
            "summary": f"Error processing speech file: {e}",
            "metadata": {
                "file_type": "speech_error",
                "error": str(e),
                "file_path": file_path,
                "processing_method": "speech_analysis_failed"
            },
            "images": [],
            "figures": [],
            "audio": None
        }
