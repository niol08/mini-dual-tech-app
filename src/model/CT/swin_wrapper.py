
import os
from typing import List, Dict, Optional
import torch
import numpy as np
from PIL import Image
import pydicom
from transformers import AutoImageProcessor, SwinForImageClassification


class SwinHFClassifier:
    """
    Wrapper for Hugging Face SwinForImageClassification models.
    Handles DICOM (.dcm) or standard image files and returns:
      {"label_id": int, "label_name": str, "confidence": float, "ai_insight": str}
    """

    def __init__(self,
        model_id: str = "Koushim/breast-cancer-swin-classifier",
        device: Optional[str] = None,
        hf_token: Optional[str] = None):
        self.model_id = model_id
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        kwargs = {"use_auth_token": hf_token} if hf_token else {}
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True, **kwargs)
        self.model = SwinForImageClassification.from_pretrained(model_id, **kwargs).to(self.device).eval()


        self.labels = ["benign", "malignant"]

    def _dicom_to_pil(self, dcm_path: str,
        apply_window: bool = False,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None) -> Image.Image:
        """Read DICOM and convert to PIL RGB using optional simple windowing."""
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        if apply_window and (window_center is not None and window_width is not None):
            low = window_center - window_width / 2.0
            high = window_center + window_width / 2.0
            arr = np.clip((arr - low) / (high - low), 0.0, 1.0) * 255.0
        else:
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0
            else:
                arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=2)

        return Image.fromarray(arr).convert("RGB")

    def _load_image(self, path: str, dicom_windowing: Optional[dict] = None) -> Image.Image:
        """Load either a DICOM (.dcm) or a normal image and return a PIL.Image RGB."""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".dcm", ".dicom"):
            if dicom_windowing:
                center = dicom_windowing.get("center")
                width = dicom_windowing.get("width")
                return self._dicom_to_pil(path, apply_window=True, window_center=center, window_width=width)
            else:
                return self._dicom_to_pil(path, apply_window=False)
        else:
            return Image.open(path).convert("RGB")

    def predict(self, path: str, top_k: int = 1, dicom_windowing: Optional[dict] = None) -> List[Dict]:
        """
        Return top_k predictions in form:
            [{"id": idx, "label_name": str, "score": float}, ...]
        """
        img = self._load_image(path, dicom_windowing=dicom_windowing)
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top_idx = probs.argsort()[-top_k:][::-1]
        results = []
        for idx in top_idx:
            label_name = self.labels[idx] if idx < len(self.labels) else str(idx)
            results.append({"id": int(idx), "label_name": label_name, "score": float(probs[idx])})
        return results

    def predict_single(self, path: str, dicom_windowing: Optional[dict] = None) -> Dict:
        """
        Convenience method returning a single top prediction payload:
        { "label_id": int, "label_name": str, "confidence": float, "ai_insight": str }
        """
        preds = self.predict(path, top_k=1, dicom_windowing=dicom_windowing)
        if not preds:
            return {"label_id": -1, "label_name": "unknown", "confidence": 0.0, "ai_insight": "no prediction"}
        top = preds[0]
        return {
            # "label_id": top["id"],
            "label_name": top["label_name"],
            "confidence": top["score"],
            "ai_insight": "swin-hf"
        }
