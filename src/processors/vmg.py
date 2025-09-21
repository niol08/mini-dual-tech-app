import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
from scipy import signal
import sys

# Import label mapping
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from label_map import apply_label_mapping

def load_emg_model_for_vmg():
    """Load EMG model for VMG classification using dynamic import"""
    try:
        # Try to import HuggingFace client dynamically
        import importlib.util
        import sys
        
        # Add the app directory to path
        app_dir = os.path.join(os.path.dirname(__file__), '..', 'app')
        if app_dir not in sys.path:
            sys.path.append(app_dir)
        
        # Import model_loader
        spec = importlib.util.spec_from_file_location("model_loader", os.path.join(app_dir, "model_loader.py"))
        model_loader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_loader_module)
        
        # Create HuggingFace client instance
        hf_client = model_loader_module.HuggingFaceClient()
        return hf_client
        
    except Exception as e:
        print(f"Could not load HuggingFace client: {e}")
        return None

def classify_vmg_with_emg_model(signal_data: np.ndarray) -> Dict[str, Any]:
    """Classify VMG signal using EMG model"""
    
    try:
        # Load EMG model client
        hf_client = load_emg_model_for_vmg()
        if hf_client is None:
            return {
                'classification': 'unknown',
                'confidence': 0.0,
                'method': 'emg_model_unavailable',
                'error': 'Could not load EMG model'
            }
        
        # Prepare signal data for EMG model (expects 1000 samples, normalized)
        data = signal_data.copy()
        
        # Resize to 1000 samples
        if len(data) > 1000:
            data = data[:1000]
        elif len(data) < 1000:
            # Pad with zeros
            data = np.pad(data, (0, 1000 - len(data)), mode='constant', constant_values=0)
        
        # Normalize the data
        normalized_data = (data - data.mean()) / (data.std() + 1e-6)
        
        # Create a temporary file-like object for the HuggingFace client
        import io
        
        # Convert signal to text format for the model
        data_text = '\n'.join([str(val) for val in normalized_data])
        temp_file = io.BytesIO(data_text.encode('utf-8'))
        temp_file.name = 'vmg_signal.txt'  # Add name attribute
        
        # Use EMG model to classify the VMG signal
        predicted_label, confidence = hf_client.predict_emg(temp_file)
        
        return {
            'classification': predicted_label,
            'confidence': confidence,
            'method': 'emg_model_classification',
            'model_classes': ["healthy", "myopathy", "neuropathy"]
        }
        
    except Exception as e:
        print(f"Error in EMG model classification: {e}")
        return {
            'classification': 'unknown',
            'confidence': 0.0,
            'method': 'emg_model_error',
            'error': str(e)
        }

def analyze_vmg_signal(signal_data: np.ndarray) -> Dict[str, float]:
    """Extract VMG signal features"""
    features = {}
    
    try:
        # Basic statistical features
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak_to_peak'] = np.max(signal_data) - np.min(signal_data)
        
        # Frequency domain features
        freqs, psd = signal.periodogram(signal_data, fs=1000)  # Assuming 1kHz sampling
        features['dominant_freq'] = freqs[np.argmax(psd)]
        features['mean_freq'] = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_centroid'] = features['mean_freq']
        
        # Energy features
        features['total_energy'] = np.sum(signal_data**2)
        features['mean_energy'] = features['total_energy'] / len(signal_data)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal_data)
        
    except Exception as e:
        print(f"Error extracting VMG features: {e}")
        features = {'error': str(e)}
        
    return features

def classify_vmg_condition(signal_data: np.ndarray) -> Dict[str, Any]:
    """Classify VMG signal using EMG model instead of reference pattern matching"""
    
    # Use EMG model for VMG classification
    classification_result = classify_vmg_with_emg_model(signal_data)
    
    # Extract features for additional metadata
    input_features = analyze_vmg_signal(signal_data)
    classification_result['features'] = input_features
    
    return classification_result

def process_vmg(file_path: str) -> Dict[str, Any]:
    """
    Process VMG (Vibrocardiogram/Vibromyogram) signals using EMG model for disease classification.
    
    Args:
        file_path: Path to the VMG data file
        
    Returns:
        Dictionary containing processing results with label mapping applied
    """
    try:
        # Read input data
        if file_path.lower().endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file_path)
        elif file_path.lower().endswith('.txt'):
            # Try different formats
            try:
                data = pd.read_csv(file_path, delimiter='\t')
            except:
                try:
                    data = pd.read_csv(file_path, delimiter=' ')
                except:
                    data = np.loadtxt(file_path)
                    data = pd.DataFrame(data)
        else:
            # Generic file reading
            try:
                data = pd.read_csv(file_path)
            except:
                data = np.loadtxt(file_path)
                data = pd.DataFrame(data)
        
        # Convert to numpy array for processing
        if isinstance(data, pd.DataFrame):
            if data.shape[1] >= 2:
                # Assume format: [time, amplitude] or [time, vmg_signal, ...]
                signal_data = data.iloc[:, 1].values
                time_data = data.iloc[:, 0].values if data.shape[1] > 1 else np.arange(len(signal_data))
            else:
                # Single column assumed to be VMG signal
                signal_data = data.iloc[:, 0].values
                time_data = np.arange(len(signal_data))
        else:
            # numpy array
            if data.ndim == 2 and data.shape[1] >= 2:
                signal_data = data[:, 1]
                time_data = data[:, 0]
            else:
                signal_data = data.flatten()
                time_data = np.arange(len(signal_data))
        
        # Perform classification using EMG model
        classification_result = classify_vmg_condition(signal_data)
        
        # Apply label mapping to results
        mapped_result = apply_label_mapping(
            classification_result['classification'], 
            classification_result['confidence'],
            processor_type='vmg'
        )
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original signal
        axes[0, 0].plot(time_data[:min(1000, len(time_data))], 
                       signal_data[:min(1000, len(signal_data))])
        axes[0, 0].set_title('VMG Signal (first 1000 samples)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # Power spectrum
        freqs, psd = signal.periodogram(signal_data, fs=1000)
        axes[0, 1].semilogy(freqs[:len(freqs)//2], psd[:len(psd)//2])
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].grid(True)
        
        # Classification results
        axes[0, 2].text(0.5, 0.7, f"Classification: {mapped_result['mapped_label']}", 
                       ha='center', va='center', transform=axes[0, 2].transAxes, 
                       fontsize=14, fontweight='bold')
        axes[0, 2].text(0.5, 0.5, f"Confidence: {mapped_result['confidence']:.3f}", 
                       ha='center', va='center', transform=axes[0, 2].transAxes, 
                       fontsize=12)
        axes[0, 2].text(0.5, 0.3, f"Original: {classification_result['classification']}", 
                       ha='center', va='center', transform=axes[0, 2].transAxes, 
                       fontsize=10)
        axes[0, 2].text(0.5, 0.1, f"Method: {classification_result.get('method', 'unknown')}", 
                       ha='center', va='center', transform=axes[0, 2].transAxes, 
                       fontsize=9, style='italic')
        axes[0, 2].set_title('Classification Results (EMG Model)')
        axes[0, 2].axis('off')
        
        # EMG Model Classes visualization
        model_classes = classification_result.get('model_classes', ["healthy", "myopathy", "neuropathy"])
        predicted_class = classification_result['classification']
        class_confidences = [classification_result['confidence'] if cls == predicted_class else 0.0 for cls in model_classes]
        
        bars = axes[1, 0].bar(model_classes, class_confidences, alpha=0.7)
        axes[1, 0].set_title('EMG Model Prediction Confidence')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Highlight predicted class
        if predicted_class in model_classes:
            best_idx = model_classes.index(predicted_class)
            bars[best_idx].set_color('red')
        
        # Signal statistics
        features = classification_result.get('features', {})
        if features and 'error' not in features:
            feature_names = ['RMS', 'Peak-to-Peak', 'Dominant Freq', 'Zero Crossing Rate']
            feature_values = [
                features.get('rms', 0),
                features.get('peak_to_peak', 0),
                features.get('dominant_freq', 0),
                features.get('zero_crossing_rate', 0)
            ]
            axes[1, 1].bar(feature_names, feature_values, alpha=0.7)
            axes[1, 1].set_title('Signal Features')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Signal preprocessing visualization (show normalized signal used by EMG model)
        normalized_signal = signal_data.copy()
        if len(normalized_signal) > 1000:
            normalized_signal = normalized_signal[:1000]
        elif len(normalized_signal) < 1000:
            normalized_signal = np.pad(normalized_signal, (0, 1000 - len(normalized_signal)), mode='constant')
        normalized_signal = (normalized_signal - normalized_signal.mean()) / (normalized_signal.std() + 1e-6)
        
        axes[1, 2].plot(normalized_signal, alpha=0.7, label='Normalized for EMG Model')
        axes[1, 2].plot(signal_data[:min(1000, len(signal_data))], alpha=0.5, label='Original Signal')
        axes[1, 2].set_title('Signal Preprocessing for EMG Model')
        axes[1, 2].set_xlabel('Sample')
        axes[1, 2].set_ylabel('Amplitude')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # Generate summary
        summary_parts = [
            "VMG Signal Analysis Complete (using EMG Model):",
            f"- Signal length: {len(signal_data)} samples",
            f"- Classification: {mapped_result['mapped_label']} (confidence: {mapped_result['confidence']:.3f})",
            f"- Original classification: {classification_result['classification']}",
            f"- Analysis method: {classification_result.get('method', 'unknown')}",
            f"- Model classes: {', '.join(model_classes)}",
        ]
            
        if features and 'error' not in features:
            summary_parts.extend([
                f"- Signal RMS: {features.get('rms', 0):.4f}",
                f"- Dominant frequency: {features.get('dominant_freq', 0):.2f} Hz"
            ])
        
        if 'error' in classification_result:
            summary_parts.append(f"- Classification error: {classification_result['error']}")
        
        summary = "\n".join(summary_parts)
        
        metadata = {
            "file_type": "vmg_signal",
            "signal_length": len(signal_data),
            "classification": mapped_result['mapped_label'],
            "original_classification": classification_result['classification'],
            "confidence": float(mapped_result['confidence']),
            "method": classification_result.get('method', 'unknown'),
            "model_classes": model_classes,
            "features": features,
            "label_mapping_applied": True,
            "uses_emg_model": True
        }
        
        return {
            "summary": summary,
            "metadata": metadata,
            "images": [],
            "figures": [fig],
            "audio": None
        }
        
    except Exception as e:
        return {
            "summary": f"Error processing VMG data with EMG model: {str(e)}",
            "metadata": {"error": str(e), "file_path": file_path, "uses_emg_model": True},
            "images": [],
            "figures": [],
            "audio": None
        }
