import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt
import pickle


ARRHYTHMIA_TYPES = {
    'Asystole': 0,
    'Bradycardia': 1, 
    'Tachycardia': 2,
    'Ventricular_Tachycardia': 3,
    'Ventricular_Flutter_Fib': 4
}

def preprocess_signal(ppg_signal, fs=250):
    """Basic signal preprocessing"""
    if ppg_signal is None or len(ppg_signal) == 0:
        return None
    

    ppg_signal = ppg_signal[~np.isnan(ppg_signal)]
    if len(ppg_signal) == 0:
        return None
    

    nyquist = fs / 2
    low_cutoff = 0.5 / nyquist
    high_cutoff = 8.0 / nyquist
    
    try:
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_signal = signal.filtfilt(b, a, ppg_signal)
        

        detrended_signal = signal.detrend(filtered_signal)
        

        mean_val = np.mean(detrended_signal)
        std_val = np.std(detrended_signal)
        outlier_mask = np.abs(detrended_signal - mean_val) < 3 * std_val
        clean_signal = np.copy(detrended_signal)
        clean_signal[~outlier_mask] = mean_val
        
        return clean_signal
    except:
        return ppg_signal

def extract_features(ppg_signal, fs=250):
    """Extract essential features from PPG signal"""
    if ppg_signal is None or len(ppg_signal) == 0:
        return {}
    
    features = {}
    

    features['mean'] = np.mean(ppg_signal)
    features['std'] = np.std(ppg_signal)
    features['var'] = np.var(ppg_signal)
    features['min'] = np.min(ppg_signal)
    features['max'] = np.max(ppg_signal)
    features['range'] = np.ptp(ppg_signal)
    features['median'] = np.median(ppg_signal)
    features['skewness'] = skew(ppg_signal)
    features['kurtosis'] = kurtosis(ppg_signal)
    features['energy'] = np.sum(ppg_signal ** 2)
    features['rms'] = np.sqrt(np.mean(ppg_signal ** 2))
    

    zero_crossings = np.where(np.diff(np.signbit(ppg_signal)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(ppg_signal)

    try:
        freqs, psd = signal.welch(ppg_signal, fs=fs, nperseg=min(len(ppg_signal)//4, 1024))
        total_power = np.sum(psd)
        
        if total_power > 0:
            vlf_band = (freqs >= 0.003) & (freqs < 0.04)
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs < 0.4)
            
            features['vlf_power'] = np.sum(psd[vlf_band]) / total_power
            features['lf_power'] = np.sum(psd[lf_band]) / total_power
            features['hf_power'] = np.sum(psd[hf_band]) / total_power
            features['lf_hf_ratio'] = features['lf_power'] / (features['hf_power'] + 1e-10)
            

            features['spectral_centroid'] = np.sum(freqs * psd) / (total_power + 1e-10)
            features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / (total_power + 1e-10))
            features['dominant_frequency'] = freqs[np.argmax(psd)]
    except:
        features.update({
            'vlf_power': 0, 'lf_power': 0, 'hf_power': 0,
            'lf_hf_ratio': 0, 'spectral_centroid': 0,
            'spectral_bandwidth': 0, 'dominant_frequency': 0
        })
    
    try:
        peaks, _ = signal.find_peaks(ppg_signal, distance=fs//4, prominence=np.std(ppg_signal)*0.1)
        
        if len(peaks) > 0:
            features['num_peaks'] = len(peaks)
            features['avg_peak_amplitude'] = np.mean(ppg_signal[peaks])
            features['std_peak_amplitude'] = np.std(ppg_signal[peaks])
            
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / fs
                heart_rate = 60 / np.mean(peak_intervals)
                features['estimated_hr'] = heart_rate
                features['hr_variability'] = np.std(peak_intervals)
                features['rmssd'] = np.sqrt(np.mean(np.diff(peak_intervals) ** 2))
            else:
                features['estimated_hr'] = 0
                features['hr_variability'] = 0
                features['rmssd'] = 0
        else:
            features.update({
                'num_peaks': 0, 'avg_peak_amplitude': 0,
                'std_peak_amplitude': 0, 'estimated_hr': 0,
                'hr_variability': 0, 'rmssd': 0
            })
    except:
        features.update({
            'num_peaks': 0, 'avg_peak_amplitude': 0,
            'std_peak_amplitude': 0, 'estimated_hr': 0,
            'hr_variability': 0, 'rmssd': 0
        })

    try:
        coeffs = pywt.wavedec(ppg_signal, 'db4', level=5)
        for i, coeff in enumerate(coeffs):
            level_name = 'approx' if i == 0 else f'detail_{i}'
            features[f'{level_name}_energy'] = np.sum(coeff ** 2)
            features[f'{level_name}_mean'] = np.mean(coeff)
            features[f'{level_name}_std'] = np.std(coeff)
    except:
        for i in range(6):
            level_name = 'approx' if i == 0 else f'detail_{i}'
            features[f'{level_name}_energy'] = 0
            features[f'{level_name}_mean'] = 0
            features[f'{level_name}_std'] = 0

    features['signal_quality'] = 1.0 if len(peaks) > 0 else 0.5
    
    return features

def predict_arrhythmia(ppg_signal, model_path='ppg_arrhythmia_model.pkl', fs=250):
    """
    Simplified prediction function for PPG arrhythmia detection
    
    Args:
        ppg_signal: Raw PPG signal (numpy array)
        model_path: Path to the saved model file
        fs: Sampling frequency (default: 250 Hz)
    
    Returns:
        Dictionary with prediction results
    """
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        scaler = model_package['scaler']
        feature_names = model_package['feature_names']
        

        clean_signal = preprocess_signal(ppg_signal, fs)
        if clean_signal is None:
            return {
                'success': False,
                'error': 'Signal preprocessing failed',
                'prediction': None,
                'arrhythmia': None,
                'confidence': 0.0
            }
        

        features = extract_features(clean_signal, fs)
        if not features:
            return {
                'success': False,
                'error': 'Feature extraction failed',
                'prediction': None,
                'arrhythmia': None,
                'confidence': 0.0
            }
     
        feature_df = pd.DataFrame([features])

        for feature in feature_names:
            if feature not in feature_df.columns:
                feature_df[feature] = 0.0
        
        feature_df = feature_df[feature_names]
        

        feature_scaled = scaler.transform(feature_df)

        prediction = model.predict(feature_scaled)[0]
        prediction_proba = model.predict_proba(feature_scaled)[0]
        

        arrhythmia_names = {v: k for k, v in ARRHYTHMIA_TYPES.items()}
        arrhythmia_name = arrhythmia_names.get(prediction, 'Unknown')
        

        confidence = float(prediction_proba[prediction])
        
        return {
            'success': True,
            'prediction': int(prediction),
            'arrhythmia': arrhythmia_name,
            'confidence': confidence,
            'probabilities': {
                arrhythmia_names[i]: float(prob) 
                for i, prob in enumerate(prediction_proba)
                if i in arrhythmia_names
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}',
            'prediction': None,
            'arrhythmia': None,
            'confidence': 0.0
        }


if __name__ == "__main__":
    fs = 250
    duration = 30  
    t = np.linspace(0, duration, fs * duration)

    heart_rate = 75
    ppg_signal = np.sin(2 * np.pi * (heart_rate/60) * t) + 0.1 * np.random.randn(len(t))
    

    result = predict_arrhythmia(ppg_signal, model_path='ppg_arrhythmia_model.pkl')
    
    if result['success']:
        print(f"Prediction successful!")
        print(f"Arrhythmia: {result['arrhythmia']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"All probabilities:")
        for arrhythmia, prob in result['probabilities'].items():
            print(f"  {arrhythmia}: {prob:.3f}")
    else:
        print(f"Prediction failed: {result['error']}")