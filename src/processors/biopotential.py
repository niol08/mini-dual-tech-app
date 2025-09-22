import os
import sys
from typing import Dict, Any

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.erp_hf_model import run_erp_detector

# Import label mapping
try:
    from label_map import apply_label_mapping
except ImportError:
    def apply_label_mapping(classification, confidence, processor_type='erp'):
        return {
            'mapped_label': classification, 
            'confidence': confidence,
            'original_label': classification,
            'mapping_applied': False,
            'processor_type': processor_type
        }

def get_ap_rp_status(epilepsy_prob):
    """Determine AP/RP status based on epilepsy probability"""
    if epilepsy_prob >= 0.7:
        return {
            "ap_status": "Hyperexcitable",
            "rp_status": "Depolarized",
            "interpretation": "High epileptiform activity with abnormal neuronal excitability"
        }
    elif epilepsy_prob >= 0.5:
        return {
            "ap_status": "Elevated",
            "rp_status": "Unstable", 
            "interpretation": "Moderate epileptiform activity with altered membrane dynamics"
        }
    elif epilepsy_prob >= 0.3:
        return {
            "ap_status": "Borderline",
            "rp_status": "Variable",
            "interpretation": "Borderline activity with some membrane instability"
        }
    else:
        return {
            "ap_status": "Normal",
            "rp_status": "Stable",
            "interpretation": "Normal neuronal membrane dynamics within expected parameters"
        }

def process_biopotential(file_path: str) -> Dict[str, Any]:
    """
    Process ERP data using the ERP detector model.
    This function serves as the main entry point for ERP processing in app.py.
    
    Args:
        file_path: Path to the ERP data file
        
    Returns:
        Dictionary containing processing results compatible with app.py structure
    """
    try:
        # Create a file-like object that matches what run_erp_detector expects
        class FileWrapper:
            def __init__(self, file_path):
                self.name = os.path.basename(file_path)
                self.file_path = file_path
            
            def getbuffer(self):
                with open(self.file_path, 'rb') as f:
                    return f.read()
        
        file_wrapper = FileWrapper(file_path)
        erp_result = run_erp_detector(file_wrapper)
        
        # Extract results from ERP detector
        prediction = erp_result.get('prediction', 'unknown')
        confidence = erp_result.get('confidence', 0.5)
        model_name = erp_result.get('model', 'ERP_Classifier')
        per_sample = erp_result.get('per_sample', [{}])
        
        # Get AP/RP status
        epilepsy_prob = per_sample[0].get("p_epilepsy", 0) if per_sample else 0
        ap_rp_info = get_ap_rp_status(epilepsy_prob)
        
        # Apply label mapping for consistent labeling
        mapped_result = apply_label_mapping(prediction, confidence, processor_type='erp')
        
        # Create comprehensive summary with version identifier and label mapping
        summary = f"""BioPotential Analysis Results :
- Original Classification: {prediction.capitalize()}
- Mapped Classification: {mapped_result['mapped_label']}
- Confidence Score: {confidence:.3f}
- {'High confidence' if confidence > 0.7 else 'Moderate confidence' if confidence > 0.5 else 'Low confidence'} in classification

Signal Analysis:
- Prediction based on converted signal imagery (128x128 RGB)
- Input file: {os.path.basename(file_path)}
- Label mapping: {'Applied' if mapped_result['mapping_applied'] else 'Not needed'}

Clinical Interpretation with AP/RP Analysis:
- Action Potential (AP) Status: {ap_rp_info['ap_status']}
- Resting Potential (RP) Status: {ap_rp_info['rp_status']}
- Neurophysiological Assessment: {ap_rp_info['interpretation']}
- {f'Probability of epilepsy: {per_sample[0].get("p_epilepsy", 0):.3f}' if per_sample else ''}
- {f'Probability of normal: {per_sample[0].get("p_normal", 0):.3f}' if per_sample else ''}
- Consider correlation with clinical symptoms and additional testing"""
        
        # Build metadata
        metadata = {
            "file_type": "ap/rp",

            "prediction": prediction,
            "mapped_prediction": mapped_result['mapped_label'],
            "confidence": float(confidence),
            "processing_method": "ERP_epilepsy_classifier_with_image_conversion",
            "n_samples": erp_result.get('n_samples', 1),
            "per_sample_results": per_sample,
            "file_name": os.path.basename(file_path),
            "ap_rp_analysis": ap_rp_info,
            "label_mapping": mapped_result,
            "version": "v2024.09.22_LabelMapped_AP_RP_Enhanced"
        }
        
        return {
            "summary": summary,
            "metadata": metadata,
            "images": [],
            "figures": [],
            "audio": None,
            "debug_info": {
                "timestamp": str(__import__('datetime').datetime.now()),
                "processor_version": "v2024.09.22_LabelMapped_AP_RP_Enhanced",
                "original_prediction": prediction,
                "mapped_prediction": mapped_result['mapped_label'],
                "ap_rp_status": f"AP:{ap_rp_info['ap_status']}, RP:{ap_rp_info['rp_status']}"
            }
        }
        
    except Exception as e:
        # Error handling
        error_summary = f"""ERP Processing Error:
- Error: {str(e)}
- File: {os.path.basename(file_path)}
- Processing failed during ERP analysis

Possible causes:
- Unsupported file format
- Corrupted data file
- Missing dependencies (mne, tensorflow, keras)
- Model loading issues

Recommended actions:
- Verify file format is supported (.edf, .fif, .set, .bdf, .csv, .txt)
- Check file integrity
- Ensure all required packages are installed
- Try a different file format"""

        error_metadata = {
            "file_type": "erp_error",
            "error": str(e),
            "file_path": file_path,
            "processing_method": "ERP_classifier_failed"
        }
        
        return {
            "summary": error_summary,
            "metadata": error_metadata,
            "images": [],
            "figures": [],
            "audio": None
        }