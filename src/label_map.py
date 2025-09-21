
LABEL_REPHRASE = {
    # Medical condition mappings
    "benign": "non-cancerous",
    "malignant": "cancerous",
    "Normal": "No abnormality detected",
    "Pneumonia": "Chest infection",
    
    # BioPotentials AP/RP mappings
    "action_potential": "AP",
    "resting_potential": "RP",
    "ap": "AP",
    "rp": "RP",
    "0": "RP",  # Class 0 -> Resting Potential
    "1": "AP",  # Class 1 -> Action Potential
    0: "RP",    # Numeric class mapping
    1: "AP",    # Numeric class mapping
    
    # ERP/EEG related mappings
    "epilepsy": "Epileptic Activity",
    "normal": "Normal Activity",
    "seizure": "Seizure Detected",
    "no_seizure": "No Seizure",
    
    # VMG (Vibromyogram) specific mappings
    "healthy": "Normal VMG Pattern",
    "myopathy": "Myopathy Detected",
    "neuropathy": "Neuropathy Detected",
    "unknown": "Pattern Unrecognized",
    
    # General biosignal mappings
    "abnormal": "Abnormal Signal",
    "pathological": "Pathological Pattern",
    "physiological": "Normal Physiological Pattern"
}

def apply_label_mapping(original_label, confidence, processor_type=None):
    """
    Apply label mapping based on processor type and original classification
    
    Args:
        original_label: Original classification result
        confidence: Classification confidence score
        processor_type: Type of processor ('biopotentials', 'vmg', 'erp', etc.)
    
    Returns:
        Dictionary with mapped label and metadata
    """
    
    # Convert to string for consistent mapping
    label_key = str(original_label).lower()
    
    # Apply general mapping first
    mapped_label = LABEL_REPHRASE.get(original_label, LABEL_REPHRASE.get(label_key, str(original_label)))
    
    # Apply processor-specific mappings
    if processor_type == 'vmg':
        # VMG specific label refinements
        if label_key in ['healthy', 'normal']:
            mapped_label = "Normal VMG - Healthy Muscle"
        elif label_key == 'myopathy':
            mapped_label = "Myopathy - Muscle Disease"
        elif label_key == 'neuropathy':
            mapped_label = "Neuropathy - Nerve Disorder"
        elif label_key == 'unknown':
            mapped_label = "Unrecognized VMG Pattern"
    
    elif processor_type == 'biopotentials':
        # BioPotentials specific refinements
        if label_key in ['0', '1'] or isinstance(original_label, (int, float)):
            mapped_label = "RP" if int(float(original_label)) == 0 else "AP"
        elif 'action' in label_key:
            mapped_label = "AP - Action Potential"
        elif 'resting' in label_key:
            mapped_label = "RP - Resting Potential"
    
    elif processor_type == 'erp':
        # ERP specific refinements
        if label_key in ['normal', 'healthy']:
            mapped_label = "Normal EEG Activity"
        elif label_key in ['epilepsy', 'seizure']:
            mapped_label = "Epileptic Activity Detected"
    
    return {
        'mapped_label': mapped_label,
        'original_label': original_label,
        'confidence': confidence,
        'mapping_applied': mapped_label != str(original_label),
        'processor_type': processor_type
    }
