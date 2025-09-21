
from utils.page import biosignal_chat_page
from processors.biopotentials import process_biopotentials
from label_map import apply_label_mapping

def run_erp_wrapper(uploaded):
    """
    ERP analysis using biopotentials processor with label mapping.
    This integrates ERP detection with AP/RP classification.
    """
    result = process_biopotentials(uploaded)
    
    # Check if processing was successful and extract classification info
    if result.get("metadata", {}).get("models_loaded", False):
        # Get the classification from biopotentials result
        metadata = result["metadata"]
        
        # Extract dominant class from biopotentials classification
        if "dominant_class" in metadata:
            dominant_class = metadata["dominant_class"]
            
            # Use class distribution to estimate confidence
            class_dist = metadata.get("class_distribution", {})
            if class_dist:
                total_predictions = sum(class_dist.values())
                dominant_count = class_dist.get(dominant_class, 0)
                confidence = dominant_count / total_predictions if total_predictions > 0 else 0.5
            else:
                confidence = 0.7  # Default confidence
            
            # Apply label mapping for ERP context
            mapped_result = apply_label_mapping(
                dominant_class, 
                confidence,
                processor_type='erp'
            )
            
            # Create ERP-specific result format
            erp_result = {
                "model": "BioPotentials_ERP_Integration",
                "prediction": "epilepsy" if dominant_class == "1" or dominant_class == 1 else "normal",
                "confidence": confidence,
                "mapped_prediction": mapped_result['mapped_label'],
                "n_samples": metadata.get("processed_samples", 1),
                "original_biopotential_class": dominant_class,
                "biopotential_confidence": confidence,
                "label_mapping_applied": True,
                "class_distribution": class_dist,
                "per_sample": [{
                    "p_normal": 1 - confidence if (dominant_class == "1" or dominant_class == 1) else confidence,
                    "p_epilepsy": confidence if (dominant_class == "1" or dominant_class == 1) else 1 - confidence,
                    "biopotential_class": dominant_class,
                    "predicted_label": 1 if (dominant_class == "1" or dominant_class == 1) else 0,
                    "mapped_label": mapped_result['mapped_label']
                }]
            }
            
            # Update the original result with ERP-specific information
            result["metadata"].update({
                "erp_analysis": erp_result,
                "label_mapping_applied": True
            })
            
            # Update summary to include ERP context
            erp_summary = f"""ERP Analysis via BioPotentials Classification:
- Mapped Result: {mapped_result['mapped_label']}
- Confidence: {confidence:.3f}
- Original BioPotentials Class: {dominant_class}
- ERP Interpretation: {erp_result['prediction']}
- Label Mapping: {'Applied' if mapped_result['mapping_applied'] else 'Not needed'}

{result['summary']}"""
            result["summary"] = erp_summary
            
            return result
    
    # Handle case where models failed to load or processing had errors
    error_result = result.copy()
    error_result["metadata"]["erp_analysis"] = {
        "model": "BioPotentials_ERP_Integration",
        "prediction": "error",
        "confidence": 0.0,
        "error": "BioPotentials models could not be loaded or processing failed",
        "label_mapping_applied": False
    }
    
    error_result["summary"] = f"""ERP Analysis Error:
- BioPotentials processing failed or models unavailable
- Cannot perform ERP classification
- Original error: {result.get('summary', 'Unknown error')}

{result['summary']}"""
    
    return error_result

biosignal_chat_page(
    biosignal_label="ERP",
    slug="erp",
    accepted_types=("edf", "fif", "set", "bdf", "csv", "txt"),
    analyzer=run_erp_wrapper,
    analyzer_label="Run ERP Analysis (with BioPotentials)"
)
