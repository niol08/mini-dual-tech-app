
from utils.page import biosignal_chat_page
from processors.biopotentials import process_biopotentials
from label_map import apply_label_mapping

def run_erp_wrapper(uploaded):
    """
    ERP analysis using biopotentials processor with AP/RP classification.
    This integrates ERP detection with Action Potential vs Resting Potential analysis.
    """
    result = process_biopotentials(uploaded)
    
    # Check if processing was successful and extract AP/RP classification info
    if result.get("metadata", {}).get("models_loaded", False):
        # Get the classification from biopotentials result
        metadata = result["metadata"]
        
        # Extract dominant class from biopotentials classification
        if "dominant_class" in metadata:
            dominant_class = metadata["dominant_class"]
            ap_rp_result = metadata.get("ap_rp_result", {})
            
            # Use class distribution to estimate confidence
            class_dist = metadata.get("class_distribution", {})
            if class_dist:
                total_predictions = sum(class_dist.values())
                dominant_count = class_dist.get(dominant_class, 0)
                confidence = dominant_count / total_predictions if total_predictions > 0 else 0.5
            else:
                confidence = 0.7  # Default confidence
            
            # Apply label mapping for biopotentials to get AP/RP results
            mapped_result = apply_label_mapping(
                dominant_class, 
                confidence,
                processor_type='biopotentials'  # Keep as biopotentials to get AP/RP mapping
            )
            
            # Create ERP-specific result format with AP/RP context
            erp_result = {
                "model": "BioPotentials_ERP_AP_RP_Analysis",
                "ap_rp_classification": ap_rp_result.get("class", "AP" if dominant_class == "1" else "RP"),
                "ap_rp_full_name": ap_rp_result.get("full_name", "Action Potential" if dominant_class == "1" else "Resting Potential"),
                "confidence": confidence,
                "mapped_prediction": mapped_result['mapped_label'],
                "n_samples": metadata.get("processed_samples", 1),
                "original_biopotential_class": dominant_class,
                "biopotential_confidence": confidence,
                "label_mapping_applied": True,
                "class_distribution": class_dist,
                "erp_interpretation": {
                    "potential_type": "Action Potential" if dominant_class == "1" else "Resting Potential",
                    "neuronal_activity": "High activity (depolarization)" if dominant_class == "1" else "Low activity (resting state)",
                    "erp_relevance": "May indicate event-related neural response" if dominant_class == "1" else "Baseline neural activity"
                },
                "per_sample": [{
                    "p_resting": 1 - confidence if dominant_class == "1" else confidence,
                    "p_action": confidence if dominant_class == "1" else 1 - confidence,
                    "biopotential_class": dominant_class,
                    "predicted_label": 1 if dominant_class == "1" else 0,
                    "mapped_label": mapped_result['mapped_label'],
                    "ap_rp_class": "AP" if dominant_class == "1" else "RP"
                }]
            }
            
            # Update the original result with ERP-specific information
            result["metadata"].update({
                "erp_analysis": erp_result,
                "label_mapping_applied": True
            })
            
            # Update summary to include ERP context with AP/RP focus
            erp_summary = f"""ERP Analysis via BioPotentials AP/RP Classification:
- Biopotential Type: {mapped_result['mapped_label']} ({ap_rp_result.get('full_name', 'Unknown')})
- Confidence: {confidence:.3f}
- Original Classification: {dominant_class} ({'AP' if dominant_class == '1' else 'RP'})
- ERP Context: {erp_result['erp_interpretation']['erp_relevance']}
- Neural Activity Level: {erp_result['erp_interpretation']['neuronal_activity']}
- Label Mapping: {'Applied' if mapped_result.get('mapping_applied', True) else 'Not needed'}

{result['summary']}"""
            result["summary"] = erp_summary
            
            return result
    
    # Handle case where models failed to load or processing had errors
    error_result = result.copy()
    if "metadata" not in error_result:
        error_result["metadata"] = {}
        
    error_result["metadata"]["erp_analysis"] = {
        "model": "BioPotentials_ERP_AP_RP_Analysis",
        "ap_rp_classification": "error",
        "confidence": 0.0,
        "error": "BioPotentials AP/RP models could not be loaded or processing failed",
        "label_mapping_applied": False
    }
    
    error_result["summary"] = f"""ERP Analysis Error:
- BioPotentials AP/RP processing failed or models unavailable
- Cannot perform Action Potential vs Resting Potential classification
- Original error: {result.get('summary', 'Unknown error')}

{result.get('summary', '')}"""
    
    return error_result

biosignal_chat_page(
    biosignal_label="ERP",
    slug="erp",
    accepted_types=("edf", "fif", "set", "bdf", "csv", "txt"),
    analyzer=run_erp_wrapper,
    analyzer_label="Run ERP Analysis (AP/RP Classification)"
)
