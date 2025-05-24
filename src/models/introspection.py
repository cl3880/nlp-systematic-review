"""
Systematic pipeline architecture introspection utilities.

This module provides robust inspection mechanisms for scikit-learn pipelines,
enabling architecture-agnostic feature extraction through capability detection
rather than explicit type checking.
"""
import logging
from collections import defaultdict
from sklearn.pipeline import Pipeline, FeatureUnion

logger = logging.getLogger(__name__)

def detect_pipeline_architecture(model):
    """
    Systematically analyze pipeline architecture through capability-based inspection.
    
    Parameters
    ----------
    model : Pipeline
        Scikit-learn pipeline to analyze
        
    Returns
    -------
    dict
        Architectural specification with the following keys:
        - type: Architecture type ('standard', 'hierarchical', 'unknown')
        - components: Dict mapping component types to their count
        - capabilities: List of detected functional capabilities
        - structure: Hierarchical description of pipeline components
    """
    if not hasattr(model, 'named_steps'):
        logger.warning("Input is not a scikit-learn Pipeline")
        return {
            'type': 'unknown',
            'components': {},
            'capabilities': [],
            'structure': None
        }
    
    # Initialize architecture specification
    architecture = {
        'type': 'standard',
        'components': defaultdict(int),
        'capabilities': [],
        'structure': {'steps': []}
    }
    
    # Analyze pipeline components and structure
    for step_name, step in model.named_steps.items():
        step_info = {
            'name': step_name,
            'type': type(step).__name__
        }
        
        architecture['components'][step_info['type']] += 1
        architecture['structure']['steps'].append(step_info)
        
        # Detect FeatureUnion for hierarchical pipelines
        if isinstance(step, FeatureUnion):
            architecture['type'] = 'hierarchical'
            architecture['capabilities'].append('feature_union')
            
            # Extract transformer details
            step_info['transformers'] = []
            for trans_name, transformer in step.transformer_list:
                trans_info = {
                    'name': trans_name,
                    'type': type(transformer).__name__
                }
                
                # Recursively analyze nested pipelines
                if isinstance(transformer, Pipeline):
                    trans_info['pipeline'] = detect_pipeline_architecture(transformer)
                
                step_info['transformers'].append(trans_info)
        
        # Detect classifier capabilities
        if step_name == 'clf':
            if hasattr(step, 'coef_'):
                architecture['capabilities'].append('linear_coefficients')
            if hasattr(step, 'feature_log_prob_'):
                architecture['capabilities'].append('feature_probabilities')
            if hasattr(step, 'feature_importances_'):
                architecture['capabilities'].append('feature_importances')
    
    logger.debug(f"Detected pipeline architecture: {architecture['type']}")
    return architecture

def validate_architecture_compatibility(architecture, required_capabilities):
    """
    Validates whether the pipeline architecture supports required capabilities.
    
    Parameters
    ----------
    architecture : dict
        Architecture specification from detect_pipeline_architecture
    required_capabilities : list
        List of capabilities required for feature extraction
        
    Returns
    -------
    bool
        True if architecture supports all required capabilities
    """
    return all(cap in architecture['capabilities'] for cap in required_capabilities)

def extract_feature_extractors(architecture):
    """
    Locate feature extraction components within pipeline architecture.
    
    Parameters
    ----------
    architecture : dict
        Architecture specification from detect_pipeline_architecture
        
    Returns
    -------
    list
        List of (name, location) tuples for feature extraction components
    """
    extractors = []
    
    if architecture['type'] == 'standard':
        # Look for vectorizer in standard pipeline
        for step in architecture['structure']['steps']:
            if 'tfidf' in step['name'] or 'vectorizer' in step['name'].lower():
                extractors.append((step['name'], 'standard'))
    
    elif architecture['type'] == 'hierarchical':
        # Find FeatureUnion step
        for step in architecture['structure']['steps']:
            if 'transformers' in step:
                for transformer in step['transformers']:
                    if 'pipeline' in transformer:
                        # Search for vectorizers in nested pipelines
                        nested_arch = transformer['pipeline']
                        for nested_step in nested_arch['structure']['steps']:
                            if 'tfidf' in nested_step['name'] or 'vectorizer' in nested_step['name'].lower():
                                extractors.append((
                                    f"{step['name']}.{transformer['name']}.{nested_step['name']}", 
                                    transformer['name']
                                ))
    
    return extractors