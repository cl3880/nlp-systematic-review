"""
Feature importance extraction with architecture-agnostic implementation.

This module provides unified feature importance extraction mechanisms
that adapt to different pipeline architectures through capability-based
inspection rather than rigid structural assumptions.
"""
import logging
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.svm import SVC

from src.models.introspection import detect_pipeline_architecture, extract_feature_extractors

logger = logging.getLogger(__name__)

def extract_feature_importance(model, n=30, class_names=["Irrelevant", "Relevant"]):
    """
    Polymorphic feature importance extraction with architecture detection.
    
    Parameters
    ----------
    model : Pipeline
        Scikit-learn pipeline with feature extraction and classifier components
    n : int
        Number of top features to extract per class
    class_names : list
        Class labels for positive and negative coefficients
        
    Returns
    -------
    pd.DataFrame
        Feature importance dataframe with source attribution when available
    """
    # Detect pipeline architecture
    architecture = detect_pipeline_architecture(model)
    
    # Select appropriate extraction method based on architecture
    if architecture['type'] == 'hierarchical':
        return extract_hierarchical_features(model, n, class_names)
    else:
        return extract_standard_features(model, n, class_names)

def extract_standard_features(model, n=30, class_names=["Irrelevant", "Relevant"]):
    """Extract feature importance from standard pipeline architecture."""
    try:
        # Validate pipeline structure
        if not hasattr(model, 'named_steps') or 'tfidf' not in model.named_steps or 'clf' not in model.named_steps:
            logger.error("Standard pipeline missing required components")
            return pd.DataFrame()
            
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']
        
        # Extract feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract coefficients based on classifier type
        coefficients = extract_coefficients(classifier, len(feature_names))
        
        if coefficients is None or len(coefficients) != len(feature_names):
            logger.error(f"Coefficient dimension mismatch: {len(coefficients) if coefficients is not None else 0} vs {len(feature_names)}")
            return pd.DataFrame()
            
        # Create dataframe with feature importance
        features_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coef': np.abs(coefficients)
        })
        
        # Add class labels based on coefficient sign
        features_df['class'] = features_df['coefficient'].apply(
            lambda x: class_names[1] if float(x) > 0 else class_names[0]
        )
        
        # Extract top features by importance
        return extract_top_features(features_df, n)
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}")
        return pd.DataFrame()

def extract_hierarchical_features(model, n=30, class_names=["Irrelevant", "Relevant"]):
    """Extract feature importance from hierarchical pipeline with source attribution."""
    try:
        # Validate hierarchical architecture
        if not hasattr(model, 'named_steps') or 'features' not in model.named_steps or 'clf' not in model.named_steps:
            logger.error("Hierarchical pipeline missing required components")
            return pd.DataFrame()
            
        feature_union = model.named_steps['features']
        classifier = model.named_steps['clf']
        
        # Extract features with source attribution
        feature_names, feature_sources = extract_hierarchical_feature_names(feature_union)
        
        if not feature_names:
            logger.error("No features extracted from hierarchical pipeline")
            return pd.DataFrame()
            
        # Extract coefficients from classifier
        coefficients = extract_coefficients(classifier, len(feature_names))
        
        if coefficients is None or len(coefficients) != len(feature_names):
            logger.error(f"Coefficient dimension mismatch: {len(coefficients) if coefficients is not None else 0} vs {len(feature_names)}")
            return pd.DataFrame()
            
        # Create dataframe with feature importance and source attribution
        features_df = pd.DataFrame({
            'feature': feature_names,
            'source': feature_sources,
            'coefficient': coefficients,
            'abs_coef': np.abs(coefficients)
        })
        
        # Add class labels based on coefficient sign
        features_df['class'] = features_df['coefficient'].apply(
            lambda x: class_names[1] if float(x) > 0 else class_names[0]
        )
        
        # Calculate source contribution metrics
        source_importance = features_df.groupby('source')['abs_coef'].sum().reset_index()
        total_importance = source_importance['abs_coef'].sum()
        source_importance['importance_percentage'] = 100 * source_importance['abs_coef'] / total_importance
        
        # Log source contribution
        logger.info("Feature importance by source:")
        for _, row in source_importance.iterrows():
            logger.info(f"  {row['source']}: {row['importance_percentage']:.2f}%")
        
        # Extract top features by importance
        return extract_top_features(features_df, n)
        
    except Exception as e:
        logger.error(f"Error extracting hierarchical feature importance: {e}")
        return pd.DataFrame()

def extract_coefficients(classifier, expected_features):
    """Extract coefficients from classifier with robust type handling."""
    try:
        if isinstance(classifier, SVC):
            if classifier.kernel != 'linear':
                logger.warning(f"Non-linear kernel ({classifier.kernel}) may produce unreliable feature importance")
                
            if hasattr(classifier, 'coef_'):
                coefficients = classifier.coef_.toarray()[0] if issparse(classifier.coef_) else classifier.coef_[0]
            elif hasattr(classifier, 'dual_coef_') and hasattr(classifier, 'support_vectors_'):
                dual_coef = classifier.dual_coef_
                support_vectors = classifier.support_vectors_
                coefficients = dual_coef.dot(support_vectors).toarray().flatten() if issparse(support_vectors) else np.dot(dual_coef, support_vectors).flatten()
            else:
                logger.error("Unable to extract coefficients from SVM model")
                return None
                
        elif hasattr(classifier, 'coef_'):
            # Handle LogisticRegression and similar models
            coefs = classifier.coef_
            coefficients = coefs[0] if coefs.ndim > 1 else coefs
            
        elif hasattr(classifier, 'feature_log_prob_'):
            # Handle Naive Bayes models
            if len(classifier.feature_log_prob_) == 2:
                coefficients = -(classifier.feature_log_prob_[0] - classifier.feature_log_prob_[1])
                coefficients = np.asarray(coefficients).flatten()
            else:
                logger.warning("Multi-class models not supported")
                return None
                
        elif hasattr(classifier, 'centroid_'):
            # Handle cosine similarity models
            centroid = classifier.centroid_
            if hasattr(centroid, 'toarray'):
                centroid = centroid.toarray()
            coefficients = np.asarray(centroid).flatten()
            
        else:
            logger.error(f"Unsupported classifier type: {type(classifier).__name__}")
            return None
            
        # Ensure consistent dimensionality
        coefficients = np.asarray(coefficients).flatten()
        if len(coefficients) != expected_features:
            logger.error(f"Coefficient dimension mismatch: {len(coefficients)} vs {expected_features}")
            return None
            
        return coefficients
        
    except Exception as e:
        logger.error(f"Error extracting coefficients: {e}")
        return None

def extract_hierarchical_feature_names(feature_union):
    """Extract feature names with source attribution from FeatureUnion."""
    feature_names = []
    feature_sources = []
    
    for name, transformer in feature_union.transformer_list:
        # Text features (TF-IDF)
        if name == 'text_features' and hasattr(transformer, 'named_steps') and 'tfidf' in transformer.named_steps:
            tfidf = transformer.named_steps['tfidf']
            text_features = tfidf.get_feature_names_out()
            feature_names.extend(text_features)
            feature_sources.extend(['text'] * len(text_features))
            
        # Criteria features
        elif name == 'criteria_features' and hasattr(transformer, 'named_steps') and 'rules' in transformer.named_steps:
            rules = transformer.named_steps['rules']
            if hasattr(rules, 'get_feature_names_out'):
                criteria_features = rules.get_feature_names_out()
                feature_names.extend(criteria_features)
                feature_sources.extend(['criteria'] * len(criteria_features))
                
        # MeSH features
        elif name == 'mesh_features' and hasattr(transformer, 'named_steps') and 'vectorizer' in transformer.named_steps:
            vectorizer = transformer.named_steps['vectorizer']
            if hasattr(vectorizer, 'get_feature_names_out'):
                mesh_features = vectorizer.get_feature_names_out()
                feature_names.extend(mesh_features)
                feature_sources.extend(['mesh'] * len(mesh_features))
                
    return feature_names, feature_sources

def extract_top_features(features_df, n):
    """Extract top n positive and negative features by coefficient value."""
    positive_features = features_df[features_df['coefficient'] > 0].sort_values('coefficient', ascending=False).head(n)
    negative_features = features_df[features_df['coefficient'] < 0].sort_values('coefficient', ascending=True).head(n)
    
    # Handle edge cases
    if len(positive_features) == 0 and len(negative_features) == 0:
        logger.error("No non-zero coefficients found")
        return pd.DataFrame()
    if len(positive_features) == 0:
        return negative_features
    if len(negative_features) == 0:
        return positive_features
        
    return pd.concat([positive_features, negative_features])

def get_feature_importance(model, n=30, class_names=["Irrelevant", "Relevant"]):
    """Backward compatibility wrapper for extract_feature_importance."""
    return extract_feature_importance(model, n, class_names)

def get_feature_importance_hierarchical(model, n=30, class_names=["Irrelevant", "Relevant"]):
    """Backward compatibility wrapper for extract_hierarchical_features."""
    return extract_hierarchical_features(model, n, class_names)