#!/usr/bin/env python3
"""
Feature importance extraction utility for systematic review classification models.

This script extracts and visualizes feature importance from trained classification models
without requiring grid search re-execution. It supports multiple classifier architectures
including LogisticRegression, SVM, CNB, and Cosine Similarity.

Usage:
    python extract_feature_importance.py [--model MODEL_PATH] [--output OUTPUT_DIR] [--debug]
"""
import argparse
import os
import sys
import joblib
import logging
import traceback
from sklearn.pipeline import FeatureUnion

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.introspection import detect_pipeline_architecture
from src.models.feature_importance import extract_feature_importance
from src.visualization.feature_importance import (
    visualize_feature_importance,
    visualize_source_stratified_features
)
from src.utils.logging_utils import setup_logging

def main():
    """Main execution function with robust error handling."""
    parser = argparse.ArgumentParser(description="Extract feature importance from trained model")
    parser.add_argument("--model", type=str, 
                        default="results_v2/baseline/svm/models/svm_model.joblib",
                        help="Path to the trained model")
    parser.add_argument("--output", type=str, 
                        default=None,
                        help="Output directory for feature importance artifacts")
    parser.add_argument("--debug", action="store_true",
                        help="Enable comprehensive model architecture analysis")
    parser.add_argument("--stratified", action="store_true",
                        help="Generate additional source-stratified visualizations")
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = setup_logging(name="feature_extraction")
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug-level logging enabled")
    
    # Determine model type from filename
    model_type = os.path.basename(args.model).split("_")[0]
    
    # Set up output directories
    if args.output is None:
        base_dir = os.path.dirname(os.path.dirname(args.model))
        args.output = os.path.join(base_dir, "metrics", "feature_importance")
        plots_dir = os.path.join(base_dir, "plots", "feature_importance")
    else:
        args.output = os.path.abspath(args.output)
        plots_dir = os.path.join(args.output, "plots")
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Load the model
        logger.info(f"Loading model from {args.model}")
        model = joblib.load(args.model)
        
        # Analyze model architecture
        architecture = detect_pipeline_architecture(model)
        if args.debug:
            logger.debug(f"Detected architecture: {architecture['type']}")
            logger.debug(f"Pipeline components: {dict(architecture['components'])}")
            logger.debug(f"Pipeline capabilities: {architecture['capabilities']}")
        
        # Extract feature importance
        logger.info(f"Extracting feature importance for {model_type} model")
        features = extract_feature_importance(model)
        
        if features.empty:
            logger.error("Failed to extract feature importance")
            return 1
            
        # Save feature importance data
        csv_path = os.path.join(args.output, "features.csv")
        features.to_csv(csv_path, index=False)
        logger.info(f"Feature importance saved to {csv_path}")
        
        try:
            is_hierarchical = architecture['type'] == 'hierarchical'
            plot_path = os.path.join(plots_dir, 'features_hierarchical.png' if is_hierarchical else 'features.png')
            
            logger.info(f"Generating feature importance visualization at {plot_path}")
            result = visualize_feature_importance(features, plot_path, model_type, hierarchical=is_hierarchical)
            
            if not result:
                logger.warning("Visualization generation failed - check logs for details")
        except Exception as e:
            logger.error(f"Unexpected error during visualization: {e}")
            logger.debug(traceback.format_exc())
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during feature importance extraction: {e}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())