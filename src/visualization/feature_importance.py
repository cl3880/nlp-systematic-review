"""
Feature importance visualization with comprehensive source attribution.

This module provides specialized visualization functions for feature importance
analysis with adaptive layouts based on data characteristics.
"""
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

logger = logging.getLogger(__name__)

def visualize_feature_importance(features_df, output_path, model_type="model", hierarchical=False):
    """
    Unified feature importance visualization with architecture-adaptive layout and robust error handling.
    """
    try:
        # Validate input integrity
        if features_df.empty:
            logger.error("Empty feature dataframe - visualization skipped")
            return None
            
        # Critical path verification
        required_columns = ['feature', 'coefficient', 'class']
        for col in required_columns:
            if col not in features_df.columns:
                logger.error(f"Required column '{col}' missing - visualization failed")
                return None
                
        # Ensure target directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Import visualization dependencies with verification
        try:
            import matplotlib
            matplotlib.use('Agg')  # Force non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            logger.error(f"Visualization dependency error: {e}")
            return None
            
        # Select appropriate visualization based on data characteristics
        has_source_info = 'source' in features_df.columns
        
        if has_source_info and hierarchical:
            result = visualize_hierarchical_layout(features_df, output_path, model_type)
        else:
            result = visualize_standard_layout(features_df, output_path, model_type)
            
        # Verify output file creation
        if result and os.path.exists(output_path):
            logger.info(f"Feature importance visualization successfully saved to {output_path}")
            return output_path
        else:
            logger.error(f"Visualization file not created at {output_path}")
            return None
            
    except Exception as e:
        logger.error(f"Visualization generation failed with exception: {e}")
        logger.debug(traceback.format_exc())
        return None

def visualize_standard_layout(features_df, output_path, model_type="model"):
    """Generate standard two-panel visualization for feature importance."""
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate visualization
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        sns.barplot(
            x="coefficient", 
            y="feature",
            data=features_df[features_df["class"] == "Relevant"].head(20)
        )
        plt.title(f"{model_type.capitalize()} Model - Top 20 Relevant Features")
        
        plt.subplot(2, 1, 2)
        sns.barplot(
            x="coefficient", 
            y="feature",
            data=features_df[features_df["class"] == "Irrelevant"].head(20)
        )
        plt.title(f"{model_type.capitalize()} Model - Top 20 Irrelevant Features")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Feature importance visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating standard visualization: {e}")
        return None

def visualize_hierarchical_layout(features_df, output_path, model_type="model"):
    """Generate four-panel visualization with source attribution for hierarchical models."""
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate visualization
        plt.figure(figsize=(15, 12))
        
        # Primary visualization: Top features by class
        plt.subplot(2, 2, 1)
        sns.barplot(
            x="coefficient", 
            y="feature",
            data=features_df[features_df["class"] == "Relevant"].head(15)
        )
        plt.title(f"Top 15 Relevant Features")
        
        plt.subplot(2, 2, 2)
        sns.barplot(
            x="coefficient", 
            y="feature",
            data=features_df[features_df["class"] == "Irrelevant"].head(15)
        )
        plt.title(f"Top 15 Irrelevant Features")
        
        # Secondary visualization: Feature importance by source
        source_importance = features_df.groupby('source')['abs_coef'].sum().reset_index()
        total_importance = source_importance['abs_coef'].sum()
        source_importance['importance_percentage'] = 100 * source_importance['abs_coef'] / total_importance
        
        plt.subplot(2, 2, 3)
        sns.barplot(
            x="source", 
            y="importance_percentage",
            data=source_importance
        )
        plt.title("Feature Importance by Source (%)")
        plt.ylabel("Contribution to Model (%)")
        
        # Tertiary visualization: Top features by source
        plt.subplot(2, 2, 4)
        source_counts = features_df['source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
        sns.barplot(
            x="source", 
            y="count",
            data=source_counts
        )
        plt.title("Feature Count by Source")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Hierarchical feature importance visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating hierarchical visualization: {e}")
        return None

def visualize_source_stratified_features(features_df, output_dir, model_type="model"):
    """Generate source-stratified visualizations for detailed analysis."""
    try:
        if 'source' not in features_df.columns:
            logger.error("Source column missing - stratified visualization skipped")
            return None
            
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate separate visualization for each source
        sources = features_df['source'].unique()
        output_paths = []
        
        for source in sources:
            source_features = features_df[features_df['source'] == source]
            
            if len(source_features) < 5:
                logger.info(f"Skipping visualization for source '{source}' with only {len(source_features)} features")
                continue
                
            output_path = os.path.join(output_dir, f"{source}_features.png")
            
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 1, 1)
            relevant = source_features[source_features["class"] == "Relevant"].head(10)
            if not relevant.empty:
                sns.barplot(
                    x="coefficient", 
                    y="feature",
                    data=relevant
                )
            plt.title(f"Top Relevant Features from '{source}'")
            
            plt.subplot(2, 1, 2)
            irrelevant = source_features[source_features["class"] == "Irrelevant"].head(10)
            if not irrelevant.empty:
                sns.barplot(
                    x="coefficient", 
                    y="feature",
                    data=irrelevant
                )
            plt.title(f"Top Irrelevant Features from '{source}'")
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Source-specific visualization saved to {output_path}")
            output_paths.append(output_path)
            
        return output_paths
        
    except Exception as e:
        logger.error(f"Error generating source-stratified visualizations: {e}")
        return None