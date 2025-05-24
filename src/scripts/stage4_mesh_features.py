#!/usr/bin/env python3
"""
Custom SVM + MeSH experiments for systematic review screening.

This script:
1. Runs SVM models with fixed baseline parameters
2. Allows for using MeSH features without requiring criteria features
3. Tests different combinations of text normalization and balancing
4. Uses the exact same parameters for all runs to isolate each effect
5. Evaluates performance for both balanced and high-recall models

Usage:
    python custom_svm_mesh.py --use-mesh [--normalization NORM] [--balancing BAL]
"""
import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.utils.logging_utils import setup_per_model_logging
from src.scripts.stage1_baseline_grid_search import preprocess_corpus

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logger = logging.getLogger(__name__)

# Custom transformers for our pipeline
class TextCombiner(BaseEstimator, TransformerMixin):
    """Combines multiple text columns into a single text field."""
    
    def __init__(self, text_columns=('title', 'abstract')):
        self.text_columns = text_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Combine specified text columns with a space between them."""
        combined_text = []
        
        for _, row in X.iterrows():
            text_parts = []
            for col in self.text_columns:
                if col in row and pd.notna(row[col]) and row[col]:
                    text_parts.append(str(row[col]).strip())
            combined_text.append(' '.join(text_parts))
            
        return combined_text

class MeSHFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract MeSH terms as features."""
    
    def __init__(self, mesh_columns=('mesh_terms',)):
        self.mesh_columns = mesh_columns
        self.encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        self.mesh_terms = None
    
    def fit(self, X, y=None):
        """Extract all unique MeSH terms and fit the one-hot encoder."""
        all_mesh_terms = []
        
        for _, row in X.iterrows():
            for col in self.mesh_columns:
                if col in row and pd.notna(row[col]) and row[col]:
                    terms = str(row[col]).split(';')
                    all_mesh_terms.extend([term.strip() for term in terms if term.strip()])
        
        # Get unique terms
        self.mesh_terms = sorted(set(all_mesh_terms))
        
        # No need for encoder if we're handling it manually
        return self
    
    def transform(self, X):
        """Transform MeSH terms into one-hot encoded features."""
        if not self.mesh_terms:
            # No MeSH terms found during fit
            return np.zeros((len(X), 0))
        
        # Initialize a matrix of zeros
        mesh_matrix = np.zeros((len(X), len(self.mesh_terms)))
        
        for i, (_, row) in enumerate(X.iterrows()):
            row_terms = set()
            for col in self.mesh_columns:
                if col in row and pd.notna(row[col]) and row[col]:
                    terms = str(row[col]).split(';')
                    row_terms.update([term.strip() for term in terms if term.strip()])
            
            # Set corresponding indices to 1 for each MeSH term present
            for term in row_terms:
                if term in self.mesh_terms:
                    term_idx = self.mesh_terms.index(term)
                    mesh_matrix[i, term_idx] = 1
        
        return mesh_matrix

def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    # Calculate F2 score (weighs recall higher than precision)
    beta = 2
    if precision + recall > 0:
        f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    else:
        f2 = 0.0
    
    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Calculate Work Saved over Sampling at 95% recall (WSS@95)
    # For this we need to simulate threshold tuning to achieve 95% recall
    thresholds = np.sort(y_prob)
    best_wss = 0
    for threshold in thresholds:
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        tn = np.sum((y_true == 0) & (y_pred_at_threshold == 0))
        fp = np.sum((y_true == 0) & (y_pred_at_threshold == 1))
        fn = np.sum((y_true == 1) & (y_pred_at_threshold == 0))
        tp = np.sum((y_true == 1) & (y_pred_at_threshold == 1))
        
        if fn == 0:  # Perfect recall
            current_recall = 1.0
        else:
            current_recall = tp / (tp + fn)
            
        if current_recall >= 0.95:
            wss = (tn + fn) / (tn + fp + fn + tp) - (1 - current_recall)
            best_wss = max(best_wss, wss)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'roc_auc': roc_auc,
        'wss_at_95': best_wss,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def optimize_threshold_for_recall(y_true, y_prob, target_recall=0.95):
    """Find optimal threshold to achieve target recall."""
    thresholds = np.sort(y_prob)
    best_threshold = 0
    best_diff = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        diff = abs(recall - target_recall)
        if diff < best_diff and recall >= target_recall:
            best_diff = diff
            best_threshold = threshold
            
    return best_threshold

def create_output_directory(model_name):
    """Create an output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Try common results directories
    for results_dir in ["results/v3", "results/v2", "results"]:
        if os.path.exists(results_dir):
            output_dir = os.path.join(results_dir, f"{model_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
    
    # Fallback to a new results directory
    output_dir = os.path.join("results", f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_confusion_matrix(y_true, y_pred, output_path, title='Confusion Matrix'):
    """Save confusion matrix visualization."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Irrelevant', 'Relevant'],
                yticklabels=['Irrelevant', 'Relevant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_pr_curve(y_true, y_prob, output_path, title='Precision-Recall Curve'):
    """Save precision-recall curve visualization."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 for each point
    f1_scores = []
    for p, r in zip(precision, recall):
        if p + r > 0:
            f1 = 2 * (p * r) / (p + r)
        else:
            f1 = 0
        f1_scores.append(f1)
    
    # Find max F1 point
    max_f1_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_idx]
    max_precision = precision[max_f1_idx]
    max_recall = recall[max_f1_idx]
    
    # Find high recall points (≥0.95)
    high_recall_idx = [i for i, r in enumerate(recall) if r >= 0.95]
    if high_recall_idx:
        # Find best F1 among high recall points
        hr_f1_scores = [f1_scores[i] for i in high_recall_idx]
        max_hr_idx = high_recall_idx[np.argmax(hr_f1_scores)]
        hr_precision = precision[max_hr_idx]
        hr_recall = recall[max_hr_idx]
        hr_f1 = f1_scores[max_hr_idx]
    else:
        hr_precision, hr_recall, hr_f1 = 0, 0, 0
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot curve
    plt.plot(recall, precision, 'b-', linewidth=2)
    
    # Mark max F1 point
    plt.plot(max_recall, max_precision, 'ro', markersize=8, 
             label=f'Best F1: {max_f1:.3f} (P={max_precision:.3f}, R={max_recall:.3f})')
    
    # Mark high recall point if exists
    if high_recall_idx:
        plt.plot(hr_recall, hr_precision, 'go', markersize=8,
                 label=f'Best F1 at R≥0.95: {hr_f1:.3f} (P={hr_precision:.3f}, R={hr_recall:.3f})')
    
    # Add F1 curves
    f1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for f1_value in f1_values:
        # Calculate the precision for each recall value to maintain this F1
        r = np.linspace(0.01, 0.99, 100)
        p = (f1_value * r) / (2 * r - f1_value)
        valid_idx = p <= 1
        plt.plot(r[valid_idx], p[valid_idx], 'k--', alpha=0.3)
        
        # Add F1 label at rightmost valid point
        idx = np.where(valid_idx)[0]
        if len(idx) > 0:
            rightmost_idx = idx[-1]
            plt.annotate(
                f'F1={f1_value}',
                xy=(r[rightmost_idx], p[rightmost_idx]),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8
            )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_custom_pipeline(
    model_type="svm",
    text_columns=("title", "abstract"),
    use_mesh=False,
    balancing=None,
):
    """Create a custom pipeline for SVM with or without MeSH features."""
    # Create text feature extractor
    text_features = Pipeline([
        ('combiner', TextCombiner(text_columns=text_columns)),
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.85,
            stop_words='english',
            sublinear_tf=True
        ))
    ])
    
    # Create feature union
    feature_extractors = [('text_features', text_features)]
    
    # Add MeSH features if requested
    if use_mesh:
        feature_extractors.append(('mesh_features', MeSHFeatureExtractor()))
    
    features = FeatureUnion(feature_extractors)
    
    # Create the classifier
    if model_type == 'svm':
        clf = SVC(
            C=1.0,
            class_weight='balanced',
            kernel='linear',
            probability=True,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create the pipeline
    steps = [
        ('features', features),
        ('clf', clf)
    ]
    
    # Apply SMOTE if requested
    if balancing == 'smote':
        steps = steps[:-1] + [('sampler', SMOTE(random_state=42))] + [steps[-1]]
        return ImbPipeline(steps)
    else:
        return Pipeline(steps)

def run_custom_svm_mesh_experiment(
    data_path,
    output_dir=None,
    normalization=None,
    balancing=None,
    use_mesh=False,
    target_recall=0.95,
    debug=False
):
    # Setup model name based on configuration
    model_name = "svm_fixed_params"
    if normalization:
        model_name = f"{normalization}_{model_name}"
    if balancing and balancing != "none":
        model_name = f"{model_name}_{balancing}"
    if use_mesh:
        model_name = f"{model_name}_mesh"
    
    # Set up logging
    log_level = logging.DEBUG if debug else logging.INFO
    logger = setup_per_model_logging(model_name, level=log_level)
    
    if output_dir is None:
        output_dir = create_output_directory(model_name)
    
    # Configure file handler for logging
    file_handler = logging.FileHandler(os.path.join(output_dir, f"{model_name}.log"))
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting SVM+MeSH experiment: {model_name}")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Load and split data
    df = load_data(data_path)
    train, val, test = make_splits(df, test_size=0.1, val_size=0.1, stratify=True, seed=42)
    
    # Apply normalization if requested
    if normalization:
        logger.info(f"Applying {normalization} normalization...")
        train = preprocess_corpus(train, technique=normalization)
        val = preprocess_corpus(val, technique=normalization)
        text_columns = ['normalized_text']
    else:
        text_columns = ['title', 'abstract']
    
    # Prepare data
    X_train = train.drop('relevant', axis=1)
    y_train = train['relevant']
    X_val = val.drop('relevant', axis=1)
    y_val = val['relevant']
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    # Create custom pipeline
    logger.info(f"Creating custom pipeline with: use_mesh={use_mesh}, balancing={balancing}")
    pipeline = create_custom_pipeline(
        model_type="svm",
        text_columns=text_columns,
        use_mesh=use_mesh,
        balancing=balancing
    )
    
    # Train model
    logger.info("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    y_pred = pipeline.predict(X_val)
    balanced_metrics = compute_metrics(y_val, y_pred, y_prob)
    
    # Find optimal threshold for high recall
    logger.info(f"Finding optimal threshold for {target_recall*100}% recall...")
    threshold = optimize_threshold_for_recall(y_val, y_prob, target_recall)
    logger.info(f"Optimal threshold: {threshold:.4f}")
    
    # Evaluate high-recall model
    y_pred_hr = (y_prob >= threshold).astype(int)
    high_recall_metrics = compute_metrics(y_val, y_pred_hr, y_prob)
    
    # Log metrics
    logger.info("Balanced model metrics:")
    for metric, value in balanced_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("High-recall model metrics:")
    for metric, value in high_recall_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        "model_name": model_name,
        "balanced": balanced_metrics,
        "high_recall": high_recall_metrics,
        "threshold": threshold,
        "params": {
            "features__text_features__tfidf__max_features": 10000,
            "features__text_features__tfidf__ngram_range": (1, 3),
            "features__text_features__tfidf__min_df": 5,
            "features__text_features__tfidf__max_df": 0.85,
            "clf__C": 1.0,
            "clf__class_weight": "balanced",
            "clf__kernel": "linear"
        }
    }
    
    # Convert numpy types to Python types for JSON serialization
    for section in ["balanced", "high_recall"]:
        for k, v in results[section].items():
            if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                results[section][k] = int(v)
            elif isinstance(v, (np.float64, np.float32, np.float16)):
                results[section][k] = float(v)
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save parameters to JSON
    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, 'w') as f:
        json.dump({"balanced": results["params"], "threshold": threshold}, f, indent=2)
    logger.info(f"Parameters saved to {params_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': y_val,
        'predicted_label': y_pred,
        'probability': y_prob,
        'high_recall_pred': y_pred_hr
    })
    predictions_path = os.path.join(output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    
    # Save visualizations
    # Confusion matrices
    save_confusion_matrix(
        y_val, y_pred,
        os.path.join(output_dir, "confusion_matrix_balanced.png"),
        title=f'Confusion Matrix - Balanced ({model_name})'
    )
    
    save_confusion_matrix(
        y_val, y_pred_hr,
        os.path.join(output_dir, "confusion_matrix_high_recall.png"),
        title=f'Confusion Matrix - High Recall ({model_name})'
    )
    
    # Precision-recall curve
    save_pr_curve(
        y_val, y_prob,
        os.path.join(output_dir, "pr_curve.png"),
        title=f'Precision-Recall Curve ({model_name})'
    )
    
    logger.info(f"Experiment complete. Results saved to {output_dir}")
    return pipeline, results

def main():
    parser = argparse.ArgumentParser(
        description="Run custom SVM+MeSH experiments for systematic review screening."
    )
    parser.add_argument(
        "--data", 
        default=os.path.join(PATHS["data_processed"], "data_final_processed.csv"),
        help="Path to the dataset"
    )
    parser.add_argument(
        "--output", 
        default=None,
        help="Directory to save results (default: auto-generated)"
    )
    parser.add_argument(
        "--normalization", 
        default=None, 
        choices=[None, "stemming", "lemmatization"],
        help="Text normalization technique (None, 'stemming', or 'lemmatization')"
    )
    parser.add_argument(
        "--balancing", 
        default=None,
        choices=[None, "smote"],
        help="Class balancing technique (None, 'smote')"
    )
    parser.add_argument(
        "--use-mesh",
        action="store_true",
        help="Include MeSH terms as features (if available)"
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target recall for high-recall model (default: 0.95)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug-level logging"
    )
    
    args = parser.parse_args()
    
    run_custom_svm_mesh_experiment(
        args.data,
        args.output,
        normalization=args.normalization,
        balancing=args.balancing,
        use_mesh=args.use_mesh,
        target_recall=args.target_recall,
        debug=args.debug
    )

if __name__ == "__main__":
    main()