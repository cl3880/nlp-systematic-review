#!/usr/bin/env python3
"""
Isolation experiments for systematic review screening.

This script:
1. Runs SVM models with fixed baseline parameters
2. Tests different combinations of text normalization and balancing
3. Uses the exact same parameters for all runs to isolate each effect
4. Evaluates performance for both balanced and high-recall models
5. Saves results for comprehensive comparison

Usage:
    python isolation_experiments.py --normalization [none|stemming|lemmatization] --balancing [none|smote]
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
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.models.model_factory import create_model
from src.utils.logging_utils import setup_per_model_logging
from src.scripts.stage1_baseline_grid_search import preprocess_corpus
# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logger = logging.getLogger(__name__)

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

def run_isolation_experiment(
    data_path,
    output_dir=None,
    normalization=None,
    balancing=None,
    target_recall=0.95,
    debug=False
):
    # Setup model name based on configuration
    model_name = "svm_fixed_params"
    if normalization:
        model_name = f"{normalization}_{model_name}"
    if balancing and balancing != "none":
        model_name = f"{model_name}_{balancing}"
    
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
    
    logger.info(f"Starting isolation experiment: {model_name}")
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
    
    # Fixed baseline parameters (from baseline SVM model)
    baseline_params = {
        "clf__C": 1,
        "clf__class_weight": "balanced",
        "clf__kernel": "linear",
        "tfidf__max_df": 0.85,
        "tfidf__max_features": 10000,
        "tfidf__min_df": 5,
        "tfidf__ngram_range": [1, 3]
    }
    
    logger.info("Using fixed baseline SVM parameters:")
    for param, value in baseline_params.items():
        logger.info(f"  {param}: {value}")
    
    # Create model with fixed parameters
    model = create_model(
        model_type="svm",
        normalization=None,  # We handle normalization separately
        balancing=balancing,
        max_features=baseline_params["max_features"],
        ngram_range=baseline_params["ngram_range"],
        min_df=baseline_params["min_df"],
        max_df=baseline_params["max_df"],
        text_columns=text_columns,
        C=baseline_params["C"],
        class_weight=baseline_params["class_weight"],
        kernel=baseline_params["kernel"]
    )
    
    # Train model
    logger.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
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
        "params": baseline_params
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
        json.dump({"balanced": baseline_params, "threshold": threshold}, f, indent=2)
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
    
    # Extract feature importance if available
    try:
        from src.scripts.feature_importance import extract_feature_importance
        feature_importance = extract_feature_importance(model, n=30)
        
        if feature_importance is not None and not feature_importance.empty:
            fi_path = os.path.join(output_dir, "feature_importance.csv")
            feature_importance.to_csv(fi_path, index=False)
            logger.info(f"Feature importance saved to {fi_path}")
            
            # Also create plot
            plt.figure(figsize=(12, 8))
            
            # Plot relevant features
            relevant_features = feature_importance[feature_importance['class'] == 'Relevant']
            irrelevant_features = feature_importance[feature_importance['class'] == 'Irrelevant']
            
            if not relevant_features.empty:
                plt.barh(
                    range(len(relevant_features)),
                    relevant_features['coefficient'],
                    color='green',
                    alpha=0.6,
                    label='Relevant'
                )
                plt.yticks(
                    range(len(relevant_features)),
                    relevant_features['feature'],
                    fontsize=10
                )
            
            # Plot irrelevant features
            if not irrelevant_features.empty:
                plt.barh(
                    range(len(relevant_features), len(relevant_features) + len(irrelevant_features)),
                    irrelevant_features['coefficient'],
                    color='red',
                    alpha=0.6,
                    label='Irrelevant'
                )
                plt.yticks(
                    list(range(len(relevant_features))) + 
                    list(range(len(relevant_features), len(relevant_features) + len(irrelevant_features))),
                    list(relevant_features['feature']) + 
                    list(irrelevant_features['feature']),
                    fontsize=10
                )
            
            plt.axhline(y=len(relevant_features) - 0.5, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel('Coefficient Value')
            plt.legend()
            plt.title(f'Top Features by Importance ({model_name})')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "feature_importance.png"))
            plt.close()
    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}")
    
    logger.info(f"Experiment complete. Results saved to {output_dir}")
    return model, results

def main():
    parser = argparse.ArgumentParser(
        description="Run isolation experiments for systematic review screening."
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
    
    run_isolation_experiment(
        args.data,
        args.output,
        normalization=args.normalization,
        balancing=args.balancing,
        target_recall=args.target_recall,
        debug=args.debug
    )

if __name__ == "__main__":
    main()