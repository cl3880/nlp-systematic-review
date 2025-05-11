# utils/evaluate.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)
import logging
import json
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)

def calculate_wss_at_recall(y_true, y_scores, target_recall=0.95):
    """
    Calculate Work Saved over Sampling at a target recall level.
    
    Args:
        y_true: Array-like of true binary labels
        y_scores: Array-like of predicted scores or probabilities
        target_recall: Target recall threshold (default: 0.95)
        
    Returns:
        float: Work saved over sampling metric value
    """
    # Ensure arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Sort by prediction score (highest first)
    sorted_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[sorted_idx]
    
    # Find minimum number of documents needed to achieve target recall
    n_pos = y_true.sum()
    target_pos = int(np.ceil(n_pos * target_recall))
    
    # Count until we find enough positive examples
    found_pos = 0
    for i, val in enumerate(y_sorted):
        found_pos += val
        if found_pos >= target_pos:
            n_reviewed = i + 1
            break
    else:
        # If we never reached target recall, we reviewed all documents
        n_reviewed = len(y_true)
    
    # Calculate work saved
    n_total = len(y_true)
    wss = 1.0 - (n_reviewed / n_total) - (1.0 - target_recall)
    
    return wss

def find_threshold_for_recall(y_true, y_scores, target_recall=0.95):
    """
    Find the threshold that gives exactly the target recall.
    
    Args:
        y_true: Array-like of true binary labels
        y_scores: Array-like of predicted scores or probabilities
        target_recall: Target recall threshold (default: 0.95)
        
    Returns:
        tuple: (threshold, achieved_recall)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find the threshold that gives us the closest recall to target
    best_idx = np.argmin(np.abs(recall - target_recall))
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0
    achieved_recall = recall[best_idx]
    
    return threshold, achieved_recall

def evaluate(y_true, y_pred, y_prob=None, base_dir=None, result_prefix="", target_recall=0.95):
    """
    Evaluate the model and save results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        base_dir: Base directory to save results (default: None)
        result_prefix: Prefix for result filenames (default: "")
        target_recall: Target recall for threshold calculation (default: 0.95)
        
    Returns:
        dict: Dictionary of metrics
    """
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        precision_score, recall_score, f1_score, roc_auc_score,
        precision_recall_curve, average_precision_score, fbeta_score
    )
    
    # Compute basic metrics
    metrics = {}
    
    # Handle cases where predictions are all negative
    if sum(y_pred) == 0:
        logger.warning("All predictions are negative!")
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
        metrics['f2'] = 0.0
    else:
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        # F2 score weights recall higher than precision
        metrics['f2'] = fbeta_score(y_true, y_pred, beta=2)
    
    # Compute AUC if probabilities are provided
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        
        # Compute threshold for target recall
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        idx = np.where(rec >= target_recall)[0]
        if len(idx) > 0:
            threshold = thresholds[idx[-1]]
            metrics['threshold_recall'] = threshold
        else:
            threshold = 0.0
            metrics['threshold_recall'] = threshold
        
        # Calculate work saved over sampling
        # This is the percentage of papers that don't need to be screened
        # when using the model to achieve 95% recall
        if rec[-1] >= target_recall:  # Check if target recall is achievable
            # Find the index where recall first exceeds the target
            recall_idx = np.where(rec >= target_recall)[0][0]
            # Calculate corresponding precision
            precision_at_recall = prec[recall_idx]
            # Calculate work saved
            n_relevant = sum(y_true)
            n_screened = int(n_relevant / precision_at_recall)
            wss = 1.0 - (n_screened / len(y_true))
            metrics['wss@95'] = max(0, wss)
        else:
            metrics['wss@95'] = 0.0
    
    # Log metrics
    logger.info(f"Evaluating model '{base_dir}' on {result_prefix} data (default threshold)")
    logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
    logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
    logger.info(f"F1 Score: {metrics.get('f1', 0):.4f}")
    logger.info(f"F2 Score: {metrics.get('f2', 0):.4f}")
    
    if 'roc_auc' in metrics:
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Avg. Precision: {metrics['avg_precision']:.4f}")
        logger.info(f"WSS@95: {metrics['wss@95']:.4f}")
        logger.info(f"Default threshold: 0.5")
        logger.info(f"Threshold for 95% recall would be: {metrics['threshold_recall']:.4f}")
    
    # If base_dir is provided, save detailed metrics
    if base_dir:
        # Create metrics directory first
        os.makedirs(base_dir, exist_ok=True)
        
        # Calculate classification report
        report = classification_report(y_true, y_pred, target_names=["Irrelevant", "Relevant"], output_dict=True)
        
        # Save classification report
        with open(os.path.join(base_dir, f"{result_prefix}_classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        
        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_dict = {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1])
        }
        with open(os.path.join(base_dir, f"{result_prefix}_confusion_matrix.json"), "w") as f:
            json.dump(cm_dict, f, indent=2)
        
        # Save all metrics
        all_metrics = {**metrics, **cm_dict}
        with open(os.path.join(base_dir, f"{result_prefix}_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2)
    
    return metrics

def plot_roc_curve(y_true, y_scores, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_scores, output_path):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Calculate no-skill line (random classifier)
    no_skill = sum(y_true) / len(y_true)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.plot([0, 1], [no_skill, no_skill], 'k--', label=f'No Skill ({no_skill:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.close()
    
    return pr_auc

def compare_models(model_results_list, filename="model_comparison", base_dir="results"):
    """
    Create a bar chart comparing multiple models across key metrics.
    
    Args:
        model_results_list: List of (name, metrics_dict) tuples for each model
        filename: Filename for the output chart (default: "model_comparison")
        base_dir: Base directory for results (default: 'results')
        
    Returns:
        None
    """
    # Set up paths for saving results
    plots_dir = get_plot_path("comparison", base_dir) 
    os.makedirs(plots_dir, exist_ok=True)
    
    # Select important metrics to compare
    metrics = ['roc_auc', 'wss@95', 'f1', 'f2', 'precision', 'recall']
    labels = ['AUC', 'WSS@95', 'F1', 'F2', 'Precision', 'Recall']
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(model_results_list)
    
    # Plot each model's metrics
    for i, (name, results) in enumerate(model_results_list):
        values = [results.get(m, 0) for m in metrics]
        plt.bar(x + (i - len(model_results_list)/2 + 0.5) * width, values, width, label=name)
    
    # Add labels and legend
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, labels)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(f"{plots_dir}/{filename}.png")
    plt.close()
    
    # Also save as CSV for easier analysis
    comparison_data = {
        'Metric': labels
    }
    
    for name, results in model_results_list:
        comparison_data[name] = [results.get(m, 0) for m in metrics]
    
    pd.DataFrame(comparison_data).to_csv(f"{plots_dir}/{filename}.csv", index=False)
    
    logger.info(f"Model comparison saved to {plots_dir}/{filename}.png and {filename}.csv")

def find_optimal_hyperparams_for_recall(model, X, y, param_grid, target_recall=0.95, cv=5):
    """
    Find optimal hyperparameters that achieve at least target recall.
    
    Args:
        model: Base model/pipeline to optimize
        X: Training data
        y: Target labels
        param_grid: Parameter grid for GridSearchCV
        target_recall: Target recall threshold (default: 0.95)
        cv: Number of cross-validation folds
        
    Returns:
        tuple: (best_model, best_params, best_score, achieved_recall)
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, recall_score
    
    # Create a custom scorer that returns -inf if recall < target
    def recall_threshold_scorer(y_true, y_pred, **kwargs):
        recall = recall_score(y_true, y_pred)
        return recall if recall >= target_recall else float('-inf')
    
    # Run grid search with custom scorer
    grid = GridSearchCV(
        model,
        param_grid,
        scoring=make_scorer(recall_threshold_scorer),
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    
    # Get best model and its recall
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    
    # Calculate achieved recall
    y_pred = best_model.predict(X)
    achieved_recall = recall_score(y, y_pred)
    
    return best_model, best_params, best_score, achieved_recall

def save_cross_validation_results(cv_results, model_name, base_dir="results"):
    """
    Save cross-validation results to a standardized location.
    
    Args:
        cv_results: Dictionary of cross-validation results
        model_name: Name of the model (logreg, svm, cosine)
        base_dir: Base directory for results (default: 'results')
        
    Returns:
        str: Path to saved CV results file
    """
    cv_dir = get_metrics_path("cross_validation", base_dir)
    os.makedirs(cv_dir, exist_ok=True)
    
    output_path = f"{cv_dir}/{model_name}_cv.json"
    with open(output_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    
    logger.info(f"Cross-validation results saved to {output_path}")
    return output_path

def save_feature_importance(features_df, model_name, base_dir="results"):
    """
    Save feature importance results to a standardized location.
    
    Args:
        features_df: DataFrame containing feature names and their importance scores
        model_name: Name of the model (logreg, svm)
        base_dir: Base directory for results (default: 'results')
        
    Returns:
        str: Path to saved feature importance file
    """
    fi_dir = get_metrics_path("feature_importance", base_dir)
    plots_dir = get_plot_path("feature_importance", base_dir)
    
    os.makedirs(fi_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save as JSON
    output_path = f"{fi_dir}/{model_name}_features.json"
    features_dict = features_df.to_dict(orient="records")
    with open(output_path, "w") as f:
        json.dump(features_dict, f, indent=2)
    
    logger.info(f"Feature importance saved to {output_path}")
    return output_path

# Add new plot function for threshold analysis
def plot_threshold_analysis(y_true, y_scores, output_path):
    """
    Plot precision, recall, F1, and F2 scores as functions of the classification threshold.
    This helps understand how threshold selection affects different metrics.
    """
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    f2_scores = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        if np.sum(y_pred) > 0:  # Check if there are any positive predictions
            precisions.append(precision_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred))
            f2_scores.append(fbeta_score(y_true, y_pred, beta=2))
        else:
            # If no positive predictions, metrics are 0
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            f2_scores.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(thresholds, f2_scores, label='F2 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold Analysis')
    plt.legend(loc='best')
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% Recall Target')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_multiple_roc_curves(model_data_list, output_path=None, title="ROC Curve Comparison", base_dir="results"):
    """
    Plot ROC curves for multiple models on the same graph.
    
    Args:
        model_data_list: List of tuples (model_name, y_true, y_scores)
        output_path: Path to save the combined ROC curve plot (optional)
        title: Title for the plot
        base_dir: Base directory for results
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, y_true, y_scores in model_data_list:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if output_path is None:
        # Use standardized path
        comparison_dir = get_plot_path("roc_curves", base_dir)
        os.makedirs(comparison_dir, exist_ok=True)
        output_path = f"{comparison_dir}/roc_curve_comparison.png"
        
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_multiple_pr_curves(model_data_list, output_path=None, title="Precision-Recall Curve Comparison", base_dir="results"):
    """
    Plot precision-recall curves for multiple models on the same graph.
    
    Args:
        model_data_list: List of tuples (model_name, y_true, y_scores)
        output_path: Path to save the combined PR curve plot (optional)
        title: Title for the plot
        base_dir: Base directory for results
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate no-skill line from the first dataset (assuming all use the same test set)
    if model_data_list:
        _, y_true_first, _ = model_data_list[0]
        no_skill = sum(y_true_first) / len(y_true_first)
        plt.plot([0, 1], [no_skill, no_skill], 'k--', label=f'No Skill ({no_skill:.3f})')
    
    for model_name, y_true, y_scores in model_data_list:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if output_path is None:
        # Use standardized path
        comparison_dir = get_plot_path("pr_curves", base_dir)
        os.makedirs(comparison_dir, exist_ok=True)
        output_path = f"{comparison_dir}/pr_curve_comparison.png"
        
    plt.savefig(output_path)
    plt.close()
    
    return output_path