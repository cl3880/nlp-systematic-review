"""
Utilities for standardizing result paths and organization.

This module provides functions to create standardized output paths 
for model results, ensuring consistent organization across experiments.
"""
import os
from pathlib import Path
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import logging
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, roc_curve

logger = logging.getLogger(__name__)

def numpy_to_python(obj):
    """Convert NumPy types to Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def ensure_dir(path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to check/create
        
    Returns:
        str: The input path
    """
    os.makedirs(path, exist_ok=True)
    return path

def create_model_directories(output_dir, model_type=None, normalization=None):
    """
    Create the standardized directory structure for a model's results.
    
    Args:
        output_dir: Output directory path
        model_type: Type of model (logreg, svm, etc.)
        normalization: Type of text normalization (stemming, lemmatization)
        
    Returns:
        dict: Dictionary with paths to different directories
    """
    model_dir = output_dir
    bal_dir = ensure_dir(os.path.join(model_dir, "baseline"))
    hr_dir = ensure_dir(os.path.join(model_dir, "recall_95"))
    
    for subdir in ["models", "metrics", "plots"]:
        ensure_dir(os.path.join(bal_dir, subdir))
        ensure_dir(os.path.join(hr_dir, subdir))
    
    ensure_dir(os.path.join(bal_dir, "metrics", "feature_importance"))
    ensure_dir(os.path.join(bal_dir, "plots", "feature_importance"))
    ensure_dir(os.path.join(hr_dir, "metrics", "feature_importance"))
    ensure_dir(os.path.join(hr_dir, "plots", "feature_importance"))
    
    return {
        "model_dir": model_dir,
        "balanced_dir": bal_dir,
        "highrecall_dir": hr_dir,
        "bal_models_dir": os.path.join(bal_dir, "models"),
        "bal_metrics_dir": os.path.join(bal_dir, "metrics"),
        "bal_plots_dir": os.path.join(bal_dir, "plots"),
        "hr_models_dir": os.path.join(hr_dir, "models"),
        "hr_metrics_dir": os.path.join(hr_dir, "metrics"),
        "hr_plots_dir": os.path.join(hr_dir, "plots")
    }

def save_model_results(bal_model, hr_model, bal_metrics, hr_metrics, bal_params, hr_params, 
                     threshold, output_dir, model_type, X_val, y_val, y_prob_bal, y_prob_hr, ngram_analysis=None):
    """
    Save all model results to disk following the v2.0.0-f1-refactor directory structure.
    
    Args:
        bal_model: Trained balanced model
        hr_model: Trained high-recall model
        bal_metrics: Metrics for balanced model
        hr_metrics: Metrics for high-recall model
        bal_params: Parameters for balanced model
        hr_params: Parameters for high-recall model
        threshold: Threshold for high-recall model
        output_dir: Base output directory
        model_type: Type of model (logreg, svm, etc.)
        X_val: Validation features
        y_val: Validation labels
        y_prob_bal: Prediction probabilities from balanced model
        y_prob_hr: Prediction probabilities from high-recall model
        ngram_analysis: Analysis of n-gram performance (optional)
    """
    dirs = create_model_directories(output_dir, model_type)
    model_dir = dirs["model_dir"]
    bal_dir = dirs["balanced_dir"]
    hr_dir = dirs["highrecall_dir"]
    
    joblib.dump(bal_model, os.path.join(bal_dir, "models", "balanced.joblib"))
    with open(os.path.join(bal_dir, "metrics", "balanced_params.json"), "w") as f:
        json.dump(bal_params, f, indent=2, default=numpy_to_python)
    with open(os.path.join(bal_dir, "metrics", "balanced_metrics.json"), "w") as f:
        json.dump(bal_metrics, f, indent=2, default=numpy_to_python)
    
    joblib.dump(hr_model, os.path.join(hr_dir, "models", "high_recall.joblib"))
    with open(os.path.join(hr_dir, "metrics", "highrecall_params.json"), "w") as f:
        json.dump(hr_params, f, indent=2, default=numpy_to_python)
    with open(os.path.join(hr_dir, "metrics", "highrecall_metrics.json"), "w") as f:
        json.dump(hr_metrics, f, indent=2, default=numpy_to_python)
    with open(os.path.join(hr_dir, "metrics", "highrecall_threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f, indent=2, default=numpy_to_python)
    
    plot_pr_curve(y_val, y_prob_bal, None, os.path.join(bal_dir, "plots", "pr_curve.png"))
    plot_roc_curve(y_val, y_prob_bal, os.path.join(bal_dir, "plots", "roc_curve.png"))
    plot_confusion_matrix(y_val, bal_model.predict(X_val), os.path.join(bal_dir, "plots", "confusion_matrix.png"))
    
    plot_pr_curve(y_val, y_prob_hr, None, os.path.join(hr_dir, "plots", "pr_curve.png"))
    plot_roc_curve(y_val, y_prob_hr, os.path.join(hr_dir, "plots", "roc_curve.png"))
    plot_confusion_matrix(y_val, (y_prob_hr >= threshold).astype(int), os.path.join(hr_dir, "plots", "confusion_matrix.png"))
    
    try:
        from src.models.classifiers import get_feature_importance
        
        features_bal = get_feature_importance(bal_model)
        if not features_bal.empty:
            features_bal.to_csv(
                os.path.join(bal_dir, "metrics", "feature_importance", "balanced_features.csv"),
                index=False
            )

            plt.figure(figsize=(12,10))
            plt.subplot(2,1,1)
            sns.barplot(
                x="coefficient", 
                y="feature",
                data=features_bal[features_bal["class"] == "Relevant"].head(20)
            )
            plt.title("Balanced Model - Top 20 Relevant Features")
            plt.subplot(2,1,2)
            sns.barplot(
                x="coefficient", 
                y="feature",
                data=features_bal[features_bal["class"] == "Irrelevant"].head(20)
            )
            plt.title("Balanced Model - Top 20 Irrelevant Features")
            plt.tight_layout()
            plt.savefig(os.path.join(bal_dir, "plots", "feature_importance", "balanced_features.png"))
            plt.close()

        features_hr = get_feature_importance(hr_model)
        if not features_hr.empty:
            features_hr.to_csv(
                os.path.join(hr_dir, "metrics", "feature_importance", "highrecall_features.csv"),
                index=False
            )

            plt.figure(figsize=(12,10))
            plt.subplot(2,1,1)
            sns.barplot(
                x="coefficient", 
                y="feature",
                data=features_hr[features_hr["class"] == "Relevant"].head(20)
            )
            plt.title("High-Recall Model - Top 20 Relevant Features")
            plt.subplot(2,1,2)
            sns.barplot(
                x="coefficient", 
                y="feature",
                data=features_hr[features_hr["class"] == "Irrelevant"].head(20)
            )
            plt.title("High-Recall Model - Top 20 Irrelevant Features")
            plt.tight_layout()
            plt.savefig(os.path.join(hr_dir, "plots", "feature_importance", "highrecall_features.png"))
            plt.close()

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
    
    create_comparison_markdown(bal_metrics, hr_metrics, bal_params, hr_params,
                             os.path.join(model_dir, f"{model_type}_comparison.md"),
                             model_type, ngram_analysis)
    
    logger.info(f"Balanced model results saved to {bal_dir}")
    logger.info(f"High-recall model results saved to {hr_dir}")

def plot_pr_curve(y_true, y_prob, y_prob_hr=None, out_path=None):
    """Plot precision-recall curve for one or two models"""
    plt.figure(figsize=(10, 6))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    plt.plot(rec, prec, label=f'Model (AUC={pr_auc:.3f})')
    
    target_recall = 0.95
    idx = np.argmin(np.abs(rec - target_recall))
    target_precision = prec[idx]
    
    if y_prob_hr is not None:
        prec_hr, rec_hr, _ = precision_recall_curve(y_true, y_prob_hr)
        hr_auc = auc(rec_hr, prec_hr)
        plt.plot(rec_hr, prec_hr, label=f'High-Recall Model (AUC={hr_auc:.3f})')
    
    plt.axvline(target_recall, linestyle='--', color='r', label='95% Recall Target')
    plt.plot([target_recall], [target_precision], 'ro')  # Red dot at intersection
    plt.annotate(f'Precision @ 95% recall: {target_precision:.3f}', 
                xy=(target_recall, target_precision),
                xytext=(target_recall-0.15, target_precision-0.1),
                arrowprops=dict(arrowstyle='->'))
    
    no_skill = sum(y_true) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], 'k--', label=f'No Skill ({no_skill:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved PR curve to {out_path}")
    else:
        plt.show()
        
    return pr_auc

def plot_roc_curve(y_true, y_prob, out_path=None):
    """Plot ROC curve"""
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved ROC curve to {out_path}")
    else:
        plt.show()
        
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, out_path=None):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Irrelevant', 'Relevant'],
               yticklabels=['Irrelevant', 'Relevant'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {out_path}")
    else:
        plt.show()
        
    return cm

def create_comparison_markdown(balanced_metrics, hr_metrics, balanced_params, hr_params,
                             output_path, model_type="logreg", ngram_analysis=None):
    """Create a markdown document comparing the balanced and high-recall models"""
    model_names = {
        "logreg": "Logistic Regression",
        "svm": "Support Vector Machine",
        "cosine": "Cosine Similarity",
        "cnb": "Complement Naive Bayes"
    }
    model_name = model_names.get(model_type, model_type.upper())
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"# {model_name} Model Comparison\n\n")
        
        f.write("## Model Configurations\n\n")
        
        f.write("### Balanced Model (Optimized for F1)\n")
        for param, value in balanced_params.items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")
        
        f.write("### High-Recall Model (95% Target)\n")
        for param, value in hr_params.items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("| Metric | Balanced Model | High-Recall Model |\n")
        f.write("|--------|---------------|-------------------|\n")
        
        def fmt(metric, metrics_dict):
            value = metrics_dict.get(metric)
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                return f"{value:.4f}"
            return str(value)
        
        metrics_to_show = ['precision', 'recall', 'f1', 'f2', 'roc_auc', 'wss@95']
        for metric in metrics_to_show:
            f.write(f"| {metric.replace('@', ' at ')} | {fmt(metric, balanced_metrics)} | {fmt(metric, hr_metrics)} |\n")
        
        if ngram_analysis and len(ngram_analysis) > 1:
            f.write("\n## N-gram Range Analysis\n\n")
            f.write("Based on our cross-validation results (see cv_results.csv), we found:\n\n")
            
            sorted_ngrams = sorted(ngram_analysis.items(), key=lambda x: x[1], reverse=True)
            
            for ngram, f1 in sorted_ngrams:
                f.write(f"- **{ngram}** n-grams: average F1 = {f1:.4f}\n")
            
            if (1, 2) in ngram_analysis and (1, 3) in ngram_analysis:
                f1_12 = ngram_analysis[(1, 2)]
                f1_13 = ngram_analysis[(1, 3)]
                diff = (f1_13 - f1_12) * 100
                
                better_ngram = "(1,3)" if f1_13 > f1_12 else "(1,2)"
                worse_ngram = "(1,2)" if f1_13 > f1_12 else "(1,3)"
                abs_diff = abs(diff)
                
                f.write(f"\nThe {better_ngram} n-gram range outperforms {worse_ngram} by **{abs_diff:.1f} percentage points** ")
                
                if abs_diff >= 9.5:
                    f.write("- approximately 10 percentage points improvement, which aligns with findings from recent literature (LREC 2020).\n")
                elif abs_diff >= 5:
                    f.write("- a substantial improvement that supports using this configuration in the final model.\n")
                else:
                    f.write("- a modest improvement that should be considered alongside other hyperparameters.\n")
            
            f.write("\nFor detailed analysis, see the full grid search results in the cv_results.csv file.\n")
        
        f.write("\n## Analysis\n\n")
        
        f.write("### Methodology\n\n")
        f.write("Following Cohen 2006 and Norman 2018, we performed a single grid search with multi-metric scoring\n")
        f.write("and extracted two models:\n\n")
        f.write("1. **Balanced model**: Optimized for F1 (best balance of precision and recall)\n")
        f.write("2. **High-recall model**: Configuration with highest F1 score among those with recall ≥ 0.95\n\n")
        
        f.write("## Conclusion\n\n")
        
        bal_f1 = balanced_metrics.get('f1', 0)
        hr_f1 = hr_metrics.get('f1', 0)
        
        if hr_f1 > bal_f1 and hr_metrics.get('recall', 0) >= 0.95:
            f.write("The high-recall model achieves both higher recall (≥95%) and higher F1 score, ")
            f.write("making it the preferred choice for systematic review screening.\n")
        else:
            f.write("The balanced model is best for general classification tasks where overall performance ")
            f.write("is important, while the high-recall model is better suited for systematic review ")
            f.write("screening where achieving high recall (≥95%) is crucial to ensure comprehensive coverage.\n")

    logger.info(f"Created comparison markdown at {output_path}")

def archive_directory(directory_path, archive_dir="results/archive"):
    """
    Archive a directory by moving it to the archive folder.
    
    Args:
        directory_path: Path to directory to archive
        archive_dir: Destination directory for archived results
        
    Returns:
        bool: True if archiving was successful, False otherwise
    """
    os.makedirs(archive_dir, exist_ok=True)
    dir_name = os.path.basename(directory_path)
    dest_path = os.path.join(archive_dir, dir_name)
    
    if not os.path.exists(directory_path):
        logger.warning(f"{directory_path} does not exist, nothing to archive")
        return False
    
    if os.path.exists(dest_path):
        timestamp = Path(directory_path).stat().st_mtime
        from datetime import datetime
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        dest_path = f"{dest_path}_{date_str}"
    
    try:
        shutil.move(directory_path, dest_path)
        logger.info(f"Archived {directory_path} to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Error archiving {directory_path}: {e}")
        return False 