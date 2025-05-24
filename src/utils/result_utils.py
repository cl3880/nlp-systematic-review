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
from src.models.feature_importance import get_feature_importance


logger = logging.getLogger(__name__)

def numpy_to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def ensure_dir(path):

    os.makedirs(path, exist_ok=True)
    return path

def create_model_directories(output_dir, model_type=None, normalization=None):
    model_dir = output_dir
    
    models_dir = ensure_dir(os.path.join(model_dir, "models"))
    metrics_dir = ensure_dir(os.path.join(model_dir, "metrics"))
    plots_dir = ensure_dir(os.path.join(model_dir, "plots"))
    ensure_dir(os.path.join(metrics_dir, "feature_importance"))
    ensure_dir(os.path.join(plots_dir, "feature_importance"))
    
    bal_dir = ensure_dir(os.path.join(model_dir, "balanced"))
    hr_dir = ensure_dir(os.path.join(model_dir, "highrecall"))
    
    for subdir in ["metrics", "plots"]:
        ensure_dir(os.path.join(bal_dir, subdir))
        ensure_dir(os.path.join(hr_dir, subdir))
    
    return {
        "model_dir": model_dir,
        "models_dir": models_dir,
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
        "balanced_dir": bal_dir,
        "highrecall_dir": hr_dir,
        "bal_metrics_dir": os.path.join(bal_dir, "metrics"),
        "bal_plots_dir": os.path.join(bal_dir, "plots"),
        "hr_metrics_dir": os.path.join(hr_dir, "metrics"),
        "hr_plots_dir": os.path.join(hr_dir, "plots")
    }

def save_model_results(bal_model, hr_model, bal_metrics, hr_metrics, bal_params, hr_params, 
                     threshold, output_dir, model_type, X_val, y_val, y_prob_bal, y_prob_hr, ngram_analysis=None):
    dirs = create_model_directories(output_dir, model_type)
    model_dir = dirs["model_dir"]
    models_dir = dirs["models_dir"]
    metrics_dir = dirs["metrics_dir"]
    plots_dir = dirs["plots_dir"]
    bal_dir = dirs["balanced_dir"]
    hr_dir = dirs["highrecall_dir"]
    
    cv_path_src = os.path.join(output_dir, "cv_results.csv")
    cv_path_dst = os.path.join(model_dir, "cv_results.csv")
    if os.path.exists(cv_path_src) and cv_path_src != cv_path_dst:
        shutil.copy(cv_path_src, cv_path_dst)
        logger.info(f"Copied CV results to {cv_path_dst}")
    
    model_path = os.path.join(models_dir, f"{model_type}_model.joblib")
    joblib.dump(bal_model, model_path)
    logger.info(f"Saved unified model to {model_path}")
    
    with open(os.path.join(metrics_dir, "model_params.json"), "w") as f:
        json.dump(bal_params, f, indent=2, default=numpy_to_python)
    
    with open(os.path.join(bal_dir, "metrics", "threshold.json"), "w") as f:
        json.dump({"threshold": 0.5}, f, indent=2)
    with open(os.path.join(bal_dir, "metrics", "metrics.json"), "w") as f:
        json.dump(bal_metrics, f, indent=2, default=numpy_to_python)
        
    with open(os.path.join(hr_dir, "metrics", "threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    with open(os.path.join(hr_dir, "metrics", "metrics.json"), "w") as f:
        json.dump(hr_metrics, f, indent=2, default=numpy_to_python)
    
    roc_path = os.path.join(plots_dir, "roc_curve.png")
    pr_path = os.path.join(plots_dir, "pr_curve.png")
    plot_roc_curve(y_val, y_prob_bal, roc_path)
    plot_pr_curve(y_val, y_prob_bal, None, pr_path)
    
    plot_confusion_matrix(y_val, bal_model.predict(X_val), 
                         os.path.join(bal_dir, "plots", "confusion_matrix.png"))
    plot_confusion_matrix(y_val, (y_prob_hr >= threshold).astype(int), 
                         os.path.join(hr_dir, "plots", "confusion_matrix.png"))
    
    try:        
        features = get_feature_importance(bal_model)
        if not features.empty:
            features.to_csv(
                os.path.join(metrics_dir, "feature_importance", "features.csv"),
                index=False
            )

            plt.figure(figsize=(12,10))
            plt.subplot(2,1,1)
            sns.barplot(
                x="coefficient", 
                y="feature",
                data=features[features["class"] == "Relevant"].head(20)
            )
            plt.title(f"{model_type.capitalize()} Model - Top 20 Relevant Features")
            plt.subplot(2,1,2)
            sns.barplot(
                x="coefficient", 
                y="feature",
                data=features[features["class"] == "Irrelevant"].head(20)
            )
            plt.title(f"{model_type.capitalize()} Model - Top 20 Irrelevant Features")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "feature_importance", "features.png"))
            plt.close()

    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    create_comparison_markdown(bal_metrics, hr_metrics, bal_params, hr_params,
                             os.path.join(model_dir, "SUMMARY.md"),
                             model_type, ngram_analysis)
    
    logger.info(f"Model results saved to {model_dir}")
    logger.info(f"Balanced configuration results saved to {bal_dir}")
    logger.info(f"High-recall configuration results saved to {hr_dir}")

def plot_pr_curve(y_true, y_prob, y_prob_hr=None, out_path=None):
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
            if param != 'optimal_threshold':
                f.write(f"- {param}: {value}\n")
        if 'optimal_threshold' in hr_params:
            f.write(f"- threshold: {hr_params['optimal_threshold']}\n")
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
            f.write("Based on cross-validation results (see cv_results.csv), we found:\n\n")
            
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
                
                f.write(f"\nThe {better_ngram} n-gram range outperforms {worse_ngram} by **{abs_diff:.1f} percentage points**.\n")

        f.write("\n## Methodological Notes\n\n")
        f.write("The high-recall configuration utilizes the same underlying model as the balanced configuration, ")
        f.write("with an adjusted decision threshold to prioritize recall (≥95%) at the expense of precision. ")
        f.write("This approach maintains identical feature coefficients while modifying only the decision boundary.\n")

    logger.info(f"Created comprehensive model summary at {output_path}")

def archive_directory(directory_path, archive_dir="results/archive"):
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