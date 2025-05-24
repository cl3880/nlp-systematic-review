#!/usr/bin/env python3
"""
Comprehensive Grid Search Experiment for Systematic Review Classification

This script runs a complete grid search that:
1. Defines a parameter grid including TF-IDF and classifier parameters
2. Uses multi-metric scoring (F1, recall, precision, ROC AUC)
3. Extracts two models from a single grid search:
   - Balanced model: Optimized for F1 score
   - High-recall model: Highest F1 among configurations with recall ≥ 0.95
4. Evaluates both models on validation data and saves results

Supports multiple model types:
- Logistic Regression
- SVM
- Cosine Similarity
- Complement Naive Bayes

Usage:
    python baseline_grid_search.py --model logreg --output results/grid_search
"""
import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import clone
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)
from sklearn.naive_bayes import ComplementNB

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.models.classifiers import (
    make_tfidf_logreg_pipeline, make_tfidf_svm_pipeline,
    make_tfidf_cosine_pipeline, TextCombiner
)
from src.utils.evaluate import evaluate, find_threshold_for_recall
from src.models.model_factory import create_model
from src.models.param_grids import get_param_grid
import matplotlib.pyplot as plt

# Configure logging
os.makedirs(os.path.join(PATHS["logs_dir"]), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PATHS["logs_dir"], "grid_search.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Helper to convert NumPy types for JSON
def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_and_split_data(data_path):
    """Load and split data for model training and evaluation"""
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    
    # Split data using the make_splits function
    train, val, test = make_splits(df, test_size=0.1, val_size=0.1, stratify=True, seed=42)
    
    X_train = train.drop('relevant', axis=1)
    y_train = train['relevant']
    X_val = val.drop('relevant', axis=1)
    y_val = val['relevant']
    X_test = test.drop('relevant', axis=1)
    y_test = test['relevant']
    
    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

],
        "tfidf__ngram_range": [(1, 2), (1, 3)],
    }
    
    # Model-specific parameters
    if model_type == "logreg":
        model_params = {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": ["balanced"],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear"],
        }
    elif model_type == "svm":
        model_params = {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": ["balanced"],
            "clf__kernel": ["linear"],
        }
    elif model_type == "cosine":
        # Cosine similarity doesn't have traditional hyperparameters to tune
        model_params = {}
    elif model_type == "cnb":
        model_params = {
            "clf__alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
            "clf__norm": [True, False],
        }
    else:
        model_params = {}
    
    # Combine parameters
    param_grid = {**tfidf_params, **model_params}
    return param_grid

def create_pipeline(model_type):
    """Create a pipeline based on model type"""
    if model_type == "logreg":
        return make_tfidf_logreg_pipeline()
    elif model_type == "svm":
        return make_tfidf_svm_pipeline()
    elif model_type == "cosine":
        return make_tfidf_cosine_pipeline()
    elif model_type == "cnb":
        from sklearn.pipeline import Pipeline
        from sklearn.naive_bayes import ComplementNB
        return Pipeline([
            ("combiner", TextCombiner()),
            ("tfidf", make_tfidf_logreg_pipeline().named_steps["tfidf"]),
            ("clf", ComplementNB()),
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def run_grid_search(X_train, y_train, model_type="logreg", cv=5):
    """Run grid search to find balanced and high-recall models"""
    # 1. Create base pipeline
    pipeline = create_model(model_type=model_type)
    logger.info(f"Using {model_type.upper()} model")
    
    # 2. Get parameter grid
    param_grid = get_param_grid(model_type)
    
    # For cosine similarity, we use a simplified approach without grid search
    if model_type == "cosine":
        logger.info("Cosine similarity model doesn't use standard grid search")
        pipeline.fit(X_train, y_train)
        # Return placeholder values for consistency
        return None, pipeline, {}, pipeline, {}
    
    # 3. Define scoring metrics
    scoring = {
        'f1': 'f1',             # For balanced model
        'recall': 'recall',     # For extracting high-recall model
        'precision': 'precision',  # Useful for examining trade-offs
        'roc_auc': 'roc_auc'    # For overall ranking quality
    }
    
    # 4. Run grid search
    logger.info("Starting grid search with multi-metric scoring...")
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='f1',  # Optimize for F1 (balanced model)
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    
    logger.info(f"Grid search complete. Best F1 score: {grid.best_score_:.4f}")
    logger.info(f"Best parameters: {grid.best_params_}")
    
    # 5. Extract balanced model
    balanced_model = grid.best_estimator_
    balanced_params = grid.best_params_
    
    # 6. Extract high-recall model
    high_recall_model, high_recall_params = extract_high_recall_model(grid, X_train, y_train)
    
    # 6a. Export full cv_results_ for deeper analysis
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_path = os.path.join("results", "baseline", model_type, "metrics", "cv_results.csv")
    os.makedirs(os.path.dirname(cv_path), exist_ok=True)
    cv_df.to_csv(cv_path, index=False)
    logger.info(f"Exported CV results to {cv_path}")
    
    # 7. Analyze n-gram performance if grid search was performed
    try:
        ngram_analysis = {}
        if 'param_tfidf__ngram_range' in cv_df.columns:
            # Group by n-gram range and calculate average F1 scores
            ngram_groups = cv_df.groupby('param_tfidf__ngram_range')['mean_test_f1'].mean().to_dict()
            # Convert tuple string representations to actual tuples
            ngram_analysis = {eval(k) if isinstance(k, str) else k: v for k, v in ngram_groups.items()}
            
            # Log the analysis
            logger.info("N-gram range analysis:")
            for ngram, f1 in sorted(ngram_analysis.items()):
                logger.info(f"  {ngram}: average F1 = {f1:.4f}")
            
            # Compare (1,2) vs (1,3) if both are present
            if (1, 2) in ngram_analysis and (1, 3) in ngram_analysis:
                f1_12 = ngram_analysis[(1, 2)]
                f1_13 = ngram_analysis[(1, 3)]
                diff = (f1_13 - f1_12) * 100  # Convert to percentage points
                logger.info(f"  (1,3) vs (1,2): {diff:.1f} percentage points difference")
    except Exception as e:
        logger.warning(f"Could not perform n-gram analysis: {e}")
        ngram_analysis = {}
    
    return grid, balanced_model, balanced_params, high_recall_model, high_recall_params, ngram_analysis

def extract_high_recall_model(grid, X_train, y_train):
    """Extract high-recall model from grid search results"""
    # Get results
    res = grid.cv_results_
    recalls = res['mean_test_recall']
    f1s = res['mean_test_f1']
    params = res['params']
    
    # Find indices where recall >= 0.95
    idxs = np.where(recalls >= 0.95)[0]
    
    if len(idxs) == 0:
        logger.warning("No config ≥95% recall; choosing closest.")
        idx = np.argmin(np.abs(recalls - 0.95))
    else:
        # Get the configuration with highest F1 among those with recall >= 0.95
        idx = idxs[np.argmax(f1s[idxs])]
    
    best_params = params[idx]
    
    # Create and fit high-recall model
    hr_model = clone(grid.best_estimator_)
    hr_model.set_params(**{k: v for k, v in best_params.items() if k in hr_model.get_params()})
    hr_model.fit(X_train, y_train)
    
    logger.info(f"High-recall model params: {best_params}")
    return hr_model, best_params

def compute_metrics(y_true, y_pred, y_prob):
    """Compute standard evaluation metrics"""
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    wss95 = (tn + fn) / len(y_true) - 0.05
    
    # Calculate F2 score (weights recall higher)
    beta = 2
    if prec + rec > 0:
        f2 = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
    else:
        f2 = 0
    
    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'f2': f2,
        'roc_auc': auc,
        'wss@95': wss95,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def evaluate_models(bal_model, hr_model, X_val, y_val):
    """Evaluate both balanced and high-recall models"""
    # Evaluate balanced model
    try:
        y_prob_bal = bal_model.predict_proba(X_val)[:, 1]
    except (AttributeError, IndexError):
        # Handle cases where predict_proba might not be available
        y_prob_bal = np.zeros(len(y_val))
        logger.warning("Could not get prediction probabilities for balanced model")
    
    y_pred_bal = bal_model.predict(X_val)
    bal_metrics = compute_metrics(y_val, y_pred_bal, y_prob_bal)
    
    # Evaluate high-recall model
    try:
        y_prob_hr = hr_model.predict_proba(X_val)[:, 1]
        
        # Find threshold for 95% recall
        prec, rec, ths = precision_recall_curve(y_val, y_prob_hr)
        idxs = np.where(rec >= 0.95)[0]
        if len(idxs) > 0:
            threshold = ths[idxs[-1]]
        else:
            threshold = 0.5
            logger.warning("Could not find threshold for 95% recall, using default 0.5")
        
        # Apply custom threshold
        y_pred_hr = (y_prob_hr >= threshold).astype(int)
        
    except (AttributeError, IndexError):
        # Handle cases where predict_proba might not be available
        y_prob_hr = np.zeros(len(y_val))
        threshold = 0.5
        y_pred_hr = hr_model.predict(X_val)
        logger.warning("Could not get prediction probabilities for high-recall model")
    
    hr_metrics = compute_metrics(y_val, y_pred_hr, y_prob_hr)
    
    # Log results
    logger.info("Balanced model metrics:")
    for k, v in bal_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("High-recall model metrics:")
    for k, v in hr_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
        
    return bal_metrics, hr_metrics, threshold, y_prob_bal, y_prob_hr

def plot_pr_curve(y_true, y_prob_bal, y_prob_hr, out_path):
    """
    Plot precision-recall curve for both models.
    
    Args:
        y_true: True labels
        y_prob_bal: Probabilities from balanced model
        y_prob_hr: Probabilities from high-recall model
        out_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate curves
    prec_b, rec_b, _ = precision_recall_curve(y_true, y_prob_bal)
    prec_h, rec_h, _ = precision_recall_curve(y_true, y_prob_hr)
    
    # Calculate AUC
    bal_auc = auc(rec_b, prec_b)
    hr_auc = auc(rec_h, prec_h)
    
    # Plot curves
    plt.plot(rec_b, prec_b, label=f'Balanced Model (AUC={bal_auc:.3f})')
    plt.plot(rec_h, prec_h, label=f'High-Recall Model (AUC={hr_auc:.3f})')
    
    # Add target recall line
    plt.axvline(0.95, linestyle='--', color='r', label='95% Recall Target')
    
    # Add styling
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    
    logger.info(f"Saved PR curve to {out_path}")

def save_results(bal_model, hr_model, bal_metrics, hr_metrics, bal_params, hr_params, 
               threshold, output_dir, model_type, X_val, y_val, y_prob_bal, y_prob_hr, ngram_analysis=None):
    """
    Save all results to disk following the specific directory structure:
    
    results/
    ├── baseline/{model_type}/
    │   ├── models/
    │   ├── metrics/
    │   └── plots/
    └── recall_95/{model_type}/
        ├── models/
        ├── metrics/
        └── plots/
    """
    import joblib
    
    # Create directory structure
    bal_dir = os.path.join(output_dir, "baseline", model_type)
    hr_dir = os.path.join(output_dir, "recall_95", model_type)
    
    # Create directories for balanced model
    os.makedirs(os.path.join(bal_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(bal_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(bal_dir, "metrics", "feature_importance"), exist_ok=True)
    os.makedirs(os.path.join(bal_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(bal_dir, "plots", "feature_importance"), exist_ok=True)
    
    # Create directories for high-recall model
    os.makedirs(os.path.join(hr_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(hr_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(hr_dir, "metrics", "feature_importance"), exist_ok=True)
    os.makedirs(os.path.join(hr_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(hr_dir, "plots", "feature_importance"), exist_ok=True)
    
    # Save balanced model
    joblib.dump(bal_model, os.path.join(bal_dir, "models", "balanced.joblib"))
    with open(os.path.join(bal_dir, "metrics", "balanced_params.json"), "w") as f:
        json.dump(bal_params, f, indent=2, default=numpy_to_python)
    with open(os.path.join(bal_dir, "metrics", "balanced_metrics.json"), "w") as f:
        json.dump(bal_metrics, f, indent=2, default=numpy_to_python)
    
    # Save high-recall model
    joblib.dump(hr_model, os.path.join(hr_dir, "models", "high_recall.joblib"))
    with open(os.path.join(hr_dir, "metrics", "highrecall_params.json"), "w") as f:
        json.dump(hr_params, f, indent=2, default=numpy_to_python)
    with open(os.path.join(hr_dir, "metrics", "highrecall_metrics.json"), "w") as f:
        json.dump(hr_metrics, f, indent=2, default=numpy_to_python)
    with open(os.path.join(hr_dir, "metrics", "highrecall_threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f, indent=2, default=numpy_to_python)
    
    # Generate plots
    # Balanced model plots
    plot_pr_curve(y_val, y_prob_bal, None, os.path.join(bal_dir, "plots", "pr_curve.png"))
    plot_roc_curve(y_val, y_prob_bal, os.path.join(bal_dir, "plots", "roc_curve.png"))
    plot_confusion_matrix(y_val, bal_model.predict(X_val), os.path.join(bal_dir, "plots", "confusion_matrix.png"))
    
    # High-recall model plots
    plot_pr_curve(y_val, y_prob_hr, None, os.path.join(hr_dir, "plots", "pr_curve.png"))
    plot_roc_curve(y_val, y_prob_hr, os.path.join(hr_dir, "plots", "roc_curve.png"))
    plot_confusion_matrix(y_val, (y_prob_hr >= threshold).astype(int), os.path.join(hr_dir, "plots", "confusion_matrix.png"))
    
    # Try to extract feature importance when available
    try:
        from src.models.classifiers import get_feature_importance
        
        # Balanced model feature importance
        features_bal = get_feature_importance(bal_model)
        if not features_bal.empty:
            features_bal.to_csv(os.path.join(bal_dir, "metrics", "feature_importance", "balanced_features.csv"), index=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 10))
            
            # Plot relevant features
            plt.subplot(2, 1, 1)
            relevant_features = features_bal[features_bal['class'] == 'Relevant'].head(20)
            sns.barplot(x='coefficient', y='feature', data=relevant_features)
            plt.title(f'Top 20 Features Indicating Relevant')
            plt.tight_layout()
            
            # Plot irrelevant features
            plt.subplot(2, 1, 2)
            irrelevant_features = features_bal[features_bal['class'] == 'Irrelevant'].head(20)
            sns.barplot(x='coefficient', y='feature', data=irrelevant_features)
            plt.title(f'Top 20 Features Indicating Irrelevant')
            plt.tight_layout()
            
            plt.savefig(os.path.join(bal_dir, "plots", "feature_importance", "balanced_features.png"))
            plt.close()
        
        # High-recall model feature importance
        features_hr = get_feature_importance(hr_model)
        if not features_hr.empty:
            features_hr.to_csv(os.path.join(hr_dir, "metrics", "feature_importance", "highrecall_features.csv"), index=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 10))
            
            # Plot relevant features
            plt.subplot(2, 1, 1)
            relevant_features = features_hr[features_hr['class'] == 'Relevant'].head(20)
            sns.barplot(x='coefficient', y='feature', data=relevant_features)
            plt.title(f'Top 20 Features Indicating Relevant')
            plt.tight_layout()
            
            # Plot irrelevant features
            plt.subplot(2, 1, 2)
            irrelevant_features = features_hr[features_hr['class'] == 'Irrelevant'].head(20)
            sns.barplot(x='coefficient', y='feature', data=irrelevant_features)
            plt.title(f'Top 20 Features Indicating Irrelevant')
            plt.tight_layout()
            
            plt.savefig(os.path.join(hr_dir, "plots", "feature_importance", "highrecall_features.png"))
            plt.close()
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
    
    # Create comparison markdown report
    create_comparison_markdown(bal_metrics, hr_metrics, bal_params, hr_params,
                             os.path.join(output_dir, f"{model_type}_comparison.md"),
                             model_type, ngram_analysis)
    
    logger.info(f"Balanced model results saved to {bal_dir}")
    logger.info(f"High-recall model results saved to {hr_dir}")os.path.join(bal_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(bal_dir, "plots", "feature_importance"), exist_ok=True)
    
    # Create directories for high-recall model
    os.makedirs(os.path.join(hr_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(hr_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(hr_dir, "metrics", "feature_importance"), exist_ok=True)
    os.makedirs(os.path.join(hr_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(hr_dir, "plots", "feature_importance"), exist_ok=True)
    
    # Save balanced model
    joblib.dump(bal_model, os.path.join(bal_dir, "models", "balanced.joblib"))
    with open(os.path.join(bal_dir, "metrics", "balanced_params.json"), "w") as f:
        json.dump(bal_params, f, indent=2, default=numpy_to_python)
    with open(os.path.join(bal_dir, "metrics", "balanced_metrics.json"), "w") as f:
        json.dump(bal_metrics, f, indent=2, default=numpy_to_python)
    
    # Save high-recall model
    joblib.dump(hr_model, os.path.join(hr_dir, "models", "high_recall.joblib"))
    with open(os.path.join(hr_dir, "metrics", "highrecall_params.json"), "w") as f:
        json.dump(hr_params, f, indent=2, default=numpy_to_python)
    with open(os.path.join(hr_dir, "metrics", "highrecall_metrics.json"), "w") as f:
        json.dump(hr_metrics, f, indent=2, default=numpy_to_python)
    with open(os.path.join(hr_dir, "metrics", "highrecall_threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f, indent=2, default=numpy_to_python)
    
    # Generate plots
    # Balanced model plots
    plot_pr_curve(y_val, y_prob_bal, None, os.path.join(bal_dir, "plots", "pr_curve.png"))
    plot_roc_curve(y_val, y_prob_bal, os.path.join(bal_dir, "plots", "roc_curve.png"))
    plot_confusion_matrix(y_val, bal_model.predict(X_val), os.path.join(bal_dir, "plots", "confusion_matrix.png"))
    
    # High-recall model plots
    plot_pr_curve(y_val, y_prob_hr, None, os.path.join(hr_dir, "plots", "pr_curve.png"))
    plot_roc_curve(y_val, y_prob_hr, os.path.join(hr_dir, "plots", "roc_curve.png"))
    plot_confusion_matrix(y_val, (y_prob_hr >= threshold).astype(int), os.path.join(hr_dir, "plots", "confusion_matrix.png"))
    
    # Try to extract feature importance when available
    try:
        from src.models.classifiers import get_feature_importance, plot_feature_importance
        
        # Balanced model feature importance
        features_bal = get_feature_importance(bal_model)
        if not features_bal.empty:
            features_bal.to_csv(os.path.join(bal_dir, "metrics", "feature_importance", "balanced_features.csv"), index=False)
            plot_feature_importance(bal_model, os.path.join(bal_dir, "plots", "feature_importance", "balanced_features.png"))
        
        # High-recall model feature importance
        features_hr = get_feature_importance(hr_model)
        if not features_hr.empty:
            features_hr.to_csv(os.path.join(hr_dir, "metrics", "feature_importance", "highrecall_features.csv"), index=False)
            plot_feature_importance(hr_model, os.path.join(hr_dir, "plots", "feature_importance", "highrecall_features.png"))
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
    
    # Create comparison markdown report
    create_comparison_markdown(bal_metrics, hr_metrics, bal_params, hr_params,
                             os.path.join(output_dir, f"{model_type}_comparison.md"),
                             model_type)
    
    logger.info(f"Balanced model results saved to {bal_dir}")
    logger.info(f"High-recall model results saved to {hr_dir}")

def create_comparison_markdown(balanced_metrics, hr_metrics, balanced_params, hr_params,
                             output_path, model_type="logreg", ngram_analysis=None):
    """Create a markdown document comparing the balanced and high-recall models"""
    # Model name mapping
    model_names = {
        "logreg": "Logistic Regression",
        "svm": "Support Vector Machine",
        "cosine": "Cosine Similarity",
        "cnb": "Complement Naive Bayes"
    }
    model_name = model_names.get(model_type, model_type.upper())
    
    # Create the directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"# {model_name} Model Comparison\n\n")
        
        # Model configurations
        f.write("## Model Configurations\n\n")
        
        f.write("### Balanced Model (Optimized for F1)\n")
        for param, value in balanced_params.items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")
        
        f.write("### High-Recall Model (95% Target)\n")
        for param, value in hr_params.items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")
        
        # Performance comparison
        f.write("## Performance Comparison\n\n")
        f.write("| Metric | Balanced Model | High-Recall Model |\n")
        f.write("|--------|---------------|-------------------|\n")
        
        # Function to safely format metrics
        def fmt(metric, metrics_dict):
            value = metrics_dict.get(metric)
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                return f"{value:.4f}"
            return str(value)
        
        # Add all metrics to the table
        metrics_to_show = ['precision', 'recall', 'f1', 'f2', 'roc_auc', 'wss@95']
        for metric in metrics_to_show:
            f.write(f"| {metric.replace('@', ' at ')} | {fmt(metric, balanced_metrics)} | {fmt(metric, hr_metrics)} |\n")
        
        # N-gram analysis section
        if ngram_analysis and len(ngram_analysis) > 1:
            f.write("\n## N-gram Range Analysis\n\n")
            f.write("Based on our cross-validation results (see cv_results.csv), we found:\n\n")
            
            # Sort the n-gram ranges by their F1 scores (descending)
            sorted_ngrams = sorted(ngram_analysis.items(), key=lambda x: x[1], reverse=True)
            
            for ngram, f1 in sorted_ngrams:
                f.write(f"- **{ngram}** n-grams: average F1 = {f1:.4f}\n")
            
            # If we have both (1,2) and (1,3), compute the difference
            if (1, 2) in ngram_analysis and (1, 3) in ngram_analysis:
                f1_12 = ngram_analysis[(1, 2)]
                f1_13 = ngram_analysis[(1, 3)]
                diff = (f1_13 - f1_12) * 100  # Convert to percentage points
                
                better_ngram = "(1,3)" if f1_13 > f1_12 else "(1,2)"
                worse_ngram = "(1,2)" if f1_13 > f1_12 else "(1,3)"
                abs_diff = abs(diff)
                
                f.write(f"\nThe {better_ngram} n-gram range outperforms {worse_ngram} by **{abs_diff:.1f} percentage points** ")
                
                if abs_diff >= 9.5:  # Close enough to 10 pp
                    f.write("- approximately 10 percentage points improvement, which aligns with findings from recent literature (LREC 2020).\n")
                elif abs_diff >= 5:
                    f.write("- a substantial improvement that supports using this configuration in the final model.\n")
                else:
                    f.write("- a modest improvement that should be considered alongside other hyperparameters.\n")
            
            f.write("\nFor detailed analysis, see the full grid search results in the cv_results.csv file.\n")
        
        # Analysis
        f.write("\n## Analysis\n\n")
        
        f.write("### Methodology\n\n")
        f.write("Following Cohen 2006 and Norman 2018, we performed a single grid search with multi-metric scoring\n")
        f.write("and extracted two models:\n\n")
        f.write("1. **Balanced model**: Optimized for F1 (best balance of precision and recall)\n")
        f.write("2. **High-recall model**: Configuration with highest F1 score among those with recall ≥ 0.95\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        # Compare F1 scores
        bal_f1 = balanced_metrics.get('f1', 0)
        hr_f1 = hr_metrics.get('f1', 0)
        
        if hr_f1 > bal_f1 and hr_metrics.get('recall', 0) >= 0.95:
            f.write("The high-recall model achieves both higher recall (≥95%) and higher F1 score, ")
            f.write("making it the preferred choice for systematic review screening.\n")
        else:
            f.write("The balanced model is best for general classification tasks where overall performance ")
            f.write("is important, while the high-recall model is better suited for systematic review ")
            f.write("screening where achieving high recall (≥95%) is crucial to ensure comprehensive coverage.\n")

def main():
    """Main function to run the grid search experiment"""
    parser = argparse.ArgumentParser(description="Run grid search experiment for systematic review classification")
    parser.add_argument("--data", type=str, default=os.path.join(PATHS["data_processed"], "data_final_processed.csv"),
                      help="Path to the dataset")
    parser.add_argument("--output", type=str, default="results/grid_search",
                      help="Directory to save results")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "svm", "cosine", "cnb"],
                      help="Type of model ('logreg', 'svm', 'cosine', 'cnb')")
    parser.add_argument("--cv", type=int, default=5,
                      help="Number of cross-validation folds")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(args.data)
    
    # Run grid search
    grid, balanced_model, balanced_params, high_recall_model, high_recall_params, ngram_analysis = run_grid_search(
        X_train, y_train, model_type=args.model, cv=args.cv
    )
    
    # Evaluate models
    bal_metrics, hr_metrics, threshold, y_prob_bal, y_prob_hr = evaluate_models(
        balanced_model, high_recall_model, X_val, y_val
    )
    
    # Plot precision-recall curve
    plots_dir = os.path.join(args.output, "baseline", args.model, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "pr_curve.png")
    plot_pr_curve(y_val, y_prob_bal, y_prob_hr, plot_path)
    
    # Save all results
    save_results(
        balanced_model, high_recall_model,
        bal_metrics, hr_metrics,
        balanced_params, high_recall_params,
        threshold, args.output, args.model,
        X_val, y_val, y_prob_bal, y_prob_hr,
        ngram_analysis
    )
    
    # Record conclusion
    logger.info(f"Grid search and evaluation complete for {args.model} model")
    logger.info(f"Balanced model F1: {bal_metrics['f1']:.4f}")
    logger.info(f"High-recall model F1: {hr_metrics['f1']:.4f}")
    logger.info(f"High-recall model recall: {hr_metrics['recall']:.4f}")
    
    return grid, balanced_model, high_recall_model


if __name__ == "__main__":
    main()