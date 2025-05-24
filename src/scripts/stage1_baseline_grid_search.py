#!/usr/bin/env python3
"""
Baseline grid search experiment for systematic review screening.

This script:
1. Loads and splits the data
2. Defines a parameter grid (TF-IDF + classifier hyperparams)
3. Uses multi-metric scoring: F1, recall, precision, ROC AUC
4. Refits on F1 and extracts two models:
   - Balanced: highest F1
   - High-recall: highest F1 among recall ≥ 0.95
5. Evaluates both models on validation set
6. Saves models, params, metrics, and plots into a structured results directory
"""
import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix, auc
)

from src.models.param_grids import logreg_param_grid, svm_param_grid, cnb_param_grid, cosine_param_grid

from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PATHS, RESULTS_V2, get_result_path_v2
from src.utils.data_utils import load_data, make_splits
from src.models.model_factory import create_pipeline, create_model
from src.utils.evaluate import evaluate
from src.models.text_processors import TextCombiner, TextNormalizer
from src.utils.result_utils import (
    save_model_results, numpy_to_python, plot_pr_curve, 
    plot_roc_curve, plot_confusion_matrix, create_comparison_markdown
)
from src.utils.logging_utils import setup_logging, setup_per_model_logging, set_debug_logging

def preprocess_corpus(df, text_columns=["title", "abstract"], technique=None):
    """
    Perform deterministic text normalization once, prior to model training.
    
    Args:
        df: DataFrame containing text data
        text_columns: Text columns to combine
        technique: Normalization technique ('stemming', 'lemmatization', or None)
        
    Returns:
        Modified DataFrame with normalized text column
    """
    if technique is None:
        return df
        
    combiner = TextCombiner(text_columns)
    combined_texts = combiner.transform(df)
    
    normalizer = TextNormalizer(technique=technique)
    normalized_texts = normalizer.transform(combined_texts)
    
    result_df = df.copy()
    result_df['normalized_text'] = normalized_texts
    
    return result_df

logger = logging.getLogger(__name__)

def get_param_grid(model_type):
    if model_type == "logreg":
        return logreg_param_grid()
    elif model_type == "svm":
        return svm_param_grid()
    elif model_type == "cnb":
        return cnb_param_grid()
    elif model_type == "cosine":
        return cosine_param_grid()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def override_threshold(pipe, thresh):
    try:
        pipe.set_params(clf__threshold=thresh)
    except ValueError:
        pass
    return pipe

def run_grid_search(X_train, y_train, model_type="logreg", normalization=None, balancing=None, cv=5, output_dir="results", cache_dir=None, text_columns=["title", "abstract"]):
    """Run grid search to find balanced and high-recall models"""
    pipeline = create_model(model_type = model_type, normalization = None, balancing = balancing, text_columns = text_columns, cache_dir=cache_dir)
    logger.info(f"Using {model_type.upper()} model")
    override_threshold(pipeline, 0.33)
    logger.info(f"Using {model_type.upper()} model with normalization={normalization}")
    
    if balancing:
        logger.info(f"Using balancing technique: {balancing}")

    param_grid = get_param_grid(model_type)
    
    scoring = {
        'f1': 'f1',                 # For balanced model
        'recall': 'recall',         # For extracting high-recall model
        'precision': 'precision',   # For examining trade-offs
        # 'roc_auc': 'roc_auc'      # For overall ranking quality
    }
    
    logger.info("Starting grid search with multi-metric scoring...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='f1',
        cv=skf,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    
    logger.info(f"Grid search complete. Best F1 score: {grid.best_score_:.4f}")
    logger.info(f"Best parameters: {grid.best_params_}")
    
    balanced_model = grid.best_estimator_
    balanced_params = grid.best_params_

    high_recall_model = None
    high_recall_params = None

    # Export full cv_results_ for deeper analysis
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_path = os.path.join(output_dir, "cv_results.csv")
    os.makedirs(os.path.dirname(cv_path), exist_ok=True)
    cv_df.to_csv(cv_path, index=False)
    logger.info(f"Exported CV results to {cv_path}")
    
    # Analyze n-gram performance if grid search was performed
    ngram_analysis = {}
    try:
        if 'param_tfidf__ngram_range' in cv_df.columns:
            # Group by n-gram range and calculate average F1 scores
            ngram_groups = cv_df.groupby('param_tfidf__ngram_range')['mean_test_f1'].mean()
            ngram_analysis = {eval(k) if isinstance(k, str) else k: v for k, v in ngram_groups.to_dict().items()}
            
            logger.info("N-gram range analysis:")
            for ngram, f1 in sorted(ngram_analysis.items()):
                logger.info(f"  {ngram}: average F1 = {f1:.4f}")
            
            # Compare (1,2) vs (1,3) ngram range
            if (1, 2) in ngram_analysis and (1, 3) in ngram_analysis:
                f1_12 = ngram_analysis[(1, 2)]
                f1_13 = ngram_analysis[(1, 3)]
                diff = (f1_13 - f1_12) * 100
                logger.info(f"  (1,3) vs (1,2): {diff:.1f} percentage points difference")
    except Exception as e:
        logger.warning(f"Could not perform n-gram analysis: {e}")
    
    return grid, balanced_model, balanced_params, high_recall_model, high_recall_params, ngram_analysis

def extract_high_recall_model(grid, X_train, y_train, target_recall=0.95):
    """
    Extract a high-recall model from grid search results.
    
    Args:
        grid: Fitted GridSearchCV object
        X_train: Training features
        y_train: Training labels
        target_recall: Target recall threshold (default: 0.95)
        
    Returns:
        tuple: (high_recall_model, high_recall_params)
    """
    # Get results
    res = grid.cv_results_
    recalls = np.array(res['mean_test_recall'])
    f1s = np.array(res['mean_test_f1'])
    params = res['params']
    
    # Find indices where recall >= target_recall
    idxs = np.where(recalls >= target_recall)[0]
    
    if len(idxs) == 0:
        logger.warning(f"No config ≥{target_recall*100}% recall; choosing closest.")
        idx = np.argmin(np.abs(recalls - target_recall))
    else:
        # Get the configuration with highest F1 among those with recall >= target_recall
        idx = idxs[np.argmax(f1s[idxs])]
    
    best_params = params[idx]
    logger.info(f"Selected high-recall config with recall {recalls[idx]:.4f} and F1 {f1s[idx]:.4f}")
    
    # Create and fit high-recall model (simply clone and set params)
    from sklearn.base import clone
    
    hr_model = clone(grid.best_estimator_)
    hr_model.set_params(**{k: v for k, v in best_params.items() if k in hr_model.get_params()})
    hr_model.fit(X_train, y_train)
    
    return hr_model, best_params

def optimize_threshold_for_recall(model, X_val, y_val, target_recall=0.95):
    """
    Optimize decision threshold on validation data to achieve target recall.
    
    Returns:
        tuple: (optimal_threshold, metrics_at_threshold)
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    thresholds = np.append(thresholds, 1.0)
    
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) == 0:
        return 0.0, None
        
    valid_precision = precision[valid_indices]
    valid_recall = recall[valid_indices]
    valid_thresholds = thresholds[valid_indices]
    
    f1_scores = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall)
    best_idx = np.argmax(f1_scores)
    
    optimal_threshold = valid_thresholds[best_idx]
    
    y_pred = (y_prob >= optimal_threshold).astype(int)
    metrics = compute_metrics(y_val, y_pred, y_prob)
    
    return optimal_threshold, metrics

def compute_metrics(y_true, y_pred, y_prob, zero_division=0):
    """Compute standard evaluation metrics"""
    prec = precision_score(y_true, y_pred, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, zero_division=zero_division)
    
    if prec + rec > 0:
        f1 = 2 * (prec * rec) / (prec + rec)
    else:
        f1 = 0.0

    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    wss95 = (tn + fn) / len(y_true) - 0.05

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
    try:
        y_prob_bal = bal_model.predict_proba(X_val)[:, 1]
    except (AttributeError, IndexError):
        y_prob_bal = np.zeros(len(y_val))
        logger.warning("Could not get prediction probabilities for balanced model")
    
    y_pred_bal = bal_model.predict(X_val)
    bal_metrics = compute_metrics(y_val, y_pred_bal, y_prob_bal)
    
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
        
        y_pred_hr = (y_prob_hr >= threshold).astype(int)
        
    except (AttributeError, IndexError):
        y_prob_hr = np.zeros(len(y_val))
        threshold = 0.5
        y_pred_hr = hr_model.predict(X_val)
        logger.warning("Could not get prediction probabilities for high-recall model")
    
    hr_metrics = compute_metrics(y_val, y_pred_hr, y_prob_hr)
    
    logger.info("Balanced model metrics:")
    for k, v in bal_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("High-recall model metrics:")
    for k, v in hr_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
        
    return bal_metrics, hr_metrics, threshold, y_prob_bal, y_prob_hr

def main():
    """Main function to run the grid search experiment"""
    parser = argparse.ArgumentParser(description="Run grid search experiment for systematic review classification")
    parser.add_argument("--data", type=str, default=os.path.join(PATHS["data_processed"], "data_final_processed.csv"),
                      help="Path to the dataset")
    parser.add_argument("--output", type=str, default="results/grid_search",
                      help="Directory to save results")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "svm", "cosine", "cnb"],
                      help="Type of model ('logreg', 'svm', 'cosine', 'cnb')")
    parser.add_argument("--normalization", type=str, default=None, choices=[None, "stemming", "lemmatization"],
                      help="Text normalization technique (None, 'stemming', or 'lemmatization')")
    parser.add_argument("--cache_dir", type=str, default="cache",
                      help="Directory for caching pipeline transformations")
    parser.add_argument("--balancing", type=str, choices=["none", "smote"], default=None,
                      help="Balancing technique (none, smote)")
    parser.add_argument("--cv", type=int, default=5,
                      help="Number of cross-validation folds")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug-level logging")
    
    args = parser.parse_args()
    
    global logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    model_type_str = args.model
    if args.normalization:
        model_type_str = f"{args.normalization}_{args.model}"
    elif args.balancing and args.balancing != "none":
        model_type_str = f"{args.balancing}_{args.model}"
    
    logger = setup_per_model_logging(model_type_str, name=__name__, level=log_level)
    
    if args.debug:
        logger.debug("Debug logging enabled")
        
    logger_msg = f"Starting grid search experiment for {args.model} model"
    if args.normalization:
        logger_msg += f" with {args.normalization} normalization"
    if args.balancing and args.balancing != "none":
        logger_msg += f" with {args.balancing} balancing"
    logger.info(logger_msg)
    
    if args.output == "results/grid_search": 
        if args.normalization:
            output_dir = get_result_path_v2(args.model, normalization=args.normalization)
        elif args.balancing and args.balancing != "none":
            output_dir = get_result_path_v2(args.model, balancing=args.balancing)
        else:
            output_dir = get_result_path_v2(args.model)
    else:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    
    cache_dir = args.cache_dir if args.cache_dir else None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
    
    df = load_data(args.data)
    train, val, test = make_splits(df, test_size=0.1, val_size=0.1, stratify=True, seed=42)
    
    if args.normalization:
        logger.info(f"Applying {args.normalization} normalization once...")
        train = preprocess_corpus(train, technique=args.normalization)
        val = preprocess_corpus(val, technique=args.normalization)
        cache_dir = args.cache_dir
        text_columns = ['normalized_text']
    else:
        cache_dir = None
        text_columns = ['title', 'abstract']
    
    X_train = train.drop('relevant', axis=1) if 'relevant' in train.columns else train
    y_train = train['relevant'] if 'relevant' in train.columns else None
    X_val = val.drop('relevant', axis=1) if 'relevant' in val.columns else val
    y_val = val['relevant'] if 'relevant' in val.columns else None
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    if args.debug:
        logger.debug(f"Column list: {X_train.columns.tolist()}")
        logger.debug(f"Label distribution - Train: {np.bincount(y_train)}")
        logger.debug(f"Label distribution - Val: {np.bincount(y_val)}")
    
    grid, balanced_model, balanced_params, _, _, ngram_analysis = run_grid_search(
        X_train, y_train, model_type=args.model, normalization=args.normalization, balancing=args.balancing,
        cv=args.cv, output_dir=output_dir, text_columns=text_columns,
        cache_dir=cache_dir
    )
    
    optimal_threshold, hr_metrics = optimize_threshold_for_recall(
        balanced_model, X_val, y_val, target_recall=0.95
    )
    
    high_recall_model = balanced_model
    high_recall_threshold = optimal_threshold
    
    # Calculate predictions using balanced model with optimal threshold
    y_prob_bal = balanced_model.predict_proba(X_val)[:, 1]
    y_pred_bal = balanced_model.predict(X_val)
    bal_metrics = compute_metrics(y_val, y_pred_bal, y_prob_bal)
    
    # Calculate high-recall predictions using inference-time threshold
    y_prob_hr = y_prob_bal  # Same probabilities, different decision threshold
    y_pred_hr = (y_prob_hr >= optimal_threshold).astype(int)
    
    high_recall_params = balanced_params.copy()
    high_recall_params['optimal_threshold'] = optimal_threshold
    
    bal_metrics, hr_metrics, threshold, y_prob_bal, y_prob_hr = evaluate_models(
        balanced_model, high_recall_model, X_val, y_val
    )
    
    save_model_results(
        balanced_model, high_recall_model,
        bal_metrics, hr_metrics,
        balanced_params, high_recall_params,
        threshold, output_dir, args.model,
        X_val, y_val, y_prob_bal, y_prob_hr,
        ngram_analysis
    )
    
    logger.info(f"Grid search and evaluation complete for {args.model} model")
    logger.info(f"Balanced model F1: {bal_metrics['f1']:.4f}")
    logger.info(f"High-recall model F1: {hr_metrics['f1']:.4f}")
    logger.info(f"High-recall model recall: {hr_metrics['recall']:.4f}")
    
    return grid, balanced_model, high_recall_model

if __name__ == "__main__":
    main()
