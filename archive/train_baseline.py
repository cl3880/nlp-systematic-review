#!/usr/bin/env python3
"""
Train a single model (logreg, svm, or cosine) for systematic review classification.
Supports optional text normalization, class balancing, and grid search or single-run hyperparameter testing.
"""
import os
import argparse
import logging
import json
from datetime import datetime

import pandas as pd
import joblib
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
import numpy as np

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.models.model_factory import create_model
from src.models.baseline_classifier import baseline_param_grid
from src.models.svm_classifier import svm_param_grid
from src.utils.evaluate import evaluate, find_threshold_for_recall

# Configure logging
log_path = os.path.join(PATHS['logs_dir'], 'train_baseline.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def prepare_data(input_path: str, output_path: str, clean_only: bool = False):
    """
    Invoke prepare_data to clean or process the raw CSV.
    """
    logger.info(f"Preparing data from {input_path} to {output_path}")
    cmd = f"python -m src.scripts.prepare_data --input {input_path} --output {output_path}"
    if clean_only:
        cmd += " --clean-only"
    os.system(cmd)

def run_grid_search(pipeline, X, y, param_grid: dict, cv: int = 5, scoring=None):
    """
    Run GridSearchCV on the provided pipeline.
    """
    logger.info("Starting grid search...")
    
    # Set up multi-metric scoring
    if scoring is None:
        scoring = {
            'f1': 'f1',
            'recall': 'recall',
            'roc_auc': 'roc_auc',
            'f2': make_scorer(fbeta_score, beta=2)
        }
    
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        refit='f1',  # Default to optimize for F1
        cv=splitter,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid.fit(X, y)
    logger.info(f"Best f1 score: {grid.best_score_:.4f}")
    logger.info(f"Best f1 params: {grid.best_params_}")
    
    # Find best parameters for high recall
    recall_idx = np.argmax(grid.cv_results_['mean_test_recall'])
    best_recall_params = grid.cv_results_['params'][recall_idx]
    best_recall_score = grid.cv_results_['mean_test_recall'][recall_idx]
    logger.info(f"Best recall score: {best_recall_score:.4f}")
    logger.info(f"Best recall params: {best_recall_params}")
    
    # Extract n-gram summary
    results_df = pd.DataFrame(grid.cv_results_)
    ngram_summary = (
        results_df
        .loc[:, ['param_tfidf__ngram_range', 'mean_test_f1', 'mean_test_recall', 'mean_test_f2']]
        .drop_duplicates()
        .sort_values('mean_test_f1', ascending=False)
    )
    logger.info("N-gram performance summary (sorted by F1):")
    logger.info(f"\n{ngram_summary}")
    
    # Save the n-gram summary
    metrics_dir = os.path.join(PATHS['results_dir'], 'analysis')
    os.makedirs(metrics_dir, exist_ok=True)
    ngram_summary.to_csv(os.path.join(metrics_dir, 'ngram_performance.csv'), index=False)
    
    # Return best model, params for F1, params for recall, and cv_results
    return grid.best_estimator_, grid.best_params_, best_recall_params, grid.cv_results_

def main():
    parser = argparse.ArgumentParser(
        description="Train baseline model for systematic review classification"
    )
    parser.add_argument('--data-mode', choices=['raw','cleaned','processed','custom'], default='processed',
                        help='Data processing mode')
    parser.add_argument('--raw-data', default=os.path.join(PATHS['data_raw'], 'data_final.csv'),
                        help='Raw input CSV path')
    parser.add_argument('--cleaned-data', default=os.path.join(PATHS['data_processed'], 'data_final_cleaned.csv'),
                        help='Cleaned data CSV path')
    parser.add_argument('--processed-data', dest='processed_data', default=os.path.join(PATHS['data_processed'], 'data_final_processed.csv'),
                        help='Processed data CSV path')
    parser.add_argument('--custom-data', default=None, help='Custom data CSV path')
    parser.add_argument('--clean-only', action='store_true', help='Only perform initial cleaning')
    parser.add_argument('--force-preprocessing', action='store_true', help='Force data preprocessing even if outputs exist')

    parser.add_argument('--model', choices=['logreg','svm','cosine', 'cnb'], required=True,
                        help='Classifier type')
    parser.add_argument('--normalization', choices=['none','stemming','lemmatization'], default='none',
                        help='Text normalization technique')
    parser.add_argument('--balancing', choices=['smote','undersample'], default=None,
                        help='Class balancing strategy')
    parser.add_argument('--no-grid-search', action='store_true', help='Train with default params; skip grid search')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds for grid search')

    parser.add_argument('--max-features', type=int, default=10000,
                        help='TF-IDF max features (single-run only)')
    parser.add_argument('--ngram-range', nargs=2, type=int, metavar=('MIN_N','MAX_N'), default=[1,3],
                        help='TF-IDF ngram range as two ints (single-run only)')
    parser.add_argument('--min-df', type=int, default=3,
                        help='TF-IDF min_df (single-run only)')

    parser.add_argument('--test-size', type=float, default=0.1, help='Test set fraction')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set fraction')
    parser.add_argument('--target-recall', type=float, default=0.95, help='Target recall for threshold selection')
    parser.add_argument('--cos-thresh', type=float, default=None, help='Manual threshold for cosine classifier')
    parser.add_argument('--output-dir', default=PATHS['baseline_dir'], help='Directory for outputs')
    args = parser.parse_args()

    dirs = [
        args.output_dir,
        os.path.join(args.output_dir, 'models'),
        os.path.join(args.output_dir, 'metrics'),
        os.path.join(args.output_dir, 'plots'),
        os.path.join(args.output_dir, 'analysis'),
        PATHS['logs_dir'],
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    if args.data_mode == 'raw':
        if args.clean_only or not os.path.exists(args.cleaned_data):
            prepare_data(args.raw_data, args.cleaned_data, clean_only=True)
        if not os.path.exists(args.processed_data):
            prepare_data(args.cleaned_data, args.processed_data)
        data_path = args.processed_data
    elif args.data_mode == 'cleaned':
        if not os.path.exists(args.processed_data):
            prepare_data(args.cleaned_data, args.processed_data)
        data_path = args.processed_data
    elif args.data_mode == 'processed':
        data_path = args.processed_data
    else:
        if not args.custom_data:
            logger.error('Custom mode requires --custom-data')
            return
        data_path = args.custom_data

    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    train, val, test = make_splits(df, test_size=args.test_size, val_size=args.val_size, stratify=True, seed=42)
    logger.info(f"Split sizes => train: {len(train)}, val: {len(val)}, test: {len(test)}")

    normalization = None if args.normalization == 'none' else args.normalization
    pipeline = create_model(
        model_type=args.model,
        normalization=normalization,
        balancing=args.balancing,
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        min_df=args.min_df,
        threshold=args.cos_thresh if args.model == 'cosine' else None
    )

    if args.model == 'cosine':
        args.no_grid_search = True

    if args.no_grid_search:
        logger.info('Training without grid search')
        pipeline.fit(train, train['relevant'])
        best_model = pipeline
        balanced_params = {
            'max_features': args.max_features,
            'ngram_range': tuple(args.ngram_range),
            'min_df': args.min_df,
            'normalization': args.normalization,
            'balancing': args.balancing,
            'threshold': args.cos_thresh if args.model == 'cosine' else None
        }
        high_recall_params = balanced_params.copy()
    else:
        # parameter grid based on model
        if args.model == 'logreg':
            grid = baseline_param_grid()
        elif args.model == 'svm':
            grid = svm_param_grid()
        elif args.model == 'cnb':
            grid = baseline_param_grid().copy()
            grid['clf'] = [ComplementNB()]
            grid['clf__alpha'] = [0.1, 1.0, 10.0]
        else:
            grid = {}
            
        # Run grid search
        best_model, balanced_params, high_recall_params, cv_results = run_grid_search(
            pipeline,
            train,
            train['relevant'],
            grid,
            cv=args.cv_folds
        )
        
        logger.info(f"Balanced model parameters: {balanced_params}")
        logger.info(f"High-recall model parameters: {high_recall_params}")

    # Save balanced (F1-optimized) model
    joblib.dump(best_model, os.path.join(args.output_dir, 'models', 'balanced_model.joblib'))
    with open(os.path.join(args.output_dir, 'models', 'balanced_params.json'), 'w') as fp:
        json.dump(balanced_params, fp, indent=2)

    # Create and save high-recall model if not using grid search
    if args.no_grid_search:
        high_recall_model = best_model
    else:
        # Create a new model with high-recall parameters
        high_recall_model = create_model(
            model_type=args.model,
            normalization=None if high_recall_params.get('normalizer__technique') is None else high_recall_params.get('normalizer__technique'),
            balancing=args.balancing,
            max_features=high_recall_params.get('tfidf__max_features', args.max_features),
            ngram_range=high_recall_params.get('tfidf__ngram_range', tuple(args.ngram_range)),
            min_df=high_recall_params.get('tfidf__min_df', args.min_df),
            C=high_recall_params.get('clf__C', 1.0),
            class_weight=high_recall_params.get('clf__class_weight', 'balanced'),
            threshold=args.cos_thresh if args.model == 'cosine' else None
        )
        high_recall_model.fit(train, train['relevant'])
    
    # Save high-recall model
    joblib.dump(high_recall_model, os.path.join(args.output_dir, 'models', 'high_recall_model.joblib'))
    with open(os.path.join(args.output_dir, 'models', 'high_recall_params.json'), 'w') as fp:
        json.dump(high_recall_params, fp, indent=2)

    # Evaluate balanced model on validation set
    logger.info('Evaluating balanced model on validation set')
    try:
        val_probs = best_model.predict_proba(val)[:, 1]
    except Exception:
        val_probs = None

    if val_probs is not None:
        # Find threshold for target recall
        thresh, achieved_recall = find_threshold_for_recall(
            val['relevant'].values, 
            val_probs, 
            target_recall=args.target_recall
        )
        logger.info(f"Threshold for {args.target_recall*100:.0f}% recall: {thresh:.4f} (achieved: {achieved_recall*100:.1f}%)")
        
        # Get predictions at optimized threshold
        val_preds_optim = (val_probs >= thresh).astype(int)
        
        # Evaluate with optimized threshold
        optim_metrics = evaluate(
            val['relevant'].values,
            val_preds_optim,
            val_probs,
            os.path.join(args.output_dir, 'metrics'),
            'balanced_optimized',
            target_recall=args.target_recall
        )
        
        # Save optimized threshold
        with open(os.path.join(args.output_dir, 'models', 'optimized_threshold.json'), 'w') as f:
            json.dump({
                'threshold': thresh,
                'target_recall': args.target_recall,
                'achieved_recall': achieved_recall,
                'validation_metrics': optim_metrics
            }, f, indent=2)
        
        # If it's a cosine model, update its threshold
        if args.model == 'cosine':
            best_model.named_steps['clf'].threshold = thresh
            logger.info(f"Updated cosine model threshold to {thresh:.4f}")

    # Original evaluation with model's default threshold
    val_preds = best_model.predict(val)
    balanced_metrics = evaluate(
        val['relevant'].values,
        val_preds,
        val_probs,
        os.path.join(args.output_dir, 'metrics'),
        'balanced',
        target_recall=args.target_recall
    )

    # Evaluate high-recall model on validation set
    logger.info('Evaluating high-recall model on validation set')
    try:
        hr_val_probs = high_recall_model.predict_proba(val)[:, 1]
    except Exception:
        hr_val_probs = None

    if hr_val_probs is not None:
        # Find threshold for target recall
        hr_thresh, hr_achieved_recall = find_threshold_for_recall(
            val['relevant'].values, 
            hr_val_probs, 
            target_recall=args.target_recall
        )
        logger.info(f"High-recall threshold for {args.target_recall*100:.0f}% recall: {hr_thresh:.4f} (achieved: {hr_achieved_recall*100:.1f}%)")
        
        # Get predictions at optimized threshold
        hr_val_preds = (hr_val_probs >= hr_thresh).astype(int)
        
        # Evaluate high-recall model with optimized threshold
        hr_metrics = evaluate(
            val['relevant'].values,
            hr_val_preds,
            hr_val_probs,
            os.path.join(args.output_dir, 'metrics'),
            'high_recall',
            target_recall=args.target_recall
        )
        
        # Save high-recall optimized threshold
        with open(os.path.join(args.output_dir, 'models', 'high_recall_threshold.json'), 'w') as f:
            json.dump({
                'threshold': hr_thresh,
                'target_recall': args.target_recall,
                'achieved_recall': hr_achieved_recall,
                'validation_metrics': hr_metrics
            }, f, indent=2)
    
    # Save comparison of balanced vs high-recall models
    with open(os.path.join(args.output_dir, 'analysis', 'model_comparison.md'), 'w') as f:
        f.write(f"# Model Comparison: Balanced vs High-Recall\n\n")
        f.write(f"## Parameters\n\n")
        f.write(f"### Balanced Model (F1-optimized)\n")
        for k, v in balanced_params.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n### High-Recall Model\n")
        for k, v in high_recall_params.items():
            f.write(f"- {k}: {v}\n")
        
        f.write(f"\n## Validation Metrics\n\n")
        f.write(f"### Balanced Model\n")
        for metric, value in balanced_metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
        
        if 'hr_metrics' in locals():
            f.write(f"\n### High-Recall Model\n")
            for metric, value in hr_metrics.items():
                f.write(f"- {metric}: {value:.4f}\n")

    if args.model in ('logreg','svm'):
        logger.info('Performing cross-validation')
        splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        scoring = { 'f1':'f1','precision':'precision','recall':'recall','roc_auc':'roc_auc' }
        cv_res = cross_validate(
            best_model,
            train,
            train['relevant'],
            cv=splitter,
            scoring=scoring,
            return_train_score=True
        )
        for metric in scoring:
            mean = cv_res[f'test_{metric}'].mean()
            std = cv_res[f'test_{metric}'].std()
            logger.info(f"CV {metric}: {mean:.4f} ±{std:.4f}")
    
    else:
        logger.info('Skipping cv for cosine pipeline')

    logger.info('Training complete.')

if __name__ == '__main__':
    main()
