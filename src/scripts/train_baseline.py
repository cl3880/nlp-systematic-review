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
    if scoring is None:
        scoring = make_scorer(fbeta_score, beta=2)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=splitter,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid.fit(X, y)
    logger.info(f"Best grid score: {grid.best_score_:.4f}")
    logger.info(f"Best grid params: {grid.best_params_}")
    return grid.best_estimator_, grid.best_params_, grid.cv_results_

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
    parser.add_argument('--ngram-range', nargs=2, type=int, metavar=('MIN_N','MAX_N'), default=[1,2],
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
        min_df=args.min_df
    )

    if args.model == 'cosine':
        args.no_grid_search = True

    if args.no_grid_search:
        logger.info('Training without grid search')
        pipeline.fit(train, train['relevant'])
        best_model = pipeline
        best_params = {
            'max_features': args.max_features,
            'ngram_range': tuple(args.ngram_range),
            'min_df': args.min_df,
            'normalization': args.normalization,
            'balancing': args.balancing
        }
    else:
        # parameter grid based on model
        if args.model == 'logreg':
            grid = baseline_param_grid()
        elif args.model == 'svm':
            grid = svm_param_grid()
        elif args.model == 'cnb':
            grid = baseline_param_grid().copy()
            grid['clf'] = [ComplementNB()]
            grid['clf__alpha'] = [0.1, 1.0, 10.0]   # Have to double check Frunza & Norman
        else:
            grid = {}
        best_model, best_params, cv_results = run_grid_search(
            pipeline,
            train,
            train['relevant'],
            grid,
            cv=args.cv_folds
        )
        pd.DataFrame(cv_results).to_csv(
            os.path.join(args.output_dir, 'metrics', 'grid_search_results.csv'),
            index=False
        )

    joblib.dump(best_model, os.path.join(args.output_dir, 'models', 'model.joblib'))
    with open(os.path.join(args.output_dir, 'models', 'params.json'), 'w') as fp:
        json.dump(best_params, fp, indent=2)

    logger.info('Evaluating on validation set')
    try:
        val_probs = best_model.predict_proba(val)[:, 1]
    except Exception:
        val_probs = None

    if args.model == 'cosine' and val_probs is not None:
        if args.cos_thresh is None:
            thresh, _ = find_threshold_for_recall(val['relevant'].values, val_probs, args.target_recall)
        else:
            thresh = args.cos_thresh
        best_model.named_steps['clf'].threshold = thresh
        logger.info(f"Set cosine threshold to {thresh:.4f}")

    val_preds = best_model.predict(val)
    evaluate(
        val['relevant'].values,
        val_preds,
        val_probs,
        os.path.join(args.output_dir, 'metrics'),
        'validation',
        target_recall=args.target_recall
    )

    logger.info('Evaluating on test set')
    test_preds = best_model.predict(test)
    try:
        test_probs = best_model.predict_proba(test)[:, 1]
    except Exception:
        test_probs = None
    evaluate(
        test['relevant'].values,
        test_preds,
        test_probs,
        os.path.join(args.output_dir, 'metrics'),
        'test',
        target_recall=args.target_recall
    )

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
