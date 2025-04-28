#!/usr/bin/env python3
"""
Comprehensive grid search for systematic review classification.
This script performs a unified grid search over:
- Model types (logreg, svm, cosine)
- Text normalization techniques (none, stemming, lemmatization)
- Class balancing methods (none, smote, undersample)
- Traditional hyperparameters
"""
import os
import argparse
import logging
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tabulate import tabulate

from sklearn.metrics import make_scorer, fbeta_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.utils.evaluate import evaluate, compare_models, find_threshold_for_recall
from src.scripts.compare_normalizations import TextNormalizer, NormalizingTextCombiner
from src.models.baseline_classifier import TextCombiner, CosineSimilarityClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PATHS["logs_dir"], "grid_search_full.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def custom_f2_scorer():
    """Create a scorer that emphasizes recall over precision (F2 score)"""
    return make_scorer(fbeta_score, beta=2)

def wss_scorer(target_recall=0.95):
    """
    Create a scorer based on Work Saved over Sampling (WSS) at a target recall level.
    WSS represents the percentage of papers that don't need to be screened manually
    when applying a classifier with the given threshold.
    
    WSS@95 = (TN + FN) / (TN + FP + FN + TP) - (1 - 0.95)
    """
    def wss_score(y_true, y_scores, **kwargs):
        # Find threshold that gives target recall
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find points with recall >= target
        valid_idx = np.where(recall >= target_recall)[0]
        if len(valid_idx) == 0:
            return -1.0
        
        # Get the precision at the threshold that gives at least target recall
        idx = valid_idx[-1]
        threshold = thresholds[idx] if idx < len(thresholds) else 0
        
        # Apply threshold
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate confusion matrix values
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate WSS
        total = TP + TN + FP + FN
        wss = (TN + FN) / total - (1 - target_recall)
        return wss
    
    return make_scorer(wss_score, needs_threshold=True, greater_is_better=True)

def run_grid_search(X, y, output_dir, cv=5, target_recall=0.95):
    """
    Run a comprehensive grid search over model types, normalization techniques,
    balancing methods, and traditional hyperparameters.
    
    Returns the best model found.
    """
    logger.info("Starting comprehensive grid search...")
    
    # Create output directories
    models_dir = os.path.join(output_dir, "models")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Define base pipeline with normalization, TF-IDF, and a classifier
    base_pipeline = Pipeline([
        ('normalizer', TextNormalizer(technique=None)),
        ('combiner', TextCombiner(['title', 'abstract'])),
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=3,
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(max_iter=5000, random_state=42))
    ])
    
    # Configure different parameter grids for different model types
    param_grids = {
        "logreg": {
            'normalizer__technique': [None, 'stemming', 'lemmatization'],
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [3, 5],
            'clf': [LogisticRegression(max_iter=5000, random_state=42)],
            'clf__C': [0.1, 1.0, 10.0],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear']
        },
        "svm": {
            'normalizer__technique': [None, 'stemming', 'lemmatization'],
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [3, 5],
            'clf': [SVC(probability=True, random_state=42)],
            'clf__C': [0.1, 1.0, 10.0],
            'clf__kernel': ['linear', 'rbf'],
            'clf__gamma': ['scale']
        },
        "cosine": {
            'normalizer__technique': [None, 'stemming', 'lemmatization'],
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__min_df': [3, 5],
            'clf': [CosineSimilarityClassifier()]
        },
        "cnb": {
            'normalizer__technique': [None, 'stemming', 'lemmatization'],
            'tfidf__max_features': [5000, 10000, 20000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'tfidf__min_df': [3,5],
            'clf': [ComplementNB()],
            'clf__alpha': [0.1, 1.0, 10.0]
        }
    }
    
    # Define scoring metrics
    scoring = {
        'f1': make_scorer(f1_score, zero_division=0),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f2': make_scorer(fbeta_score, beta=2, zero_division=0),
        'wss': wss_scorer(target_recall)
    }
    
    # Create stratified k-fold cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Track results for all configurations
    all_results = []
    best_score = -1
    best_model = None
    best_params = None
    best_model_name = None
    
    # For each sampling strategy (none, SMOTE, undersample)
    for sampling in [None, 'smote', 'undersample']:
        sampling_name = 'none' if sampling is None else sampling
        logger.info(f"Evaluating sampling strategy: {sampling_name}")
        
        # For each model type (logreg, svm, cosine)
        for model_type, param_grid in param_grids.items():
            logger.info(f"Running grid search for model type: {model_type}")
            
            # Create a pipeline with the current sampling strategy
            if sampling is None:
                # Regular scikit-learn pipeline without sampling
                current_pipeline = base_pipeline
            else:
                # imblearn pipeline with sampling
                steps = list(base_pipeline.steps)
                if sampling == 'smote':
                    sampler = SMOTE(random_state=42)
                else:  # undersample
                    sampler = RandomUnderSampler(random_state=42)
                
                # Insert sampler before classifier
                steps.insert(-1, ('sampler', sampler))
                current_pipeline = ImbPipeline(steps)
            
            # Run grid search
            grid = GridSearchCV(
                current_pipeline,
                param_grid,
                cv=cv_splitter,
                scoring=scoring,
                refit='wss',  # Or 'precision_at_95' to maximize precision among models >= 95% recall
                error_score=0.0,
                n_jobs=-1,
                verbose=1,
                return_train_score=True
            )
            
            # Fit grid search
            grid.fit(X, y)
            
            # Save grid search results
            model_name = f"{model_type}_{sampling_name}"
            cv_results_df = pd.DataFrame(grid.cv_results_)
            cv_results_df.to_csv(os.path.join(metrics_dir, f"{model_name}_cv_results.csv"), index=False)
            
            # Save best model from this grid search
            joblib.dump(grid.best_estimator_, os.path.join(models_dir, f"{model_name}_model.joblib"))
            
            # Save best parameters
            with open(os.path.join(models_dir, f"{model_name}_params.json"), 'w') as f:
                # Convert parameters to JSON-serializable format
                best_params_dict = grid.best_params_.copy()
                for key, value in best_params_dict.items():
                    if hasattr(value, '__name__'):
                        best_params_dict[key] = value.__name__
                    elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                        best_params_dict[key] = str(value)
                
                json.dump(best_params_dict, f, indent=2)
            
            # Store key metrics for this configuration
            for i, (params, mean_test_f2, mean_test_wss) in enumerate(zip(
                    grid.cv_results_['params'],
                    grid.cv_results_['mean_test_f2'],
                    grid.cv_results_['mean_test_wss'])):
                
                # Only record the best configuration for each combination
                if i == grid.best_index_:
                    config_result = {
                        'model': model_type,
                        'sampling': sampling_name,
                        'normalization': str(params.get('normalizer__technique', 'none')),
                        'max_features': params.get('tfidf__max_features', 'default'),
                        'ngram_range': str(params.get('tfidf__ngram_range', 'default')),
                        'min_df': params.get('tfidf__min_df', 'default'),
                        'f2_score': mean_test_f2,
                        'wss_score': mean_test_wss
                    }
                    
                    # Add model-specific parameters
                    if model_type == 'logreg':
                        config_result.update({
                            'C': params.get('clf__C', 'default'),
                            'penalty': params.get('clf__penalty', 'default')
                        })
                    elif model_type == 'svm':
                        config_result.update({
                            'C': params.get('clf__C', 'default'),
                            'kernel': params.get('clf__kernel', 'default'),
                            'gamma': params.get('clf__gamma', 'default')
                        })
                    
                    all_results.append(config_result)
                    
                    # Track overall best model
                    if mean_test_f2 > best_score:
                        best_score = mean_test_f2
                        best_model = grid.best_estimator_
                        best_params = grid.best_params_
                        best_model_name = model_name
            
            logger.info(f"Best F2 score for {model_name}: {grid.best_score_:.4f}")
    
    # Save all results to a tabular format
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
    
    # Sort results by F2 score
    results_df = results_df.sort_values('f2_score', ascending=False)
    
    # Create formatted output tables (similar to train_grid.py)
    formatted_table = tabulate(
        results_df.values.tolist(),
        headers=results_df.columns,
        tablefmt="grid"
    )
    
    # Save formatted tables
    with open(os.path.join(output_dir, "all_results_formatted.txt"), "w") as f:
        f.write(formatted_table)
    
    # Save best overall model
    joblib.dump(best_model, os.path.join(output_dir, "best_model.joblib"))
    
    # Save performance summary
    with open(os.path.join(output_dir, "performance_summary.txt"), "w") as f:
        f.write("===== COMPREHENSIVE GRID SEARCH RESULTS =====\n\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"Best F2 score: {best_score:.4f}\n\n")
        f.write("Top 5 configurations:\n")
        f.write(tabulate(
            results_df.head(5).values.tolist(),
            headers=results_df.columns,
            tablefmt="simple"
        ))
    
    logger.info(f"Comprehensive grid search complete. Best F2 score: {best_score:.4f}")
    logger.info(f"Best model: {best_model_name}")
    
    return best_model

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive grid search for systematic review classification"
    )
    parser.add_argument(
        "--data",
        default=os.path.join(PATHS["data_processed"], "data_final_processed.csv"),
        help="Path to processed data CSV"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PATHS["results_dir"], "grid_search_full"),
        help="Directory for outputs"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target recall level (default: 0.95)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Proportion of data for test split"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of data for validation split"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = load_data(args.data)
    
    # Split data
    train, val, test = make_splits(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        stratify=True,
        seed=42
    )
    
    logger.info(f"Dataset: {len(df)} examples ({df['relevant'].sum()} relevant)")
    logger.info(f"Train: {len(train)} examples ({train['relevant'].sum()} relevant)")
    logger.info(f"Validation: {len(val)} examples ({val['relevant'].sum()} relevant)")
    logger.info(f"Test: {len(test)} examples ({test['relevant'].sum()} relevant)")
    
    # Run grid search on training data
    X_train = train.drop('relevant', axis=1)
    y_train = train['relevant']
    
    # Record start time
    start_time = datetime.now()
    logger.info(f"Starting grid search at {start_time}")
    
    # Run comprehensive grid search
    best_model = run_grid_search(
        X_train,
        y_train,
        args.output_dir,
        cv=args.cv,
        target_recall=args.target_recall
    )
    
    # Record end time and duration
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Completed grid search at {end_time}, duration: {duration}")
    
    # Evaluate on validation set
    logger.info("Evaluating best model on validation set")
    X_val = val.drop('relevant', axis=1)
    y_val = val['relevant']
    
    val_preds = best_model.predict(X_val)
    val_probs = best_model.predict_proba(X_val)[:, 1]
    
    val_metrics = evaluate(
        y_val,
        val_preds,
        val_probs,
        os.path.join(args.output_dir, "metrics"),
        "validation",
        target_recall=args.target_recall
    )
    
    # Evaluate on test set
    logger.info("Evaluating best model on test set")
    X_test = test.drop('relevant', axis=1)
    y_test = test['relevant']
    
    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)[:, 1]
    
    test_metrics = evaluate(
        y_test,
        test_preds,
        test_probs,
        os.path.join(args.output_dir, "metrics"),
        "test",
        target_recall=args.target_recall
    )
    
    # Update performance summary with final metrics
    with open(os.path.join(args.output_dir, "performance_summary.txt"), "a") as f:
        f.write("\n\nValidation metrics:\n")
        for metric, value in val_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\nTest metrics:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    logger.info("Grid search and evaluation complete.")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()