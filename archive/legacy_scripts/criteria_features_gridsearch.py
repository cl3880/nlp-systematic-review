import os
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import PATHS, get_result_path_v2
from src.utils.data_utils import load_data, make_splits
from src.models.criteria_features_model import create_criteria_pipeline
from src.models.param_grids import criteria_param_grid
from src.utils.logging_utils import setup_per_model_logging
from src.scripts.baseline_grid_search import (
    extract_high_recall_model, 
    optimize_threshold_for_recall, 
    evaluate_models,
    compute_metrics
)
from src.utils.result_utils import save_model_results

logger = logging.getLogger(__name__)

def run_criteria_experiment(
    data_path,
    output_dir,
    model_type="logreg",
    use_mesh=False,
    normalization=None,
    cv=5,
    debug=False
):
    log_level = logging.DEBUG if debug else logging.INFO
    model_name = f"criteria_{model_type}"
    if normalization:
        model_name = f"{normalization}_{model_name}"
    if use_mesh:
        model_name = f"{model_name}_mesh"
    
    logger = setup_per_model_logging(model_name, level=log_level)
    logger.info(f"Starting criteria-enhanced experiment: {model_name}")
    
    df = load_data(data_path)
    train, val, test = make_splits(df, test_size=0.1, val_size=0.1, stratify=True, seed=42)
    
    if normalization:
        from src.scripts.baseline_grid_search import preprocess_corpus
        logger.info(f"Applying {normalization} normalization...")
        train = preprocess_corpus(train, technique=normalization)
        val = preprocess_corpus(val, technique=normalization)
        text_columns = ['normalized_text']
    else:
        text_columns = ['title', 'abstract']
    
    X_train = train.drop('relevant', axis=1)
    y_train = train['relevant']
    X_val = val.drop('relevant', axis=1)
    y_val = val['relevant']
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    available_columns = df.columns.tolist()
    pipeline = create_criteria_pipeline(
        model_type=model_type,
        text_columns=text_columns,
        use_mesh=use_mesh,
        available_columns=available_columns
    )
    
    param_grid = criteria_param_grid(model_type)
    
    scoring = {
        'f1': 'f1',
        'recall': 'recall',
        'precision': 'precision'
    }
    
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
    
    logger.info("Starting grid search...")
    grid.fit(X_train, y_train)
    
    logger.info(f"Grid search complete. Best F1: {grid.best_score_:.4f}")
    logger.info(f"Best parameters: {grid.best_params_}")
    
    cv_results = pd.DataFrame(grid.cv_results_)
    os.makedirs(output_dir, exist_ok=True)
    cv_results.to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    
    balanced_model = grid.best_estimator_
    balanced_params = grid.best_params_
    
    high_recall_model, high_recall_params = extract_high_recall_model(
        grid, X_train, y_train, target_recall=0.95
    )
    
    bal_metrics, hr_metrics, threshold, y_prob_bal, y_prob_hr = evaluate_models(
        balanced_model, high_recall_model, X_val, y_val
    )
    
    ngram_analysis = {}
    try:
        if 'param_features__text_features__tfidf__ngram_range' in cv_results.columns:
            ngram_col = 'param_features__text_features__tfidf__ngram_range'
            ngram_groups = cv_results.groupby(ngram_col)['mean_test_f1'].mean()
            ngram_analysis = {eval(k) if isinstance(k, str) else k: v 
                            for k, v in ngram_groups.to_dict().items()}
            
            logger.info("N-gram range analysis:")
            for ngram, f1 in sorted(ngram_analysis.items()):
                logger.info(f"  {ngram}: average F1 = {f1:.4f}")
    except Exception as e:
        logger.warning(f"Could not perform n-gram analysis: {e}")
    
    save_model_results(
        balanced_model, high_recall_model,
        bal_metrics, hr_metrics,
        balanced_params, high_recall_params,
        threshold, output_dir, model_name,
        X_val, y_val, y_prob_bal, y_prob_hr,
        ngram_analysis
    )
    
    logger.info(f"Experiment complete. Results saved to {output_dir}")
    return grid, balanced_model, high_recall_model

def main():
    parser = argparse.ArgumentParser(
        description="Run criteria-enhanced grid search experiment"
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
        "--model", 
        default="logreg", 
        choices=["logreg", "svm"],
        help="Type of model ('logreg', 'svm')"
    )
    parser.add_argument(
        "--normalization", 
        default=None, 
        choices=[None, "stemming", "lemmatization"],
        help="Text normalization technique (None, 'stemming', or 'lemmatization')"
    )
    parser.add_argument(
        "--use-mesh", 
        action="store_true",
        help="Include MeSH terms as features (if available)"
    )
    parser.add_argument(
        "--cv", 
        type=int, 
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug-level logging"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        model_name = f"criteria_{args.model}"
        if args.normalization:
            model_name = f"{args.normalization}_{model_name}"
        if args.use_mesh:
            model_name = f"{model_name}_mesh"
        output_dir = get_result_path_v2(model_name)
    else:
        output_dir = args.output
    
    run_criteria_experiment(
        args.data,
        output_dir,
        model_type=args.model,
        use_mesh=args.use_mesh,
        normalization=args.normalization,
        cv=args.cv,
        debug=args.debug
    )

if __name__ == "__main__":
    main()