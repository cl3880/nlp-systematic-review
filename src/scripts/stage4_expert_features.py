import os
import argparse
import logging
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.models.criteria_features_model import create_criteria_pipeline
from src.utils.logging_utils import setup_per_model_logging

logger = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Calculate F2 score (weighs recall higher than precision)
    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
    
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
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'roc_auc': roc_auc,
        'wss_at_95': best_wss
    }

def optimize_threshold_for_recall(y_true, y_prob, target_recall=0.95):
    """Find optimal threshold to achieve target recall."""
    thresholds = np.sort(y_prob)
    best_threshold = 0
    best_diff = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        diff = abs(recall - target_recall)
        if diff < best_diff and recall >= target_recall:
            best_diff = diff
            best_threshold = threshold
            
    return best_threshold

def create_output_directory(model_name):
    """Create an output directory with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Try common results directories
    for results_dir in ["results/v2.3", "results/v2.2.1-skf", "results/v2", "results"]:
        if os.path.exists(results_dir):
            output_dir = os.path.join(results_dir, f"{model_name}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
    
    # Fallback to a new results directory
    output_dir = os.path.join("results", f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def run_criteria_experiment_no_grid(
    data_path,
    output_dir=None,
    model_type="svm",
    use_mesh=False,
    normalization=None,
    debug=False,
    save_features=False,
    balancing=None
):
    log_level = logging.DEBUG if debug else logging.INFO
    model_name = f"criteria_{model_type}_direct"
    if normalization:
        model_name = f"{normalization}_{model_name}"
    if use_mesh:
        model_name = f"{model_name}_mesh"
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = create_output_directory(model_name)
        
    # Setup logging without log_path parameter
    logger = setup_per_model_logging(model_name, level=log_level)
    logger.info(f"Starting criteria-enhanced direct experiment: {model_name}")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Configure file handler for logging
    file_handler = logging.FileHandler(os.path.join(output_dir, f"{model_name}.log"))
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
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
        available_columns=available_columns,
        balancing=balancing,
    )
    
    # Using baseline SVM parameters directly instead of grid search
    # These parameters are based on the provided baseline SVM configuration
    baseline_params = {
        "features__text_features__tfidf__max_features": 10000,
        "features__text_features__tfidf__ngram_range": (1, 3),
        "features__text_features__tfidf__min_df": 5,
        "features__text_features__tfidf__max_df": 0.85,
        "clf__C": 1,
        "clf__class_weight": "balanced",
        "clf__kernel": "linear"
    }
    
    # Set parameters directly
    logger.info("Setting baseline parameters:")
    for param, value in baseline_params.items():
        logger.info(f"  {param}: {value}")
        try:
            pipeline.set_params(**{param: value})
        except Exception as e:
            logger.warning(f"Could not set parameter {param}: {e}")
    
    logger.info("Fitting model with baseline parameters...")
    pipeline.fit(X_train, y_train)
    import joblib
    joblib.dump(pipeline, os.path.join(output_dir, "pipeline.joblib"))
    logger.info("Model saved to {os.path.join(output_dir, 'model.joblib)}")
    
    # Save featurized data if requested
    if save_features:
        logger.info("Saving featurized data...")
        try:
            # Get the features from each step
            text_features = pipeline.named_steps['features'].transformer_list[0][1].transform(X_train)
            criteria_features = pipeline.named_steps['features'].transformer_list[1][1].transform(X_train)
            
            # Save to files
            if isinstance(text_features, np.ndarray):
                np.save(os.path.join(output_dir, "text_features_train.npy"), text_features)
            elif hasattr(text_features, "toarray"):
                np.save(os.path.join(output_dir, "text_features_train.npy"), text_features.toarray())
            
            np.save(os.path.join(output_dir, "criteria_features_train.npy"), criteria_features)
            
            # Save validation features too
            text_features_val = pipeline.named_steps['features'].transformer_list[0][1].transform(X_val)
            criteria_features_val = pipeline.named_steps['features'].transformer_list[1][1].transform(X_val)
            
            if isinstance(text_features_val, np.ndarray):
                np.save(os.path.join(output_dir, "text_features_val.npy"), text_features_val)
            elif hasattr(text_features_val, "toarray"):
                np.save(os.path.join(output_dir, "text_features_val.npy"), text_features_val.toarray())
                
            np.save(os.path.join(output_dir, "criteria_features_val.npy"), criteria_features_val)
            
            # Save the labels
            np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
            np.save(os.path.join(output_dir, "y_val.npy"), y_val.values)
            
            logger.info("Featurized data saved successfully.")
        except Exception as e:
            logger.error(f"Error saving featurized data: {e}")
    
    # Compute predictions and metrics on validation set
    logger.info("Evaluating model on validation set...")
    y_prob_val = pipeline.predict_proba(X_val)[:, 1]
    y_pred_val = pipeline.predict(X_val)
    balanced_metrics = compute_metrics(y_val, y_pred_val, y_prob_val)
    
    # Find optimal threshold for high recall
    logger.info("Finding optimal threshold for high recall...")
    threshold = optimize_threshold_for_recall(y_val, y_prob_val, target_recall=0.95)
    logger.info(f"Optimal threshold for 95% recall: {threshold:.4f}")
    
    # Create high-recall predictions
    y_pred_hr = (y_prob_val >= threshold).astype(int)
    hr_metrics = compute_metrics(y_val, y_pred_hr, y_prob_val)
    
    # Print metrics
    logger.info("Balanced model performance:")
    for metric, value in balanced_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("High-recall model performance:")
    for metric, value in hr_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "model_name": model_name,
        "balanced_metrics": balanced_metrics,
        "hr_metrics": hr_metrics,
        "hr_threshold": threshold,
        "balanced_params": baseline_params,
    }
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': y_val,
        'predicted_label': y_pred_val,
        'probability': y_prob_val,
        'high_recall_pred': y_pred_hr
    })
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    # Save metrics
    pd.DataFrame([{
        'model': model_name,
        'precision': balanced_metrics['precision'],
        'recall': balanced_metrics['recall'],
        'f1': balanced_metrics['f1'],
        'f2': balanced_metrics['f2'],
        'roc_auc': balanced_metrics['roc_auc'],
        'wss_at_95': balanced_metrics['wss_at_95'],
        'hr_precision': hr_metrics['precision'],
        'hr_recall': hr_metrics['recall'],
        'hr_f1': hr_metrics['f1'],
        'hr_f2': hr_metrics['f2'],
        'hr_roc_auc': hr_metrics['roc_auc'],
        'hr_wss_at_95': hr_metrics['wss_at_95'],
        'threshold': threshold
    }]).to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)
    
    logger.info(f"Experiment complete. Results saved to {output_dir}")
    return pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run criteria-enhanced direct experiment with baseline parameters"
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
        default="svm", 
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
        "--save-features",
        action="store_true",
        help="Save the featurized data for future use"
    )
    parser.add_argument(
        "--balancing",
        default=None,
        choices=[None, "smote"],
        help="Technique to apply for class balancing (currently supports 'smote')"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug-level logging"
    )
    
    args = parser.parse_args()
    
    run_criteria_experiment_no_grid(
        args.data,
        args.output,
        model_type=args.model,
        use_mesh=args.use_mesh,
        normalization=args.normalization,
        debug=args.debug,
        save_features=args.save_features,
        balancing=args.balancing
    )

if __name__ == "__main__":
    main()