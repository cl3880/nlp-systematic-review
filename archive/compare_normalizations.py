#!/usr/bin/env python3
"""
Compare text normalization techniques using independent grid searches for each method.
This ensures fair comparison by allowing each technique to use its optimal hyperparameters.
"""
import os
import argparse
import logging
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.utils.evaluate import evaluate, compare_models
from src.models.text_processors import NormalizingTextCombiner
from src.models.model_factory import create_pipeline
from src.models.param_grids import get_param_grid

# Configure logging
log_file = os.path.join(PATHS["logs_dir"], "normalization_comparison.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_normalized_pipeline(model_type, technique, text_columns=["title", "abstract"]):
    """Create a pipeline with specified normalization technique.
    
    Args:
        model_type: Type of classifier to use
        technique: Normalization technique (None, 'stemming', or 'lemmatization')
        text_columns: List of text column names to combine for feature extraction
        
    Returns:
        Configured sklearn Pipeline object
    """
    logger.info(f"Building {model_type} pipeline with technique={technique}")

    # Create base pipeline for the specified model type
    base_pipeline = create_pipeline(model_type)
    
    # Replace the text combiner with normalizing text combiner
    steps = base_pipeline.steps.copy()
    steps[0] = ("normalizer", NormalizingTextCombiner(text_columns=text_columns, technique=technique))
    
    return Pipeline(steps)


def run_grid_search(pipeline, X_train, y_train, param_grid, cv=5, output_dir=None, technique_name="baseline"):
    """Run grid search to find optimal parameters for a normalization technique.
    
    Args:
        pipeline: Base pipeline to use
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for search
        cv: Number of cross-validation folds
        output_dir: Directory to save results
        technique_name: Name of technique for logging
        
    Returns:
        tuple: (best_params, best_model, best_score, high_recall_params, high_recall_model, high_recall_score)
    """
    logger.info(f"Starting grid search for {technique_name}...")
    
    # Define scoring metrics
    scoring = {
        'f1': 'f1',             # For balanced model
        'recall': 'recall',     # For extracting high-recall model
        'precision': 'precision',  # For trade-off analysis
    }
    
    # Run grid search
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
    
    # Get best model and parameters
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_
    
    logger.info(f"Grid search complete for {technique_name}. Best F1: {best_score:.4f}")
    
    # Extract high-recall model
    high_recall_model, high_recall_params, high_recall_score = extract_high_recall_model(
        grid, X_train, y_train, target_recall=0.95
    )
    
    # Export CV results if output directory is provided
    if output_dir:
        cv_df = pd.DataFrame(grid.cv_results_)
        os.makedirs(output_dir, exist_ok=True)
        cv_path = os.path.join(output_dir, f"{technique_name}_cv_results.csv")
        cv_df.to_csv(cv_path, index=False)
        logger.info(f"Exported CV results to {cv_path}")
        
        # Save best parameters
        params_path = os.path.join(output_dir, f"{technique_name}_best_params.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
            
        # Save high-recall parameters
        hr_params_path = os.path.join(output_dir, f"{technique_name}_hr_params.json")
        with open(hr_params_path, 'w') as f:
            json.dump(high_recall_params, f, indent=2)
    
    return best_params, best_model, best_score, high_recall_params, high_recall_model, high_recall_score


def extract_high_recall_model(grid, X_train, y_train, target_recall=0.95):
    """Extract a high-recall model from grid search results."""
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
    best_score = f1s[idx]
    logger.info(f"Selected high-recall config with recall {recalls[idx]:.4f} and F1 {best_score:.4f}")
    
    # Create and fit high-recall model
    hr_model = clone(grid.best_estimator_)
    hr_model.set_params(**{k: v for k, v in best_params.items() if k in hr_model.get_params()})
    hr_model.fit(X_train, y_train)
    
    return hr_model, best_params, best_score


def evaluate_model(model, X, y, metrics_dir, name, target_recall=0.95):
    """Evaluate a model and save results."""
    logger.info(f"Evaluating {name} model...")
    
    # Generate predictions
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
    except (AttributeError, IndexError):
        probs = None
        logger.warning(f"Could not get prediction probabilities for {name}")
    
    # Get metrics
    metrics = evaluate(y, preds, probs, metrics_dir, name, target_recall=target_recall)
    
    # Save metrics
    metrics_path = os.path.join(metrics_dir, f"{name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics for {name}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    return metrics


def save_model(model, model_dir, name):
    """Save a model to disk."""
    model_path = os.path.join(model_dir, f"{name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")


def create_summary_report(results, output_dir, model_type):
    """Create a summary report of normalization comparison results."""
    with open(os.path.join(output_dir, "normalization_summary.md"), "w") as f:
        f.write(f"# Text Normalization Comparison for {model_type.upper()} Model\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Experimental Design\n\n")
        f.write("Each normalization technique was evaluated with its own optimized hyperparameters ")
        f.write("through an independent grid search. This ensures a fair comparison by allowing each technique ")
        f.write("to perform at its best rather than using hyperparameters optimized for non-normalized text.\n\n")
        
        f.write("## Balanced Model Results\n\n")
        f.write("| Technique | F1 | Precision | Recall | AUC | WSS@95 |\n")
        f.write("|-----------|----|-----------|---------|----|--------|\n")
        
        for tech, metrics in results["balanced"].items():
            display_name = tech.capitalize() if tech != "baseline" else "No normalization"
            f.write(f"| {display_name} | {metrics['f1']:.4f} | {metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | {metrics.get('roc_auc', 'N/A'):.4f} | {metrics.get('wss@95', 'N/A'):.4f} |\n")
        
        f.write("\n## High-Recall Model Results\n\n")
        f.write("| Technique | F1 | Precision | Recall | AUC | WSS@95 |\n")
        f.write("|-----------|----|-----------|---------|----|--------|\n")
        
        for tech, metrics in results["highrecall"].items():
            display_name = tech.capitalize() if tech != "baseline" else "No normalization"
            f.write(f"| {display_name} | {metrics['f1']:.4f} | {metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | {metrics.get('roc_auc', 'N/A'):.4f} | {metrics.get('wss@95', 'N/A'):.4f} |\n")
        
        # Find best techniques
        best_bal_f1 = max(results["balanced"].items(), key=lambda x: x[1]["f1"])[0]
        best_hr_f1 = max(results["highrecall"].items(), key=lambda x: x[1]["f1"])[0]
        best_bal_wss = max(results["balanced"].items(), key=lambda x: x[1].get("wss@95", 0))[0]
        best_hr_wss = max(results["highrecall"].items(), key=lambda x: x[1].get("wss@95", 0))[0]
        
        f.write("\n## Key Findings\n\n")
        f.write(f"- Best balanced model (F1): **{best_bal_f1.capitalize() if best_bal_f1 != 'baseline' else 'No normalization'}**\n")
        f.write(f"- Best high-recall model (F1): **{best_hr_f1.capitalize() if best_hr_f1 != 'baseline' else 'No normalization'}**\n")
        f.write(f"- Best work savings (balanced): **{best_bal_wss.capitalize() if best_bal_wss != 'baseline' else 'No normalization'}**\n")
        f.write(f"- Best work savings (high-recall): **{best_hr_wss.capitalize() if best_hr_wss != 'baseline' else 'No normalization'}**\n\n")
        
        f.write("## Detailed Analysis\n\n")
        f.write("### Hyperparameter Analysis\n\n")
        f.write("Different normalization techniques often benefit from different hyperparameter settings. ")
        f.write("Below are the optimal hyperparameters found for each technique:\n\n")
        
        for tech, params in results["best_params"].items():
            display_name = tech.capitalize() if tech != "baseline" else "No normalization"
            f.write(f"#### {display_name}\n\n")
            f.write("```json\n")
            f.write(json.dumps(params, indent=2))
            f.write("\n```\n\n")


def main():
    """Main execution function for normalization comparison."""
    parser = argparse.ArgumentParser(description="Compare text normalization approaches with fair methodology")
    parser.add_argument(
        "--data",
        default=os.path.join(PATHS["data_processed"], "data_final_processed.csv"),
        help="Path to processed data CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PATHS["grid_results_dir"], "normalization"),
        help="Directory for normalization outputs",
    )
    parser.add_argument(
        "--model-type",
        default="svm",
        choices=["logreg", "svm", "cnb", "cosine"],
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target recall for high-recall evaluation",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    args = parser.parse_args()

    # Prepare output dirs
    model_dir = os.path.join(args.output_dir, args.model_type)
    models_dir = os.path.join(model_dir, "models")
    metrics_dir = os.path.join(model_dir, "metrics")
    plots_dir = os.path.join(model_dir, "plots")
    cv_results_dir = os.path.join(model_dir, "cv_results")
    
    for directory in [models_dir, metrics_dir, plots_dir, cv_results_dir]:
        os.makedirs(directory, exist_ok=True)

    # Load and split data
    logger.info(f"Loading data from {args.data}")
    df = load_data(args.data)
    train, val, test = make_splits(df, stratify=True, seed=42)
    
    # Log dataset stats
    logger.info(f"Dataset: {len(df)} records, {df['relevant'].sum()} relevant ({df['relevant'].mean()*100:.1f}%)")
    logger.info(f"Train/Val/Test splits: {len(train)}/{len(val)}/{len(test)} records")

    # Define normalization techniques to evaluate
    techniques = ["baseline", "stemming", "lemmatization"]
    text_columns = ["title", "abstract"]
    
    # Save experiment configuration
    config = {
        "data_path": args.data,
        "model_type": args.model_type,
        "text_columns": text_columns,
        "normalization_techniques": techniques,
        "target_recall": args.target_recall,
        "cross_validation_folds": args.cv,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "class_distribution": {
            "total": {"relevant": int(df['relevant'].sum()), "total": len(df)},
            "train": {"relevant": int(train['relevant'].sum()), "total": len(train)},
            "val": {"relevant": int(val['relevant'].sum()), "total": len(val)},
            "test": {"relevant": int(test['relevant'].sum()), "total": len(test)}
        }
    }
    
    with open(os.path.join(model_dir, "experiment_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Get parameter grid for the model type
    param_grid = get_param_grid(args.model_type)
    logger.info(f"Using parameter grid with {len(param_grid)} parameters")
    
    # Results dictionaries to store metrics
    balanced_results = {}
    highrecall_results = {}
    best_params_dict = {}
    
    # Run grid search for each normalization technique
    for technique in techniques:
        technique_name = technique if technique != "baseline" else "baseline"
        logger.info(f"Processing {technique_name} normalization technique")
        
        # Create pipeline with appropriate normalization
        if technique == "baseline":
            pipeline = create_pipeline(args.model_type)
        else:
            pipeline = create_normalized_pipeline(args.model_type, technique, text_columns)
        
        # Run grid search
        best_params, best_model, best_score, hr_params, hr_model, hr_score = run_grid_search(
            pipeline=pipeline,
            X_train=train,
            y_train=train["relevant"],
            param_grid=param_grid,
            cv=args.cv,
            output_dir=cv_results_dir,
            technique_name=technique_name
        )
        
        # Save best parameters
        best_params_dict[technique_name] = best_params
        
        # Save models
        save_model(best_model, models_dir, f"{technique_name}_balanced")
        save_model(hr_model, models_dir, f"{technique_name}_highrecall")
        
        # Evaluate on test set
        bal_metrics = evaluate_model(
            best_model, test, test["relevant"], 
            metrics_dir, f"{technique_name}_balanced", 
            target_recall=args.target_recall
        )
        hr_metrics = evaluate_model(
            hr_model, test, test["relevant"], 
            metrics_dir, f"{technique_name}_highrecall", 
            target_recall=args.target_recall
        )
        
        # Store results
        balanced_results[technique_name] = bal_metrics
        highrecall_results[technique_name] = hr_metrics
    
    # Save combined results
    all_results = {
        "balanced": balanced_results,
        "highrecall": highrecall_results,
        "best_params": best_params_dict
    }
    with open(os.path.join(metrics_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comparison plots
    try:
        # Create plots for both settings
        balanced_models = [(f"{tech}_balanced", metrics) for tech, metrics in balanced_results.items()]
        compare_models(balanced_models, filename="balanced_comparison", base_dir=model_dir)
        
        highrecall_models = [(f"{tech}_highrecall", metrics) for tech, metrics in highrecall_results.items()]
        compare_models(highrecall_models, filename="highrecall_comparison", base_dir=model_dir)
        logger.info("Generated comparison plots")
    except Exception as e:
        logger.error(f"Error generating comparison plots: {e}")
    
    # Create summary report
    create_summary_report(all_results, model_dir, args.model_type)
    logger.info(f"Normalization comparison complete. Results saved to {model_dir}")


if __name__ == "__main__":
    main()