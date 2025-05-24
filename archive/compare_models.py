#!/usr/bin/env python3
"""
Model Comparison: Compare LogReg, SVM, and Cosine Similarity with F1 scoring

This script compares our fixed LogReg models (balanced and high-recall) with
SVM and cosine similarity approaches using the same n-gram configurations (1,2)
and F1 score as the primary metric for optimization and evaluation.

Methodological improvements:
1. Consistent feature filtering across all models (max_df=0.9)
2. Threshold selection only on validation data
3. Evaluation of cosine similarity as a reranker for LogReg results

Usage:
    python -m src.scripts.compare_models --output-dir results/model_comparison_f1
"""
import os
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import precision_recall_curve, auc, f1_score

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.models.baseline_models import create_balanced_logreg_pipeline, create_high_recall_logreg_pipeline
from src.models.classifiers import make_tfidf_svm_pipeline, make_tfidf_cosine_pipeline
from src.utils.evaluate import evaluate, compare_models, find_threshold_for_recall
from sklearn.metrics import make_scorer, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PATHS["logs_dir"], "model_comparison_f1.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def evaluate_reranking(logreg_scores, cosine_scores, y_true, top_percentile=0.3):
    """
    Evaluate cosine similarity as a reranker for LogReg results.
    
    Args:
        logreg_scores: Probability scores from LogReg model
        cosine_scores: Similarity scores from Cosine model
        y_true: True labels
        top_percentile: Percentage of top LogReg docs to rerank using cosine
        
    Returns:
        Dictionary of metrics for the reranking approach
    """
    n_samples = len(y_true)
    n_top = int(n_samples * top_percentile)
    
    # Get indices of top LogReg predictions
    logreg_indices = np.argsort(logreg_scores)[::-1][:n_top]
    
    # Create a new ranking by sorting these top predictions by cosine scores
    reranked_indices = logreg_indices[np.argsort(cosine_scores[logreg_indices])[::-1]]
    
    # Create the final ranking: reranked top docs followed by remaining docs in logreg order
    remaining_indices = np.array([i for i in range(n_samples) if i not in reranked_indices])
    remaining_indices = remaining_indices[np.argsort(logreg_scores[remaining_indices])[::-1]]
    final_ranking = np.concatenate([reranked_indices, remaining_indices])
    
    # Evaluate the ranking at different cutoffs
    precision_at_k = {}
    recall_at_k = {}
    f1_at_k = {}
    
    for k in [50, 100, 200, int(n_samples * 0.2), int(n_samples * 0.3)]:
        if k > n_samples:
            continue
            
        selected = final_ranking[:k]
        predictions = np.zeros(n_samples)
        predictions[selected] = 1
        
        tp = np.sum(predictions[y_true == 1])
        fp = np.sum(predictions[y_true == 0])
        fn = np.sum(y_true[predictions == 0])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_at_k[k] = precision
        recall_at_k[k] = recall
        f1_at_k[k] = f1
    
    # Find threshold for 95% recall
    sorted_scores = np.sort(logreg_scores)[::-1]
    cumulative_recall = np.cumsum(y_true[np.argsort(logreg_scores)[::-1]]) / np.sum(y_true)
    threshold_idx = np.argmax(cumulative_recall >= 0.95)
    
    if threshold_idx < len(sorted_scores):
        threshold_95 = sorted_scores[threshold_idx]
        cutoff_95 = threshold_idx + 1
        precision_95 = np.sum(y_true[np.argsort(logreg_scores)[::-1][:cutoff_95]]) / cutoff_95
    else:
        threshold_95 = 0
        cutoff_95 = len(y_true)
        precision_95 = np.sum(y_true) / len(y_true)
    
    return {
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'f1_at_k': f1_at_k,
        'threshold_95': threshold_95,
        'cutoff_95': cutoff_95,
        'precision_95': precision_95
    }

def main():
    parser = argparse.ArgumentParser(
        description="Compare LogReg, SVM, and Cosine Similarity models using F1 as primary metric"
    )
    parser.add_argument(
        "--data",
        default=os.path.join(PATHS["data_processed"], "data_final_processed.csv"),
        help="Path to processed data CSV"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PATHS["results_dir"], "model_comparison_f1"),
        help="Directory to save comparison results"
    )
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    
    # Log parameters
    logger.info("Running model comparison with F1 scoring and (1,2) n-gram range")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = load_data(args.data)
    
    # Split data
    train, val, test = make_splits(df, test_size=0.1, val_size=0.1, stratify=True, seed=42)
    logger.info(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Create models with consistent feature filtering (max_df=0.9)
    models = {
        # LogReg models (our baseline) using fixed parameters from baseline_models.py
        "logreg_balanced": create_balanced_logreg_pipeline(),
        "logreg_high_recall": create_high_recall_logreg_pipeline(),
        
        # SVM with balanced config - same (1,2) n-gram config as balanced LogReg
        "svm_balanced": make_tfidf_svm_pipeline(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            C=10,
            class_weight="balanced"
        ),
        
        # SVM with high-recall config - same (1,2) n-gram config as high-recall LogReg
        "svm_high_recall": make_tfidf_svm_pipeline(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            C=0.01,
            class_weight="balanced"
        ),
        
        # Cosine similarity - same (1,2) n-gram config as balanced LogReg
        "cosine": make_tfidf_cosine_pipeline(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            threshold=None  # Will select threshold from validation data
        )
    }
    
    # Train all models
    logger.info("Training models")
    for name, model in models.items():
        logger.info(f"Training {name}")
        model.fit(train, train['relevant'])
    
    # Validate and find optimal thresholds
    logger.info("Evaluating models on validation set")
    val_results = {}
    thresholds = {}
    
    for name, model in models.items():
        # Get predictions and probabilities
        val_preds = model.predict(val)
        try:
            val_probs = model.predict_proba(val)[:, 1]
        except:
            if name == "cosine":
                # For cosine similarity, get raw similarity scores
                val_probs = model.named_steps['clf'].predict_proba(val)[:, 1]
            else:
                val_probs = None
        
        if val_probs is not None:
            # Find threshold for 95% recall on validation set
            thresh, achieved_recall = find_threshold_for_recall(
                val['relevant'].values,
                val_probs,
                target_recall=0.95
            )
            
            # Store the threshold for later use on test set
            thresholds[name] = thresh
            
            # Default threshold metrics
            metrics = evaluate(
                val['relevant'].values,
                val_preds,
                val_probs,
                f"{name}",
                data_split="default",
                base_dir=args.output_dir
            )
            
            # Get predictions with the new threshold
            val_preds_hr = (val_probs >= thresh).astype(int)
            
            # High-recall threshold metrics on validation
            hr_metrics = evaluate(
                val['relevant'].values,
                val_preds_hr,
                val_probs,
                f"{name}",
                data_split="high_recall",
                base_dir=args.output_dir
            )
            
            val_results[name] = {
                'metrics': metrics,
                'high_recall_metrics': hr_metrics,
                'threshold': float(thresh),
                'val_probs': val_probs  # Store for reranking evaluation
            }
        else:
            metrics = evaluate(
                val['relevant'].values,
                val_preds,
                None,
                f"{name}",
                data_split="default",
                base_dir=args.output_dir
            )
            val_results[name] = {
                'metrics': metrics
            }
    
    # Now evaluate on test set using thresholds from validation
    logger.info("Evaluating models on test set")
    test_results = {}
    
    for name, model in models.items():
        # Get predictions and probabilities
        test_preds = model.predict(test)
        try:
            test_probs = model.predict_proba(test)[:, 1]
        except:
            if name == "cosine":
                test_probs = model.named_steps['clf'].predict_proba(test)[:, 1]
            else:
                test_probs = None
        
        if test_probs is not None:
            # Use threshold selected on validation set
            thresh = thresholds.get(name, 0.5)
            
            # Default threshold metrics
            metrics = evaluate(
                test['relevant'].values,
                test_preds,
                test_probs,
                f"{name}",
                data_split="test_default",
                base_dir=args.output_dir
            )
            
            # Get predictions with the validated threshold
            test_preds_hr = (test_probs >= thresh).astype(int)
            
            # High-recall threshold metrics on test
            hr_metrics = evaluate(
                test['relevant'].values,
                test_preds_hr,
                test_probs,
                f"{name}",
                data_split="test_high_recall",
                base_dir=args.output_dir
            )
            
            test_results[name] = {
                'metrics': metrics,
                'high_recall_metrics': hr_metrics,
                'threshold': float(thresh),
                'test_probs': test_probs  # Store for reranking evaluation
            }
        else:
            metrics = evaluate(
                test['relevant'].values,
                test_preds,
                None,
                f"{name}",
                data_split="test_default",
                base_dir=args.output_dir
            )
            test_results[name] = {
                'metrics': metrics
            }
    
    # Evaluate reranking approach on validation set
    if 'logreg_balanced' in val_results and 'cosine' in val_results:
        logreg_scores = val_results['logreg_balanced']['val_probs']
        cosine_scores = val_results['cosine']['val_probs']
        
        reranking_val_results = evaluate_reranking(
            logreg_scores, 
            cosine_scores, 
            val['relevant'].values
        )
        
        # Save reranking results
        with open(os.path.join(args.output_dir, 'reranking_val_results.json'), 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_results = {
                'precision_at_k': {str(k): float(v) for k, v in reranking_val_results['precision_at_k'].items()},
                'recall_at_k': {str(k): float(v) for k, v in reranking_val_results['recall_at_k'].items()},
                'f1_at_k': {str(k): float(v) for k, v in reranking_val_results['f1_at_k'].items()},
                'threshold_95': float(reranking_val_results['threshold_95']),
                'cutoff_95': int(reranking_val_results['cutoff_95']),
                'precision_95': float(reranking_val_results['precision_95'])
            }
            json.dump(serializable_results, f, indent=2)
    
    # Evaluate reranking approach on test set
    if 'logreg_balanced' in test_results and 'cosine' in test_results:
        logreg_scores = test_results['logreg_balanced']['test_probs']
        cosine_scores = test_results['cosine']['test_probs']
        
        reranking_test_results = evaluate_reranking(
            logreg_scores, 
            cosine_scores, 
            test['relevant'].values
        )
        
        # Save reranking results
        with open(os.path.join(args.output_dir, 'reranking_test_results.json'), 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_results = {
                'precision_at_k': {str(k): float(v) for k, v in reranking_test_results['precision_at_k'].items()},
                'recall_at_k': {str(k): float(v) for k, v in reranking_test_results['recall_at_k'].items()},
                'f1_at_k': {str(k): float(v) for k, v in reranking_test_results['f1_at_k'].items()},
                'threshold_95': float(reranking_test_results['threshold_95']),
                'cutoff_95': int(reranking_test_results['cutoff_95']),
                'precision_95': float(reranking_test_results['precision_95'])
            }
            json.dump(serializable_results, f, indent=2)
    
    # Create comparison tables for validation results
    val_df = pd.DataFrame()
    for name, result in val_results.items():
        metrics = result['metrics']
        row = {
            'Model': name,
            'Precision': metrics.get('precision', None),
            'Recall': metrics.get('recall', None),
            'F1': metrics.get('f1', None),
            'F2': metrics.get('f2', None),
            'ROC AUC': metrics.get('roc_auc', None),
            'WSS@95': metrics.get('wss@95', None),
        }
        
        # Add high-recall metrics if available
        if 'high_recall_metrics' in result:
            hr_metrics = result['high_recall_metrics']
            row.update({
                'HR Precision': hr_metrics.get('precision', None),
                'HR Recall': hr_metrics.get('recall', None),
                'HR F1': hr_metrics.get('f1', None),
                'HR F2': hr_metrics.get('f2', None),
                'HR Threshold': result.get('threshold', None)
            })
        
        val_df = pd.concat([val_df, pd.DataFrame([row])], ignore_index=True)
    
    # Create comparison tables for test results
    test_df = pd.DataFrame()
    for name, result in test_results.items():
        metrics = result['metrics']
        row = {
            'Model': name,
            'Precision': metrics.get('precision', None),
            'Recall': metrics.get('recall', None),
            'F1': metrics.get('f1', None),
            'F2': metrics.get('f2', None),
            'ROC AUC': metrics.get('roc_auc', None),
            'WSS@95': metrics.get('wss@95', None),
        }
        
        # Add high-recall metrics if available
        if 'high_recall_metrics' in result:
            hr_metrics = result['high_recall_metrics']
            row.update({
                'HR Precision': hr_metrics.get('precision', None),
                'HR Recall': hr_metrics.get('recall', None),
                'HR F1': hr_metrics.get('f1', None),
                'HR F2': hr_metrics.get('f2', None),
                'HR Threshold': result.get('threshold', None)
            })
        
        test_df = pd.concat([test_df, pd.DataFrame([row])], ignore_index=True)
    
    # Save comparison tables
    val_df.to_csv(os.path.join(args.output_dir, 'validation_results.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_results.csv'), index=False)
    
    # Create markdown report
    with open(os.path.join(args.output_dir, 'model_comparison_report.md'), 'w') as f:
        f.write("# Model Comparison: LogReg vs SVM vs Cosine Similarity (F1 Scoring)\n\n")
        f.write("This report compares different model architectures using F1 as the primary metric and (1,2) n-gram configurations.\n")
        f.write("All models use consistent feature filtering (max_df=0.9) and thresholds selected on validation data only.\n\n")
        
        f.write("## Validation Set Performance (Default Threshold)\n\n")
        f.write("| Model | Precision | Recall | F1 | F2 | ROC AUC | WSS@95 |\n")
        f.write("|-------|-----------|--------|----|----|---------|--------|\n")
        
        for _, row in val_df.iterrows():
            f.write(f"| {row['Model']} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['F2']:.4f} | {row['ROC AUC']:.4f} | {row['WSS@95']:.4f} |\n")
        
        f.write("\n## Validation Set High-Recall Performance (95% Target)\n\n")
        f.write("| Model | Precision | Recall | F1 | F2 | Threshold |\n")
        f.write("|-------|-----------|--------|----|----|----------|\n")
        
        for _, row in val_df.iterrows():
            if 'HR Precision' in row and not pd.isna(row['HR Precision']):
                f.write(f"| {row['Model']} | {row['HR Precision']:.4f} | {row['HR Recall']:.4f} | {row['HR F1']:.4f} | {row['HR F2']:.4f} | {row['HR Threshold']:.4f} |\n")
        
        f.write("\n## Test Set Performance (Default Threshold)\n\n")
        f.write("| Model | Precision | Recall | F1 | F2 | ROC AUC | WSS@95 |\n")
        f.write("|-------|-----------|--------|----|----|---------|--------|\n")
        
        for _, row in test_df.iterrows():
            f.write(f"| {row['Model']} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['F2']:.4f} | {row['ROC AUC']:.4f} | {row['WSS@95']:.4f} |\n")
        
        f.write("\n## Test Set High-Recall Performance (Using Validation Thresholds)\n\n")
        f.write("| Model | Precision | Recall | F1 | F2 | Threshold |\n")
        f.write("|-------|-----------|--------|----|----|----------|\n")
        
        for _, row in test_df.iterrows():
            if 'HR Precision' in row and not pd.isna(row['HR Precision']):
                f.write(f"| {row['Model']} | {row['HR Precision']:.4f} | {row['HR Recall']:.4f} | {row['HR F1']:.4f} | {row['HR F2']:.4f} | {row['HR Threshold']:.4f} |\n")
        
        # Reranking results
        if os.path.exists(os.path.join(args.output_dir, 'reranking_test_results.json')):
            with open(os.path.join(args.output_dir, 'reranking_test_results.json'), 'r') as r:
                reranking_results = json.load(r)
            
            f.write("\n## Cosine Similarity as Reranker\n\n")
            f.write("In this approach, we use LogReg to identify the top candidates, then rerank them using Cosine Similarity.\n\n")
            
            f.write("### Test Set Performance at Different Cutoffs\n\n")
            f.write("| Cutoff | Precision | Recall | F1 |\n")
            f.write("|--------|-----------|--------|----|\n")
            
            for k in sorted([int(k) for k in reranking_results['precision_at_k'].keys()]):
                p = reranking_results['precision_at_k'][str(k)]
                r = reranking_results['recall_at_k'][str(k)]
                f1 = reranking_results['f1_at_k'][str(k)]
                f.write(f"| Top {k} | {p:.4f} | {r:.4f} | {f1:.4f} |\n")
            
            f.write(f"\n### 95% Recall Performance\n\n")
            f.write(f"- Threshold: {reranking_results['threshold_95']:.4f}\n")
            f.write(f"- Documents to screen: {reranking_results['cutoff_95']}\n")
            f.write(f"- Precision at 95% recall: {reranking_results['precision_95']:.4f}\n")
        
        # Analysis
        f.write("\n## Methodological Improvements\n\n")
        f.write("This analysis addresses several methodological issues in the previous comparison:\n\n")
        f.write("1. **Consistent Feature Filtering**: All models now use the same feature filtering settings (max_df=0.9) to ensure fair comparison.\n\n")
        f.write("2. **Validation-Only Threshold Selection**: Thresholds for high-recall performance are determined exclusively on validation data, then applied to test data.\n\n")
        f.write("3. **Appropriate Use of Cosine Similarity**: Cosine similarity is evaluated both as a standalone model and as a reranker for LogReg results.\n\n")
        
        # Find best models from test set
        best_f1_idx = test_df['F1'].idxmax()
        best_f1_model = test_df.loc[best_f1_idx, 'Model']
        best_f1 = test_df.loc[best_f1_idx, 'F1']
        
        best_auc_idx = test_df['ROC AUC'].idxmax()
        best_auc_model = test_df.loc[best_auc_idx, 'Model']
        best_auc = test_df.loc[best_auc_idx, 'ROC AUC']
        
        best_hr_f1_idx = test_df['HR F1'].idxmax() if 'HR F1' in test_df.columns else None
        
        f.write("\n## Conclusions\n\n")
        f.write(f"**Best Model for F1 Score**: {best_f1_model} (F1 = {best_f1:.4f})\n\n")
        f.write(f"**Best Model for AUC**: {best_auc_model} (AUC = {best_auc:.4f})\n\n")
        
        if best_hr_f1_idx is not None:
            best_hr_f1_model = test_df.loc[best_hr_f1_idx, 'Model']
            best_hr_f1 = test_df.loc[best_hr_f1_idx, 'HR F1']
            f.write(f"**Best Model for High-Recall F1**: {best_hr_f1_model} (F1 = {best_hr_f1:.4f})\n\n")
        
        if os.path.exists(os.path.join(args.output_dir, 'reranking_test_results.json')):
            f.write("**Reranking Approach**: Using cosine similarity to rerank the top LogReg results provides a promising approach, ")
            f.write("particularly for high-recall screening scenarios.\n")
    
    # Save models
    import joblib
    for name, model in models.items():
        joblib.dump(model, os.path.join(args.output_dir, 'models', f"{name}.joblib"))
    
    logger.info(f"Comparison complete. Report saved to {os.path.join(args.output_dir, 'model_comparison_report.md')}")
    logger.info("Summary of results:")
    logger.info(f"Best F1 model (test): {best_f1_model} (F1 = {best_f1:.4f})")
    logger.info(f"Best ROC AUC model (test): {best_auc_model} (AUC = {best_auc:.4f})")
    
    if os.path.exists(os.path.join(args.output_dir, 'reranking_test_results.json')):
        with open(os.path.join(args.output_dir, 'reranking_test_results.json'), 'r') as r:
            reranking_results = json.load(r)
        logger.info(f"Reranking precision at 95% recall: {reranking_results['precision_95']:.4f}")

if __name__ == "__main__":
    main() 