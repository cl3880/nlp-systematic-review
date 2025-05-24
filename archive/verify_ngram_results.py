#!/usr/bin/env python3
"""
Verify N-gram Results

This script performs a direct comparison between (1,2) and (1,3) n-gram configurations 
using cross-validation to verify our grid search findings.

Usage:
    python -m src.scripts.verify_ngram_results
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, roc_auc_score

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits
from src.models.model_factory import create_model

# Configure logging
log_path = os.path.join(PATHS['logs_dir'], 'verify_ngram.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def main():
    """Run verification of n-gram configurations"""
    # Load data
    logger.info("Loading processed data")
    data_path = os.path.join(PATHS['data_processed'], 'data_final_processed.csv')
    df = load_data(data_path)
    
    # Split data
    train, val, _ = make_splits(df, test_size=0.1, val_size=0.1, stratify=True, seed=42)
    logger.info(f"Split sizes: train={len(train)}, val={len(val)}")
    
    # Define n-gram configurations to test
    configs = [
        {
            'name': 'Unigrams+Bigrams (1,2)',
            'ngram_range': (1, 2),
            'color': 'blue'
        },
        {
            'name': 'Unigrams+Bigrams+Trigrams (1,3)',
            'ngram_range': (1, 3),
            'color': 'red'
        }
    ]
    
    # Use best hyperparameters from grid search for balanced model
    balanced_params = {
        'max_features': 5000,
        'min_df': 2,
        'max_df': 0.9,
        'C': 10,
    }
    
    # Use best hyperparameters from grid search for high-recall model
    high_recall_params = {
        'max_features': 10000,
        'min_df': 1,
        'max_df': 0.9,
        'C': 0.01,
    }
    
    # Define scoring metrics
    scoring = {
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc'
    }
    
    # Create CV splitter
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Results storage
    results = []
    
    # Run cross-validation for each configuration
    logger.info("Running cross-validation for balanced models")
    for config in configs:
        logger.info(f"Testing {config['name']}")
        
        # Create balanced model
        balanced_model = create_model(
            model_type="logreg",
            max_features=balanced_params['max_features'],
            ngram_range=config['ngram_range'],
            min_df=balanced_params['min_df'],
            max_df=balanced_params['max_df'],
            C=balanced_params['C']
        )
        
        # Cross-validate
        cv_results = cross_validate(
            balanced_model,
            train,
            train['relevant'],
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Extract results
        result = {
            'name': f"Balanced {config['name']}",
            'ngram_range': str(config['ngram_range']),
            'model_type': 'Balanced (F1-optimized)'
        }
        
        # Add test metrics
        for metric in scoring:
            mean_score = cv_results[f'test_{metric}'].mean()
            std_score = cv_results[f'test_{metric}'].std()
            result[f'{metric}_mean'] = mean_score
            result[f'{metric}_std'] = std_score
            logger.info(f"{metric}: {mean_score:.4f} (±{std_score:.4f})")
        
        results.append(result)
    
    # Run for high-recall models
    logger.info("Running cross-validation for high-recall models")
    for config in configs:
        logger.info(f"Testing {config['name']}")
        
        # Create high-recall model
        high_recall_model = create_model(
            model_type="logreg",
            max_features=high_recall_params['max_features'],
            ngram_range=config['ngram_range'],
            min_df=high_recall_params['min_df'],
            max_df=high_recall_params['max_df'],
            C=high_recall_params['C']
        )
        
        # Cross-validate
        cv_results = cross_validate(
            high_recall_model,
            train,
            train['relevant'],
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Extract results
        result = {
            'name': f"High-recall {config['name']}",
            'ngram_range': str(config['ngram_range']),
            'model_type': 'High-recall'
        }
        
        # Add test metrics
        for metric in scoring:
            mean_score = cv_results[f'test_{metric}'].mean()
            std_score = cv_results[f'test_{metric}'].std()
            result[f'{metric}_mean'] = mean_score
            result[f'{metric}_std'] = std_score
            logger.info(f"{metric}: {mean_score:.4f} (±{std_score:.4f})")
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create output directory
    verification_dir = os.path.join(PATHS['results_dir'], 'verification')
    os.makedirs(verification_dir, exist_ok=True)
    
    # Save CSV results
    results_df.to_csv(os.path.join(verification_dir, 'ngram_verification.csv'), index=False)
    
    # Create markdown report
    with open(os.path.join(verification_dir, 'ngram_verification.md'), 'w') as f:
        f.write("# N-gram Verification Results\n\n")
        f.write("This report compares the performance of (1,2) vs (1,3) n-gram ranges ")
        f.write("on both balanced and high-recall model configurations using 5-fold cross-validation.\n\n")
        
        # Balanced models table
        f.write("## Balanced Models (F1-optimized)\n\n")
        f.write("| N-gram Range | F1 Score | Precision | Recall | ROC AUC |\n")
        f.write("|-------------|----------|-----------|--------|--------|\n")
        
        balanced_df = results_df[results_df['model_type'] == 'Balanced (F1-optimized)']
        for _, row in balanced_df.iterrows():
            f.write(f"| {row['ngram_range']} | {row['f1_mean']:.4f} (±{row['f1_std']:.4f}) | ")
            f.write(f"{row['precision_mean']:.4f} (±{row['precision_std']:.4f}) | ")
            f.write(f"{row['recall_mean']:.4f} (±{row['recall_std']:.4f}) | ")
            f.write(f"{row['roc_auc_mean']:.4f} (±{row['roc_auc_std']:.4f}) |\n")
        
        # High-recall models table
        f.write("\n## High-Recall Models\n\n")
        f.write("| N-gram Range | F1 Score | Precision | Recall | ROC AUC |\n")
        f.write("|-------------|----------|-----------|--------|--------|\n")
        
        high_recall_df = results_df[results_df['model_type'] == 'High-recall']
        for _, row in high_recall_df.iterrows():
            f.write(f"| {row['ngram_range']} | {row['f1_mean']:.4f} (±{row['f1_std']:.4f}) | ")
            f.write(f"{row['precision_mean']:.4f} (±{row['precision_std']:.4f}) | ")
            f.write(f"{row['recall_mean']:.4f} (±{row['recall_std']:.4f}) | ")
            f.write(f"{row['roc_auc_mean']:.4f} (±{row['roc_auc_std']:.4f}) |\n")
        
        # Analysis
        f.write("\n## Analysis\n\n")
        
        # Get the best F1 scores for each range
        best_12 = balanced_df[balanced_df['ngram_range'] == '(1, 2)']['f1_mean'].values[0]
        best_13 = balanced_df[balanced_df['ngram_range'] == '(1, 3)']['f1_mean'].values[0]
        diff = abs(best_12 - best_13) * 100  # Convert to percentage points
        
        f.write(f"The performance difference between (1,2) and (1,3) n-grams is {diff:.2f} percentage points ")
        if best_12 > best_13:
            f.write("in favor of (1,2) n-grams. ")
        else:
            f.write("in favor of (1,3) n-grams. ")
        
        f.write("This verification ")
        if best_12 > best_13:
            f.write("confirms our previous finding that (1,2) n-grams perform better than (1,3) n-grams ")
            f.write("on our systematic review screening task, contradicting the literature claim that ")
            f.write("(1,3) n-grams should provide a 10 percentage point F1 boost over unigrams alone.\n\n")
        else:
            f.write("does not align with our grid search results. In this verification test, ")
            f.write("(1,3) n-grams showed better performance, which may warrant further investigation.\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        if best_12 > best_13:
            f.write("This verification confirms our decision to use (1,2) n-grams as the foundation for our baseline models.")
        else:
            f.write("While our grid search indicated (1,2) n-grams were superior, this verification shows that ")
            f.write("(1,3) n-grams may be competitive. We should conduct more thorough testing before finalizing our decision.")
    
    logger.info(f"Verification complete. Results saved to {verification_dir}")

if __name__ == "__main__":
    main() 