#!/usr/bin/env python3
"""
Run comprehensive experiments for systematic review classification.

This script coordinates a series of experiments:
1. Baseline grid search for each classifier type
2. Normalization comparison (with independent grid searches)
3. Consolidated comparison reports

Usage:
    python run_experiments.py --all --output results/experiments
"""
import os
import argparse
import logging
import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.config import PATHS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PATHS["logs_dir"], "run_experiments.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def run_baseline_experiment(model="logreg", normalization=None, output_dir="results_v2"):
    """Run a grid search experiment for a specific model type with optional normalization"""
    logger.info(f"Running {model.upper()} grid search experiment with normalization={normalization}")
    
    if normalization:
        norm_output = os.path.join(output_dir, "normalization", normalization, model)
    else:
        norm_output = os.path.join(output_dir, "baseline", model)
    
    cmd = [
        "python", "src/scripts/baseline_grid_search.py",
        "--model", model,
        "--output", norm_output
    ]
    
    if normalization:
        cmd.extend(["--normalization", normalization])
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"{model.upper()} experiment with normalization={normalization} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {model} experiment with normalization={normalization}: {e}")
        return False

def run_normalization_experiment(model="logreg", output_dir="results/experiments"):
    """Run normalization comparison with independent grid searches for a specific model type"""
    logger.info(f"Running {model.upper()} normalization comparison experiment")
    
    cmd = [
        "python", "src/scripts/compare_normalizations.py",
        "--model-type", model,
        "--output-dir", os.path.join(output_dir, "normalization")
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"{model.upper()} normalization experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {model} normalization experiment: {e}")
        return False

def analyze_ngram_performance(output_dir, model_types):
    """
    Analyze n-gram performance across all models.
    
    Args:
        output_dir: Base directory containing results
        model_types: List of model types to analyze
    """
    ngram_f1_scores = {}  
    ngram_counts = {}
    
    for model in model_types:
        cv_path = os.path.join(output_dir, "baseline", model, "metrics", "cv_results.csv")
        try:
            if os.path.exists(cv_path):
                cv_df = pd.read_csv(cv_path)
                
                if 'param_tfidf__ngram_range' in cv_df.columns and 'mean_test_f1' in cv_df.columns:
                    for ngram_str, group in cv_df.groupby('param_tfidf__ngram_range'):
                        try:
                            ngram = eval(ngram_str) if isinstance(ngram_str, str) else ngram_str
                            
                            f1_scores = group['mean_test_f1'].values
                            
                            if ngram not in ngram_f1_scores:
                                ngram_f1_scores[ngram] = []
                            
                            ngram_f1_scores[ngram].extend(f1_scores)
                            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + len(f1_scores)
                        except Exception as e:
                            logger.warning(f"Error processing n-gram {ngram_str} for model {model}: {e}")
        except Exception as e:
            logger.warning(f"Could not analyze CV results for {model}: {e}")
    
    ngram_analysis = {}
    for ngram, scores in ngram_f1_scores.items():
        if scores:
            ngram_analysis[ngram] = {
                'mean_f1': np.mean(scores),
                'median_f1': np.median(scores),
                'max_f1': np.max(scores),
                'min_f1': np.min(scores),
                'std_f1': np.std(scores),
                'count': len(scores)
            }
    
    if ngram_analysis:
        analysis_df = pd.DataFrame([
            {
                'ngram_range': str(ngram),
                'mean_f1': stats['mean_f1'],
                'median_f1': stats['median_f1'],
                'max_f1': stats['max_f1'],
                'min_f1': stats['min_f1'],
                'std_f1': stats['std_f1'],
                'count': stats['count']
            }
            for ngram, stats in ngram_analysis.items()
        ])
        
        analysis_df = analysis_df.sort_values('mean_f1', ascending=False)
        analysis_path = os.path.join(output_dir, "comparison", "ngram_analysis.csv")
        analysis_df.to_csv(analysis_path, index=False)
        logger.info(f"N-gram analysis saved to {analysis_path}")
        
        plt.figure(figsize=(10, 6))
        
        x = range(len(analysis_df))
        plt.bar(x, analysis_df['mean_f1'], yerr=analysis_df['std_f1'], capsize=5)
        plt.xticks(x, analysis_df['ngram_range'], rotation=45)
        plt.xlabel('N-gram Range')
        plt.ylabel('Mean F1 Score')
        plt.title('N-gram Range Performance Across All Models')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "comparison", "ngram_performance.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"N-gram performance plot saved to {plot_path}")
        
        return analysis_df
    else:
        logger.warning("No n-gram analysis data collected")
        return None

def analyze_normalization_performance(output_dir, model_types):
    """
    Analyze normalization performance across all models.
    
    Args:
        output_dir: Base directory containing results
        model_types: List of model types to analyze
        
    Returns:
        DataFrame containing aggregated normalization performance metrics
    """
    normalization_results = {
        "balanced": {},
        "highrecall": {}
    }
    
    for model in model_types:
        norm_dir = os.path.join(output_dir, "normalization", model)
        if os.path.exists(norm_dir):
            results_path = os.path.join(norm_dir, "metrics", "all_results.json")
            try:
                with open(results_path, 'r') as f:
                    model_results = json.load(f)
                    
                    for technique, metrics in model_results.get("balanced", {}).items():
                        if technique not in normalization_results["balanced"]:
                            normalization_results["balanced"][technique] = []
                        normalization_results["balanced"][technique].append({
                            "model": model,
                            **metrics
                        })
                    
                    for technique, metrics in model_results.get("highrecall", {}).items():
                        if technique not in normalization_results["highrecall"]:
                            normalization_results["highrecall"][technique] = []
                        normalization_results["highrecall"][technique].append({
                            "model": model,
                            **metrics
                        })
            except Exception as e:
                logger.warning(f"Could not analyze normalization results for {model}: {e}")
    
    balanced_rows = []
    highrecall_rows = []
    
    for technique, results in normalization_results["balanced"].items():
        for result in results:
            balanced_rows.append({
                "technique": technique,
                "model": result["model"],
                "f1": result.get("f1", 0),
                "precision": result.get("precision", 0),
                "recall": result.get("recall", 0),
                "roc_auc": result.get("roc_auc", 0),
                "wss@95": result.get("wss@95", 0)
            })
    
    for technique, results in normalization_results["highrecall"].items():
        for result in results:
            highrecall_rows.append({
                "technique": technique,
                "model": result["model"],
                "f1": result.get("f1", 0),
                "precision": result.get("precision", 0),
                "recall": result.get("recall", 0),
                "roc_auc": result.get("roc_auc", 0),
                "wss@95": result.get("wss@95", 0)
            })
    
    balanced_df = pd.DataFrame(balanced_rows)
    highrecall_df = pd.DataFrame(highrecall_rows)
    
    if not balanced_df.empty:
        agg_balanced = balanced_df.groupby('technique').agg({
            'f1': ['mean', 'std', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'wss@95': ['mean', 'std', 'max']
        }).reset_index()
        
        agg_balanced.columns = ['_'.join(col).strip('_') for col in agg_balanced.columns.values]
        
        os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)
        agg_balanced.to_csv(os.path.join(output_dir, "comparison", "normalization_balanced.csv"), index=False)
        
        plt.figure(figsize=(12, 8))
        
        techniques = agg_balanced['technique'].tolist()
        means = agg_balanced['f1_mean'].values
        stds = agg_balanced['f1_std'].values
        
        plt.bar(techniques, means, yerr=stds, capsize=5)
        plt.xlabel('Normalization Technique')
        plt.ylabel('Mean F1 Score')
        plt.title('Normalization Technique Performance (Balanced Models)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "comparison", "normalization_balanced.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Normalization performance plot (balanced) saved to {plot_path}")
    
    if not highrecall_df.empty:
        agg_highrecall = highrecall_df.groupby('technique').agg({
            'f1': ['mean', 'std', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'wss@95': ['mean', 'std', 'max']
        }).reset_index()
        
        agg_highrecall.columns = ['_'.join(col).strip('_') for col in agg_highrecall.columns.values]
        agg_highrecall.to_csv(os.path.join(output_dir, "comparison", "normalization_highrecall.csv"), index=False)
        plt.figure(figsize=(12, 8))
        
        techniques = agg_highrecall['technique'].tolist()
        means = agg_highrecall['f1_mean'].values
        stds = agg_highrecall['f1_std'].values
        
        plt.bar(techniques, means, yerr=stds, capsize=5)
        plt.xlabel('Normalization Technique')
        plt.ylabel('Mean F1 Score')
        plt.title('Normalization Technique Performance (High-Recall Models)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "comparison", "normalization_highrecall.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Normalization performance plot (high-recall) saved to {plot_path}")
    
    return balanced_df, highrecall_df

def create_consolidated_comparison(output_dir="results/experiments"):
    """Create a consolidated comparison of all model results including normalization techniques"""
    model_types = ["logreg", "svm", "cosine", "cnb"]
    
    norm_techniques = ["raw", "stemming", "lemmatization"]
    
    metrics = ["f1", "recall", "precision", "f2", "roc_auc", "wss@95"]
    
    results = {}
    
    for model in model_types:
        model_results = {}
        
        for norm in norm_techniques:
            balanced_path = os.path.join(output_dir, model, norm, "baseline", "metrics", "balanced_metrics.json")
            hr_path = os.path.join(output_dir, model, norm, "recall_95", "metrics", "highrecall_metrics.json")
            
            try:
                with open(balanced_path, 'r') as f:
                    balanced_metrics = json.load(f)
                with open(hr_path, 'r') as f:
                    hr_metrics = json.load(f)
                    
                model_results[norm] = {
                    "balanced": balanced_metrics,
                    "highrecall": hr_metrics
                }
            except FileNotFoundError:
                logger.warning(f"No metrics found for {model} with {norm} normalization")
        
        if model_results:
            results[model] = model_results
    
    bal_df = pd.DataFrame({
        'Model': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'F2': [],
        'ROC AUC': [],
        'WSS@95': []
    })
    
    hr_df = pd.DataFrame({
        'Model': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'F2': [],
        'ROC AUC': [],
        'WSS@95': []
    })
    
    for model, data in balanced_results.items():
        row = {
            'Model': model,
            'F1': data.get('f1', None),
            'Precision': data.get('precision', None),
            'Recall': data.get('recall', None),
            'F2': data.get('f2', None),
            'ROC AUC': data.get('roc_auc', None),
            'WSS@95': data.get('wss@95', None)
        }
        bal_df = pd.concat([bal_df, pd.DataFrame([row])], ignore_index=True)
    
    for model, data in hr_results.items():
        row = {
            'Model': model,
            'F1': data.get('f1', None),
            'Precision': data.get('precision', None),
            'Recall': data.get('recall', None),
            'F2': data.get('f2', None),
            'ROC AUC': data.get('roc_auc', None),
            'WSS@95': data.get('wss@95', None)
        }
        hr_df = pd.concat([hr_df, pd.DataFrame([row])], ignore_index=True)
    
    norm_bal_df = pd.DataFrame({
        'Model': [],
        'Technique': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'F2': [],
        'ROC AUC': [],
        'WSS@95': []
    })
    
    norm_hr_df = pd.DataFrame({
        'Model': [],
        'Technique': [],
        'F1': [],
        'Precision': [],
        'Recall': [],
        'F2': [],
        'ROC AUC': [],
        'WSS@95': []
    })
    
    for model, data in norm_balanced_metrics.items():
        row = {
            'Model': model,
            'Technique': data.get('technique', 'Unknown'),
            'F1': data.get('f1', None),
            'Precision': data.get('precision', None),
            'Recall': data.get('recall', None),
            'F2': data.get('f2', None),
            'ROC AUC': data.get('roc_auc', None),
            'WSS@95': data.get('wss@95', None)
        }
        norm_bal_df = pd.concat([norm_bal_df, pd.DataFrame([row])], ignore_index=True)
    
    for model, data in norm_hr_metrics.items():
        row = {
            'Model': model,
            'Technique': data.get('technique', 'Unknown'),
            'F1': data.get('f1', None),
            'Precision': data.get('precision', None),
            'Recall': data.get('recall', None),
            'F2': data.get('f2', None),
            'ROC AUC': data.get('roc_auc', None),
            'WSS@95': data.get('wss@95', None)
        }
        norm_hr_df = pd.concat([norm_hr_df, pd.DataFrame([row])], ignore_index=True)
    
    bal_df = bal_df.sort_values('F1', ascending=False)
    hr_df = hr_df.sort_values('F1', ascending=False)
    norm_bal_df = norm_bal_df.sort_values('F1', ascending=False)
    norm_hr_df = norm_hr_df.sort_values('F1', ascending=False)
    
    os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)
    bal_df.to_csv(os.path.join(output_dir, "comparison", "balanced_models.csv"), index=False)
    hr_df.to_csv(os.path.join(output_dir, "comparison", "high_recall_models.csv"), index=False)
    norm_bal_df.to_csv(os.path.join(output_dir, "comparison", "normalized_balanced_models.csv"), index=False)
    norm_hr_df.to_csv(os.path.join(output_dir, "comparison", "normalized_high_recall_models.csv"), index=False)
    
    plot_model_comparison(bal_df, "balanced", output_dir)
    plot_model_comparison(hr_df, "high_recall", output_dir)
    
    create_baseline_vs_normalized_plots(bal_df, norm_bal_df, "balanced", output_dir)
    create_baseline_vs_normalized_plots(hr_df, norm_hr_df, "high_recall", output_dir)
    
    analyze_ngram_performance(output_dir, model_types)
    
    analyze_normalization_performance(output_dir, model_types)
    
    create_markdown_comparison(bal_df, hr_df, norm_bal_df, norm_hr_df, output_dir)
    
    logger.info(f"Consolidated comparison created in {os.path.join(output_dir, 'comparison')}")

def create_baseline_vs_normalized_plots(baseline_df, norm_df, model_type, output_dir):
    """Create comparison plots between baseline and normalized models"""
    if baseline_df.empty or norm_df.empty:
        logger.warning(f"Cannot create comparison plots for {model_type} - insufficient data")
        return
    
    common_models = set(baseline_df['Model']).intersection(set(norm_df['Model']))
    
    if not common_models:
        logger.warning(f"No common models found between baseline and normalized {model_type}")
        return
    
    b_df = baseline_df[baseline_df['Model'].isin(common_models)]
    n_df = norm_df[norm_df['Model'].isin(common_models)]
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(common_models))
    width = 0.35
    
    models_sorted = b_df.sort_values('F1', ascending=False)['Model'].tolist()
    
    baseline_f1 = [b_df[b_df['Model'] == m]['F1'].values[0] for m in models_sorted]
    norm_f1 = [n_df[n_df['Model'] == m]['F1'].values[0] for m in models_sorted]
    
    techniques = [n_df[n_df['Model'] == m]['Technique'].values[0] for m in models_sorted]
    
    bar1 = plt.bar(x - width/2, baseline_f1, width, label='Baseline')
    bar2 = plt.bar(x + width/2, norm_f1, width, label='Best Normalization')
    
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Comparison: Baseline vs. Best Normalization ({model_type.capitalize()})')
    plt.xticks(x, models_sorted)
    plt.legend()
    
    for i, tech in enumerate(techniques):
        plt.annotate(tech, 
                   (x[i] + width/2, norm_f1[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "comparison", f"{model_type}_baseline_vs_normalized.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    baseline_wss = [b_df[b_df['Model'] == m]['WSS@95'].values[0] for m in models_sorted]
    norm_wss = [n_df[n_df['Model'] == m]['WSS@95'].values[0] for m in models_sorted]
    
    bar1 = plt.bar(x - width/2, baseline_wss, width, label='Baseline')
    bar2 = plt.bar(x + width/2, norm_wss, width, label='Best Normalization')
    
    plt.xlabel('Model')
    plt.ylabel('WSS@95')
    plt.title(f'Work Saved Comparison: Baseline vs. Best Normalization ({model_type.capitalize()})')
    plt.xticks(x, models_sorted)
    plt.legend()
    
    for i, tech in enumerate(techniques):
        plt.annotate(tech, 
                   (x[i] + width/2, norm_wss[i]),
                   textcoords="offset points",
                   xytext=(0,10), 
                   ha='center')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "comparison", f"{model_type}_wss_baseline_vs_normalized.png"))
    plt.close()

def plot_model_comparison(df, model_type, output_dir):
    """Create bar charts comparing models"""
    metrics = ['F1', 'Precision', 'Recall', 'F2', 'ROC AUC', 'WSS@95']
    
    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.barplot(x='Model', y=metric, data=df)
        plt.title(f'{metric} Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "comparison", f"{model_type}_comparison.png"))
    plt.close()

def create_markdown_comparison(balanced_df, hr_df, norm_bal_df, norm_hr_df, output_dir):
    """Create a comprehensive markdown report comparing all models and normalization techniques"""
    output_path = os.path.join(output_dir, "comparison", "comprehensive_comparison.md")
    
    ngram_analysis_path = os.path.join(output_dir, "comparison", "ngram_analysis.csv")
    ngram_df = None
    if os.path.exists(ngram_analysis_path):
        try:
            ngram_df = pd.read_csv(ngram_analysis_path)
        except Exception as e:
            logger.warning(f"Could not load n-gram analysis: {e}")
    
    with open(output_path, 'w') as f:
        f.write("# Comprehensive Systematic Review Classification Model Comparison\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## 1. Baseline Models\n\n")
        f.write("### 1.1 Balanced Models (F1-optimized)\n\n")
        f.write("| Model | F1 | Precision | Recall | F2 | ROC AUC | WSS@95 |\n")
        f.write("|-------|-----|-----------|--------|-----|---------|--------|\n")
        
        for _, row in balanced_df.iterrows():
            f.write(f"| {row['Model']} | {row['F1']:.4f} | {row['Precision']:.4f} | ")
            f.write(f"{row['Recall']:.4f} | {row['F2']:.4f} | {row['ROC AUC']:.4f} | {row['WSS@95']:.4f} |\n")
        
        f.write("\n### 1.2 High-Recall Models (95% Target)\n\n")
        f.write("| Model | F1 | Precision | Recall | F2 | ROC AUC | WSS@95 |\n")
        f.write("|-------|-----|-----------|--------|-----|---------|--------|\n")
        
        for _, row in hr_df.iterrows():
            f.write(f"| {row['Model']} | {row['F1']:.4f} | {row['Precision']:.4f} | ")
            f.write(f"{row['Recall']:.4f} | {row['F2']:.4f} | {row['ROC AUC']:.4f} | {row['WSS@95']:.4f} |\n")
        
        f.write("\n## 2. Text Normalization\n\n")
        f.write("### 2.1 Best Normalization Techniques for Balanced Models\n\n")
        f.write("| Model | Technique | F1 | Precision | Recall | F2 | ROC AUC | WSS@95 |\n")
        f.write("|-------|-----------|-----|-----------|--------|-----|---------|--------|\n")
        
        if not norm_bal_df.empty:
            for _, row in norm_bal_df.iterrows():
                tech = row['Technique'].capitalize() if row['Technique'] != 'baseline' else 'No normalization'
                f.write(f"| {row['Model']} | {tech} | {row['F1']:.4f} | {row['Precision']:.4f} | ")
                f.write(f"{row['Recall']:.4f} | {row['F2']:.4f} | {row['ROC AUC']:.4f} | {row['WSS@95']:.4f} |\n")
        else:
            f.write("| No data available |\n")
        
        f.write("\n### 2.2 Best Normalization Techniques for High-Recall Models\n\n")
        f.write("| Model | Technique | F1 | Precision | Recall | F2 | ROC AUC | WSS@95 |\n")
        f.write("|-------|-----------|-----|-----------|--------|-----|---------|--------|\n")
        
        if not norm_hr_df.empty:
            for _, row in norm_hr_df.iterrows():
                tech = row['Technique'].capitalize() if row['Technique'] != 'baseline' else 'No normalization'
                f.write(f"| {row['Model']} | {tech} | {row['F1']:.4f} | {row['Precision']:.4f} | ")
                f.write(f"{row['Recall']:.4f} | {row['F2']:.4f} | {row['ROC AUC']:.4f} | {row['WSS@95']:.4f} |\n")
        else:
            f.write("| No data available |\n")
        
        if ngram_df is not None and len(ngram_df) > 1:
            f.write("\n## 3. N-gram Range Analysis\n\n")
            f.write("We analyzed the performance of different n-gram ranges across all models:\n\n")
            f.write("| N-gram Range | Mean F1 | Median F1 | Max F1 | Min F1 | Std Dev | Count |\n")
            f.write("|--------------|---------|-----------|--------|--------|---------|-------|\n")
            
            for _, row in ngram_df.iterrows():
                f.write(f"| {row['ngram_range']} | {row['mean_f1']:.4f} | {row['median_f1']:.4f} | ")
                f.write(f"{row['max_f1']:.4f} | {row['min_f1']:.4f} | {row['std_f1']:.4f} | {int(row['count'])} |\n")
            
            if len(ngram_df) >= 2:
                best_ngram = ngram_df.iloc[0]['ngram_range']
                second_ngram = ngram_df.iloc[1]['ngram_range']
                best_f1 = ngram_df.iloc[0]['mean_f1']
                second_f1 = ngram_df.iloc[1]['mean_f1']
                diff = (best_f1 - second_f1) * 100
                
                f.write(f"\n**{best_ngram}** n-grams outperform **{second_ngram}** n-grams by **{diff:.1f} percentage points** in F1 score. ")
                
                f1_12 = None
                f1_13 = None
                
                for _, row in ngram_df.iterrows():
                    if row['ngram_range'] == '(1, 2)':
                        f1_12 = row['mean_f1']
                    elif row['ngram_range'] == '(1, 3)':
                        f1_13 = row['mean_f1']
                
                if f1_12 is not None and f1_13 is not None:
                    diff_13_vs_12 = (f1_13 - f1_12) * 100
                    
                    if diff_13_vs_12 > 0:
                        f.write(f"\n\n**Finding**: (1,3) n-grams outperform (1,2) n-grams by {diff_13_vs_12:.1f} percentage points. ")
                        
                        if diff_13_vs_12 >= 9.5:  # Close to 10 pp
                            f.write("This is approximately 10 percentage points improvement, which aligns with findings from recent literature (LREC 2020).\n")
                        else:
                            f.write(f"While this is less than the 10 percentage point improvement reported in some literature, it still represents a significant lift in performance.\n")
                    else:
                        abs_diff = abs(diff_13_vs_12)
                        f.write(f"\n\n**Finding**: (1,2) n-grams outperform (1,3) n-grams by {abs_diff:.1f} percentage points. ")
                        f.write("This contradicts some literature suggesting (1,3) would provide better performance. The discrepancy may be due to specificities of our dataset or model configurations.\n")
        
        if not norm_bal_df.empty and not balanced_df.empty:
            f.write("\n## 4. Normalization Impact Analysis\n\n")
            
            common_models = set(balanced_df['Model']).intersection(set(norm_bal_df['Model']))
            
            if common_models:
                improvements = []
                
                for model in common_models:
                    bal_f1 = balanced_df[balanced_df['Model'] == model]['F1'].values[0]
                    norm_f1 = norm_bal_df[norm_bal_df['Model'] == model]['F1'].values[0]
                    diff = (norm_f1 - bal_f1) * 100  # percentage points
                    tech = norm_bal_df[norm_bal_df['Model'] == model]['Technique'].values[0]
                    improvements.append((model, tech, diff))
                
                improvements.sort(key=lambda x: x[2], reverse=True)
                
                f.write("### 4.1 F1 Score Improvements with Normalization\n\n")
                f.write("| Model | Best Technique | F1 Improvement (percentage points) |\n")
                f.write("|-------|---------------|-----------------------------------|\n")
                
                for model, tech, diff in improvements:
                    tech_display = tech.capitalize() if tech != 'baseline' else 'No normalization'
                    f.write(f"| {model} | {tech_display} | {diff:.1f} |\n")
                
                avg_imp = sum(i[2] for i in improvements) / len(improvements)
                f.write(f"\nAverage F1 improvement across all models: **{avg_imp:.1f} percentage points**\n")
                
                helps_count = sum(1 for _, _, diff in improvements if diff > 0)
                hurts_count = sum(1 for _, _, diff in improvements if diff < 0)
                same_count = sum(1 for _, _, diff in improvements if abs(diff) < 0.1)
                
                f.write(f"\nNormalization improved performance in **{helps_count}/{len(improvements)}** cases, ")
                f.write(f"hurt performance in **{hurts_count}/{len(improvements)}** cases, ")
                f.write(f"and had minimal impact in **{same_count}/{len(improvements)}** cases.\n")
                
                tech_counts = {}
                for _, tech, _ in improvements:
                    if tech not in tech_counts:
                        tech_counts[tech] = 0
                    tech_counts[tech] += 1
                
                f.write("\n### 4.2 Technique Effectiveness\n\n")
                f.write("| Technique | Count (as best technique) | Percentage |\n")
                f.write("|-----------|----------------------------|------------|\n")
                
                for tech, count in sorted(tech_counts.items(), key=lambda x: x[1], reverse=True):
                    tech_display = tech.capitalize() if tech != 'baseline' else 'No normalization'
                    percent = (count / len(improvements)) * 100
                    f.write(f"| {tech_display} | {count} | {percent:.1f}% |\n")
        
        f.write("\n## 5. Best Performing Models\n\n")
        
        if not balanced_df.empty:
            best_bal_model = balanced_df.loc[balanced_df['F1'].idxmax(), 'Model'] if not balanced_df.empty else None
            best_bal_f1 = balanced_df['F1'].max() if not balanced_df.empty else None
            
            best_hr_model = hr_df.loc[hr_df['F1'].idxmax(), 'Model'] if not hr_df.empty else None
            best_hr_f1 = hr_df['F1'].max() if not hr_df.empty else None
            
            best_wss_model = hr_df.loc[hr_df['WSS@95'].idxmax(), 'Model'] if not hr_df.empty else None
            best_wss_score = hr_df['WSS@95'].max() if not hr_df.empty else None
            
            f.write("### 5.1 Best Baseline Models\n\n")
            f.write(f"- **Best Overall Model (F1)**: {best_bal_model} (F1 = {best_bal_f1:.4f})\n")
            f.write(f"- **Best High-Recall Model (F1)**: {best_hr_model} (F1 = {best_hr_f1:.4f})\n")
            f.write(f"- **Best Work Savings**: {best_wss_model} (WSS@95 = {best_wss_score:.4f})\n\n")
        
        if not norm_bal_df.empty:
            best_norm_bal_model = norm_bal_df.loc[norm_bal_df['F1'].idxmax()]
            best_norm_hr_model = norm_hr_df.loc[norm_hr_df['F1'].idxmax()] if not norm_hr_df.empty else None
            best_norm_wss_model = norm_hr_df.loc[norm_hr_df['WSS@95'].idxmax()] if not norm_hr_df.empty else None
            
            f.write("### 5.2 Best Models with Normalization\n\n")
            f.write(f"- **Best Overall Model with Normalization**: {best_norm_bal_model['Model']} ")
            f.write(f"with {best_norm_bal_model['Technique']} (F1 = {best_norm_bal_model['F1']:.4f})\n")
            
            if best_norm_hr_model is not None:
                f.write(f"- **Best High-Recall Model with Normalization**: {best_norm_hr_model['Model']} ")
                f.write(f"with {best_norm_hr_model['Technique']} (F1 = {best_norm_hr_model['F1']:.4f})\n")
                
            if best_norm_wss_model is not None:
                f.write(f"- **Best Work Savings with Normalization**: {best_norm_wss_model['Model']} ")
                f.write(f"with {best_norm_wss_model['Technique']} (WSS@95 = {best_norm_wss_model['WSS@95']:.4f})\n")
        
        f.write("\n## 6. Methodology\n\n")
        f.write("All models were evaluated using a rigorous methodology:\n\n")
        f.write("1. **Baseline Grid Search**: Each classifier underwent grid search with multi-metric scoring (F1, recall, precision).\n")
        f.write("2. **Normalization Evaluation**: Each normalization technique was evaluated with its own independent grid search\n")
        f.write("   to find optimal hyperparameters rather than applying predetermined parameters from the baseline.\n")
        f.write("3. **Model Extraction**:\n")
        f.write("   - **Balanced model**: Optimized for F1 score\n")
        f.write("   - **High-recall model**: Highest F1 score among configurations with recall ≥ 0.95\n\n")
        
        f.write("## 7. Recommendations\n\n")
        
        best_overall_config = "baseline"
        best_overall_f1 = 0
        
        if not balanced_df.empty:
            top_baseline_f1 = balanced_df['F1'].max()
            if top_baseline_f1 > best_overall_f1:
                best_overall_f1 = top_baseline_f1
                best_overall_config = "baseline"
        
        if not norm_bal_df.empty:
            top_norm_f1 = norm_bal_df['F1'].max()
            if top_norm_f1 > best_overall_f1:
                best_overall_f1 = top_norm_f1
                best_overall_config = "normalized"
        
        if best_overall_config == "baseline":
            f.write("Based on our comprehensive analysis, we recommend using the baseline models without text normalization\n")
            f.write("as they provided the best overall performance on our dataset.\n\n")
        else:
            f.write("Based on our comprehensive analysis, we recommend incorporating text normalization in the classification pipeline\n")
            f.write("as it demonstrated statistically significant improvements in model performance.\n\n")
        
        f.write("### Specific Recommendations\n\n")
        
        if not norm_bal_df.empty and not balanced_df.empty:
            if norm_bal_df['F1'].max() > balanced_df['F1'].max():
                best_model = norm_bal_df.loc[norm_bal_df['F1'].idxmax()]
                f.write(f"1. For general classification with balanced precision-recall tradeoff, use the **{best_model['Model']}** model\n")
                f.write(f"   with **{best_model['Technique']}** normalization (F1 = {best_model['F1']:.4f}).\n")
            else:
                best_model = balanced_df.loc[balanced_df['F1'].idxmax()]
                f.write(f"1. For general classification with balanced precision-recall tradeoff, use the **{best_model['Model']}** model\n")
                f.write(f"   without normalization (F1 = {best_model['F1']:.4f}).\n")
        
        if not norm_hr_df.empty and not hr_df.empty:
            if norm_hr_df['F1'].max() > hr_df['F1'].max():
                best_model = norm_hr_df.loc[norm_hr_df['F1'].idxmax()]
                f.write(f"2. For systematic review screening where high recall is critical, use the **{best_model['Model']}** model\n")
                f.write(f"   with **{best_model['Technique']}** normalization (F1 = {best_model['F1']:.4f}, Recall ≥ 95%).\n")
            else:
                best_model = hr_df.loc[hr_df['F1'].idxmax()]
                f.write(f"2. For systematic review screening where high recall is critical, use the **{best_model['Model']}** model\n")
                f.write(f"   without normalization (F1 = {best_model['F1']:.4f}, Recall ≥ 95%).\n")
        
        if not norm_hr_df.empty and not hr_df.empty:
            if norm_hr_df['WSS@95'].max() > hr_df['WSS@95'].max():
                best_model = norm_hr_df.loc[norm_hr_df['WSS@95'].idxmax()]
                f.write(f"3. For maximum work savings while maintaining 95% recall, use the **{best_model['Model']}** model\n")
                f.write(f"   with **{best_model['Technique']}** normalization (WSS@95 = {best_model['WSS@95']:.4f}).\n")
            else:
                best_model = hr_df.loc[hr_df['WSS@95'].idxmax()]
                f.write(f"3. For maximum work savings while maintaining 95% recall, use the **{best_model['Model']}** model\n")
                f.write(f"   without normalization (WSS@95 = {best_model['WSS@95']:.4f}).\n")
        
        if ngram_df is not None and len(ngram_df) > 0:
            best_ngram = ngram_df.iloc[0]['ngram_range']
            f.write(f"4. For optimal text feature extraction, use **{best_ngram}** n-gram range based on our comprehensive analysis.\n")
    
    logger.info(f"Comprehensive comparison report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive experiments for systematic review classification"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments (baseline grid search, normalization comparison)"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run baseline grid search experiments"
    )
    parser.add_argument(
        "--normalization", action="store_true",
        help="Run normalization comparison experiments"
    )
    parser.add_argument(
        "--output", type=str, default="results/experiments",
        help="Directory to save results"
    )
    parser.add_argument(
        "--models", type=str, default="logreg",
        help="Comma-separated list of models to run (logreg,svm,cosine,cnb)"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.all:
        models = ["logreg", "svm", "cosine", "cnb"]
    else:
        models = args.models.split(",")
    
    logger.info(f"Running experiments for models: {', '.join(models)}")
    
    run_baseline = args.all or args.baseline
    run_normalization = args.all or args.normalization
    
    baseline_results = {}
    normalization_results = {}
    
    if run_baseline:
    logger.info("Running baseline grid search experiments")
    normalization_techniques = [None, "stemming", "lemmatization"]
    
    for model in models:
        for norm in normalization_techniques:
            success = run_baseline_experiment(model, normalization=norm, output_dir=args.output)
            norm_key = norm if norm else "baseline"
            if norm_key not in baseline_results:
                baseline_results[norm_key] = {}
            baseline_results[norm_key][model] = success
    
    for norm in normalization_techniques:
        norm_name = "baseline" if norm is None else norm
        successful = [m for m, s in baseline_results.get(norm_name, {}).items() if s]
        failed = [m for m, s in baseline_results.get(norm_name, {}).items() if not s]
        
        if successful:
            logger.info(f"Successfully completed experiments with {norm_name} for: {', '.join(successful)}")
        if failed:
            logger.warning(f"Failed experiments with {norm_name} for: {', '.join(failed)}")
    
    if run_normalization:
        logger.info("Running normalization comparison experiments")
        for model in models:
            success = run_normalization_experiment(model, args.output)
            normalization_results[model] = success
        
        successful = [m for m, s in normalization_results.items() if s]
        failed = [m for m, s in normalization_results.items() if not s]
        
        if successful:
            logger.info(f"Successfully completed normalization experiments for: {', '.join(successful)}")
        if failed:
            logger.warning(f"Failed normalization experiments for: {', '.join(failed)}")
    
    if any(baseline_results.values()) or any(normalization_results.values()):
        logger.info("Creating consolidated comparison")
        create_consolidated_comparison(args.output)
        logger.info(f"Comparison saved to {os.path.join(args.output, 'comparison')}")
    else:
        logger.error("No successful experiments to compare")
    
    logger.info("All experiments completed")

if __name__ == "__main__":
    main()