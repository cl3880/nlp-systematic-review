#!/usr/bin/env python3
"""
Generate final comparison tables for the paper.

This script:
1. Loads results from all experiment stages
2. Creates LaTeX and Markdown tables for the paper
3. Performs significance testing between key models
4. Generates summary statistics and insights
"""
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mcnemar
from tabulate import tabulate

def find_all_results():
    """Find result files from all experiment stages."""
    result_files = []
    
    # Search for metrics.json files in all results directories
    for base_dir in ["results/v2", "results/v3", "results/v2.2", "results"]:
        if os.path.exists(base_dir):
            result_files.extend(glob.glob(f"{base_dir}/**/metrics.json", recursive=True))
    
    return result_files

def determine_experiment_type(model_dir, model_name):
    """Determine the experiment type and configuration."""
    experiment_info = {
        "stage": None,
        "experiment_type": None,
        "base_model": "svm",  # Default
        "normalization": "none",
        "balancing": "none",
        "features": "text"  # Default to text-only
    }
    
    # Determine stage based on directory path
    if "v3" in model_dir:
        experiment_info["stage"] = "Stage 3 (Isolation)"
        experiment_info["experiment_type"] = "isolation"
    elif "v2.2" in model_dir or "criteria" in model_name:
        experiment_info["stage"] = "Stage 4 (Criteria)"
        experiment_info["experiment_type"] = "criteria"
    else:
        experiment_info["stage"] = "Stage 2 (Grid Search)"
        experiment_info["experiment_type"] = "grid_search"
    
    # Determine model configuration
    if "stemming" in model_name:
        experiment_info["normalization"] = "stemming"
    elif "lemmatization" in model_name:
        experiment_info["normalization"] = "lemmatization"
    
    if "smote" in model_name:
        experiment_info["balancing"] = "smote"
    
    # Determine feature types
    if "criteria" in model_name:
        experiment_info["features"] = "text+criteria"
        if "mesh" in model_name:
            experiment_info["features"] = "text+criteria+mesh"
    
    # Extract base model type
    if "logreg" in model_name:
        experiment_info["base_model"] = "logreg"
    elif "svm" in model_name:
        experiment_info["base_model"] = "svm"
    elif "cnb" in model_name:
        experiment_info["base_model"] = "cnb"
    elif "cosine" in model_name:
        experiment_info["base_model"] = "cosine"
    
    return experiment_info

def load_and_process_results(result_files):
    """Load and process all result files into a DataFrame."""
    rows = []
    
    for rf in result_files:
        model_dir = os.path.dirname(rf)
        model_name = os.path.basename(model_dir)
        
        try:
            # Load metrics
            with open(rf, 'r') as f:
                metrics = json.load(f)
            
            # Skip if missing required data
            if "balanced" not in metrics or "high_recall" not in metrics:
                continue
            
            # Determine experiment type and configuration
            experiment_info = determine_experiment_type(model_dir, model_name)
            
            # Create row with standardized data
            row = {
                "model_name": model_name,
                "model_directory": model_dir,
                "stage": experiment_info["stage"],
                "experiment_type": experiment_info["experiment_type"],
                "base_model": experiment_info["base_model"],
                "normalization": experiment_info["normalization"],
                "balancing": experiment_info["balancing"],
                "features": experiment_info["features"],
                
                # Balanced metrics
                "balanced_precision": metrics["balanced"]["precision"],
                "balanced_recall": metrics["balanced"]["recall"],
                "balanced_f1": metrics["balanced"]["f1"],
                "balanced_f2": metrics["balanced"].get("f2", None),
                "balanced_roc_auc": metrics["balanced"]["roc_auc"],
                "balanced_wss_at_95": metrics["balanced"].get("wss@95", 
                                     metrics["balanced"].get("wss_at_95", None)),
                
                # High-recall metrics
                "hr_precision": metrics["high_recall"]["precision"],
                "hr_recall": metrics["high_recall"]["recall"],
                "hr_f1": metrics["high_recall"]["f1"],
                "hr_f2": metrics["high_recall"].get("f2", None),
                "hr_roc_auc": metrics["high_recall"]["roc_auc"],
                "hr_wss_at_95": metrics["high_recall"].get("wss@95",
                               metrics["high_recall"].get("wss_at_95", None)),
                
                # Threshold information
                "threshold": metrics.get("threshold", None)
            }
            
            rows.append(row)
            
        except Exception as e:
            print(f"Error processing {rf}: {e}")
    
    # Create DataFrame and clean up any NaN values
    df = pd.DataFrame(rows)
    
    # Fill NaN values in numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def create_formatted_name(row):
    """Create a standardized model name for tables."""
    components = []
    
    # Base model
    base_model = row["base_model"].upper()
    
    # Normalization
    if row["normalization"] == "stemming":
        components.append("Stem")
    elif row["normalization"] == "lemmatization":
        components.append("Lemma")
    
    # Balancing
    if row["balancing"] == "smote":
        components.append("SMOTE")
    
    # Features
    if row["features"] == "text+criteria":
        components.append("Criteria")
    elif row["features"] == "text+criteria+mesh":
        components.append("Criteria+MeSH")
    
    if components:
        name = f"{base_model} + {' + '.join(components)}"
    else:
        name = f"{base_model} (baseline)"
    
    # Add experiment type
    if row["experiment_type"] == "isolation":
        name += " [fixed]"
    elif row["experiment_type"] == "criteria":
        name += " [fixed]"
    
    return name

def create_comprehensive_table(df):
    """Create a comprehensive comparison table of all models."""
    # Add formatted model name
    df["formatted_name"] = df.apply(create_formatted_name, axis=1)
    
    # Select relevant columns
    cols = [
        "formatted_name",
        "balanced_precision", "balanced_recall", "balanced_f1", 
        "balanced_roc_auc", "balanced_wss_at_95",
        "hr_precision", "hr_recall", "hr_f1", 
        "hr_roc_auc", "hr_wss_at_95"
    ]
    
    # Create table DataFrame
    table_df = df[cols].copy()
    
    # Rename columns for readability
    table_df.columns = [
        "Model",
        "Precision", "Recall", "F₁", "AUC", "WSS@95",
        "HR Precision", "HR Recall", "HR F₁", "HR AUC", "HR WSS@95"
    ]
    
    return table_df

def format_latex_table(df, highlight_best=True):
    """Format DataFrame as a LaTeX table with optional best value highlighting."""
    # Copy DataFrame to avoid modifying original
    latex_df = df.copy()
    
    # Columns to find best values for (excluding "Model")
    metric_cols = [col for col in latex_df.columns if col != "Model"]
    
    if highlight_best:
        # Find best value for each metric column
        best_values = {}
        for col in metric_cols:
            if "Precision" in col or "F" in col or "AUC" in col or "WSS" in col:
                best_values[col] = latex_df[col].max()
            elif "Recall" in col:
                best_values[col] = latex_df[col].max()
    
    # Format numeric columns
    for col in metric_cols:
        if highlight_best:
            # Apply formatting with bold for best values
            latex_df[col] = latex_df[col].apply(
                lambda x: f"\\textbf{{{x:.4f}}}" if x == best_values[col] else f"{x:.4f}"
            )
        else:
            # Simple formatting without highlighting
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4f}")
    
    # Convert to LaTeX table
    latex_table = latex_df.to_latex(index=False, escape=False)
    
    # Add LaTeX table formatting
    latex_table = latex_table.replace("tabular", "tabular*{\\textwidth}")
    latex_table = latex_table.replace("llllllllll", "@{\\extracolsep{\\fill}}lrrrrrrrrrr")
    
    return latex_table

def format_markdown_table(df, highlight_best=True):
    """Format DataFrame as a Markdown table with optional best value highlighting."""
    # Copy DataFrame to avoid modifying original
    md_df = df.copy()
    
    # Columns to find best values for (excluding "Model")
    metric_cols = [col for col in md_df.columns if col != "Model"]
    
    if highlight_best:
        # Find best value for each metric column
        best_values = {}
        for col in metric_cols:
            if "Precision" in col or "F" in col or "AUC" in col or "WSS" in col:
                best_values[col] = md_df[col].max()
            elif "Recall" in col:
                best_values[col] = md_df[col].max()
    
    # Format numeric columns
    for col in metric_cols:
        if highlight_best:
            # Apply formatting with bold for best values
            md_df[col] = md_df[col].apply(
                lambda x: f"**{x:.4f}**" if x == best_values[col] else f"{x:.4f}"
            )
        else:
            # Simple formatting without highlighting
            md_df[col] = md_df[col].apply(lambda x: f"{x:.4f}")
    
    # Convert to Markdown table
    md_table = tabulate(md_df, headers="keys", tablefmt="pipe", showindex=False)
    
    return md_table

def create_compact_tables(df):
    """Create compact tables for different stages/experiments."""
    tables = {}
    
    # Table 1: Baseline Classifier Comparison (Stage 1)
    baseline_df = df[(df["experiment_type"] == "grid_search") & 
                     (df["normalization"] == "none") & 
                     (df["balancing"] == "none") &
                     (df["features"] == "text")]
    
    if not baseline_df.empty:
        tables["baseline"] = create_comprehensive_table(baseline_df)
    
    # Table 2: Normalization Impact (Stage 2)
    normalization_df = df[(df["experiment_type"] == "grid_search") & 
                          (df["normalization"] != "none") & 
                          (df["balancing"] == "none") &
                          (df["features"] == "text")]
    
    if not normalization_df.empty:
        # Add baseline SVM for comparison
        baseline_svm = baseline_df[baseline_df["base_model"] == "svm"]
        if not baseline_svm.empty:
            normalization_df = pd.concat([baseline_svm, normalization_df])
        
        tables["normalization"] = create_comprehensive_table(normalization_df)
    
    # Table 3: SMOTE Impact (Stage 2)
    smote_df = df[(df["experiment_type"] == "grid_search") & 
                  (df["balancing"] == "smote") &
                  (df["features"] == "text")]
    
    if not smote_df.empty:
        # Add baseline SVM for comparison
        baseline_svm = baseline_df[baseline_df["base_model"] == "svm"]
        if not baseline_svm.empty:
            smote_df = pd.concat([baseline_svm, smote_df])
        
        tables["smote"] = create_comprehensive_table(smote_df)
    
    # Table 4: Isolation Experiments (Stage 3)
    isolation_df = df[df["experiment_type"] == "isolation"]
    
    if not isolation_df.empty:
        tables["isolation"] = create_comprehensive_table(isolation_df)
    
    # Table 5: Criteria Experiments (Stage 4)
    criteria_df = df[df["experiment_type"] == "criteria"]
    
    if not criteria_df.empty:
        # Add best isolation model for comparison
        best_isolation = isolation_df.loc[isolation_df["balanced_f1"].idxmax()]
        criteria_df = pd.concat([pd.DataFrame([best_isolation]), criteria_df])
        
        tables["criteria"] = create_comprehensive_table(criteria_df)
    
    return tables

def perform_significance_tests(df, predictions_dir="results"):
    """Perform significance tests between key model pairs."""
    # We need the actual predictions for this, which should be in predictions.csv files
    # First, find all the prediction files
    pred_files = []
    for root, dirs, files in os.walk(predictions_dir):
        if "predictions.csv" in files:
            model_dir = root
            model_name = os.path.basename(model_dir)
            
            # Match with our results DataFrame
            matching_rows = df[df["model_directory"] == model_dir]
            if not matching_rows.empty:
                pred_files.append((
                    model_dir,
                    os.path.join(root, "predictions.csv"),
                    matching_rows.iloc[0]["formatted_name"] 
                    if "formatted_name" in matching_rows.columns else model_name
                ))
    
    if not pred_files:
        print("No prediction files found for significance testing.")
        return {}
    
    print(f"Found {len(pred_files)} prediction files for significance testing.")
    
    # Load predictions
    predictions = {}
    for model_dir, pred_file, model_name in pred_files:
        try:
            pred_df = pd.read_csv(pred_file)
            predictions[model_name] = {
                "true_labels": pred_df["true_label"].values,
                "predicted_labels": pred_df["predicted_label"].values,
                "hr_predicted_labels": pred_df["high_recall_pred"].values if "high_recall_pred" in pred_df.columns else None
            }
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
    
    # Perform McNemar's test between key pairs
    test_results = {}
    
    # Get model names
    model_names = list(predictions.keys())
    
    # Compare each model with each other
    for i, model1 in enumerate(model_names):
        for model2 in enumerate(model_names[i+1:], i+1):
            model2 = model_names[model2[0]]
            
            # Skip if either model is missing predictions
            if model1 not in predictions or model2 not in predictions:
                continue
            
            # Get predictions
            true_labels = predictions[model1]["true_labels"]
            pred1 = predictions[model1]["predicted_labels"]
            pred2 = predictions[model2]["predicted_labels"]
            
            # Create contingency table for McNemar's test
            # Both correct, model1 correct & model2 wrong, model1 wrong & model2 correct, both wrong
            contingency = np.zeros((2, 2), dtype=int)
            
            for y_true, y1, y2 in zip(true_labels, pred1, pred2):
                contingency[int(y1 != y_true), int(y2 != y_true)] += 1
            
            # Perform test if there are disagreements
            if contingency[0, 1] + contingency[1, 0] > 0:
                try:
                    result = mcnemar(contingency, exact=False, correction=True)
                    test_results[f"{model1} vs {model2}"] = {
                        "statistic": result.statistic,
                        "p_value": result.pvalue,
                        "contingency": contingency.tolist(),
                        "significant": result.pvalue < 0.05
                    }
                except Exception as e:
                    print(f"Error performing McNemar's test for {model1} vs {model2}: {e}")
            else:
                test_results[f"{model1} vs {model2}"] = {
                    "statistic": 0,
                    "p_value": 1.0,
                    "contingency": contingency.tolist(),
                    "significant": False,
                    "note": "No disagreements between models"
                }
    
    return test_results

def main():
    """Main function to generate final paper tables."""
    # Create output directory
    os.makedirs("paper_results", exist_ok=True)
    
    # Find and load all results
    result_files = find_all_results()
    
    if not result_files:
        print("No result files found!")
        return
    
    print(f"Found {len(result_files)} result files.")
    
    # Load and process results
    df = load_and_process_results(result_files)
    print(f"Loaded data for {len(df)} models.")
    
    # Create comprehensive table
    print("Creating comprehensive comparison table...")
    comp_table = create_comprehensive_table(df)
    
    # Sort by balanced F1 score
    comp_table = comp_table.sort_values("F₁", ascending=False)
    
    # Format and save as LaTeX
    latex_table = format_latex_table(comp_table)
    with open("paper_results/comprehensive_table.tex", "w") as f:
        f.write(latex_table)
    
    # Format and save as Markdown
    md_table = format_markdown_table(comp_table)
    with open("paper_results/comprehensive_table.md", "w") as f:
        f.write("# Comprehensive Model Comparison\n\n")
        f.write(md_table)
    
    # Create compact tables for different stages
    print("Creating compact tables for different stages...")
    tables = create_compact_tables(df)
    
    # Save each table
    for table_name, table_df in tables.items():
        # Sort by balanced F1 score
        table_df = table_df.sort_values("F₁", ascending=False)
        
        # Format and save as LaTeX
        latex_table = format_latex_table(table_df)
        with open(f"paper_results/{table_name}_table.tex", "w") as f:
            f.write(latex_table)
        
        # Format and save as Markdown
        md_table = format_markdown_table(table_df)
        with open(f"paper_results/{table_name}_table.md", "w") as f:
            f.write(f"# {table_name.title()} Comparison\n\n")
            f.write(md_table)
    
    # Perform significance tests
    print("Performing significance tests between models...")
    significance_results = perform_significance_tests(df)
    
    # Save significance test results
    if significance_results:
        with open("paper_results/significance_tests.md", "w") as f:
            f.write("# Significance Test Results\n\n")
            
            # Format and write results
            for comparison, result in significance_results.items():
                f.write(f"## {comparison}\n\n")
                f.write(f"* Statistic: {result['statistic']:.4f}\n")
                f.write(f"* p-value: {result['p_value']:.4f}\n")
                f.write(f"* Significant: {'Yes' if result['significant'] else 'No'}\n")
                
                if "note" in result:
                    f.write(f"* Note: {result['note']}\n")
                
                # Add contingency table explanation
                f.write("\nContingency table:\n\n")
                f.write("```\n")
                f.write("                  | Model 2 Correct | Model 2 Wrong |\n")
                f.write("------------------|-----------------|---------------|\n")
                f.write(f"Model 1 Correct  | {result['contingency'][0][0]:15d} | {result['contingency'][0][1]:14d} |\n")
                f.write(f"Model 1 Wrong    | {result['contingency'][1][0]:15d} | {result['contingency'][1][1]:14d} |\n")
                f.write("```\n\n")
                
                # Add interpretation
                if result['significant']:
                    # Determine which model is better
                    if result['contingency'][0][1] > result['contingency'][1][0]:
                        better_model = comparison.split(" vs ")[0]
                    else:
                        better_model = comparison.split(" vs ")[1]
                    
                    f.write(f"**Interpretation**: There is a statistically significant difference between the models (p<0.05). {better_model} performs significantly better.\n\n")
                else:
                    f.write("**Interpretation**: There is no statistically significant difference between the models (p≥0.05).\n\n")
                
                f.write("---\n\n")
        
        print("Significance test results saved to paper_results/significance_tests.md")
    
    print("All paper tables generated and saved to paper_results/ directory.")

if __name__ == "__main__":
    main()