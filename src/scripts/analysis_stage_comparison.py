#!/usr/bin/env python3
"""
Analyze Stage 2 and 3 results to prepare for Stage 4.

This script:
1. Loads results from grid search experiments (Stage 2)
2. Loads results from isolation experiments (Stage 3)
3. Creates comparative visualizations and tables
4. Identifies the best models for Stage 4 criteria experiments
"""
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def find_result_files(base_dirs=["results/v2", "results/v3", "results"]):
    """Find all metrics.json files from experiments."""
    metrics_files = []
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            # Find all metrics.json files in subdirectories
            metrics_files.extend(glob.glob(f"{base_dir}/**/metrics.json", recursive=True))
    
    if not metrics_files:
        print("No result files found!")
    else:
        print(f"Found {len(metrics_files)} result files.")
    
    return metrics_files

def load_results(metrics_files):
    """Load metrics from json files into a structured format."""
    results = []
    
    for mf in metrics_files:
        model_dir = os.path.dirname(mf)
        model_name = os.path.basename(model_dir)
        
        # Extract experiment type (grid search or isolation)
        if "v3" in model_dir:
            experiment_type = "isolation"
        else:
            experiment_type = "grid_search"
        
        # Determine model configuration
        has_stemming = "stemming" in model_name
        has_lemmatization = "lemmatization" in model_name
        has_smote = "smote" in model_name
        has_criteria = "criteria" in model_name
        has_mesh = "mesh" in model_name
        
        # Load metrics
        try:
            with open(mf, 'r') as f:
                metrics = json.load(f)
                
            # Check if model has balanced and high-recall results
            if "balanced" in metrics and "high_recall" in metrics:
                results.append({
                    "model_name": model_name,
                    "experiment_type": experiment_type,
                    "has_stemming": has_stemming,
                    "has_lemmatization": has_lemmatization,
                    "has_smote": has_smote,
                    "has_criteria": has_criteria,
                    "has_mesh": has_mesh,
                    "balanced_precision": metrics["balanced"]["precision"],
                    "balanced_recall": metrics["balanced"]["recall"],
                    "balanced_f1": metrics["balanced"]["f1"],
                    "balanced_f2": metrics["balanced"]["f2"] if "f2" in metrics["balanced"] else None,
                    "balanced_roc_auc": metrics["balanced"]["roc_auc"],
                    "balanced_wss_at_95": metrics["balanced"]["wss@95"] if "wss@95" in metrics["balanced"] else (
                        metrics["balanced"]["wss_at_95"] if "wss_at_95" in metrics["balanced"] else None
                    ),
                    "hr_precision": metrics["high_recall"]["precision"],
                    "hr_recall": metrics["high_recall"]["recall"],
                    "hr_f1": metrics["high_recall"]["f1"],
                    "hr_f2": metrics["high_recall"]["f2"] if "f2" in metrics["high_recall"] else None,
                    "hr_roc_auc": metrics["high_recall"]["roc_auc"],
                    "hr_wss_at_95": metrics["high_recall"]["wss@95"] if "wss@95" in metrics["high_recall"] else (
                        metrics["high_recall"]["wss_at_95"] if "wss_at_95" in metrics["high_recall"] else None
                    ),
                    "threshold": metrics.get("threshold", None),
                    "model_directory": model_dir
                })
        except Exception as e:
            print(f"Error loading {mf}: {e}")
    
    return results

def format_model_name(result):
    """Create a standardized model name for comparison."""
    components = []
    
    if result["has_stemming"]:
        components.append("Stemming")
    elif result["has_lemmatization"]:
        components.append("Lemmatization")
    else:
        components.append("Raw")
        
    if result["has_smote"]:
        components.append("SMOTE")
        
    if result["has_criteria"]:
        components.append("Criteria")
        
    if result["has_mesh"]:
        components.append("MeSH")
        
    components.append("SVM")
    
    if result["experiment_type"] == "isolation":
        components.append("(fixed params)")
    elif result["experiment_type"] == "grid_search":
        components.append("(tuned)")
        
    return " + ".join(components)

def create_results_dataframe(results):
    """Create a clean DataFrame for analysis."""
    df = pd.DataFrame(results)
    
    # Add formatted model name
    df["formatted_name"] = df.apply(format_model_name, axis=1)
    
    # Fix any NaN values
    for col in df.columns:
        if df[col].dtype == float:
            df[col] = df[col].fillna(0)
    
    return df

def plot_model_comparison(df, metric="balanced_f1", top_n=10, title=None, output_path=None):
    """Plot horizontal bar chart comparing models by a metric."""
    # Sort and select top N models
    sorted_df = df.sort_values(metric, ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar chart
    ax = sns.barplot(
        y="formatted_name",
        x=metric,
        data=sorted_df,
        hue="experiment_type",
        dodge=False
    )
    
    # Add value labels
    for i, v in enumerate(sorted_df[metric]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    
    # Set title and labels
    if title:
        plt.title(title)
    else:
        plt.title(f"Top {top_n} Models by {metric.replace('_', ' ').title()}")
        
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Model")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_precision_recall_comparison(df, output_path=None):
    """Create a scatter plot showing precision-recall trade-offs across models."""
    plt.figure(figsize=(12, 10))
    
    # Plot balanced models
    sns.scatterplot(
        x="balanced_recall",
        y="balanced_precision",
        hue="formatted_name",
        style="experiment_type",
        s=100,
        data=df
    )
    
    # Add F1 contours
    x = np.linspace(0.01, 1, 100)
    for f1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y = (f1 * x) / (2 * x - f1)
        plt.plot(x, y, '--', color='gray', alpha=0.5, linewidth=1)
        # Add F1 label at rightmost valid point
        valid_idx = np.where(y <= 1)[0]
        if len(valid_idx) > 0:
            rightmost_idx = valid_idx[-1]
            plt.annotate(
                f'F1={f1}',
                xy=(x[rightmost_idx], y[rightmost_idx]),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8
            )
    
    plt.title("Precision-Recall Trade-offs Across Models")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def create_summary_table(df, metric="balanced_f1", top_n=8):
    """Create a summary table of the top models."""
    # Select columns for the table
    cols = [
        "formatted_name", 
        "balanced_precision", "balanced_recall", "balanced_f1", "balanced_roc_auc", "balanced_wss_at_95",
        "hr_precision", "hr_recall", "hr_f1", "hr_roc_auc", "hr_wss_at_95"
    ]
    
    # Sort by the specified metric and select top N
    sorted_df = df.sort_values(metric, ascending=False).head(top_n)
    
    # Select columns and format for display
    table_df = sorted_df[cols].copy()
    
    # Rename columns for readability
    table_df.columns = [
        "Model",
        "Prec.", "Recall", "F1", "AUC", "WSS@95",
        "HR Prec.", "HR Recall", "HR F1", "HR AUC", "HR WSS@95"
    ]
    
    # Format floats
    for col in table_df.columns:
        if col != "Model":
            table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")
    
    # Return as tabulate format
    return tabulate(table_df, headers="keys", tablefmt="pipe", showindex=False)

def find_best_models(df):
    """Identify the best models for different objectives."""
    best_models = {
        "balanced_f1": {
            "grid_search": None,
            "isolation": None
        },
        "hr_f1": {
            "grid_search": None,
            "isolation": None
        }
    }
    
    # Find best models from grid search
    grid_df = df[df["experiment_type"] == "grid_search"]
    if not grid_df.empty:
        best_balanced_grid_idx = grid_df["balanced_f1"].idxmax()
        best_hr_grid_idx = grid_df["hr_f1"].idxmax()
        
        best_models["balanced_f1"]["grid_search"] = grid_df.loc[best_balanced_grid_idx]
        best_models["hr_f1"]["grid_search"] = grid_df.loc[best_hr_grid_idx]
    
    # Find best models from isolation experiments
    iso_df = df[df["experiment_type"] == "isolation"]
    if not iso_df.empty:
        best_balanced_iso_idx = iso_df["balanced_f1"].idxmax()
        best_hr_iso_idx = iso_df["hr_f1"].idxmax()
        
        best_models["balanced_f1"]["isolation"] = iso_df.loc[best_balanced_iso_idx]
        best_models["hr_f1"]["isolation"] = iso_df.loc[best_hr_iso_idx]
    
    return best_models

def generate_stage4_recommendations(best_models):
    """Generate recommendations for Stage 4 experiments."""
    recommendations = []
    
    # Generate recommendations based on best balanced F1 model
    if best_models["balanced_f1"]["isolation"] is not None:
        best_bal_iso = best_models["balanced_f1"]["isolation"]
        
        # Base configuration for criteria experiments
        base_cmd = "python src/scripts/criteria_features_nogrid.py --model svm"
        
        # Add normalization if needed
        if best_bal_iso["has_stemming"]:
            base_cmd += " --normalization stemming"
        elif best_bal_iso["has_lemmatization"]:
            base_cmd += " --normalization lemmatization"
            
        # Add SMOTE if needed
        if best_bal_iso["has_smote"]:
            base_cmd += " --balancing smote"
            
        # Create recommendations
        recommendations.append(
            f"# Best balanced model: {best_bal_iso['formatted_name']}\n"
            f"# F1: {best_bal_iso['balanced_f1']:.4f}, Precision: {best_bal_iso['balanced_precision']:.4f}, Recall: {best_bal_iso['balanced_recall']:.4f}\n"
            f"\n"
            f"# Add criteria features\n"
            f"{base_cmd}\n"
            f"\n"
            f"# Add criteria + MeSH features\n"
            f"{base_cmd} --use-mesh\n"
        )
    
    # Generate recommendations based on best high-recall F1 model
    if best_models["hr_f1"]["isolation"] is not None:
        best_hr_iso = best_models["hr_f1"]["isolation"]
        
        # Only add if different from balanced model
        if (best_models["balanced_f1"]["isolation"] is None or 
            best_hr_iso["formatted_name"] != best_models["balanced_f1"]["isolation"]["formatted_name"]):
            
            # Base configuration for criteria experiments
            base_cmd = "python src/scripts/criteria_features_nogrid.py --model svm"
            
            # Add normalization if needed
            if best_hr_iso["has_stemming"]:
                base_cmd += " --normalization stemming"
            elif best_hr_iso["has_lemmatization"]:
                base_cmd += " --normalization lemmatization"
                
            # Add SMOTE if needed
            if best_hr_iso["has_smote"]:
                base_cmd += " --balancing smote"
                
            # Create recommendations
            recommendations.append(
                f"# Best high-recall model: {best_hr_iso['formatted_name']}\n"
                f"# F1: {best_hr_iso['hr_f1']:.4f}, Precision: {best_hr_iso['hr_precision']:.4f}, Recall: {best_hr_iso['hr_recall']:.4f}\n"
                f"\n"
                f"# Add criteria features\n"
                f"{base_cmd}\n"
                f"\n"
                f"# Add criteria + MeSH features\n"
                f"{base_cmd} --use-mesh\n"
            )
    
    return "\n".join(recommendations)

def main():
    """Main function to analyze results and generate recommendations."""
    # Create output directory
    os.makedirs("analysis", exist_ok=True)
    
    # Find and load result files
    metrics_files = find_result_files()
    results = load_results(metrics_files)
    
    if not results:
        print("No results to analyze!")
        return
    
    # Create DataFrame
    df = create_results_dataframe(results)
    print(f"Loaded data for {len(df)} models.")
    
    # Save to CSV
    df.to_csv("analysis/model_comparison.csv", index=False)
    print("Saved model comparison to analysis/model_comparison.csv")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Balanced F1 comparison
    plot_model_comparison(
        df, metric="balanced_f1", 
        title="Top Models by Balanced F1 Score",
        output_path="analysis/balanced_f1_comparison.png"
    )
    
    # High-recall F1 comparison
    plot_model_comparison(
        df, metric="hr_f1", 
        title="Top Models by High-Recall F1 Score",
        output_path="analysis/high_recall_f1_comparison.png"
    )
    
    # WSS@95 comparison
    plot_model_comparison(
        df, metric="balanced_wss_at_95", 
        title="Top Models by WSS@95",
        output_path="analysis/wss_comparison.png"
    )
    
    # Precision-recall comparison
    plot_precision_recall_comparison(df, output_path="analysis/precision_recall_comparison.png")
    
    # Create summary tables
    print("\nTop Models by Balanced F1:")
    balanced_table = create_summary_table(df, metric="balanced_f1")
    print(balanced_table)
    
    print("\nTop Models by High-Recall F1:")
    hr_table = create_summary_table(df, metric="hr_f1")
    print(hr_table)
    
    # Save tables to files
    with open("analysis/balanced_f1_table.md", "w") as f:
        f.write("# Top Models by Balanced F1\n\n")
        f.write(balanced_table)
    
    with open("analysis/high_recall_f1_table.md", "w") as f:
        f.write("# Top Models by High-Recall F1\n\n")
        f.write(hr_table)
    
    # Find best models and generate recommendations
    best_models = find_best_models(df)
    
    print("\nBest Models:")
    for objective in ["balanced_f1", "hr_f1"]:
        for exp_type in ["grid_search", "isolation"]:
            if best_models[objective][exp_type] is not None:
                model = best_models[objective][exp_type]
                obj_name = "Balanced F1" if objective == "balanced_f1" else "High-Recall F1"
                exp_name = "Grid Search" if exp_type == "grid_search" else "Isolation"
                
                precision_key = "balanced_precision" if objective == "balanced_f1" else "hr_precision"
                recall_key = "balanced_recall" if objective == "balanced_f1" else "hr_recall"
                                
                print(f"Best {obj_name} - {exp_name}: {model['formatted_name']}")
                print(f"  F1: {model[objective]:.4f}, Precision: {model[precision_key]:.4f}, Recall: {model[recall_key]:.4f}")    
    # Generate Stage 4 recommendations
    recommendations = generate_stage4_recommendations(best_models)
    
    with open("analysis/stage4_recommendations.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Stage 4 Experiment Recommendations\n")
        f.write("# Run these commands to execute the Stage 4 criteria experiments\n\n")
        f.write(recommendations)
    
    print("\nStage 4 recommendations saved to analysis/stage4_recommendations.sh")
    print("You can run chmod +x analysis/stage4_recommendations.sh to make it executable")

if __name__ == "__main__":
    main()