# config.py
"""
Configuration settings for the systematic review classification project.
Version 3.0.0 - Cleaned and organized structure
"""

# Project version
VERSION = "3.0.0"

# Data paths
PATHS = {
    "data_raw": "data/raw",
    "data_processed": "data/processed", 
    "results_dir": "results_final",
    "logs_dir": "results_final/logs",
    "archive_dir": "archive"
}

# Organized results structure matching README stages and actual methodology
RESULTS_FINAL = {
    "stage1_baseline_gridsearch_skf": "results_final/stage1_baseline_gridsearch_skf",
    "stage2_normalization_smote_grid_search": "results_final/stage2_normalization_smote_grid_search",
    "stage3_svm_fixed_params_skf": "results_final/stage3_svm_fixed_params_skf", 
    "stage4_svm_fixed_params_features_skf": "results_final/stage4_svm_fixed_params_features_skf",
    "final_test_evaluation": "results_final/final_test_evaluation"
}

# Legacy results structure (for backward compatibility)
RESULTS_V2 = {
    "base_svm": "results/v2.0.0-f1-refactor/base_svm",
    "base_logreg": "results/v2.0.0-f1-refactor/logreg", 
    "base_cnb": "results/v2.0.0-f1-refactor/base_cnb",
    "normalization": "results/v2.0.0-f1-refactor/normalization",
    "smote_svm": "results/v2.0.0-f1-refactor/smote_svm"
}

# Model configurations
MODEL_CONFIGS = {
    "baseline_svm_params": {
        "clf__C": 1,
        "clf__class_weight": "balanced", 
        "clf__kernel": "linear",
        "tfidf__max_df": 0.85,
        "tfidf__max_features": 5000,
        "tfidf__min_df": 1,
        "tfidf__ngram_range": (1, 2)
    }
}

def get_result_path_final(stage):
    """Get path for final results by stage."""
    return RESULTS_FINAL.get(stage, "results_final/unknown")

def get_result_path_v2(model_type, normalization=None, balancing=None):
    """Get path for v2 results structure (legacy)."""
    if normalization:
        return f"results/v2.0.0-f1-refactor/normalization/{normalization}_{model_type}"
    elif balancing:
        return f"results/v2.0.0-f1-refactor/{balancing}_{model_type}"  
    else:
        return RESULTS_V2.get(f"base_{model_type}", f"results/v2.0.0-f1-refactor/base_{model_type}")

# Default cross-validation parameters
DEFAULT_CV_PARAMS = {
    "n_splits": 5,
    "shuffle": True, 
    "random_state": 42,
    "stratified": True  # All experiments use StratifiedKFold
}

# Text processing parameters 
TEXT_PROCESSING = {
    "default_text_columns": ["title", "abstract"],
    "normalization_options": ["stemming", "lemmatization", None],
    "balancing_options": ["smote", None]
}

# Feature extraction parameters
FEATURE_PARAMS = {
    "expert_criteria_features": [
        "population_brain_avm", "intervention_treatment", "outcome_clinical",
        "study_design_rct", "language_english", "full_text_available"
    ],
    "mesh_terms_column": "mesh_terms"
}

# Evaluation metrics
METRICS = {
    "primary": ["precision", "recall", "f1", "f2", "roc_auc"],
    "specialized": ["wss_at_95"],  # Work Saved over Sampling at 95% recall
    "thresholds": {
        "high_recall_target": 0.95,
        "balanced_threshold": 0.5
    }
}

# Plotting configuration
PLOT_CONFIG = {
    "figure_size": (10, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "husl"
}