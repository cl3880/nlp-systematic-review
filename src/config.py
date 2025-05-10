# config.py
"""
Configuration settings for the systematic review classification project.
"""
PATHS = {
    "data_raw":        "data/raw",
    "data_processed":  "data/processed",
    "grid_results_dir":     "results",
    "logs_dir":        "results/logs",
    "baseline_dir":    "results/baseline",
    "build_data_dir":  "results/build_data",
    "normalization_dir": "results/normalization"
}

# v2.0.0 results structure
RESULTS_V2 = {
    "base_dir": "results/v2.0.0-f1-refactor",
    "base_models": {
        "logreg": "results/v2.0.0-f1-refactor/base_logreg",
        "svm": "results/v2.0.0-f1-refactor/base_svm",
        "cnb": "results/v2.0.0-f1-refactor/base_cnb",
        "cosine": "results/v2.0.0-f1-refactor/base_cosine"
    },
    "normalization": {
        "stemming": {
            "logreg": "results/v2.0.0-f1-refactor/normalization/stemming_logreg",
            "svm": "results/v2.0.0-f1-refactor/normalization/stemming_svm",
            "cnb": "results/v2.0.0-f1-refactor/normalization/stemming_cnb",
            "cosine": "results/v2.0.0-f1-refactor/normalization/stemming_cosine"
        },
        "lemmatization": {
            "logreg": "results/v2.0.0-f1-refactor/normalization/lemmatization_logreg",
            "svm": "results/v2.0.0-f1-refactor/normalization/lemmatization_svm",
            "cnb": "results/v2.0.0-f1-refactor/normalization/lemmatization_cnb",
            "cosine": "results/v2.0.0-f1-refactor/normalization/lemmatization_cosine"
        }
    }
}

def get_result_path_v2(model_type, normalization=None):
    """
    Get the path for a specific model and normalization in the v2.0.0 structure.
    
    Args:
        model_type: Model type (logreg, svm, cnb, cosine)
        normalization: Normalization technique (stemming, lemmatization, or None)
        
    Returns:
        str: Path to the results directory
    """
    if normalization:
        return RESULTS_V2["normalization"][normalization][model_type]
    else:
        return RESULTS_V2["base_models"][model_type]

MODEL_CONFIG = {
    "paths": {
        "output_dir":  "results/models",
        "cache_dir":   "cache",
        "log_file":    "results/logs/training.log",
        "models_dir":  "results/baseline/models",
        "metrics_dir": "results/baseline/metrics",
        "plots_dir":   "results/baseline/plots",
        "analysis_dir":"results/baseline/analysis"
    },
    
    "tfidf": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 3,
        "sublinear_tf": True,
        "stop_words": "english"
    },
    
    "classifier": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": 42
    },
    
    "evaluation": {
        "recall_threshold": 0.95,
        "cv_folds": 5
    },
    
    "regex": {
        "sample_size": r"(?:n\s*=\s*|sample\s+size\s*(?:of|was|:|=)\s*)(\d+)",
        "occlusion": r"(?:occlusion|obliteration)\s+(?:rate|percentage|ratio)",
        "adult_age": r"(?:\b(?:adult|mature|grown)\b|(?:older|elderly)\s+(?:than|adults?)|\b(?:age[ds]?|older)\s+(?:\d+|\w+teen))",
        "pediatric": r"\b(?:child|children|pediatric|infant|adolescent|neonatal|juvenile)\b",
        "age_range": r"(?:age|aged)\s+(?:range|between|from)?\s*(?:was|:)?\s*(\d+)(?:\s*[-–]\s*|\s+to\s+)(\d+)",
        "radiosurgery": r"\b(?:radiosurg|gamma\s+knife|cyberknife|novalis|linear\s+accelerator\s+(?:based\s+)?radiosurg)\w*\b",
        "brain_avm": r"\b(?:brain|cerebral|intracranial)\s+(?:arteriovenous\s+malformation|avm)\b"
    },
    
    "criteria": {
        "min_year": 2000,
        "sample_size_min": 10,
        "adult_age_min": 18
    }
}