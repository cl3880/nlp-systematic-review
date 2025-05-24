# src/models/param_grids.py
"""
Parameter grids for grid search experiments.

This module defines parameter grids for different model types
to be used in grid search experiments.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from src.models.classifiers import CosineSimilarityClassifier

def get_common_tfidf_grid():
    """Common TF-IDF parameters for all models."""
    return {
        "tfidf__min_df": [1, 2, 3, 5, 10],
        "tfidf__max_df": [0.85, 0.9, 0.95, 1.0],
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 2), (1, 3)],
    }

def logreg_param_grid():
    """Parameter grid for Logistic Regression model."""
    grid = get_common_tfidf_grid()
    grid.update({
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": ["balanced"],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    })
    return grid

def svm_param_grid():
    """Parameter grid for SVM model."""
    grid = get_common_tfidf_grid()
    grid.update({
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": ["balanced"],
        "clf__kernel": ["linear"],
    })
    return grid

def cnb_param_grid():
    """Parameter grid for Complement Naive Bayes model."""
    grid = get_common_tfidf_grid()
    grid.update({
        "clf__alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
        "clf__norm": [True, False],
    })
    return grid

def cosine_param_grid():
    """Parameter grid for Cosine Similarity model."""
    g = get_common_tfidf_grid()
    g["clf__threshold"] = [0.1, 0.2, 0.25,0.3, 0.35, 0.4, 0.5]
    return g

def criteria_param_grid(model_type="svm"):
    """
    Define scientifically rigorous parameter grid for criteria-enhanced models.
    
    Parameters:
    -----------
    model_type : str
        Type of classifier to use ('logreg', 'svm')
        
    Returns:
    --------
    dict
        Parameter grid dictionary for GridSearchCV
    """
    if model_type == "svm":
        return {
            "features__text_features__tfidf__max_features": [5000, 10000, 20000],
            "features__text_features__tfidf__ngram_range": [(1, 2), (1, 3)],
            "features__text_features__tfidf__min_df": [3, 5, 10],
            "features__text_features__tfidf__max_df": [0.85, 0.9, 0.95],
            "clf__C": [0.1, 1, 10],
            "clf__class_weight": ["balanced"],
            "clf__kernel": ["linear"]
        }
    elif model_type == "logreg":
        return {
            "features__text_features__tfidf__max_features": [5000, 10000, 20000],
            "features__text_features__tfidf__ngram_range": [(1, 2), (1, 3)],
            "features__text_features__tfidf__min_df": [1, 3, 5],
            "features__text_features__tfidf__max_df": [0.85, 0.9, 0.95],
            "clf__C": [0.1, 1, 10],
            "clf__class_weight": ["balanced"],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear"]
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_param_grid(model_type):
    """
    Get parameter grid based on model type.
    
    Args:
        model_type: Type of model ('logreg', 'svm', 'cosine', 'cnb')
        
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    if model_type == "logreg":
        return logreg_param_grid()
    elif model_type == "svm":
        return svm_param_grid()
    elif model_type == "cosine":
        return cosine_param_grid()
    elif model_type == "cnb":
        return cnb_param_grid()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def test_param_grid():
    """
    A minimal parameter grid for testing purposes.
    Total combinations: 2 features × 1 ngram × 1 classifier = 2 combinations
    """
    return {
        "tfidf__max_features": [5000, 10000],
        "tfidf__ngram_range": [(1, 2)],
        "tfidf__min_df": [3],
        "tfidf__max_df": [0.95],
        "clf__C": [1.0],
        "clf__class_weight": ["balanced"],
    }