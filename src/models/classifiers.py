"""
Classification models for systematic review classification.

This module contains all classifier implementations:
- TextCombiner: Utility for combining text fields
- LogisticRegressionClassifier: Parameters and pipeline for logistic regression
- SVMClassifier: Parameters and pipeline for SVM
- CosineSimilarityClassifier: Centroid-based similarity classifier
"""
import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

class TextCombiner(BaseEstimator, TransformerMixin):
    """
    Transformer to combine multiple text columns into a single text field.
    """
    def __init__(self, text_columns=["title", "abstract"]):
        self.text_columns = text_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Combine text columns with a space between them."""
        if hasattr(X, "iloc"):
            logger.debug(f"Combining text columns: {self.text_columns}")
            combined = X[self.text_columns[0]].fillna("")
            for col in self.text_columns[1:]:
                if col in X.columns:
                    combined = combined + " " + X[col].fillna("")
            return combined.values
        return X

class CosineSimilarityClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple 'classifier' that fits TF-IDF, computes centroid of positive docs,
    then scores new docs by cosine similarity to that centroid.
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def fit(self, X, y):
        mask = np.asarray(y, dtype=bool)
        positives = X[mask]

        if positives.shape[0] == 0:
            logger.warning("No positive examples—using zero-vector centroid")
            self.centroid_ = np.zeros((1, X.shape[1]))
        else:
            # sparse mean returns a 1×n_features matrix
            centroid_sparse = positives.mean(axis=0)
            # convert only the centroid to dense
            self.centroid_ = centroid_sparse.A if hasattr(centroid_sparse, "A") else centroid_sparse

        return self

    def predict_proba(self, X):
        # compute cosine similarity in sparse space
        sims = 1 - pairwise_distances(X, self.centroid_, metric="cosine", n_jobs=1).ravel()
        return np.vstack([1 - sims, sims]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

def make_tfidf_logreg_pipeline(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=3,
    text_columns=["title", "abstract"],
    C=1.0,
    class_weight="balanced",
):
    """
    Create a scikit-learn pipeline combining TF-IDF and logistic regression.

    Args:
        max_features: Maximum number of features for TF-IDF (default: 10000)
        ngram_range: Range of n-grams to include (default: (1, 3))
        min_df: Minimum document frequency for terms (default: 3)
        text_columns: List of text columns to combine (default: ['title', 'abstract'])
        C: Inverse regularization strength (default: 1.0)
        class_weight: Class weights for imbalanced data (default: 'balanced')

    Returns:
        A scikit-learn Pipeline object
    """
    logger.info("Creating TF-IDF + LogReg pipeline")
    logger.debug(
        f"Parameters: max_features={max_features}, ngram_range={ngram_range}, "
        f"min_df={min_df}, C={C}, class_weight={class_weight}"
    )

    return Pipeline(
        [
            ("combiner", TextCombiner(text_columns)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

def make_tfidf_svm_pipeline(
    max_features=10000, 
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.9,
    text_columns=["title", "abstract"],
    C=1.0,
    class_weight="balanced",
    kernel="linear"
):
    """
    Create a scikit-learn pipeline combining TF-IDF and SVM.
    
    Args:
        max_features: Maximum number of features for TF-IDF (default: 10000)
        ngram_range: Range of n-grams to include (default: (1, 3))
        min_df: Minimum document frequency for terms (default: 3)
        max_df: Maximum document frequency for terms (default: 0.9)
        text_columns: List of text columns to combine (default: ['title', 'abstract'])
        C: Regularization parameter (default: 1.0)
        class_weight: Class weights for imbalanced data (default: 'balanced')
        kernel: Kernel type to be used in the algorithm (default: 'linear')
        
    Returns:
        A scikit-learn Pipeline object
    """
    logger.info(f"Creating TF-IDF + SVM pipeline with {kernel} kernel")
    logger.debug(
        f"Parameters: max_features={max_features}, ngram_range={ngram_range}, "
        f"min_df={min_df}, max_df={max_df}, C={C}, class_weight={class_weight}"
    )
    
    return Pipeline([
        ("combiner", TextCombiner(text_columns)),
        (
            "tfidf", 
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words="english",
                sublinear_tf=True,
            )
        ),
        (
            "clf", 
            SVC(
                C=C,
                class_weight=class_weight,
                kernel=kernel,
                probability=True,
                random_state=42
            )
        )
    ])

def make_tfidf_cosine_pipeline(
    max_features=10000, 
    ngram_range=(1, 3), 
    min_df=3,
    max_df=0.9,
    text_columns=["title", "abstract"],
    threshold=None
):
    """
    Combines TextCombiner, TfidfVectorizer, and CosineSimilarityClassifier.
    """
    return Pipeline(
        [
            ("combiner", TextCombiner(text_columns)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_df=max_df,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("clf", CosineSimilarityClassifier(threshold=threshold)),
        ]
    )

def logreg_param_grid():
    """
    Define parameter grid for logistic regression hyperparameter search.

    Returns:
        dict: Parameter grid for GridSearchCV
    """
    return {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (2, 3)],
        "tfidf__min_df": [1, 2, 3, 5],
        "tfidf__max_df": [0.9, 0.95, 1.0],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": ["balanced"],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    }

def svm_param_grid():
    """
    Define parameter grid for SVM model hyperparameter search.
    
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    return {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3), (2, 3)],
        "tfidf__min_df": [2, 3, 5],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": ["balanced"],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],
    }