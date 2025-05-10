# src/models/model_factory.py
"""
Model factory for creating different classifier pipelines.

This module provides a unified interface for creating model pipelines
with different classifiers, text normalization, and class balancing.

Supported classifiers:
- Logistic Regression
- SVM
- Cosine Similarity
- Complement Naive Bayes (CNB)
"""
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.models.classifiers import (
    make_tfidf_logreg_pipeline, 
    make_tfidf_svm_pipeline,
    make_tfidf_cosine_pipeline,
)

from src.models.classifiers import TextCombiner, CosineSimilarityClassifier
from src.models.text_processors import NormalizingTextCombiner

class ClassifierPipeline(Pipeline, ClassifierMixin):
    _estimator_type = "classifier"
    
    def __init__(self, steps, memory=None):
        super().__init__(steps, memory=memory)
    
    def predict(self, X):
        clf = self.named_steps["clf"]
        if hasattr(clf, "threshold"):
            prob = self.predict_proba(X)[:, 1]
            return (prob >= clf.threshold).astype(int)
        else:
            return super().predict(X)

def make_tfidf_cnb_pipeline(
    max_features=10000,
    ngram_range=(1, 2), min_df=1,
    max_df=0.9, text_columms=["title", "abstract"], alpha=1.0, norm=True
):
    base = make_tfidf_logreg_pipeline(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        text_columns=text_columns,
    )
    return Pipeline([
        ("combiner", base.named_steps["combiner"]),
        ("tfidf", base.named_steps["tfidf"]),
        ("clf", ComplementNB(alpha=alpha, norm=norm)),
    ])

def create_model(
    model_type="logreg",
    normalization=None,
    balancing=None,
    max_features=10000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.9,
    text_columns=("title", "abstract"),
    C=1.0,
    class_weight="balanced",
    threshold=None,
    kernel="linear",
    alpha=1.0,
    cache_dir=None,
):
    """
    Create a model pipeline with the specified configuration.
    
    Args:
        model_type: Type of model ('logreg', 'svm', 'cosine', 'cnb')
        normalization: Type of text normalization (None, 'stemming', or 'lemmatization')
        balancing: Class balancing strategy (None, 'smote', or 'undersample')
        max_features: Maximum features for TF-IDF
        ngram_range: N-gram range for TF-IDF
        min_df: Minimum document frequency for TF-IDF
        max_df: Maximum document frequency for TF-IDF
        text_columns: Text columns to combine
        C: Regularization parameter for logistic regression or SVM
        class_weight: Class weight strategy
        threshold: Decision threshold for cosine similarity
        kernel: Kernel type for SVM
        alpha: Smoothing parameter for Complement Naive Bayes
        
    Returns:
        Pipeline: Scikit-learn pipeline configured with the specified parameters
    """
    base = []
    memory = cache_dir if normalization is not None else None
    pipeline = ClassifierPipeline(base, memory=memory)
    
    if normalization:
        base.append(
            (
                "normalizer",
                NormalizingTextCombiner(
                    text_columns=text_columns, technique=normalization
                ),
            )
        )
    else:
        base.append(("combiner", TextCombiner(text_columns)))
    
    base.append(
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
        )
    )
    
    if model_type == "logreg":
        base.append(
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=5000,
                    random_state=42,
                ),
            )
        )
    elif model_type == "svm":
        base.append(
            (
                "clf",
                SVC(
                    C=C,
                    class_weight=class_weight,
                    kernel=kernel,
                    probability=True,
                    random_state=42,
                ),
            )
        )
    elif model_type == "cosine":
        base.append(("clf", CosineSimilarityClassifier(threshold=threshold)))
    elif model_type == "cnb":
        base.append(
            (
                "clf",
                ComplementNB(alpha=alpha),
            )
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    

    if balancing:
        if balancing == "smote":
            sampler = SMOTE(random_state=42)
        elif balancing == "undersample":
            sampler = RandomUnderSampler(random_state=42)
        else:
            raise ValueError(f"Unsupported balancing strategy: {balancing}")
        
        steps_with_sampler = pipeline.base[:-1] + [("sampler", sampler)] + [pipeline.base[-1]]
        pipeline = ImbPipeline(steps_with_sampler)
    
    return pipeline

def create_pipeline(model_type, **kwargs):
    if model_type == "logreg":
        base = make_tfidf_logreg_pipeline(**kwargs)
    elif model_type == "svm":
        base = make_tfidf_svm_pipeline(**kwargs)
    elif model_type == "cosine":
        base = make_tfidf_cosine_pipeline(**kwargs)
    elif model_type == "cnb":
        base = make_tfidf_cnb_pipeline(**kwargs)
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    return ClassifierPipeline(base.steps)