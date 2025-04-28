# models/svm_classifier.py
"""
Support Vector Machine classifier implementation for systematic review classification.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances

import logging
from src.models.baseline_classifier import TextCombiner

logger = logging.getLogger(__name__)

def make_tfidf_svm_pipeline(max_features=10000, 
                           ngram_range=(1, 2),
                           min_df=3,
                           text_columns=['title', 'abstract'],
                           C=1.0,
                           class_weight='balanced',
                           kernel='linear'):
    """
    Create a scikit-learn pipeline combining TF-IDF and SVM.
    
    Args:
        max_features: Maximum number of features for TF-IDF (default: 10000)
        ngram_range: Range of n-grams to include (default: (1, 2))
        min_df: Minimum document frequency for terms (default: 3)
        text_columns: List of text columns to combine (default: ['title', 'abstract'])
        C: Regularization parameter (default: 1.0)
        class_weight: Class weights for imbalanced data (default: 'balanced')
        kernel: Kernel type to be used in the algorithm (default: 'linear')
        
    Returns:
        A scikit-learn Pipeline object
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    logger.info(f"Creating TF-IDF + SVM pipeline with {kernel} kernel")
    logger.debug(f"Parameters: max_features={max_features}, ngram_range={ngram_range}, "
                f"min_df={min_df}, C={C}, class_weight={class_weight}")
    
    return Pipeline([
        ('combiner', TextCombiner(text_columns)),
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english',
            sublinear_tf=True,
        )),
        ('clf', SVC(
            C=C,
            class_weight=class_weight,
            kernel=kernel,
            probability=True,
            random_state=42
        ))
    ])

def svm_param_grid():
    """
    Define parameter grid for SVM model hyperparameter search.
    
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    return {
        'tfidf__max_features': [5000, 10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [2, 3, 5],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__class_weight': ['balanced'],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': ['scale', 'auto'],
    }

def get_svm_feature_importance(model, n=20, class_names=['Irrelevant', 'Relevant']):
    """
    Extract the most important features from a trained SVM model.
    Only works with linear SVM.
    
    Args:
        model: Trained pipeline with TfidfVectorizer and SVM
        n: Number of top features to extract (default: 20)
        class_names: Names of the classes (default: ['Irrelevant', 'Relevant'])
        
    Returns:
        DataFrame containing feature names and their coefficients
    """
    import pandas as pd
    
    try:
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']
        
        # Check if SVM is linear
        if classifier.kernel != 'linear':
            logger.warning("Feature importance only available for linear SVM")
            return pd.DataFrame()
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients - for binary classification, we can use the first class
        coefficients = classifier.coef_[0]
        
        features_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        })
        
        features_df['abs_coef'] = features_df['coefficient'].abs()
        features_df = features_df.sort_values('abs_coef', ascending=False)
        
        features_df['class'] = features_df['coefficient'].apply(
            lambda x: class_names[1] if x > 0 else class_names[0]
        )
        
        top_relevant = features_df[features_df['coefficient'] > 0].sort_values(
            'coefficient', ascending=False
        ).head(n)
        
        top_irrelevant = features_df[features_df['coefficient'] < 0].sort_values(
            'coefficient', ascending=True
        ).head(n)
        
        return pd.concat([top_relevant, top_irrelevant])
        
    except Exception as e:
        logger.error(f"Error extracting SVM feature importance: {e}")
        return pd.DataFrame()

def plot_svm_features(model, output_path, n=20, class_names=['Irrelevant', 'Relevant']):
    """
    Plot and save the most important features for a linear SVM model.
    
    Args:
        model: Trained pipeline with TfidfVectorizer and SVM
        output_path: Path to save the plot
        n: Number of top features to show (default: 20)
        class_names: Names of the classes (default: ['Irrelevant', 'Relevant'])
        
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    features_df = get_svm_feature_importance(model, n, class_names)
    
    if features_df.empty:
        logger.warning("No features to plot for SVM model")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    relevant_features = features_df[features_df['class'] == class_names[1]].head(n)
    sns.barplot(x='coefficient', y='feature', data=relevant_features)
    plt.title(f'Top {n} Features Indicating {class_names[1]}')
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    irrelevant_features = features_df[features_df['class'] == class_names[0]].head(n)
    sns.barplot(x='coefficient', y='feature', data=irrelevant_features)
    plt.title(f'Top {n} Features Indicating {class_names[0]}')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    