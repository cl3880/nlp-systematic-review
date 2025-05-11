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

#
# Feature Importance Functions
#

def get_logreg_feature_importance(model, n=20, class_names=["Irrelevant", "Relevant"]):
    """
    Extract the most important features for each class from a trained logistic regression model.

    Args:
        model: Trained pipeline with TfidfVectorizer and LogisticRegression
        n: Number of top features to extract (default: 20)
        class_names: Names of the classes (default: ['Irrelevant', 'Relevant'])

    Returns:
        DataFrame containing feature names and their coefficients
    """
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["clf"]

        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]

        features_df = pd.DataFrame(
            {"feature": feature_names, "coefficient": coefficients}
        )

        features_df["abs_coef"] = features_df["coefficient"].abs()
        features_df = features_df.sort_values("abs_coef", ascending=False)

        features_df["class"] = features_df["coefficient"].apply(
            lambda x: class_names[1] if x > 0 else class_names[0]
        )

        top_relevant = (
            features_df[features_df["coefficient"] > 0]
            .sort_values("coefficient", ascending=False)
            .head(n)
        )

        top_irrelevant = (
            features_df[features_df["coefficient"] < 0]
            .sort_values("coefficient", ascending=True)
            .head(n)
        )

        return pd.concat([top_relevant, top_irrelevant])

    except Exception as e:
        logger.error(f"Error extracting top features: {e}")
        return pd.DataFrame()

def get_svm_feature_importance(model, n=20, class_names=["Irrelevant", "Relevant"]):
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
    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["clf"]
        
        # Check if SVM is linear
        if classifier.kernel != "linear":
            logger.warning("Feature importance only available for linear SVM")
            return pd.DataFrame()
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients - for binary classification, we can use the first class
        coefficients = classifier.coef_[0]
        
        features_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients
        })
        
        features_df["abs_coef"] = features_df["coefficient"].abs()
        features_df = features_df.sort_values("abs_coef", ascending=False)
        
        features_df["class"] = features_df["coefficient"].apply(
            lambda x: class_names[1] if x > 0 else class_names[0]
        )
        
        top_relevant = features_df[features_df["coefficient"] > 0].sort_values(
            "coefficient", ascending=False
        ).head(n)
        
        top_irrelevant = features_df[features_df["coefficient"] < 0].sort_values(
            "coefficient", ascending=True
        ).head(n)
        
        return pd.concat([top_relevant, top_irrelevant])
        
    except Exception as e:
        logger.error(f"Error extracting SVM feature importance: {e}")
        return pd.DataFrame()

def get_feature_importance(model, n=30, class_names=["Irrelevant", "Relevant"]):
    """
    Extract the most important features from a trained model.
    Works with both LogisticRegression and linear SVM models.
    
    Args:
        model: Trained pipeline with TfidfVectorizer and classifier
        n: Number of top features to extract (default: 30)
        class_names: Names of the classes (default: ['Irrelevant', 'Relevant'])
        
    Returns:
        DataFrame containing feature names and their coefficients
    """
    try:
        if 'tfidf' not in model.named_steps or 'clf' not in model.named_steps:
            return pd.DataFrame()
            
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']
        
        if not hasattr(classifier, 'coef_'):
            return pd.DataFrame()
            
        if hasattr(classifier, 'kernel') and classifier.kernel != 'linear':
            return pd.DataFrame()
        
        feature_names = vectorizer.get_feature_names_out()
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
        
        top_relevant = (
            features_df[features_df['coefficient'] > 0]
            .sort_values('coefficient', ascending=False)
            .head(n)
        )
        
        top_irrelevant = (
            features_df[features_df['coefficient'] < 0]
            .sort_values('coefficient', ascending=True)
            .head(n)
        )
        
        return pd.concat([top_relevant, top_irrelevant])
        
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        return pd.DataFrame()

def plot_feature_importance(model, model_name, base_dir="results", n=20, class_names=["Irrelevant", "Relevant"]):
    """
    Plot and save the most important features for a model.
    
    Args:
        model: Trained pipeline with TfidfVectorizer and classifier
        model_name: Name of the model (logreg, svm)
        base_dir: Base directory for results (default: 'results')
        n: Number of top features to show (default: 20)
        class_names: Names of the classes (default: ['Irrelevant', 'Relevant'])
        
    Returns:
        tuple: (DataFrame of features, path to saved plot)
    """
    features_df = get_feature_importance(model, n, class_names)
    
    if features_df.empty:
        print(f"No features to plot for {model_name} model")
        return features_df, None
    
    plots_dir = os.path.join(base_dir, "plots", "feature_importance")
    os.makedirs(plots_dir, exist_ok=True)
    
    metrics_dir = os.path.join(base_dir, "metrics", "feature_importance")
    os.makedirs(metrics_dir, exist_ok=True)
    
    features_dict = features_df.to_dict(orient="records")
    with open(f"{metrics_dir}/{model_name}_features.json", "w") as f:
        import json
        json.dump(features_dict, f, indent=2)
    
    output_path = f"{plots_dir}/{model_name}_top_features.png"
    
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
    
    print(f"Feature importance plot saved to {output_path}")
    
    return features_df, output_path 