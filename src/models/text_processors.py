# src/models/text_processors.py
"""
Text processing utilities for systematic review classification.

This module centralizes text processing classes used across the project:
- TextCombiner: Combines multiple text columns into a single text field
- TextNormalizer: Applies text normalization techniques (stemming, lemmatization)
- NormalizingTextCombiner: Combines text columns and applies normalization
"""
import logging
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer, WordNetLemmatizer

logger = logging.getLogger(__name__)

def ensure_nltk_resources():
    """Make sure NLTK resources are downloaded."""
    checks = {
        "punkt": "tokenizers/punkt",
        "wordnet": "corpora/wordnet",
    }
    for pkg, path in checks.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

class TextCombiner(BaseEstimator, TransformerMixin):
    """
    Transformer to combine multiple text columns into a single text field.
    Useful when processing DataFrames with title and abstract columns.
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

class TextNormalizer(BaseEstimator, TransformerMixin):
    """Class for normalizing text using stemming or lemmatization."""
    def __init__(self, technique=None):
        self.technique = technique
        ensure_nltk_resources()
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Apply the selected normalization technique to each document."""
        if self.technique is None:
            return X
        return [self.normalize_doc(doc) for doc in X]

    def normalize_doc(self, doc):
        """Apply normalization to a single document."""
        if isinstance(doc, str):
            tokens = doc.split()
            if self.technique == 'stemming':
                tokens = [self.stemmer.stem(t) for t in tokens]
            elif self.technique == 'lemmatization':
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            return ' '.join(tokens)
        return ""

class NormalizingTextCombiner(BaseEstimator, TransformerMixin):
    """Combines text columns and applies normalization in one step."""
    def __init__(self, text_columns=["title", "abstract"], technique=None):
        self.text_columns = text_columns
        self.technique = technique
        self.combiner = TextCombiner(text_columns)
        self.normalizer = TextNormalizer(technique)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Combine text columns and apply normalization."""
        combined = self.combiner.transform(X)
        return self.normalizer.transform(combined)