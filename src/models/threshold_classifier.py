"""
Threshold customization wrapper for classification with serializable implementation.
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    A wrapper classifier that applies a custom threshold to the prediction probabilities
    while maintaining the pipeline's fitted state.
    """
    
    def __init__(self, base_classifier, threshold=0.5):
        self.base_classifier = base_classifier
        self.threshold = threshold
        self._fitted_classifier = None
        
    def fit(self, X, y):
        """Properly clone and fit the base classifier."""
        self._fitted_classifier = clone(self.base_classifier)
        self._fitted_classifier.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Ensure we use the fitted instance."""
        if self._fitted_classifier is None:
            raise ValueError("ThresholdClassifier not fitted. Call fit before predict_proba.")
        return self._fitted_classifier.predict_proba(X)
    
    def predict(self, X):
        """Apply threshold to fitted classifier's probabilities."""
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= self.threshold).astype(int)