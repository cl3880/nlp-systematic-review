# Logistic Regression Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__C: 100
- clf__class_weight: balanced
- clf__penalty: l2
- clf__solver: liblinear
- tfidf__max_df: 0.85
- tfidf__max_features: 20000
- tfidf__min_df: 1
- tfidf__ngram_range: (1, 3)

### High-Recall Model (95% Target)
- clf__C: 100
- clf__class_weight: balanced
- clf__penalty: l2
- clf__solver: liblinear
- tfidf__max_df: 0.85
- tfidf__max_features: 20000
- tfidf__min_df: 1
- tfidf__ngram_range: (1, 3)
- optimal_threshold: 0.11899092942874724

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.5667 | 0.4182 |
| recall | 0.7083 | 0.9583 |
| f1 | 0.6296 | 0.5823 |
| f2 | 0.6746 | 0.7616 |
| roc_auc | 0.9476 | 0.9476 |
| wss at 95 | 0.8124 | 0.6977 |

## N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 2)** n-grams: average F1 = 0.5222
- **(1, 3)** n-grams: average F1 = 0.5200

The (1,2) n-gram range outperforms (1,3) by **0.2 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

For detailed analysis, see the full grid search results in the cv_results.csv file.

## Analysis

### Methodology

Following Cohen 2006 and Norman 2018, we performed a single grid search with multi-metric scoring
and extracted two models:

1. **Balanced model**: Optimized for F1 (best balance of precision and recall)
2. **High-recall model**: Configuration with highest F1 score among those with recall ≥ 0.95

## Conclusion

The balanced model is best for general classification tasks where overall performance is important, while the high-recall model is better suited for systematic review screening where achieving high recall (≥95%) is crucial to ensure comprehensive coverage.
