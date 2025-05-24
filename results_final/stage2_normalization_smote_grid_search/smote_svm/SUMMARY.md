# Support Vector Machine Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- tfidf__max_df: 0.85
- tfidf__max_features: 5000
- tfidf__min_df: 3
- tfidf__ngram_range: (1, 2)

### High-Recall Model (95% Target)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- tfidf__max_df: 0.85
- tfidf__max_features: 5000
- tfidf__min_df: 3
- tfidf__ngram_range: (1, 2)
- optimal_threshold: 0.0007712271692358987

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.7083 | 0.4510 |
| recall | 0.7083 | 0.9583 |
| f1 | 0.7083 | 0.6133 |
| f2 | 0.7083 | 0.7823 |
| roc_auc | 0.9543 | 0.9543 |
| wss at 95 | 0.8399 | 0.7161 |

## N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 3)** n-grams: average F1 = 0.6183
- **(1, 2)** n-grams: average F1 = 0.6180

The (1,3) n-gram range outperforms (1,2) by **0.0 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

For detailed analysis, see the full grid search results in the cv_results.csv file.

## Analysis

### Methodology

Following Cohen 2006 and Norman 2018, we performed a single grid search with multi-metric scoring
and extracted two models:

1. **Balanced model**: Optimized for F1 (best balance of precision and recall)
2. **High-recall model**: Configuration with highest F1 score among those with recall ≥ 0.95

## Conclusion

The balanced model is best for general classification tasks where overall performance is important, while the high-recall model is better suited for systematic review screening where achieving high recall (≥95%) is crucial to ensure comprehensive coverage.
