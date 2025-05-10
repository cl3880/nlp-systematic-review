# Complement Naive Bayes Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__alpha: 0.1
- clf__norm: True
- tfidf__max_df: 0.9
- tfidf__max_features: 10000
- tfidf__min_df: 2
- tfidf__ngram_range: (1, 2)

### High-Recall Model (95% Target)
- clf__alpha: 0.1
- clf__norm: True
- tfidf__max_df: 0.9
- tfidf__max_features: 10000
- tfidf__min_df: 2
- tfidf__ngram_range: (1, 2)
- optimal_threshold: 0.49999564235682203

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.3333 | 0.2323 |
| recall | 0.9167 | 0.9583 |
| f1 | 0.4889 | 0.3740 |
| f2 | 0.6790 | 0.5897 |
| roc_auc | 0.9223 | 0.9223 |
| wss at 95 | 0.6472 | 0.4959 |

## N-gram Range Analysis

Based on our cross-validation results (see cv_results.csv), we found:

- **(1, 3)** n-grams: average F1 = 0.2656
- **(1, 2)** n-grams: average F1 = 0.2562

The (1,3) n-gram range outperforms (1,2) by **0.9 percentage points** - a modest improvement that should be considered alongside other hyperparameters.

For detailed analysis, see the full grid search results in the cv_results.csv file.

## Analysis

### Methodology

Following Cohen 2006 and Norman 2018, we performed a single grid search with multi-metric scoring
and extracted two models:

1. **Balanced model**: Optimized for F1 (best balance of precision and recall)
2. **High-recall model**: Configuration with highest F1 score among those with recall ≥ 0.95

## Conclusion

The balanced model is best for general classification tasks where overall performance is important, while the high-recall model is better suited for systematic review screening where achieving high recall (≥95%) is crucial to ensure comprehensive coverage.
