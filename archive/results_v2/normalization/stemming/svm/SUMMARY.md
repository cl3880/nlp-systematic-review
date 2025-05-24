# Support Vector Machine Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- tfidf__max_df: 0.85
- tfidf__max_features: 10000
- tfidf__min_df: 5
- tfidf__ngram_range: (1, 2)

### High-Recall Model (95% Target)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- tfidf__max_df: 0.85
- tfidf__max_features: 10000
- tfidf__min_df: 5
- tfidf__ngram_range: (1, 2)
- threshold: 0.11995871290252608

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.5455 | 0.4107 |
| recall | 0.7500 | 0.9583 |
| f1 | 0.6316 | 0.5750 |
| f2 | 0.6977 | 0.7566 |
| roc_auc | 0.9478 | 0.9478 |
| wss at 95 | 0.7986 | 0.6931 |

## N-gram Range Analysis

Based on cross-validation results (see cv_results.csv), we found:

- **(1, 2)** n-grams: average F1 = 0.5287
- **(1, 3)** n-grams: average F1 = 0.5285

The (1,2) n-gram range outperforms (1,3) by **0.0 percentage points**.

## Methodological Notes

The high-recall configuration utilizes the same underlying model as the balanced configuration, with an adjusted decision threshold to prioritize recall (≥95%) at the expense of precision. This approach maintains identical feature coefficients while modifying only the decision boundary.
