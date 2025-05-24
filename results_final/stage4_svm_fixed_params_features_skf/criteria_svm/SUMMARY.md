# CRITERIA_SVM Model Comparison

## Model Configurations

### Balanced Model (Optimized for F1)
- clf__C: 1
- clf__class_weight: balanced
- clf__kernel: linear
- features__text_features__tfidf__max_df: 0.85
- features__text_features__tfidf__max_features: 20000
- features__text_features__tfidf__min_df: 3
- features__text_features__tfidf__ngram_range: (1, 2)

### High-Recall Model (95% Target)
- clf__C: 0.1
- clf__class_weight: balanced
- clf__kernel: linear
- features__text_features__tfidf__max_df: 0.85
- features__text_features__tfidf__max_features: 5000
- features__text_features__tfidf__min_df: 3
- features__text_features__tfidf__ngram_range: (1, 2)

## Performance Comparison

| Metric | Balanced Model | High-Recall Model |
|--------|---------------|-------------------|
| precision | 0.7407 | 0.4792 |
| recall | 0.8333 | 0.9583 |
| f1 | 0.7843 | 0.6389 |
| f2 | 0.8130 | 0.7986 |
| roc_auc | 0.9487 | 0.9412 |
| wss at 95 | 0.8261 | 0.7298 |

## N-gram Range Analysis

Based on cross-validation results (see cv_results.csv), we found:

- **(1, 2)** n-grams: average F1 = 0.6572
- **(1, 3)** n-grams: average F1 = 0.6534

The (1,2) n-gram range outperforms (1,3) by **0.4 percentage points**.

## Methodological Notes

The high-recall configuration utilizes the same underlying model as the balanced configuration, with an adjusted decision threshold to prioritize recall (≥95%) at the expense of precision. This approach maintains identical feature coefficients while modifying only the decision boundary.
