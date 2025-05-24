# Systematic Review Automation: Experiment Plan

## 1. Baseline Configuration Freezing

Our initial grid search resulted in the following optimal configuration:

### Logistic Regression Balanced Model (Pipeline A)
- C: 1
- class_weight: balanced
- penalty: l2
- solver: liblinear
- tfidf__max_df: 0.9
- tfidf__max_features: 5000
- tfidf__min_df: 5
- tfidf__ngram_range: (1, 2)
- Default threshold: 0.5

### Logistic Regression High-Recall Model (Pipeline B)
- Same parameters as above
- Optimized threshold: 0.2886 (for 95% recall)

These configurations will be frozen as our reference baselines for all subsequent experiments.

## 2. Experimental Conditions

### 2.1 N-gram Range Comparison
We'll test the following n-gram ranges without running grid search again:
- (1, 1): Unigrams only
- (1, 2): Unigrams & bigrams (baseline)
- (1, 3): Unigrams, bigrams & trigrams (literature recommendation)
- (2, 3): Bigrams & trigrams only

```bash
# Run these commands to test each configuration
python -m src.scripts.train_baseline --model logreg --ngram-range 1 1 --no-grid-search --output-dir results/experiments/ngram_1_1
python -m src.scripts.train_baseline --model logreg --ngram-range 1 2 --no-grid-search --output-dir results/experiments/ngram_1_2
python -m src.scripts.train_baseline --model logreg --ngram-range 1 3 --no-grid-search --output-dir results/experiments/ngram_1_3
python -m src.scripts.train_baseline --model logreg --ngram-range 2 3 --no-grid-search --output-dir results/experiments/ngram_2_3
```

### 2.2 Text Normalization Techniques
We'll test two normalization techniques while keeping other parameters fixed:
- Stemming
- Lemmatization

```bash
# Run these commands to test each normalization technique
python -m src.scripts.train_baseline --model logreg --normalization stemming --no-grid-search --output-dir results/experiments/stemming
python -m src.scripts.train_baseline --model logreg --normalization lemmatization --no-grid-search --output-dir results/experiments/lemmatization
```

### 2.3 Class Balancing Methods
We'll test two class balancing methods while keeping other parameters fixed:
- SMOTE (synthetic minority over-sampling)
- RandomUnderSampler (majority class under-sampling)

```bash
# Run these commands to test each balancing method
python -m src.scripts.train_baseline --model logreg --balancing smote --no-grid-search --output-dir results/experiments/smote
python -m src.scripts.train_baseline --model logreg --balancing undersample --no-grid-search --output-dir results/experiments/undersample
```

### 2.4 Alternative Models
We'll test different model types with the same TF-IDF parameters:
- SVM (linear kernel)
- Cosine Similarity

```bash
# Run these commands to test alternative models
python -m src.scripts.train_baseline --model svm --no-grid-search --output-dir results/experiments/svm
python -m src.scripts.train_baseline --model cosine --no-grid-search --output-dir results/experiments/cosine
```

## 3. Combinations of Best Techniques
After identifying the best-performing techniques in each category, we'll test combinations:

```bash
# Example (assuming stemming and SMOTE are best):
python -m src.scripts.train_baseline --model logreg --normalization stemming --balancing smote --no-grid-search --output-dir results/experiments/stemming_smote
```

## 4. Evaluation Metrics
For each experiment, we'll track and compare:
- Precision, Recall, F1, F2 scores
- ROC AUC and Precision-Recall AUC
- WSS@95 (Work Saved over Sampling at 95% recall)
- Threshold required for 95% recall

## 5. Results Documentation
Results for each experiment will be documented in:
- `results/baseline_experiments_ngram_comparison.md` (for n-gram experiments)
- `results/baseline_experiments_normalization.md` (for text normalization experiments)
- `results/baseline_experiments_balancing.md` (for class balancing experiments)
- `results/baseline_experiments_models.md` (for alternative models)

## 6. Final Model Selection
Based on these experiments, we'll select:
1. The best anchor classifier for future work
2. The optimal n-gram range configuration
3. Whether to include text normalization
4. Whether to include class balancing methods

This anchor model will be used as the reference point for any future enhancements or feature additions. 