# Systematic Review Classification Refactoring Guide

This guide explains the refactoring changes made to the codebase and how to use the new grid search functionality.

## Changes Made

1. **Code Consolidation**:
   - Deleted redundant files:
     - `baseline_classifier.py` ✅
     - `svm_classifier.py` ✅
     - `grid_search.py` ✅
     - `ngram_comparison.py` ✅
   - All classifier functionality is now in the consolidated `classifiers.py` file
   - Grid search functionality is consolidated in `ngram_experiment.py`

2. **Methodological Improvements**:
   - Implemented a single grid search that extracts both balanced and high-recall models
   - Added consistent feature filtering (max_df=0.9) across all models
   - Implemented proper threshold selection on validation data only
   - Added functionality to select the model with highest F1 among configurations with recall ≥ 0.95

3. **New Features**:
   - Comprehensive performance evaluation for both balanced and high-recall models
   - Automatic threshold optimization for high-recall model
   - Precision-recall curve visualization
   - Detailed markdown reports generation

## Using the Refactored Code

### 1. Activate the Virtual Environment

Always start by activating the virtual environment:

```bash
source venv/bin/activate
```

### 2. Run the Grid Search Experiment

To run the Logistic Regression grid search:

```bash
python -m src.scripts.ngram_experiment --model logreg --output-dir results/grid_search
```

To run the SVM grid search:

```bash
python -m src.scripts.ngram_experiment --model svm --output-dir results/grid_search
```

### 3. Run Multiple Experiments

You can use the `run_grid_search.py` wrapper script to run both models:

```bash
python -m src.scripts.run_grid_search --all
```

### 4. View Results

The experiment generates the following output:
- **Models**: Saved in `results/grid_search/{model}/models/`
- **Metrics**: Saved in `results/grid_search/{model}/metrics/`
- **Plots**: Precision-recall curves in `results/grid_search/{model}/plots/`
- **Reports**: Markdown comparison reports in `results/grid_search/{model}/`

## Key Functions in the Implementation

- `run_grid_search()`: Performs the main grid search with multi-metric scoring
- `extract_high_recall_model_params()`: Extracts parameters for the high-recall model (highest F1 with recall ≥ 0.95)
- `evaluate_model()`: Evaluates model performance with optional custom threshold
- `find_optimal_threshold_for_recall()`: Finds the threshold that achieves 95% recall with highest precision

## Methodology

The implementation follows the methodology from Cohen 2006 and Norman 2018:

1. A single grid search evaluates all hyperparameter combinations using 5-fold cross-validation
2. The balanced model is selected based on the highest F1 score (best trade-off between precision and recall)
3. The high-recall model is selected as the configuration with highest F1 score among those achieving ≥95% recall
4. Threshold optimization is performed on validation data only to avoid data leakage
5. Both models are evaluated on the same validation set for fair comparison

This approach ensures both scientific validity and computational efficiency by extracting two optimized models from a single grid search. 