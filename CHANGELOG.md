# Changelog: Systematic Review Classification Implementation

## [2.1.0] - 2025-05-11

### SMOTE Implementation

- **Model factory**: Added a `balancing` parameter (`None` | `"smote"` | `"undersample"`) to `create_model()`, wiring in Imbalanced-Learn’s `SMOTE(random_state=42)` and `RandomUnderSampler(random_state=42)` steps just before the classifier node via an `ImbPipeline`.
- **Result directory logic**: Refactored `config.py` to own all path definitions and updated `baseline_grid_search.py` to consume these settings; results folders have been manually reorganized to match the desired structure, deferring automated directory creation for a future refactor.
   - SMOTE-SVM run was executed manually and its outputs live under `results/v2.0.0-f1-refactor/smote_svm/`
- **CLI & grid search**: Modified `baseline_grid_search.py` to accept a `--balancing` flag, propagate it into `run_grid_search()`/`create_model()`, and label logs, model names, and output folders accordingly (e.g. `smote_logreg`).  
- **Imports & cleanup**: Refactored deprecated or unused imports in both `model_factory.py` and `config.py` to align with the new sampling logic.  

## [2.0.0] - 2025-05-10

1. **Threshold tuning bug (v1)**:
   We fell back to CV-based thresholds when no validation threshold hit 95% recall. This contradicts Cohen et al.’s strict “validation-only” protocol and led to nonsensical WSS\@95 = 0 outcomes.

2. **Objective misalignment**:
   v1’s use of F₂ diverged from the literature’s focus on F₁ under a recall floor (Gutiérrez et al. 2020, Frunza et al. 2010, Norman et al. 2018) . We corrected this by using `scoring={'precision','recall','f1'}` and `refit='f1'`.

3. **Normalization confound**:
   Reusing raw-text hyperparameters for stemming/lemmatization conflated two variables. Literature offers both fixed-parameter (for direct comparability) and per-mode re-optimization approaches ; our v2 implements **separate** grid searches per normalization.

4. **N-gram re-sweep**:
   Rezapour & Diesner (LREC 2020) show bigrams+trigrams can add \~10 pp F₁ in deductive settings . In v2 we systematically re-tuned `(1,2)` vs. `(1,3)` alongside all TF-IDF and classifier parameters.

5. **Sparse optimization for cosine**:
   v1’s dense TF-IDF→cosine routine crashed on high-dimensional data; we refactored to use sparse SciPy dot-products, eliminating memory failures.

### Key results

- **95 % recall guaranteed**  
  Thresholds are now calibrated on the validation set and held fixed for test, so every high-recall model truly meets 95 % recall on unseen data.

- **+11.8 pp F₁ improvement at 95 % recall**  
  Compared to v1, the v2 pipeline delivers up to an 11.77 percentage-point boost in F₁ when operating at the 95 % recall target.

- **–6.8 pp false-positive rate**  
  At equivalent recall levels, the false-positive rate dropped by approximately 6.8 percentage points.

- **Normalization insights**  
  - Raw text (no normalization) remains best for balanced F₁/AUC.  
  - Stemming yields the highest precision and F₁ at the 95 % recall operating point.  

## [1.0.0] - 2025-05-01

### Initial Implementation
- "Monolithic grid": a single 3×2×2 grid (classifier×normalization×balancing) over all hyperparameters, which proved unmanageable and obscured individual effects.
- Multi-classifier framework implementation (LogReg, SVM, CNB, Cosine)
- Separate hyperparameter optimization for balanced and high-recall models
- Text normalization comparison framework (stemming, lemmatization)
- Baseline evaluation metrics framework