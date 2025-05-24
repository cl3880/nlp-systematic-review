# Changelog: Systematic Review Classification Implementation

## [3.0.0] - 2025-05-24

### Major Feature Development and Project Reorganization

#### Phase 1: Advanced Features Implementation (2025-05-12)
- **Hard Filters**: Implemented domain-specific filters (DP, PT, LA) for data preprocessing
- **StratifiedKFold Standardization**: Explicit declaration and consistent use across all experiments for fair cross-validation
- **Expert Criteria Features**: Added inclusion/exclusion criteria as binary features for enhanced model performance
- **Feature Engineering**: Implemented `criteria_features_nogrid.py` using fixed SVM baseline parameters to avoid compounding variables from grid search
- **Visualization Framework**: Created comprehensive feature importance extraction and visualization tools
  - Support for CNB and cosine similarity feature extraction
  - Specialized handling for FeatureCombiner architecture differences

#### Phase 2: Project Structure Reorganization (2025-05-24)
- **Results Structure**: Reorganized into `results_final/`
- **Script Standardization**: Renamed all workflow scripts to reflect experimental stages:
  - `baseline_grid_search.py` → `stage1_baseline_grid_search.py`
  - `isolated_experiments.py` → `stage3_isolated_experiments.py`
  - `criteria_features_nogrid.py` → `stage4_expert_features.py`
- **Archive Organization**: Moved legacy scripts to `archive/legacy_scripts/`

#### Scientific Experimental Design Clarification
- **Controlled Feature Experiments**: Documented proper experimental design for Stage 4
  - `stage4_expert_features.py`: SVM + SMOTE + Expert Criteria (isolated)
  - `stage4_mesh_features.py`: SVM + SMOTE + MeSH Terms (isolated)

#### Documentation and Quality Improvements
- **README.md**: Comprehensive workflow guide with prerequisites, CLI examples, and usage patterns
- **Configuration**: Updated `src/config.py` with accurate path mappings and modular structure
- **Code Quality**: Fixed import references, verified dependencies, enhanced .gitignore
- **Professional Structure**: Repository ready for publication and collaboration

## [2.1.0] - 2025-05-11

### SMOTE Implementation and Infrastructure Improvements

- **Model Factory Enhancement**: Added `balancing` parameter (`None` | `"smote"` | `"undersample"`) to `create_model()`, integrating Imbalanced-Learn's `SMOTE(random_state=42)` and `RandomUnderSampler(random_state=42)` via `ImbPipeline`
- **Configuration Refactoring**: Centralized path definitions in `config.py` and updated `baseline_grid_search.py` to consume these settings
  - SMOTE-SVM results organized under `results/v2.0.0-f1-refactor/smote_svm/`
- **CLI Enhancement**: Added `--balancing` flag to `baseline_grid_search.py` for systematic balancing technique evaluation
- **Code Cleanup**: Refactored deprecated imports and aligned code with new sampling logic

## [2.0.0] - 2025-05-10

### Methodology Refinement and Performance Improvements

**Major Bug Fixes and Methodology Alignment**:

1. **Threshold Tuning Correction**: Fixed validation-only threshold calibration following Cohen et al.'s protocol, eliminating nonsensical WSS@95 = 0 outcomes
2. **Objective Function Alignment**: Switched from F₂ to F₁ optimization with recall floor, aligning with systematic review literature (Gutiérrez et al. 2020, Frunza et al. 2010, Norman et al. 2018)
3. **Normalization Decoupling**: Implemented separate grid searches per normalization technique to avoid hyperparameter confounding
4. **N-gram Optimization**: Systematic re-tuning of (1,2) vs. (1,3) n-gram ranges based on Rezapour & Diesner (LREC 2020) findings
5. **Sparse Processing**: Refactored cosine similarity to use sparse SciPy operations, eliminating memory failures on high-dimensional data

**Performance Improvements**:
- **+11.8pp F₁ improvement** at 95% recall target compared to v1.0.0
- **-6.8pp false-positive rate** reduction at equivalent recall levels
- **95% recall guarantee** through proper validation-based threshold calibration

**Methodology Insights**:
- Raw text optimal for balanced F₁/AUC performance
- Stemming yields highest precision and F₁ at 95% recall operating point

## [1.0.0] - 2025-05-01

### Initial Framework Implementation

- **Multi-Classifier Framework**: Implemented LogReg, SVM, CNB, and Cosine Similarity classifiers
- **Dual Optimization**: Separate hyperparameter optimization for balanced and high-recall scenarios
- **Text Processing**: Comprehensive normalization comparison (stemming, lemmatization)
- **Evaluation Infrastructure**: Baseline metrics framework and grid search methodology
- **Initial Challenge**: "Monolithic grid" approach (3×2×2 grid) proved unmanageable and obscured individual effects