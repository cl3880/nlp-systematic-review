# Changelog

## [3.0.1] – 2025-05-25
### Added
- Basic test suite under `tests/` (pytest)  
- GitHub Actions workflow (`.github/workflows/ci.yml`)

## [3.0.0] – 2025-05-24
### Added
- Domain-specific **hard filters** (DP, PT, LA) in data preprocessing  
- **Expert criteria features**: Inclusion/exclusion criteria as binary features for enhanced model performance
- `stage4_expert_features.py` for isolated expert criteria experiments (fixed SVM params)
- `stage4_mesh_features.py` for isolated MeSH-only feature experiments  
- **Feature importance** extraction & visualization tools  
  - Support for CNB and cosine-similarity feature extractors  
  - Specialized handling for `FeatureCombiner` outputs
  
### Changed
- **Scripts renamed** to reflect experimental stages:
  - `baseline_grid_search.py` → `stage1_baseline_grid_search.py`  
  - `isolated_experiments.py` → `stage3_isolated_experiments.py`  
  - `criteria_features_nogrid.py` → `stage4_expert_features.py`  
- **Results directory** reorganized from `results/` → `results_final/`  
- Legacy code moved into `archive/legacy_scripts/`

### Refactored
- Centralized all path definitions in `src/config.py`  
- Cleaned up imports, verified dependencies, tightened `.gitignore`

### Documentation
- Expanded **README.md** with:
  - Installation & prerequisites  
  - CLI examples for each stage  
  - End-to-end workflow guide  

---

## [2.1.0] – 2025-05-11
### Added
- `--balancing` flag to `baseline_grid_search.py` for choosing `None`, `"smote"`, or `"undersample"`  
- ImbPipeline integration of `SMOTE(random_state=42)` and `RandomUnderSampler(random_state=42)`

### Changed
- Centralized result paths in `config.py`; reorganized SMOTE-SVM outputs under `results/v2.0.0-f1-refactor/smote_svm/`

### Fixed
- Deprecated imports aligned with latest `imbalanced-learn` API

---

## [2.0.0] – 2025-05-10
### Added
- Separate grid searches per normalization technique (stemming vs. lemmatization)  
- N-gram re-tuning scripts following Rezapour & Diesner (LREC 2020)

### Changed
- Switched objective from F₂ to F₁ with recall floor, per Gutiérrez et al. 2020, Frunza et al. 2010, Norman et al. 2018  
- Refactored cosine-similarity to use sparse SciPy operations

### Fixed
- Threshold calibration bug (eliminated zero WSS@95 artifacts)  
- Validation-only threshold tuning now matches Cohen et al.’s protocol

### Performance
- +11.8 pp F₁ at 95% recall vs. v1.0.0  
- –6.8 pp false-positive rate at equivalent recall

---

## [1.0.0] – 2025-05-01
### Added
- Multi-classifier framework: LogReg, SVM, CNB, Cosine Similarity  
- Dual optimization pipelines for balanced vs. high-recall objectives  
- Full "monolithic" grid search (3×2×2 combinations) over normalization, sampling, and classifier hyperparameters

> **Note:** this first "production" version grew out of my initial exploratory grid-search experiments. I wanted to test every combination end-to-end, but quickly discovered it made isolating the impact of each change difficult—hence v1.0.0's move to more controlled, stage-based scripts.