# NLP Final Project: Systematic Review Automation

This repository contains code for automating systematic review article classification to determine relevance for medical literature reviews.

## Project Overview

The system classifies medical research articles as relevant or irrelevant based on specific inclusion/exclusion criteria. It uses machine learning approaches with TF-IDF vectorization and various classification models to automate the screening process.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
nlp_final_project/
├── data/
│   ├── raw/            # Raw PubMed export and relevant title lists
│   └── processed/      # Cleaned, processed datasets
│       ├── data_final.csv          # Original dataset
│       ├── data_final_cleaned.csv  # Dataset with Unicode normalized
│       └── data_final_processed.csv # Final processed dataset
├── results/            # Experiment results
├── src/
│   ├── config.py       # Configuration settings
│   ├── models/         # Model implementations
│   │   ├── baseline_classifier.py  # Baseline models
│   │   ├── imbalance_handling.py   # Class imbalance handling
│   │   ├── model_factory.py        # Model creation factory
│   │   └── svm_classifier.py       # SVM implementation
│   ├── scripts/        # Main scripts
│   │   ├── analyze_errors.py       # Error analysis script
│   │   ├── compare_normalizations.py # Compare text normalization
│   │   ├── run_all.py              # Run all experiments
│   │   ├── run_baseline.py         # Pipeline runner
│   │   ├── run.sh                  # Shell script for common operations
│   │   └── train_baseline.py       # Model training script
│   └── utils/          # Utility functions
├── README.md
└── requirements.txt
```

# Running the Systematic Review Automation Pipeline

## Run All Experiments
To run all experiments (baseline, SVM, normalization, imbalance handling):

```bash
./src/scripts/run.sh all
```

## Run Individual Experiments

```bash
# Baseline model (TF-IDF + Logistic Regression + Cosine Similarity)
./src/scripts/run.sh baseline

# SVM model
./src/scripts/run.sh svm

# With normalization (stemming or lemmatization)
./src/scripts/run.sh logreg_stemming
./src/scripts/run.sh logreg_lemmatization

# With class balancing (SMOTE or undersampling)
./src/scripts/run.sh logreg_smote
./src/scripts/run.sh logreg_undersample
```

## Features
* **Multiple models**: Logistic Regression, SVM, Cosine Similarity
* **Text normalization**: Stemming, Lemmatization
* **Class imbalance handling**: SMOTE, Random Undersampling
* **Grid search**: Automatic hyperparameter optimization
* **Comprehensive evaluation**: Precision, Recall, F1, ROC AUC, WSS@95
* **Feature analysis**: Visualization of most important features

## Key Results

| Approach                        | Precision | Recall  | F1 Score | ROC AUC | WSS@95 |
|---------------------------------|-----------|---------|----------|---------|--------|
| Logistic Regression (Baseline)  | 0.4667    | 0.8750  | 0.6087   | 0.9467  | 0.6977 |
| SVM (Linear Kernel)             | 0.4000    | 0.9167  | 0.5570   | 0.9431  | 0.6839 |
| LR + Stemming                   | 0.4889    | 0.9167  | 0.6377   | 0.9515  | 0.6977 |
| LR + Lemmatization              | 0.5000    | 0.9167  | 0.6471   | 0.9482  | 0.6885 |
| LR + SMOTE                      | 0.6250    | 0.6250  | 0.6250   | 0.9463  | 0.6748 |
| LR + Undersampling              | 0.4000    | 1.0000  | 0.5714   | 0.9405  | 0.6977 |
| Cosine Similarity (Thresh=0.1)  | 0.1379    | 1.0000  | 0.2424   | 0.9197  | 0.6518 |
| Cosine Similarity (Thresh=0.2)  | 0.2963    | 1.0000  | 0.4571   | 0.9197  | 0.6518 |
| Cosine Similarity (Thresh=0.3)  | 0.5185    | 0.5833  | 0.5490   | 0.9197  | 0.6518 |

## Conclusions

- **Best Overall Model**: Logistic Regression with Lemmatization (highest F1 score: 0.6471)
- **Best for High Recall**: Logistic Regression with Undersampling (perfect recall: 1.0000)
- **Best for Precision**: Logistic Regression with SMOTE (highest precision: 0.6250)
- **Best for Work Savings**: Baseline, Stemming, and Undersampling all tied (WSS@95: 0.6977)

Detailed results and analysis can be found in the `results/` directory and `RESULTS_SUM.md`.

---

## Medical Student's Systematic Review Screening Criteria

| Category                         | Include                                                                          | Exclude                                                                                                                                                   |
|----------------------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Population**                   | Patients ≥18 years old with brain AVMs                                           | Only pediatric populations, pregnant patients, dural/pial arteriovenous fistulas, vein of Galen malformations, cavernous malformations                    |
| **Age Mention**                  | No mention of patients under 18                                                  | Any mention of patients under 18 (e.g., "age range 4–98")                                                                                                 |
| **Automatic Exclusion Keywords** | —                                                                                | hypofractionated, proton beam therapy, fractionated stereotactic radiotherapy/surgery, tomotherapy                                                        |
| **Minimum Sample Size**          | If no patient number mentioned, **keep**                                         | If explicitly <30 patients, **exclude**                                                                                                                   |
| **Publication Year**             | Published ≥2000                                                                  | Published before 2000                                                                                                                                     |
| **Language**                     | English or French                                                               | Non-English, non-French                                                                                                                                   |
| **Outcome Reporting**            | Mentions methodology for AVM obliteration rate                                  | No obliteration rate mentioned                                                                                                                            |
| **Study Type**                   | Clinical trials, cohort studies, case series, systematic reviews                | Meta-analyses, literature reviews, case reports <10 patients                                                                                              |
| **Treatment Method**             | Radiosurgery methods: Gamma Knife, CyberKnife, Novalis, LINAC-based techniques   | Non-radiosurgical treatments                                                                                                                              |

---