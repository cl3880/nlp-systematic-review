#!/bin/bash
# src/scripts/run.sh

function run_baseline() {
    python -m src.scripts.train_baseline --data-mode processed \
        --processed-data data/processed/data_final_processed.csv \
        --model logreg \
        --output-dir results/baseline
}

function run_grid() {
    python -m src.scripts.train_grid \
        --data data/processed/data_final_processed.csv \
        --output-dir results/grid_all
}

function run_model_with_normalization() {
    MODEL=$1
    NORM=$2
    OUTPUT_DIR="results/normalization/${NORM}"
    
    python -m src.scripts.train_baseline \
        --data-mode processed \
        --processed-data data/processed/data_final_processed.csv \
        --model $MODEL \
        --normalization $NORM \
        --output-dir $OUTPUT_DIR
}

function run_model_with_balancing() {
    MODEL=$1
    BALANCE=$2
    OUTPUT_DIR="results/imbalance/${BALANCE}"
    
    python -m src.scripts.train_baseline \
        --data-mode processed \
        --processed-data data/processed/data_final_processed.csv \
        --model $MODEL \
        --balancing $BALANCE \
        --output-dir $OUTPUT_DIR
}

function run_svm() {
    python -m src.scripts.train_baseline \
        --data-mode processed \
        --processed-data data/processed/data_final_processed.csv \
        --model svm \
        --output-dir results/svm
}

function run_cnb() {
  python -m src.scripts.train_baseline \
    --data-mode processed \
    --processed-data data/processed/data_final_processed.csv \
    --model cnb \
    --output-dir results/cnb
}

function run_cnb_stemming() {
  python -m src.scripts.train_baseline \
    --data-mode processed \
    --processed-data data/processed/data_final_processed.csv \
    --model cnb \
    --normalization stemming \
    --output-dir results/cnb_stemming
}

function run_cnb_lemmatization() {
  python -m src.scripts.train_baseline \
    --data-mode processed \
    --processed-data data/processed/data_final_processed.csv \
    --model cnb \
    --normalization lemmatization \
    --output-dir results/cnb_lemmatization
}

function run_cnb_smote() {
  python -m src.scripts.train_baseline \
    --data-mode processed \
    --processed-data data/processed/data_final_processed.csv \
    --model cnb \
    --balancing smote \
    --output-dir results/cnb_smote
}

function run_cnb_undersample() {
  python -m src.scripts.train_baseline \
    --data-mode processed \
    --processed-data data/processed/data_final_processed.csv \
    --model cnb \
    --balancing undersample \
    --output-dir results/cnb_undersample
}

function run_all() {
    python -m src.scripts.run_all --all
}

function run_full_grid() {
    python -m src.scripts.grid_search \
        --data data/processed/data_final_processed.csv \
        --output-dir results/grid_search_full \
        --target-recall 0.95
}

function run_full_grid_high_recall() {
    python -m src.scripts.grid_search \
        --data data/processed/data_final_processed.csv \
        --output-dir results/grid_search_full_high_recall \
        --target-recall 0.98
}

function run() {
    case $1 in
        "baseline")
            run_baseline
            ;;
        "svm")
            run_svm
            ;;
        "grid")
            run_grid
            ;;
        "all")
            run_all
            ;;
        "full_grid")
            run_full_grid
            ;;
        "full_grid_high_recall")
            run_full_grid_high_recall
            ;;
        "logreg_stemming")
            run_model_with_normalization "logreg" "stemming"
            ;;
        "logreg_lemmatization")
            run_model_with_normalization "logreg" "lemmatization"
            ;;
        "svm_stemming")
            run_model_with_normalization "svm" "stemming"
            ;;
        "svm_lemmatization")
            run_model_with_normalization "svm" "lemmatization"
            ;;
        "logreg_smote")
            run_model_with_balancing "logreg" "smote"
            ;;
        "logreg_undersample")
            run_model_with_balancing "logreg" "undersample"
            ;;
        "svm_smote")
            run_model_with_balancing "svm" "smote"
            ;;
        "svm_undersample")
            run_model_with_balancing "svm" "undersample"
            ;;
        "cnb")              run_cnb ;;
        "cnb_stemming")     run_cnb_stemming ;;
        "cnb_lemmatization")run_cnb_lemmatization ;;
        "cnb_smote")        run_cnb_smote ;;
        "cnb_undersample")  run_cnb_undersample ;;
        *)
            echo "Unknown experiment: $1"
            echo "Valid options: baseline, svm, grid, all, full_grid, full_grid_high_recall,"
            echo "               logreg_stemming, logreg_lemmatization, svm_stemming, svm_lemmatization,"
            echo "               logreg_smote, logreg_undersample, svm_smote, svm_undersample"
            exit 1
            ;;
    esac
}

if [ $# -eq 0 ]; then
    echo "Please provide an experiment to run"
    echo "Valid options: baseline, svm, grid, all, full_grid, full_grid_high_recall,"
    echo "               logreg_stemming, logreg_lemmatization, svm_stemming, svm_lemmatization,"
    echo "               logreg_smote, logreg_undersample, svm_smote, svm_undersample"
    exit 1
fi

run $1