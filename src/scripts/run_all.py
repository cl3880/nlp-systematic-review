# src/scripts/run_all.py
#!/usr/bin/env python3
"""
Run all experiments for systematic review classification.
"""
import os
import logging
import argparse
import subprocess
from datetime import datetime

from src.config import PATHS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PATHS["logs_dir"], "all_experiments.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_experiment(cmd, description):
    logger.info(f"Starting {description}")
    start = datetime.now()
    try:
        subprocess.run(cmd, shell=True, check=True)
        status = True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        status = False
    duration = datetime.now() - start
    if status:
        logger.info(f"✔ Completed {description} in {duration}")
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Run all systematic review experiments"
    )
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument(
        "--data",
        default=os.path.join(PATHS["data_processed"], "data_final_processed.csv"),
        help="Path to processed data CSV",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip baseline models"
    )
    parser.add_argument(
        "--skip-cosine",
        action="store_true",
        help="Skip the cosine-similarity @ custom threshold experiment",
    )
    parser.add_argument(
        "--cos-thresh",
        type=float,
        default=0.3,
        help="Threshold to use for the cosine similarity classifier",
    )
    parser.add_argument("--skip-svm", action="store_true", help="Skip SVM models")
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip normalization comparison",
    )
    parser.add_argument(
        "--skip-imbalance",
        action="store_true",
        help="Skip imbalance handling techniques",
    )
    parser.add_argument(
        "--skip-grid", action="store_true", help="Skip the full grid-search script"
    )
    args = parser.parse_args()

    # Ensure results directories exist
    os.makedirs(PATHS["baseline_dir"], exist_ok=True)
    os.makedirs(os.path.join(PATHS["results_dir"], "svm"), exist_ok=True)
    os.makedirs(os.path.join(PATHS["results_dir"], "normalization"), exist_ok=True)
    os.makedirs(os.path.join(PATHS["results_dir"], "imbalance"), exist_ok=True)

    success_count = 0
    total_experiments = 0

    if args.all:
        args.skip_baseline = False
        args.skip_svm = False
        args.skip_grid = False
        args.skip_normalization = False
        args.skip_imbalance = False
        args.skip_cosine = False

    if not args.skip_baseline:
        total_experiments += 1
        if run_experiment(
            f"python -m src.scripts.train_baseline --data-mode processed --processed-data {args.data} --model logreg --output-dir {PATHS['baseline_dir']}",
            "baseline models (logistic regression and cosine similarity)",
        ):
            success_count += 1

    if not args.skip_grid:
        total_experiments += 1
        grid_dir = os.path.join(PATHS["results_dir"], "grid_all")
        os.makedirs(grid_dir, exist_ok=True)
        if run_experiment(
            f"python -m src.scripts.train_grid --data {args.data} --output-dir {grid_dir}",
            "full 3×3×2×2 grid search",
        ):
            success_count += 1

    # For SVM models
    if not args.skip_svm:
        total_experiments += 1
        if run_experiment(
            f"python -m src.scripts.train_baseline --data-mode processed --processed-data {args.data} --model svm --output-dir {os.path.join(PATHS['results_dir'], 'svm')}",
            "SVM models",
        ):
            success_count += 1

    # For imbalance techniques
    if not args.skip_imbalance:
        for technique in ["smote", "undersample"]:
            total_experiments += 1
            output_dir = os.path.join(PATHS["results_dir"], "imbalance", technique)
            os.makedirs(output_dir, exist_ok=True)

            if run_experiment(
                f"python -m src.scripts.train_baseline --data-mode processed --processed-data {args.data} --model logreg --balancing {technique} --output-dir {output_dir}",
                f"imbalance handling ({technique})",
            ):
                success_count += 1

    # For cosine similarity
    if not args.skip_cosine:
        total_experiments += 1
        cos_dir = os.path.join(PATHS["results_dir"], f"cosine_thresh_{args.cos_thresh}")
        os.makedirs(cos_dir, exist_ok=True)
        if run_experiment(
            f"python -m src.scripts.train_baseline "
            f"--data-mode processed "
            f"--processed-data {args.data} "
            f"--model cosine "
            f"--cos-thresh {args.cos_thresh} "
            f"--output-dir {cos_dir}",
            f"cosine similarity @ {args.cos_thresh}",
        ):
            success_count += 1

    logger.info(
        f"All experiments completed: {success_count}/{total_experiments} successful"
    )


if __name__ == "__main__":
    main()