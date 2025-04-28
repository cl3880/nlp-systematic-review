#!/usr/bin/env python3
"""
Wrapper script to analyze prediction errors for systematic review models.
"""
import os
import argparse
import logging

from src.config import PATHS
from src.utils.error_analysis import analyze_errors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PATHS["logs_dir"], "error_analysis.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Analyze prediction errors for systematic review models')
    parser.add_argument('--predictions', 
                      default=os.path.join(PATHS["baseline_dir"], "analysis/logreg_predictions.csv"),
                      help='Path to predictions CSV file')
    parser.add_argument('--output-dir', 
                      default=os.path.join(PATHS["baseline_dir"], "error_analysis"),
                      help='Directory for output files')
    parser.add_argument('--model-name', default='baseline',
                      help='Name of the model being analyzed (for reporting)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Starting error analysis for {args.model_name} model predictions")
    logger.info(f"Reading predictions from {args.predictions}")
    logger.info(f"Output will be saved to {args.output_dir}")
    
    # Run error analysis
    analyze_errors(args.predictions, args.output_dir)
    
    logger.info("Error analysis complete")

if __name__ == "__main__":
    main()