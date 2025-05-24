#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
import json
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src import config
target_log = config.PATHS.get("logs_dir", "results/logs")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(target_log, "data_preparation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    return text.strip()

def extract_year(date_str):
    if pd.isna(date_str) or date_str is None:
        return np.nan
    
    year_match = re.search(r'(19|20)\d{2}', str(date_str))
    if year_match:
        return int(year_match.group(0))
    
    return np.nan

def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
    return path

def prepare_data(input_file, output_file, apply_filters=False, filter_log_dir=None):
    try:
        ensure_directory(os.path.dirname(output_file))
        
        if filter_log_dir is None:
            filter_log_dir = os.path.join(os.path.dirname(output_file), "filter_logs")
        ensure_directory(filter_log_dir)
        
        log_file = os.path.join(filter_log_dir, "data_preparation.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        original_count = len(df)
        logger.info(f"Loaded {original_count} records")
        
        if apply_filters:
            logger.info("Applying metadata-based filtering")
            from src.utils.hard_filters import apply_filters_to_dataframe
            df, filter_stats = apply_filters_to_dataframe(df, filter_log_dir)
            logger.info(f"Filtering complete: {len(df)} of {original_count} records retained ({len(df)/original_count:.2%})")
            
            with open(os.path.join(filter_log_dir, "filter_summary.json"), "w") as f:
                json.dump({
                    "input_file": input_file,
                    "original_count": original_count,
                    "filtered_count": len(df),
                    "retention_rate": len(df)/original_count,
                    "filter_stats": filter_stats,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
        
        text_cols = ['title', 'abstract']
        for col in text_cols:
            if col in df.columns:
                logger.info(f"Normalizing {col} column")
                df[col] = df[col].apply(clean_text)
        
        if 'publication_year' not in df.columns and 'publication_date' in df.columns:
            logger.info("Extracting publication year from publication date")
            df['publication_year'] = df['publication_date'].apply(extract_year)
        elif 'publication_year' in df.columns:
            logger.info("Standardizing publication_year column")
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
        
        if 'relevant' in df.columns:
            logger.info("Standardizing relevance column")
            df['relevant'] = df['relevant'].astype(bool)
        
        if 'title' in df.columns:
            missing_title = df['title'].isna() | (df['title'] == '')
            if missing_title.any():
                missing_count = missing_title.sum()
                logger.warning(f"Removing {missing_count} records with missing titles")
                df = df[~missing_title]
        
        for col in text_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}. Adding empty column.")
                df[col] = ""
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} processed records to {output_file}")
        
        final_count = len(df)
        processing_removed = original_count - final_count
        logger.info(f"Processing complete: {processing_removed} records removed ({final_count/original_count:.2%} retention)")
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Pre-process systematic review data with optional metadata filtering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        default="data/processed/data_final.csv",
        help='Path to input CSV dataset (default: data/processed/data_final.csv)'
    )
    parser.add_argument(
        '--output',
        help=('Path for processed output dataset. '\
              'If not provided, defaults to data/processed/data_filtered.csv ' \
              'when --apply-filters is used, otherwise data/processed/data_unfiltered.csv.')
    )
    parser.add_argument(
        '--apply-filters',
        action='store_true',
        help='Apply metadata-based hard filtering'
    )
    parser.add_argument(
        '--filter-log-dir',
        default=None,
        help='Directory for filter logs (default: alongside output file)'
    )
    args = parser.parse_args()

    if not args.output:
        args.output = (
            "data/processed/data_filtered.csv"
            if args.apply_filters else
            "data/processed/data_unfiltered.csv"
        )

    filter_log_dir = args.filter_log_dir or os.path.join(os.path.dirname(args.output), "logs")

    df = prepare_data(
        input_file=args.input,
        output_file=args.output,
        apply_filters=args.apply_filters,
        filter_log_dir=filter_log_dir
    )

    logger.info("Data preparation completed successfully.")

if __name__ == "__main__":
    main()