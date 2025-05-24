import os
import pandas as pd
import numpy as np
import logging
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_publication_year(date_str):
    if pd.isna(date_str) or date_str is None:
        return None
        
    year_match = re.search(r'(19|20)\d{2}', str(date_str))
    if year_match:
        return int(year_match.group(0))
    
    return None

# def check_language(language_str):
#     if pd.isna(language_str) or language_str is None:
#         return True
    
#     language_str = str(language_str).lower().strip()
#     acceptable_languages = ["eng", "fre", "english", "french", "en", "fr"]
#     return any(lang in language_str for lang in acceptable_languages)

def check_publication_type(pub_type_field):
    if pd.isna(pub_type_field) or pub_type_field is None:
        return True, None

    if isinstance(pub_type_field, str):
        pub_types = [pub_type_field.strip()]
    elif isinstance(pub_type_field, list):
        pub_types = pub_type_field
    else:
        pub_types = [str(pub_type_field)]

    pub_types_lower = [pt.lower() for pt in pub_types]

    for pt in pub_types_lower:
        if "meta-analysis" in pt or "meta analysis" in pt:
            return False, f"Excluded meta-analysis: {pt}"

    has_plain_review = any(pt == "review" for pt in pub_types_lower)
    has_systematic = any("systematic review" in pt for pt in pub_types_lower)
    if has_plain_review and not has_systematic:
        return False, "Excluded generic literature review"

    return True, None

def apply_filters_to_dataframe(df, filter_log_dir="results/logs/hard_filters"):
    original_count = len(df)
    logger.info(f"Applying filters to {original_count} records")
    
    filter_stats = {
        "original_count": original_count,
        "excluded_total": 0,
        "excluded_by_year": 0,
        "excluded_by_language": 0,
        "excluded_by_pub_type": 0,
        "included_count": 0
    }
    
    filter_log = []
    
    if 'publication_year' in df.columns:
        year_mask = (df['publication_year'] >= 2000) | df['publication_year'].isna()
        excluded_by_year = (~year_mask).sum()
        filter_stats["excluded_by_year"] = int(excluded_by_year)
        
        if excluded_by_year > 0:
            for _, row in df[~year_mask].iterrows():
                filter_log.append({
                    "pmid": row.get("pmid", "unknown"),
                    "title": row.get("title", "")[:100],
                    "relevant": row.get("relevant", "ERROR"),
                    "criterion": "publication_year",
                    "value": row.get("publication_year"),
                    "reason": f"Published before 2000 (year: {row.get('publication_year')})"
                })
        
        df = df[year_mask].copy()
        logger.info(f"Year filter: excluded {excluded_by_year} records published before 2000")
    elif 'publication_date' in df.columns:
        df['temp_year'] = df['publication_date'].apply(extract_publication_year)
        year_mask = (df['temp_year'] >= 2000) | df['temp_year'].isna()
        excluded_by_year = (~year_mask).sum()
        filter_stats["excluded_by_year"] = int(excluded_by_year)
        
        if excluded_by_year > 0:
            for _, row in df[~year_mask].iterrows():
                filter_log.append({
                    "pmid": row.get("pmid", "unknown"),
                    "title": row.get("title", "")[:100],
                    "relevant": row.get("relevant", "ERROR"),
                    "criterion": "publication_year",
                    "value": row.get("temp_year"),
                    "reason": f"Published before 2000 (extracted year: {row.get('temp_year')})"
                })
        
        df = df[year_mask].copy()
        logger.info(f"Year filter: excluded {excluded_by_year} records published before 2000")
        
        if 'temp_year' in df.columns:
            df.drop('temp_year', axis=1, inplace=True)
    
    # if 'language' in df.columns:
    #     language_mask = df['language'].apply(check_language)
    #     excluded_by_language = (~language_mask).sum()
    #     filter_stats["excluded_by_language"] = int(excluded_by_language)
        
    #     if excluded_by_language > 0:
    #         for _, row in df[~language_mask].iterrows():
    #             filter_log.append({
    #                 "pmid": row.get("pmid", "unknown"),
    #                 "title": row.get("title", "")[:100],
    #                 "relevant": row.get("relevant", "ERROR"),
    #                 "criterion": "language",
    #                 "value": row.get("language"),
    #                 "reason": f"Non-English/non-French (language: {row.get('language')})"
    #             })
        
    #     df = df[language_mask].copy()
    #     logger.info(f"Language filter: excluded {excluded_by_language} non-English/non-French records")
    
    if 'publication_types' in df.columns:
        pub_type_results = df['publication_types'].apply(check_publication_type)
        pub_type_mask = [result[0] for result in pub_type_results]
        pub_type_reasons = [result[1] for result in pub_type_results]
        
        excluded_by_pub_type = sum(not mask for mask in pub_type_mask)
        filter_stats["excluded_by_pub_type"] = int(excluded_by_pub_type)
        
        if excluded_by_pub_type > 0:
            excluded_indices = [i for i, mask in enumerate(pub_type_mask) if not mask]
            for idx in excluded_indices:
                row = df.iloc[idx]
                filter_log.append({
                    "pmid": row.get("pmid", "unknown"),
                    "title": row.get("title", "")[:100],
                    "relevant": row.get("relevant", "ERROR"),
                    "criterion": "publication_type",
                    "value": row.get("publication_types"),
                    "reason": pub_type_reasons[idx] or "Excluded publication type"
                })
        
        df = df[[mask for mask in pub_type_mask]].copy()
        logger.info(f"Publication type filter: excluded {excluded_by_pub_type} records with excluded publication types")
    
    filter_stats["included_count"] = len(df)
    filter_stats["excluded_total"] = original_count - len(df)
    
    if filter_log_dir and filter_log:
        os.makedirs(filter_log_dir, exist_ok=True)
        log_df = pd.DataFrame(filter_log)
        log_path = os.path.join(filter_log_dir, "hard_filter_log.csv")
        log_df.to_csv(log_path, index=False)
        logger.info(f"Filter log saved to {log_path}")
        
        excluded_df = log_df.copy()
        excl_path = os.path.join(filter_log_dir, "excluded_titles.csv")
        excluded_df.to_csv(excl_path, index=False)
        logger.info(f"Excluded titles saved to {excl_path}")
        
        stats_path = os.path.join(filter_log_dir, "filter_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(filter_stats, f, indent=2)
    
    logger.info(f"Filtering complete: {filter_stats['excluded_total']} records excluded")
    logger.info(f"Retention rate: {len(df)/original_count:.2%}")
    
    return df, filter_stats