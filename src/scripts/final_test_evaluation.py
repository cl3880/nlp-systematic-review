import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.config import PATHS
from src.utils.data_utils import load_data, make_splits

def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Calculate F2 score (weighs recall higher than precision)
    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
    
    # ROC AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Calculate Work Saved over Sampling at 95% recall (WSS@95)
    thresholds = np.sort(y_prob)
    best_wss = 0
    for threshold in thresholds:
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        tn = np.sum((y_true == 0) & (y_pred_at_threshold == 0))
        fp = np.sum((y_true == 0) & (y_pred_at_threshold == 1))
        fn = np.sum((y_true == 1) & (y_pred_at_threshold == 0))
        tp = np.sum((y_true == 1) & (y_pred_at_threshold == 1))
        
        if fn == 0:  # Perfect recall
            current_recall = 1.0
        else:
            current_recall = tp / (tp + fn)
            
        if current_recall >= 0.95:
            wss = (tn + fn) / (tn + fp + fn + tp) - (1 - current_recall)
            best_wss = max(best_wss, wss)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'roc_auc': roc_auc,
        'wss_at_95': best_wss
    }

def optimize_threshold_for_recall(y_true, y_prob, target_recall=0.95):
    """Find optimal threshold to achieve target recall."""
    thresholds = np.sort(y_prob)
    best_threshold = 0
    best_diff = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        diff = abs(recall - target_recall)
        if diff < best_diff and recall >= target_recall:
            best_diff = diff
            best_threshold = threshold
            
    return best_threshold

def evaluate_model_on_test_set(model_path, model_name):
    """Evaluate a model on the test set."""
    print(f"Evaluating model: {model_name}")
    print(f"Loading model from: {model_path}")
    
    # Load the model
    pipeline = joblib.load(model_path)
    
    # Load and prepare test data
    df = load_data(os.path.join(PATHS["data_processed"], "data_final_processed.csv"))
    _, _, test = make_splits(df, test_size=0.1, val_size=0.1, stratify=True, seed=42)
    
    X_test = test.drop('relevant', axis=1)
    y_test = test['relevant']
    
    # Use normalized_text column if it exists
    if 'normalized_text' in X_test.columns:
        # This depends on how your model was trained
        # If you need to apply stemming/lemmatization here, you would need to do that
        pass
    
    # Make predictions
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    # Calculate balanced metrics
    balanced_metrics = compute_metrics(y_test, y_pred, y_prob)
    
    # Find optimal threshold for high recall
    threshold = optimize_threshold_for_recall(y_test, y_prob, target_recall=0.95)
    
    # Calculate high-recall metrics
    y_pred_hr = (y_prob >= threshold).astype(int)
    hr_metrics = compute_metrics(y_test, y_pred_hr, y_prob)
    
    # Print results
    print(f"\nResults for {model_name}:")
    print("\nBalanced model metrics:")
    for metric, value in balanced_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nHigh-recall model metrics:")
    for metric, value in hr_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  threshold: {threshold:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'model': [model_name],
        'precision': [balanced_metrics['precision']],
        'recall': [balanced_metrics['recall']],
        'f1': [balanced_metrics['f1']],
        'f2': [balanced_metrics['f2']],
        'roc_auc': [balanced_metrics['roc_auc']],
        'wss_at_95': [balanced_metrics['wss_at_95']],
        'hr_precision': [hr_metrics['precision']],
        'hr_recall': [hr_metrics['recall']],
        'hr_f1': [hr_metrics['f1']],
        'hr_f2': [hr_metrics['f2']],
        'hr_roc_auc': [hr_metrics['roc_auc']],
        'hr_wss_at_95': [hr_metrics['wss_at_95']],
        'threshold': [threshold]
    })
    
    results_path = f"results/test_evaluation_{model_name.replace(' ', '_')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return balanced_metrics, hr_metrics

if __name__ == "__main__":
    # Replace these paths with your actual model paths
    models_to_evaluate = {
        "Raw + SMOTE + Criteria": "results/v2.2.1-skf/svm_critieria_smote/pipeline.joblib", 
        "+ Criteria + MeSH": "results/v2.2.1-skf/criteria_svm_direct_mesh_20250513_195035/pipeline.joblib"
    }
    
    for model_name, model_path in models_to_evaluate.items():
        evaluate_model_on_test_set(model_path, model_name)
        print("\n" + "="*50 + "\n")