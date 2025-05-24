import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

def load_model(model_path):
    """Load a trained model from a .joblib file."""
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_data(data_path):
    """Load the dataset containing the papers."""
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_predictions(df, model, model_name):
    """Generate predictions using the loaded model."""
    try:
        # Special handling for cosine model
        if model_name == "cosine":
            try:
                # Try to set the threshold directly on the model
                if hasattr(model.named_steps['clf'], 'threshold'):
                    model.named_steps['clf'].threshold = 0.3
                    print(f"Set cosine model threshold to 0.3")
                # Get probabilities
                probabilities = model.predict_proba(df)[:, 1]
                # Apply threshold manually
                predictions = (probabilities >= 0.3).astype(int)
                print(f"Applied manual threshold of 0.3 to cosine predictions")
                return predictions, probabilities
            except Exception as e:
                print(f"Error applying threshold to cosine model: {e}")
                # Fall back to regular prediction
        
        # Regular prediction for all models
        predictions = model.predict(df)
        probabilities = None
        
        # Try to get probabilities if the model supports it
        try:
            probabilities = model.predict_proba(df)[:, 1]
        except:
            print(f"Model {model_name} doesn't support predict_proba. Using binary predictions only.")
        
        return predictions, probabilities
    except Exception as e:
        print(f"Error generating predictions for {model_name}: {e}")
        return None, None

def categorize_papers(df, student_labels, system_predictions, model_name):
    """
    Categorize papers based on student and system labels:
    - True Positives: Both labeled as relevant
    - False Negatives: Student labeled as relevant, system did not
    - True Negatives: Both labeled as irrelevant
    - False Positives: System labeled as relevant, student did not
    """
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Add prediction column
    result_df[f'{model_name}_prediction'] = system_predictions
    
    # Create category column
    result_df[f'{model_name}_category'] = 'unknown'
    
    # True positives: Both labeled as relevant
    mask = (student_labels == 1) & (system_predictions == 1)
    result_df.loc[mask, f'{model_name}_category'] = 'true_positive'
    
    # False negatives: Student labeled as relevant, system did not
    mask = (student_labels == 1) & (system_predictions == 0)
    result_df.loc[mask, f'{model_name}_category'] = 'false_negative'
    
    # True negatives: Both labeled as irrelevant
    mask = (student_labels == 0) & (system_predictions == 0)
    result_df.loc[mask, f'{model_name}_category'] = 'true_negative'
    
    # False positives: System labeled as relevant, student did not
    mask = (student_labels == 0) & (system_predictions == 1)
    result_df.loc[mask, f'{model_name}_category'] = 'false_positive'
    
    return result_df

def generate_category_stats(df, category_column):
    """Generate statistics for each category."""
    categories = df[category_column].unique()
    stats = {}
    
    for category in categories:
        count = df[df[category_column] == category].shape[0]
        percentage = count / df.shape[0] * 100
        stats[category] = {
            'count': count,
            'percentage': percentage
        }
    
    return stats

def print_confusion_matrix(student_labels, system_predictions, model_name):
    """Print confusion matrix for a model."""
    cm = confusion_matrix(student_labels, system_predictions)
    print(f"\nConfusion Matrix for {model_name}:")
    print("               | Predicted Negative | Predicted Positive |")
    print("---------------|-------------------|-------------------|")
    print(f"Actual Negative | {cm[0][0]:17d} | {cm[0][1]:17d} |")
    print(f"Actual Positive | {cm[1][0]:17d} | {cm[1][1]:17d} |")
    
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(student_labels, system_predictions))

def save_category_titles(df, category_column, title_column, output_dir, model_name):
    """Save titles from each category to separate CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    categories = df[category_column].unique()
    result = {}
    
    for category in categories:
        category_df = df[df[category_column] == category]
        output_file = os.path.join(output_dir, f"{model_name}_{category}_titles.csv")
        category_df.to_csv(output_file, index=False)
        print(f"Saved {category_df.shape[0]} {category} titles to {output_file}")
        
        # Add to result dictionary for return
        result[category] = category_df[title_column].tolist()
    
    return result

def create_example_file(combined_df, model_names, output_dir, title_column="title", label_column="relevant"):
    """
    Create a text file with examples of:
    1. 3 titles where student labeled as relevant and all models agreed (true positives)
    2. 3 titles where student labeled as relevant but all models disagreed (false negatives)
    3. 3 titles where student labeled as irrelevant and all models agreed (true negatives)
    """
    output_file = os.path.join(output_dir, "example_cases.txt")
    
    # Case 1: All true positives (student: relevant, all models: relevant)
    true_positives = combined_df.copy()
    for model_name in model_names:
        true_positives = true_positives[true_positives[f'{model_name}_prediction'] == 1]
    true_positives = true_positives[true_positives[label_column] == 1]
    
    # Case 2: All false negatives (student: relevant, all models: irrelevant)
    false_negatives = combined_df.copy()
    for model_name in model_names:
        false_negatives = false_negatives[false_negatives[f'{model_name}_prediction'] == 0]
    false_negatives = false_negatives[false_negatives[label_column] == 1]
    
    # Case 3: All true negatives (student: irrelevant, all models: irrelevant)
    true_negatives = combined_df.copy()
    for model_name in model_names:
        true_negatives = true_negatives[true_negatives[f'{model_name}_prediction'] == 0]
    true_negatives = true_negatives[true_negatives[label_column] == 0]

    if len(true_positives) < 3:
        print(f"Warning: Only found {len(true_positives)} true positive examples where all models agree")
        # Use what we have or get examples where at least 2 models agree if needed

    if len(false_negatives) < 3:
        print(f"Warning: Only found {len(false_negatives)} false negative examples where all models agree")
        # Use what we have or get examples where at least 2 models agree if needed

    if len(true_negatives) < 3:
        print(f"Warning: Only found {len(true_negatives)} true negative examples where all models agree")
        # Use what we have or get examples where at least 2 models agree if needed
    
    with open(output_file, 'w') as f:
        f.write("==== EXAMPLE CASES FROM MODEL COMPARISON ====\n\n")
        
        # Write true positives
        f.write("1. TITLES WHERE STUDENT LABELED AS RELEVANT AND ALL MODELS AGREED (TRUE POSITIVES):\n")
        if len(true_positives) >= 3:
            for i in range(3):
                f.write(f"   {i+1}. {true_positives.iloc[i][title_column]}\n")
        else:
            f.write("   Not enough examples found (need at least 3)\n")
        f.write("\n")
        
        # Write false negatives
        f.write("2. TITLES WHERE STUDENT LABELED AS RELEVANT BUT ALL MODELS DISAGREED (FALSE NEGATIVES):\n")
        if len(false_negatives) >= 3:
            for i in range(3):
                f.write(f"   {i+1}. {false_negatives.iloc[i][title_column]}\n")
        else:
            f.write("   Not enough examples found (need at least 3)\n")
        f.write("\n")
        
        # Write true negatives
        f.write("3. TITLES WHERE STUDENT LABELED AS IRRELEVANT AND ALL MODELS AGREED (TRUE NEGATIVES):\n")
        if len(true_negatives) >= 3:
            for i in range(3):
                f.write(f"   {i+1}. {true_negatives.iloc[i][title_column]}\n")
        else:
            f.write("   Not enough examples found (need at least 3)\n")
        f.write("\n")
        
        # Add statistics
        f.write("==== STATISTICS ====\n")
        f.write(f"Total papers: {len(combined_df)}\n")
        f.write(f"Papers where student labeled as relevant and all models agreed: {len(true_positives)} ({len(true_positives)/len(combined_df)*100:.2f}%)\n")
        f.write(f"Papers where student labeled as relevant but all models disagreed: {len(false_negatives)} ({len(false_negatives)/len(combined_df)*100:.2f}%)\n")
        f.write(f"Papers where student labeled as irrelevant and all models agreed: {len(true_negatives)} ({len(true_negatives)/len(combined_df)*100:.2f}%)\n")
    
    print(f"\nSaved example cases to {output_file}")
    return {
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }

def analyze_all_models(df, models_data, output_dir, title_column="title", label_column="relevant"):
    """Analyze all models and generate comparison statistics."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for each model
    all_results = {}
    
    # Process each model
    for model_name, model_info in models_data.items():
        model = load_model(model_info['path'])
        if model is None:
            continue
            
        print(f"\nProcessing model: {model_name}")
        predictions, probabilities = generate_predictions(df, model, model_name)
        
        if predictions is None:
            continue
            
        # Categorize papers
        result_df = categorize_papers(df, df[label_column], predictions, model_name)
        
        # Print confusion matrix and classification report
        print_confusion_matrix(df[label_column], predictions, model_name)
        
        # Generate category statistics
        category_stats = generate_category_stats(result_df, f'{model_name}_category')
        print(f"\nCategory statistics for {model_name}:")
        for category, stats in category_stats.items():
            print(f"  - {category}: {stats['count']} ({stats['percentage']:.2f}%)")
        
        # Save categorized titles
        categories = save_category_titles(
            result_df, 
            f'{model_name}_category', 
            title_column, 
            output_dir, 
            model_name
        )
        
        # Add to all results
        all_results[model_name] = {
            'categories': categories,
            'stats': category_stats,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    # Create a combined dataframe with all model predictions
    combined_df = df.copy()
    for model_name in models_data.keys():
        if model_name in all_results:
            combined_df[f'{model_name}_prediction'] = all_results[model_name]['predictions']
            if all_results[model_name]['probabilities'] is not None:
                combined_df[f'{model_name}_probability'] = all_results[model_name]['probabilities']
            combined_df[f'{model_name}_category'] = 'unknown'
            
            # Set categories
            for i, row in combined_df.iterrows():
                student_label = row[label_column]
                system_prediction = row[f'{model_name}_prediction']
                
                if student_label == 1 and system_prediction == 1:
                    combined_df.loc[i, f'{model_name}_category'] = 'true_positive'
                elif student_label == 1 and system_prediction == 0:
                    combined_df.loc[i, f'{model_name}_category'] = 'false_negative'
                elif student_label == 0 and system_prediction == 0:
                    combined_df.loc[i, f'{model_name}_category'] = 'true_negative'
                elif student_label == 0 and system_prediction == 1:
                    combined_df.loc[i, f'{model_name}_category'] = 'false_positive'
    
    # Save the combined results
    combined_output_path = os.path.join(output_dir, "all_models_comparison.csv")
    combined_df.to_csv(combined_output_path, index=False)
    print(f"\nSaved combined comparison to {combined_output_path}")
    
    # Generate model agreement analysis
    generate_model_agreement_analysis(combined_df, all_results.keys(), label_column, output_dir)
    
    # Create examples file
    create_example_file(combined_df, all_results.keys(), output_dir, title_column, label_column)
    
    return all_results, combined_df

def generate_model_agreement_analysis(df, model_names, label_column, output_dir):
    """
    Analyze where models agree and disagree with each other and with student labels.
    """
    if len(model_names) < 2:
        print("Need at least 2 models for agreement analysis.")
        return
    
    # Count where all models agree and are correct
    all_agree_correct = df.copy()
    for model_name in model_names:
        all_agree_correct = all_agree_correct[
            all_agree_correct[f'{model_name}_prediction'] == all_agree_correct[label_column]
        ]
    
    # Count where all models agree but are incorrect
    all_agree_incorrect = df.copy()
    all_agree = True
    model_names_list = list(model_names)
    first_model = model_names_list[0]
    for model_name in model_names_list[1:]:
        all_agree_incorrect = all_agree_incorrect[
            all_agree_incorrect[f'{model_name}_prediction'] == all_agree_incorrect[f'{first_model}_prediction']
        ]
    all_agree_incorrect = all_agree_incorrect[
        all_agree_incorrect[f'{first_model}_prediction'] != all_agree_incorrect[label_column]
    ]
    
    # Create a new dataframe for instances where models disagree
    disagreement_df = df.copy()
    model_names_list = list(model_names)  # Convert dict_keys to list
    for i in range(len(model_names_list)):
        for j in range(i+1, len(model_names_list)):
            model_i = model_names_list[i]
            model_j = model_names_list[j]
            disagreement_df[f'{model_i}_vs_{model_j}'] = (
                disagreement_df[f'{model_i}_prediction'] != disagreement_df[f'{model_j}_prediction']
            )
    
    # Filter to keep only rows where at least one pair disagrees
    any_disagreement = disagreement_df[list(filter(lambda x: '_vs_' in x, disagreement_df.columns))].any(axis=1)
    disagreement_df = disagreement_df[any_disagreement]
    
    # Save disagreement analysis
    if not disagreement_df.empty:
        disagreement_path = os.path.join(output_dir, "model_disagreements.csv")
        disagreement_df.to_csv(disagreement_path, index=False)
        print(f"\nSaved model disagreement analysis to {disagreement_path}")
        print(f"Found {disagreement_df.shape[0]} instances where models disagree")
    
    # Print agreement statistics
    print("\nModel Agreement Analysis:")
    print(f"  - All models agree and are correct: {all_agree_correct.shape[0]} instances ({all_agree_correct.shape[0]/df.shape[0]*100:.2f}%)")
    print(f"  - All models agree but are incorrect: {all_agree_incorrect.shape[0]} instances ({all_agree_incorrect.shape[0]/df.shape[0]*100:.2f}%)")
    print(f"  - Models have some disagreement: {disagreement_df.shape[0]} instances ({disagreement_df.shape[0]/df.shape[0]*100:.2f}%)")
    
    # Create interesting subsets for analysis
    
    # 1. All models think it's relevant, but student says it's not
    all_fp = df.copy()
    for model_name in model_names:
        all_fp = all_fp[all_fp[f'{model_name}_prediction'] == 1]
    all_fp = all_fp[all_fp[label_column] == 0]
    
    # 2. All models think it's irrelevant, but student says it is
    all_fn = df.copy()
    for model_name in model_names:
        all_fn = all_fn[all_fn[f'{model_name}_prediction'] == 0]
    all_fn = all_fn[all_fn[label_column] == 1]
    
    # Save these interesting subsets
    if not all_fp.empty:
        all_fp_path = os.path.join(output_dir, "all_models_false_positive.csv")
        all_fp.to_csv(all_fp_path, index=False)
        print(f"\nSaved {all_fp.shape[0]} papers where all models predict relevant but student says irrelevant")
    
    if not all_fn.empty:
        all_fn_path = os.path.join(output_dir, "all_models_false_negative.csv")
        all_fn.to_csv(all_fn_path, index=False)
        print(f"\nSaved {all_fn.shape[0]} papers where all models predict irrelevant but student says relevant")

if __name__ == "__main__":
    # Paths to your data and models - using relative paths now
    data_path = "data/processed/data_final_processed.csv"
    models_data = {
        "logreg": {
            "path": "results/final/baseline/logreg/models/model.joblib",
            "params": "results/final/baseline/logreg/models/params.json"
        },
        "svm": {
            "path": "results/final/baseline/svm/models/model.joblib",
            "params": "results/final/baseline/svm/models/params.json"
        },
        "cosine": {
            "path": "results/final/baseline/cosine/models/model.joblib",
            "params": "results/final/baseline/cosine/models/params.json"
        }
    }
    
    # Output directory for results
    output_dir = "results/model_comparison"
    
    # Load the dataset
    df = load_data(data_path)
    
    if df is not None:
        # Run analysis for all models
        results, combined_df = analyze_all_models(
            df, 
            models_data, 
            output_dir, 
            title_column="title",
            label_column="relevant"
        )