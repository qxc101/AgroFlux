import pandas as pd
import numpy as np

def generate_latex_table_with_highlights(csv_file, metric='R2'):
    """
    Generate a LaTeX table with highlighted best values for the specified metric.
    
    Parameters:
    csv_file (str): Path to the CSV file containing evaluation results
    metric (str): Metric to display in the table. Options: 'R2', 'RMSE', 'MAE'
    
    Returns:
    str: LaTeX code for the table
    """
    # Read the CSV data
    df = pd.read_csv(csv_file)
    
    # Get unique values
    models = sorted(df['Model'].unique())
    subsets = sorted(df['Subset'].unique())
    experiments = sorted(df['Experiment'].unique())
    
    # Determine features for each subset
    subset_features = {}
    for subset in subsets:
        subset_df = df[df['Subset'] == subset]
        subset_features[subset] = sorted(subset_df['Feature'].unique())
    
    # Display names for features
    feature_names = {
        "Feature 1": "N$_2$O" if subset == "n2o" else "CO$_2$",
        "Feature 2": "GPP"
    }
    
    # Determine if higher or lower values are better
    higher_is_better = metric == 'R2'  # For R2, higher is better; for RMSE and MAE, lower is better
    
    # Function to calculate average metric value for experiments
    def calculate_avg_metric(model, subset, experiment, feature):
        filtered_df = df[(df['Model'] == model) & 
                          (df['Subset'] == subset) & 
                          (df['Experiment'] == experiment) &
                          (df['Feature'] == feature)]
        
        if experiment == 'spatial':
            # Calculate average across five folds
            return filtered_df[metric].mean() if len(filtered_df) > 0 else None
        else:
            # For temporal, just return the single value
            return filtered_df[metric].iloc[0] if len(filtered_df) > 0 else None
    
    # Find the best value for each column
    best_values = {}
    for subset in subsets:
        best_values[subset] = {}
        for experiment in experiments:
            best_values[subset][experiment] = {}
            for feature in subset_features[subset]:
                if higher_is_better:
                    best_value = -float('inf')
                    for model in models:
                        value = calculate_avg_metric(model, subset, experiment, feature)
                        if value is not None and value > best_value:
                            best_value = value
                else:
                    best_value = float('inf')
                    for model in models:
                        value = calculate_avg_metric(model, subset, experiment, feature)
                        if value is not None and value < best_value:
                            best_value = value
                            
                best_values[subset][experiment][feature] = best_value
    
    # Start building the LaTeX table
    latex_code = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{Model evaluation results on observation datasets in {metric}}}",
        f"\\label{{tab:t1-model-evaluation-{metric.lower()}}}"
    ]
    
    # Calculate total columns - for each subset, we have experiment_count * feature_count columns
    n2o_cols = len(experiments) * len(subset_features['n2o'])
    co2_cols = len(experiments) * len(subset_features['co2'])
    
    # Create column specifications with vertical lines
    col_spec = "|l|"  # Start with left-aligned first column with vertical borders
    
    # CO2 features with vertical lines
    for feature_idx, feature in enumerate(subset_features['co2']):
        col_spec += "c" * len(experiments)
        if feature_idx < len(subset_features['co2']) - 1:
            col_spec += "|"  # Add vertical line between features
    
    # Add vertical line between CO2 and N2O sections
    col_spec += "|"
    
    # N2O features
    col_spec += "c" * n2o_cols + "|"  # Add vertical line at the end
    
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_code.append("\\hline")
    
    # First header row with multirow for Model spanning 2 rows
    header_rows = 2  # Number of header rows
    header = [f"\\multirow{{{header_rows}}}{{*}}{{Model}}"]
    
    # For CO2 section (has feature 1 and 2)
    for feature in subset_features['co2']:
        feature_name = "CO$_2$" if feature == "Feature 1" else "GPP"
        header.append(f"\\multicolumn{{{len(experiments)}}}{{|c|}}{{{feature_name}}}")
        
    # For N2O section (only has feature 1)
    header.append(f"\\multicolumn{{{len(experiments)}}}{{|c|}}{{N$_2$O}}")
    
    latex_code.append(" & ".join(header) + " \\\\")
    
    # Second header row - temporal and spatial for each feature
    subheader = [""]  # Empty cell for the Model column which is already covered by multirow
    
    # For CO2's features
    for feature in subset_features['co2']:
        for experiment in experiments:
            # Capitalize experiment names
            display_experiment = experiment.capitalize()
            subheader.append(display_experiment)
            
    # For N2O's single feature
    for experiment in experiments:
        # Capitalize experiment names
        display_experiment = experiment.capitalize()
        subheader.append(display_experiment)
        
    latex_code.append("\\cline{2-" + str(1 + co2_cols + n2o_cols) + "}")
    latex_code.append(" & ".join(subheader) + " \\\\ \\hline")
    
    # Add data rows with properly capitalized model names
    for model in models:
        # Convert model name to proper display format
        if model == "ealstm":
            display_model = "EALSTM"
        elif model == "itransformer":
            display_model = "iTransformer"
        elif model == "lstm":
            display_model = "LSTM"
        elif model == "pyraformer":
            display_model = "Pyraformer"
        elif model == "tcn":
            display_model = "TCN"
        elif model == "transformer":
            display_model = "Transformer"
        else:
            display_model = model.capitalize()  # Fallback for any other model names
        
        row = [display_model]
        
        # CO2 data (Feature 1 and Feature 2)
        for feature in subset_features['co2']:
            for experiment in experiments:
                value = calculate_avg_metric(model, 'co2', experiment, feature)
                
                # Check if this is the best value for this column
                is_best = False
                if value is not None:
                    if higher_is_better and abs(value - best_values['co2'][experiment][feature]) < 1e-6:
                        is_best = True
                    elif not higher_is_better and abs(value - best_values['co2'][experiment][feature]) < 1e-6:
                        is_best = True
                
                # Format the value (bold if it's the best)
                if is_best:
                    formatted_value = f"\\textbf{{{value:.3f}}}"
                else:
                    formatted_value = f"{value:.3f}" if value is not None else "N/A"
                    
                row.append(formatted_value)
        
        # N2O data (Feature 1 only)
        for experiment in experiments:
            feature = "Feature 1"  # N2O only has Feature 1
            value = calculate_avg_metric(model, 'n2o', experiment, feature)
            
            # Check if this is the best value for this column
            is_best = False
            if value is not None:
                if higher_is_better and abs(value - best_values['n2o'][experiment][feature]) < 1e-6:
                    is_best = True
                elif not higher_is_better and abs(value - best_values['n2o'][experiment][feature]) < 1e-6:
                    is_best = True
            
            # Format the value (bold if it's the best)
            if is_best:
                formatted_value = f"\\textbf{{{value:.3f}}}"
            else:
                formatted_value = f"{value:.3f}" if value is not None else "N/A"
                
            row.append(formatted_value)
        
        latex_code.append(" & ".join(row) + " \\\\")
    
    # Close the table
    latex_code.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

# Generate tables for each metric
if __name__ == "__main__":
    for metric in ['R2', 'RMSE', 'MAE']:
        table = generate_latex_table_with_highlights("evaluation_results_t1.csv", metric=metric)
        with open(f"t1_evaluation_table_{metric.lower()}.tex", "w") as f:
            f.write(table)
    
    print("Generated all tables successfully.")