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
    
    # Define the features we want
    selected_features = ["Feature 3", "Feature 2", "Feature 16"]
    selected_features_name = ["CO$_2$", "GPP", "N$_2$O"]
    # Get unique values
    models = sorted(df['Model'].unique())
    subsets = sorted(df['Subset'].unique())
    experiments = sorted(df['Experiment'].unique())
    
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
            return filtered_df[metric].mean()
        else:
            # For temporal, just return the single value
            return filtered_df[metric].iloc[0] if len(filtered_df) > 0 else None
    
    # Find the best value for each column
    best_values = {}
    for subset in subsets:
        best_values[subset] = {}
        for experiment in experiments:
            best_values[subset][experiment] = {}
            for feature in selected_features:
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
        f"\\caption{{Model evaluation results on simulated datasets in {metric}}}",
        f"\\label{{tab:t0-model-evaluation-{metric.lower()}}}",
        "\\resizebox{\\textwidth}{!}{%"
    ]
    
    # Create column specifications with vertical lines
    # First column for model names, then add vertical lines for subsets and experiments
    col_spec = "|l|"  # Start with left-aligned first column with vertical borders
    
    for subset_idx, subset in enumerate(subsets):
        for exp_idx, experiment in enumerate(experiments):
            # Add the selected features with vertical lines for each experiment
            col_spec += "c" * len(selected_features)
            # Add vertical line after each experiment (if not the last one)
            if exp_idx < len(experiments) - 1 or subset_idx < len(subsets) - 1:
                col_spec += "|"
    
    # Add final vertical line
    col_spec += "|"
    
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_code.append("\\hline")
    
    # Add header row with subsets - make Model span all rows vertically
    header_rows = 3  # Number of header rows
    header = [f"\\multirow{{{header_rows}}}{{*}}{{Model}}"]
    for subset in subsets:
        col_span = len(experiments) * len(selected_features)
        header.append(f"\\multicolumn{{{col_span}}}{{|c|}}{{{subset.upper()}}}")
    latex_code.append(" & ".join(header) + " \\\\")
    
    # Add sub-header row with experiment types (without \hline after first header)
    subheader = [""]  # Empty cell for the Model column which is already covered by multirow
    for subset_idx, subset in enumerate(subsets):
        for exp_idx, experiment in enumerate(experiments):
            col_span = len(selected_features)
            # Add vertical line if it's not the first experiment in the subset
            border_spec = "|c|" if exp_idx == len(experiments) - 1 else "|c"
            # For the first experiment in the subset, use a different border specification
            if exp_idx == 0:
                border_spec = "c|"
                
            # Capitalize experiment names
            display_experiment = experiment.capitalize()
            subheader.append(f"\\multicolumn{{{col_span}}}{{{border_spec}}}{{{display_experiment}}}")
    latex_code.append("\\cline{2-" + str(1 + len(subsets) * len(experiments) * len(selected_features)) + "}")
    latex_code.append(" & ".join(subheader) + " \\\\")
    
    # Add sub-sub-header row with feature types
    subsubheader = [""]  # Empty cell for the Model column which is already covered by multirow
    for subset_idx, subset in enumerate(subsets):
        for exp_idx, experiment in enumerate(experiments):
            for feat_idx, feature in enumerate(selected_features_name):
                # Add vertical line markers for appropriate cells
                if feat_idx == 0 and exp_idx == 0:  # First feature in first experiment of a subset
                    subsubheader.append(feature)
                elif feat_idx == len(selected_features) - 1 and exp_idx == len(experiments) - 1:  # Last feature in last experiment
                    subsubheader.append(feature)
                else:
                    subsubheader.append(feature)
    latex_code.append("\\cline{2-" + str(1 + len(subsets) * len(experiments) * len(selected_features)) + "}")
    latex_code.append(" & ".join(subsubheader) + " \\\\ \\hline")
    
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
        for subset in subsets:
            for experiment in experiments:
                for feature in selected_features:
                    value = calculate_avg_metric(model, subset, experiment, feature)
                    
                    # Check if this is the best value for this column
                    is_best = False
                    if value is not None:
                        if higher_is_better and abs(value - best_values[subset][experiment][feature]) < 1e-6:
                            is_best = True
                        elif not higher_is_better and abs(value - best_values[subset][experiment][feature]) < 1e-6:
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
        "\\end{tabular}%",
        "}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

# Generate all three tables
if __name__ == "__main__":
    # Generate R^2 table
    r2_table = generate_latex_table_with_highlights("evaluation_results.csv", metric='R2')
    with open("t0_evaluation_table_r2.tex", "w") as f:
        f.write(r2_table)
    
    # Generate RMSE table
    rmse_table = generate_latex_table_with_highlights("evaluation_results.csv", metric='RMSE')
    with open("t0_evaluation_table_rmse.tex", "w") as f:
        f.write(rmse_table)
    
    # Generate MAE table
    mae_table = generate_latex_table_with_highlights("evaluation_results.csv", metric='MAE')
    with open("t0_evaluation_table_mae.tex", "w") as f:
        f.write(mae_table)
    
    print("Generated all three tables successfully.")