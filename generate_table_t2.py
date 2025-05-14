import pandas as pd
import numpy as np

def generate_latex_table_with_highlights(csv_file, metric='R2', experiment_filter='all'):
    """
    Generate a LaTeX table with highlighted best values for the specified metric and experiment type.
    
    Parameters:
    csv_file (str): Path to the CSV file containing evaluation results
    metric (str): Metric to display in the table. Options: 'R2', 'RMSE', 'MAE'
    experiment_filter (str): Filter for experiment type. Options: 'all', 'spatial', 'temporal'
    
    Returns:
    str: LaTeX code for the table
    """
    # Read the CSV data
    df = pd.read_csv(csv_file)
    
    # Filter by experiment if needed
    if experiment_filter != 'all':
        df = df[df['Experiment'] == experiment_filter]
    
    # Get unique values
    models = sorted(df['Model'].unique())
    t0_subsets = sorted(df['t0_Subset'].unique())
    t1_subsets = sorted(df['t1_Subset'].unique())
    
    # Determine features for each subset
    subset_features = {}
    for t1_subset in t1_subsets:
        subset_df = df[df['t1_Subset'] == t1_subset]
        subset_features[t1_subset] = sorted(subset_df['Feature'].unique())
    
    # Determine if higher or lower values are better
    higher_is_better = metric == 'R2'  # For R2, higher is better; for RMSE and MAE, lower is better
    
    # Function to calculate metric value
    def get_metric_value(model, t0_subset, t1_subset, feature):
        filtered_df = df[(df['Model'] == model) & 
                         (df['t0_Subset'] == t0_subset) &
                         (df['t1_Subset'] == t1_subset) & 
                         (df['Feature'] == feature)]
        
        return filtered_df[metric].iloc[0] if len(filtered_df) > 0 else None
    
    # Find the best value for each column
    best_values = {}
    for t0_subset in t0_subsets:
        best_values[t0_subset] = {}
        for t1_subset in t1_subsets:
            best_values[t0_subset][t1_subset] = {}
            for feature in subset_features[t1_subset]:
                if higher_is_better:
                    best_value = -float('inf')
                    for model in models:
                        value = get_metric_value(model, t0_subset, t1_subset, feature)
                        if value is not None and value > best_value:
                            best_value = value
                else:
                    best_value = float('inf')
                    for model in models:
                        value = get_metric_value(model, t0_subset, t1_subset, feature)
                        if value is not None and value < best_value:
                            best_value = value
                            
                best_values[t0_subset][t1_subset][feature] = best_value
    
    # Start building the LaTeX table
    exp_name = "spatial" if experiment_filter == "spatial" else "temporal" if experiment_filter == "temporal" else "all"
    latex_code = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{Model evaluation results for {exp_name} experiments in {metric}}}",
        f"\\label{{tab:t2-model-evaluation-{exp_name}-{metric.lower()}}}",
    ]
    
    # Calculate total columns for this table
    col_count = 1  # Model column
    for t0_subset in t0_subsets:
        for t1_subset in t1_subsets:
            col_count += len(subset_features[t1_subset])
    
    # Create column specifications
    col_spec = "l" + "c" * (col_count - 1)
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_code.append("\\hline")
    
    # First header row - Model and t0_subsets
    header = ["Model"]
    for t0_subset in t0_subsets:
        column_count = 0
        for t1_subset in t1_subsets:
            column_count += len(subset_features[t1_subset])
        display_name = t0_subset.upper()
        header.append(f"\\multicolumn{{{column_count}}}{{c}}{{{display_name}}}")
    latex_code.append(" & ".join(header) + " \\\\")
    
    # Second header row - N2O and CO2 subsets within each t0_subset
    subheader = [""]
    for t0_subset in t0_subsets:
        for t1_subset in t1_subsets:
            display_name = "N$_2$O" if t1_subset == "n2o" else "CO$_2$"
            feature_count = len(subset_features[t1_subset])
            subheader.append(f"\\multicolumn{{{feature_count}}}{{c}}{{{display_name}}}")
    latex_code.append(" & ".join(subheader) + " \\\\")
    
    # Third header row - features within each subset
    subsubheader = [""]
    for t0_subset in t0_subsets:
        for t1_subset in t1_subsets:
            for feature in subset_features[t1_subset]:
                feature_name = "N$_2$O" if t1_subset == "n2o" and feature == "Feature 1" else "CO$_2$" if feature == "Feature 1" else "GPP"
                subsubheader.append(feature_name)
    latex_code.append(" & ".join(subsubheader) + " \\\\ \\hline")
    
    # Add data rows
    for model in models:
        row = [model]
        
        # Add data for each t0_subset, t1_subset, feature combination
        for t0_subset in t0_subsets:
            for t1_subset in t1_subsets:
                for feature in subset_features[t1_subset]:
                    value = get_metric_value(model, t0_subset, t1_subset, feature)
                    
                    # Check if this is the best value for this column
                    is_best = False
                    if value is not None:
                        if higher_is_better and abs(value - best_values[t0_subset][t1_subset][feature]) < 1e-6:
                            is_best = True
                        elif not higher_is_better and abs(value - best_values[t0_subset][t1_subset][feature]) < 1e-6:
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
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

# Generate tables for each metric and experiment type
if __name__ == "__main__":
    for metric in ['R2', 'RMSE', 'MAE']:
        for experiment in ['spatial', 'temporal']:
            table = generate_latex_table_with_highlights("evaluation_results_t2.csv", 
                                                        metric=metric, 
                                                        experiment_filter=experiment)
            
            with open(f"t2_evaluation_table_{experiment}_{metric.lower()}.tex", "w") as f:
                f.write(table)
    
    for metric in ['R2', 'RMSE', 'MAE']:
        for experiment in ['spatial', 'temporal']:
            table = generate_latex_table_with_highlights("evaluation_results_t2ad.csv", 
                                                        metric=metric, 
                                                        experiment_filter=experiment)
            
            with open(f"t2ad_evaluation_table_{experiment}_{metric.lower()}.tex", "w") as f:
                f.write(table)
    print("Generated all tables successfully.")