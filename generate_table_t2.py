import pandas as pd
import numpy as np

def generate_latex_table_with_highlights(csv_file, metric='R2', experiment_filter='all', ad=False):
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
    exp_name = experiment_filter  # Capitalize experiment name 
    if ad:
        latex_code = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{Transfer learning results for {exp_name} experiments in {metric} (Pretrain-finetune)}}",
            f"\\label{{tab:t2ad-model-evaluation-{experiment_filter}-{metric.lower()}}}"
        ]
    else:
        latex_code = [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{Transfer learning results for {exp_name} experiments in {metric} (Adversarial Training)}}",
            f"\\label{{tab:t2-model-evaluation-{experiment_filter}-{metric.lower()}}}"
        ]
    
    # Calculate total columns for this table
    total_cols = 1  # Start with Model column
    t0_col_count = {}
    for t0_subset in t0_subsets:
        t0_col_count[t0_subset] = 0
        for t1_subset in t1_subsets:
            t0_col_count[t0_subset] += len(subset_features[t1_subset])
            total_cols += len(subset_features[t1_subset])  # Add to total column count
    
    # Create column specifications with vertical lines
    col_spec = "|l|"  # Start with left-aligned first column with vertical border
    
    for t0_idx, t0_subset in enumerate(t0_subsets):
        for t1_idx, t1_subset in enumerate(t1_subsets):
            col_spec += "c" * len(subset_features[t1_subset])
            # # Add vertical line after each t1_subset (if not the last one)
            # if t1_idx < len(t1_subsets) - 1:
            #     col_spec += "|"
        # Add vertical line after each t0_subset (if not the last one)
        if t0_idx < len(t0_subsets) - 1:
            col_spec += "|"
    
    # Add final vertical line
    col_spec += "|"
    
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_code.append("\\hline")
    
    # First header row - Model and t0_subsets, with Model spanning 3 rows
    header_rows = 2  # Number of header rows
    header = [f"\\multirow{{{header_rows}}}{{*}}{{Model}}"]
    
    for t0_subset in t0_subsets:
        print(f"t0_subset: {t0_subset}")
        if t0_subset == "mw":
            display_name = "Ecosys"
        else:
            display_name = t0_subset.upper()
        header.append(f"\\multicolumn{{{t0_col_count[t0_subset]}}}{{|c|}}{{{display_name}}}")
    
    latex_code.append(" & ".join(header) + " \\\\")
    
    # Second header row - N2O and CO2 subsets within each t0_subset
    subheader = [""]  # Empty cell for the Model column which is already covered by multirow
    
    # for t0_subset in t0_subsets:
    #     for t1_subset in t1_subsets:
    #         display_name = "N$_2$O Data" if t1_subset == "n2o" else "CO$_2$ Data"
    #         feature_count = len(subset_features[t1_subset])
    #         # Add border specification with vertical line
    #         border_spec = "|c|" if t1_subset == list(t1_subsets)[-1] else "|c"
    #         subheader.append(f"\\multicolumn{{{feature_count}}}{{{border_spec}}}{{{display_name}}}")
    
    # latex_code.append("\\cline{2-" + str(total_cols) + "}")
    # latex_code.append(" & ".join(subheader) + " \\\\")
    
    # Third header row - features within each subset
    subsubheader = [""]  # Empty cell for the Model column which is already covered by multirow
    for t0_subset in t0_subsets:
        for t1_subset in t1_subsets:
            for feature in subset_features[t1_subset]:
                feature_name = "N$_2$O" if t1_subset == "n2o" and feature == "Feature 1" else "CO$_2$" if feature == "Feature 1" else "GPP"
                subsubheader.append(feature_name)
    
    latex_code.append("\\cline{2-" + str(total_cols) + "}")
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
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_code)

# Generate tables for each metric and experiment type
if __name__ == "__main__":
    for metric in ['R2', 'RMSE', 'MAE']:
        for experiment in ['spatial', 'temporal']:
            table = generate_latex_table_with_highlights("evaluation_results_t2.csv", 
                                                        metric=metric, 
                                                        experiment_filter=experiment,
                                                        ad=False)
            
            with open(f"t2_evaluation_table_{experiment}_{metric.lower()}.tex", "w") as f:
                f.write(table)
    
    for metric in ['R2', 'RMSE', 'MAE']:
        for experiment in ['spatial', 'temporal']:
            table = generate_latex_table_with_highlights("evaluation_results_t2ad.csv", 
                                                        metric=metric, 
                                                        experiment_filter=experiment,
                                                        ad=True)
            
            with open(f"t2ad_evaluation_table_{experiment}_{metric.lower()}.tex", "w") as f:
                f.write(table)
    print("Generated all tables successfully.")