import os
import json
import numpy as np
import pandas as pd


def calculate_average_metrics(data):
    """
    Calculate average metrics across multiple samples for llm_1st, kg, and llm_2nd.
    
    Args:
        data (dict): Dictionary where keys are sample IDs and values are dictionaries
                    containing evaluation_results in the specified format
    
    Returns:
        dict: Dictionary containing average metrics for each model type
    """
    # Initialize counters for each metric and model
    metrics = {
        'gat': {'precision': 0, 'recall': 0, 'f1_score': 0,
                     'overlap_count': 0, 'jaccard': 0, 
                     'precision@5': 0, 'precision@10': 0},
        'plain_omics': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'plain_omicskg': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'bmgc_omics': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'bmgc_omicskg': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'qallm_omics': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'qallm_omicskg': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'gretriever': {'precision': 0, 'recall': 0, 'f1_score': 0,
                    'overlap_count': 0, 'jaccard': 0, 
                    'precision@5': 0, 'precision@10': 0},
        'llm_1st': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'llm_2nd': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                    'overlap_count': 0, 'jaccard': 0, 
                    'precision@5': 0, 'precision@10': 0},
        'llm_2nd_2nd': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                     'overlap_count': 0, 'jaccard': 0, 
                     'precision@5': 0, 'precision@10': 0}
    }
    
    # Count samples with valid metrics
    valid_samples = 0
    
    # Sum up all metrics
    for sample_id, sample_data in data.items():
        if 'evaluation_results' in sample_data:
            valid_samples += 1
            # for loop through metrics keys
            for model_type in metrics.keys():
                if model_type in sample_data['evaluation_results']:
                    model_metrics = sample_data['evaluation_results'][model_type]
                    for metric in metrics[model_type]:
                        if metric in model_metrics:
                            metrics[model_type][metric] += model_metrics[metric]
    
    # Calculate averages
    if valid_samples > 0:
        for model_type in metrics:
            for metric in metrics[model_type]:
                metrics[model_type][metric] /= valid_samples
    
    return {
        'average_metrics': metrics,
        'samples_count': valid_samples
    }


def calculate_average_metrics_with_std(data_list):
    """
    Calculate average metrics and standard deviation across multiple runs.
    
    Args:
        data_list (list): List of data dictionaries from different runs
    
    Returns:
        dict: Dictionary containing average metrics with std for each model type
    """
    # Initialize structure to collect metrics from all runs
    all_metrics = {}
    
    # Process each run's data
    for run_data in data_list:
        run_metrics = calculate_average_metrics(run_data)
        
        # Initialize structure on first run
        if not all_metrics:
            for model_type in run_metrics['average_metrics']:
                all_metrics[model_type] = {}
                for metric in run_metrics['average_metrics'][model_type]:
                    all_metrics[model_type][metric] = []
        
        # Collect metrics from this run
        for model_type in run_metrics['average_metrics']:
            for metric in run_metrics['average_metrics'][model_type]:
                all_metrics[model_type][metric].append(run_metrics['average_metrics'][model_type][metric])
    
    # Calculate mean and std for each metric
    final_metrics = {}
    for model_type in all_metrics:
        final_metrics[model_type] = {}
        for metric in all_metrics[model_type]:
            values = all_metrics[model_type][metric]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                final_metrics[model_type][metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'values': values,
                    'formatted': f"{mean_val:.4f} ± {std_val:.4f}"
                }
            else:
                final_metrics[model_type][metric] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'values': [],
                    'formatted': "0.0000 ± 0.0000"
                }
    
    return {
        'average_metrics_with_std': final_metrics,
        'num_runs': len(data_list)
    }


def get_latest_files_by_pattern():
    """
    Get the latest files for each model type based on available files.
    
    Returns:
        dict: Dictionary mapping model types to their latest file paths
    """
    import glob
    import os
    
    # Define patterns for each model type
    patterns = {
        'bmgc_omics': './TargetQA_Results/bmgc_omics_results_*.json',
        'bmgc_omicskg': './TargetQA_Results/bmgc_omicskg_results_*.json',
        'plain_omics': './TargetQA_Results/plain_omics_results_*.json',
        'plain_omicskg': './TargetQA_Results/plain_omicskg_results_*.json',
        'qallm_omics': './TargetQA_Results/qallm_omics_results_*.json',
        'qallm_omicskg': './TargetQA_Results/qallm_omicskg_results_*.json',
        'gretriever': './TargetQA_Results/gretriever_results_*.json',
        'galax_2nd_add_1st': './TargetQA_Results/galax_2nd_step_results_add1st_*.json'
    }
    
    latest_files = {}
    
    for model_type, pattern in patterns.items():
        files = glob.glob(pattern)
        if files:
            # Sort by modification time and get the most recent
            latest_files[model_type] = sorted(files, key=os.path.getmtime)
            print(f"{model_type}: Found {len(files)} files")
            for f in latest_files[model_type]:
                print(f"  {f}")
        else:
            print(f"No files found for pattern: {pattern}")
    
    return latest_files


def concatenate_json_files(json_paths, test_samples):
    """
    Concatenate multiple JSON files into a single dictionary.
    
    Args:
        json_paths (list): List of paths to JSON files
        test_samples (list): List of sample IDs to include
        
    Returns:
        dict: Combined dictionary with sample_id as key and evaluation_results from all files
    """
    import os, traceback  # Import at the top of function for directory creation
    combined_data = {}
    
    # Initialize the combined data structure with sample IDs
    for sample_id in test_samples:
        combined_data[sample_id] = {"evaluation_results": {}}
    
    # Process each JSON file
    for json_path in json_paths:
        try:
            # Check if the file exists
            if not os.path.exists(json_path):
                print(f"⚠️ Warning: File does not exist: {json_path}")
                continue
                
            # Print file size for debugging
            file_size = os.path.getsize(json_path)
            print(f"Processing {json_path} (size: {file_size} bytes)")
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as je:
                print(f"❌ JSON decode error in {json_path}: {je}")
                with open(json_path, 'r') as f:
                    file_content = f.read()
                print(f"First 200 characters of file: {file_content[:200]}")
                continue
                
            if data is None:
                print(f"⚠️ Warning: JSON data is None for {json_path}")
                continue
                
            if not isinstance(data, dict):
                print(f"⚠️ Warning: JSON data is not a dictionary for {json_path}. Type: {type(data)}")
                continue
                
            # Print some debug info
            print(f"Data type: {type(data)}")
            print(f"Number of keys: {len(data.keys())}")
            
            # Determine model type from file path more precisely using exact pattern matching
            model_type = None
            if 'motasg_gat_results' in json_path:
                model_type = 'gat'
            elif 'plain_omicskg_results' in json_path:
                model_type = 'plain_omicskg'
            elif 'plain_omics_results' in json_path:
                model_type = 'plain_omics'
            elif 'bmgc_omicskg_results' in json_path:
                model_type = 'bmgc_omicskg'
            elif 'bmgc_omics_results' in json_path:
                model_type = 'bmgc_omics'
            elif 'qallm_omicskg_results' in json_path:
                model_type = 'qallm_omicskg'
            elif 'qallm_omics_results' in json_path:
                model_type = 'qallm_omics'
            elif 'gretriever_results' in json_path:
                model_type = 'gretriever'
            elif 'galax_results' in json_path:
                model_type = 'galax'  # This file contains both llm_1st and llm_2nd
            elif 'galax_2nd_step_results' in json_path:
                model_type = 'galax_2nd'  # This file contains llm_2nd_1st and llm_2nd_2nd
            
            print(f"File {json_path} identified as model type: {model_type}")
            
            if len(data) > 0:
                first_key = list(data.keys())[0]
                print(f"First sample key: {first_key}")
                print(f"First sample data structure: {data[first_key].keys() if isinstance(data[first_key], dict) else 'not a dict'}")
            
            # Extract data for each sample and add to combined data
            sample_count = 0
            for sample_id, sample_data in data.items():
                if not isinstance(sample_data, dict):
                    print(f"⚠️ Warning: Sample data is not a dictionary for {sample_id}. Type: {type(sample_data)}")
                    continue
                    
                sample_count += 1
                if sample_id in test_samples:
                    if 'galax' in model_type:
                        # Handle special case for files containing multiple model results
                        if model_type == 'galax':
                            if 'llm_1st' in sample_data['evaluation_results']:
                                combined_data[sample_id]['evaluation_results']['llm_1st'] = sample_data['evaluation_results']['llm_1st']
                            else:
                                print(f"⚠️ Warning: No llm_1st found for {sample_id} in {json_path}")
                            if 'llm_2nd' in sample_data['evaluation_results']:
                                combined_data[sample_id]['evaluation_results']['llm_2nd'] = sample_data['evaluation_results']['llm_2nd']
                            else:
                                print(f"⚠️ Warning: No llm_2nd found for {sample_id} in {json_path}")
                        elif model_type == 'galax_2nd':
                            if 'llm_2nd_2nd' in sample_data['evaluation_results']:
                                combined_data[sample_id]['evaluation_results']['llm_2nd_2nd'] = sample_data['evaluation_results']['llm_2nd_2nd']
                            else:
                                print(f"⚠️ Warning: No llm_2nd_2nd found for {sample_id} in {json_path}")
                    elif 'gat' in model_type:
                        if 'gat' in sample_data['evaluation_results']:
                            combined_data[sample_id]['evaluation_results']['gat'] = sample_data['evaluation_results']['gat']
                        else:
                            print(f"⚠️ Warning: No gat found for {sample_id} in {json_path}")
                    else:
                        # For other files with single model results, standardize to use llm_1st
                        if 'llm_1st' in sample_data['evaluation_results']:
                            combined_data[sample_id]['evaluation_results'][model_type] = sample_data['evaluation_results']['llm_1st']
                        else:
                            print(f"⚠️ Warning: No evaluation_results found for {sample_id} of model type {model_type}")
            
            print(f"Processed {sample_count} samples from {json_path}")
            
        except Exception as e:
            print(f"❌ Error processing {json_path}: {e}")
            traceback.print_exc()
    
    # Save the combined data to a new JSON file
    combined_data_path = './TargetQA_Results/combined_data.json'
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(combined_data_path), exist_ok=True)
    
    with open(combined_data_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"Combined data saved to {combined_data_path}")
    
    return combined_data


def combined_eval_data(combined_data):
    """
    Evaluate the combined data and calculate average metrics.
    
    Args:
        combined_data (dict): Combined data dictionary
    
    Returns:
        dict: Dictionary containing average metrics for each model type
    """
    # Calculate average metrics
    average_metrics = calculate_average_metrics(combined_data)
    
    # Save the average metrics to a new JSON file
    with open('./TargetQA_Results/combined_average_metrics.json', 'w') as f:
        json.dump(average_metrics, f, indent=2)
    
    return average_metrics


def sep_eval_test_data(json_data):
    """
    Organize test data by disease and calculate disease-specific metrics.
    
    Args:
        json_data (dict): Combined JSON data from all models
        
    Returns:
        dict: Disease-organized data with metrics
    """
    # Load disease mapping
    test_samples_detailed_df = pd.read_csv('./data/TargetQA/test_samples_detailed.csv')
    dti_combined_samples_df = pd.read_csv('./data/TargetScreen/dti_combined_samples.csv')
    disease_mapping_df = pd.merge(test_samples_detailed_df, dti_combined_samples_df[['depMapID', 'BMGC_Disease_ID', 'BMGC_Disease_name']], on='depMapID', how='left')
    
    # Create disease dictionary - GROUP BY tcga_code instead of BMGC_Disease_ID
    disease_dict = {}
    
    for _, row in disease_mapping_df.iterrows():
        sample_id = row['depMapID']
        disease_tcga_code = row['tcga_code']
        
        # Use tcga_code as the primary key for grouping
        if disease_tcga_code not in disease_dict:
            disease_dict[disease_tcga_code] = []
        
        disease_dict[disease_tcga_code].append(sample_id)
    
    # Initialize the disease organized data structure
    disease_organized_data = {
        "disease_info": {},
        "disease_metrics": {}
    }
    
    # Define metrics template for initialization
    metrics_template = {
        'gat': {'precision': 0, 'recall': 0, 'f1_score': 0,
                'overlap_count': 0, 'jaccard': 0, 
                'precision@5': 0, 'precision@10': 0},
        'plain_omics': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                       'overlap_count': 0, 'jaccard': 0, 
                       'precision@5': 0, 'precision@10': 0},
        'plain_omicskg': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                         'overlap_count': 0, 'jaccard': 0, 
                         'precision@5': 0, 'precision@10': 0},
        'bmgc_omics': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                      'overlap_count': 0, 'jaccard': 0, 
                      'precision@5': 0, 'precision@10': 0},
        'bmgc_omicskg': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                        'overlap_count': 0, 'jaccard': 0, 
                        'precision@5': 0, 'precision@10': 0},
        'qallm_omics': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                       'overlap_count': 0, 'jaccard': 0, 
                       'precision@5': 0, 'precision@10': 0},
        'qallm_omicskg': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                         'overlap_count': 0, 'jaccard': 0, 
                         'precision@5': 0, 'precision@10': 0},
        'gretriever': {'precision': 0, 'recall': 0, 'f1_score': 0,
                      'overlap_count': 0, 'jaccard': 0, 
                      'precision@5': 0, 'precision@10': 0},
        'llm_1st': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'llm_2nd': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0},
        'llm_2nd_2nd': {'precision': 0, 'recall': 0, 'f1_score': 0, 
                       'overlap_count': 0, 'jaccard': 0, 
                       'precision@5': 0, 'precision@10': 0}
    }
    
    # Add disease info - now keyed by tcga_code
    for disease_tcga_code, samples in disease_dict.items():
        print(f"Processing disease {disease_tcga_code} with {len(samples)} samples")
        
        # Create disease information entry
        disease_organized_data["disease_info"][disease_tcga_code] = {
            "disease_tcga_code": disease_tcga_code,
            "sample_count": len(samples),
            "samples": samples
        }
        
        # Create disease metrics entry initialized from template
        disease_organized_data["disease_metrics"][disease_tcga_code] = {
            "average_metrics": {model: metrics.copy() for model, metrics in metrics_template.items()},
            "samples_count": 0,
            "disease_tcga_code": disease_tcga_code
        }
        
        # Initialize counters for valid samples per disease
        valid_samples = 0
        
        # Process each sample for this disease
        for sample_id in samples:
            if sample_id in json_data:
                # Check if the sample has evaluation results
                if 'evaluation_results' in json_data[sample_id]:
                    valid_samples += 1
                    sample_results = json_data[sample_id]['evaluation_results']
                    
                    # Add sample data to disease-organized data
                    if disease_tcga_code not in disease_organized_data:
                        disease_organized_data[disease_tcga_code] = {}
                    
                    disease_organized_data[disease_tcga_code][sample_id] = json_data[sample_id]
                    
                    # Accumulate metrics for each model type
                    for model_type in metrics_template.keys():
                        if model_type in sample_results:
                            model_metrics = sample_results[model_type]
                            for metric in metrics_template[model_type]:
                                if metric in model_metrics:
                                    disease_organized_data["disease_metrics"][disease_tcga_code]["average_metrics"][model_type][metric] += model_metrics[metric]
        
        # Calculate average for this disease's metrics
        if valid_samples > 0:
            for model_type in metrics_template.keys():
                for metric in metrics_template[model_type]:
                    disease_organized_data["disease_metrics"][disease_tcga_code]["average_metrics"][model_type][metric] /= valid_samples
            
            disease_organized_data["disease_metrics"][disease_tcga_code]["samples_count"] = valid_samples
        else:
            print(f"Warning: No valid samples found for disease {disease_tcga_code}")
    
    # Create directory if it doesn't exist
    output_dir = './TargetQA_Results/disease_organized'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the organized data to JSON files
    # 1. Save complete disease organized data
    with open(f'{output_dir}/disease_organized_data.json', 'w') as f:
        json.dump(disease_organized_data, f, indent=2)
    
    # 2. Save just the disease metrics for easier analysis
    with open(f'{output_dir}/disease_metrics.json', 'w') as f:
        json.dump(disease_organized_data["disease_metrics"], f, indent=2)
    
    print(f"Disease-organized data and metrics saved to {output_dir}/")
    
    return disease_organized_data


def generate_disease_comparison_table(disease_data, avg_std_metrics=None, all_runs_disease_data=None):
    """
    Generate a comparison table of model performance across diseases,
    arranged by number of samples per disease (descending).
    Also includes overall average metrics as the first column.
    If avg_std_metrics and all_runs_disease_data are provided, uses mean±std format for all columns.
    
    Args:
        disease_data (dict): Disease-organized data from sep_eval_test_data (single run)
        avg_std_metrics (dict, optional): Average metrics with std from multiple runs
        all_runs_disease_data (list, optional): List of disease data from all runs for calculating disease-specific mean±std
        
    Returns:
        pd.DataFrame: Table with diseases as columns and models as rows
    """
    import pandas as pd
    import numpy as np
    import os
    
    # Extract disease metrics from single run for structure
    disease_metrics = disease_data.get("disease_metrics", {})
    
    if not disease_metrics:
        print("No disease metrics found in the provided data")
        return None
    
    # Create a list of diseases with sample counts for sorting
    # Now disease_id is actually tcga_code
    diseases = []
    for disease_tcga_code, data in disease_metrics.items():
        samples_count = data.get("samples_count", 0)
        diseases.append({
            "tcga_code": disease_tcga_code,
            "samples": samples_count
        })
    
    # Sort diseases by sample count (descending)
    diseases.sort(key=lambda x: x["samples"], reverse=True)
    disease_tcga_codes = [d["tcga_code"] for d in diseases]
    
    # Define model types in original order
    model_types = [
        'gat',
        'plain_omics',
        'plain_omicskg',
        'bmgc_omics', 
        'bmgc_omicskg',
        'qallm_omics',
        'qallm_omicskg',
        'gretriever',
        'llm_1st',
        'llm_2nd',
        'llm_2nd_2nd'
    ]
    
    # Define metrics to include
    metrics = ['precision', 'recall', 'f1_score', 'jaccard', 'precision@5', 'precision@10']
    
    # Create a multi-index for rows (model_type, metric)
    rows = []
    for model in model_types:
        for metric in metrics:
            rows.append((model, metric))
    
    # Calculate disease-specific mean±std if multiple runs provided
    disease_stats = {}
    if all_runs_disease_data:
        print(f"Calculating disease-specific mean±std across {len(all_runs_disease_data)} runs...")
        
        for disease_tcga_code in disease_tcga_codes:
            disease_stats[disease_tcga_code] = {}
            print(f"Processing disease {disease_tcga_code}...")
            
            for model in model_types:
                disease_stats[disease_tcga_code][model] = {}
                for metric in metrics:
                    # Collect values across all runs for this disease-model-metric combination
                    values = []
                    for run_idx, run_data in enumerate(all_runs_disease_data):
                        run_disease_metrics = run_data.get("disease_metrics", {})
                        if (disease_tcga_code in run_disease_metrics and 
                            model in run_disease_metrics[disease_tcga_code]["average_metrics"] and
                            metric in run_disease_metrics[disease_tcga_code]["average_metrics"][model]):
                            values.append(run_disease_metrics[disease_tcga_code]["average_metrics"][model][metric])
                        else:
                            # Add 0 for missing values to maintain consistency
                            values.append(0.0)
                    
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                        
                        # Format based on metric type
                        if 'precision' in metric or metric in ['recall', 'f1_score', 'jaccard']:
                            formatted_value = f"{mean_val*100:.1f} ± {std_val*100:.1f}"
                        else:
                            formatted_value = f"{mean_val:.3f} ± {std_val:.3f}"
                            
                        disease_stats[disease_tcga_code][model][metric] = {
                            'mean': mean_val,
                            'std': std_val,
                            'formatted': formatted_value,
                            'raw': f"{mean_val:.6f} ± {std_val:.6f}"
                        }
                    else:
                        disease_stats[disease_tcga_code][model][metric] = {
                            'mean': 0.0,
                            'std': 0.0,
                            'formatted': "0.0 ± 0.0",
                            'raw': "0.000000 ± 0.000000"
                        }
        
        print(f"Disease stats calculated for {len(disease_stats)} diseases")
    
    # Create empty DataFrame to store raw values for both overall averages and disease-specific metrics
    df_data_raw = []
    # Create empty DataFrame for formatted display values
    df_data = []
    
    # Calculate overall average metrics across all diseases (for single run fallback)
    overall_metrics = {model: {metric: 0.0 for metric in metrics} for model in model_types}
    total_samples = 0
    
    for disease_tcga_code, data in disease_metrics.items():
        samples_count = data.get("samples_count", 0)
        total_samples += samples_count
        
        for model in model_types:
            if model in data["average_metrics"]:
                for metric in metrics:
                    if metric in data["average_metrics"][model]:
                        # Weighted average based on sample count
                        overall_metrics[model][metric] += data["average_metrics"][model][metric] * samples_count
    
    # Normalize by total number of samples
    if total_samples > 0:
        for model in model_types:
            for metric in metrics:
                overall_metrics[model][metric] /= total_samples
    
    # Fill DataFrame with data (including overall average as first column)
    for model, metric in rows:
        row_data = []  # Formatted data for display
        row_data_raw = []  # Raw data with full precision
        
        # Add overall average for this model and metric as first column
        if avg_std_metrics and model in avg_std_metrics['average_metrics_with_std']:
            # Use mean±std format from multiple runs
            if metric in avg_std_metrics['average_metrics_with_std'][model]:
                formatted_overall = avg_std_metrics['average_metrics_with_std'][model][metric]['formatted']
                mean_value = avg_std_metrics['average_metrics_with_std'][model][metric]['mean']
                std_value = avg_std_metrics['average_metrics_with_std'][model][metric]['std']
                
                row_data_raw.append(f"{mean_value:.6f} ± {std_value:.6f}")
                row_data.append(formatted_overall)
            else:
                row_data_raw.append("N/A")
                row_data.append("N/A")
        else:
            # Use single run overall average
            overall_value = overall_metrics[model][metric]
            row_data_raw.append(overall_value)
            
            # Format value for display
            if 'precision' in metric or metric in ['recall', 'f1_score', 'jaccard']:
                formatted_value = round(overall_value * 100, 1)  # Convert to percentage and round
            elif metric == 'overlap_count':
                formatted_value = round(overall_value, 1)  # Round to 1 decimal place
            else:
                formatted_value = overall_value
                
            row_data.append(formatted_value)
        
        # Add disease-specific metrics
        for disease_tcga_code in disease_tcga_codes:
            # Force use of mean±std when multiple runs are available
            if all_runs_disease_data:
                # Ensure we have stats for this disease
                if disease_tcga_code in disease_stats and model in disease_stats[disease_tcga_code] and metric in disease_stats[disease_tcga_code][model]:
                    stats = disease_stats[disease_tcga_code][model][metric]
                    row_data.append(stats['formatted'])
                    row_data_raw.append(stats['raw'])
                else:
                    # Fallback to 0.0 ± 0.0 if no data
                    if 'precision' in metric or metric in ['recall', 'f1_score', 'jaccard']:
                        fallback_formatted = "0.0 ± 0.0"
                    else:
                        fallback_formatted = "0.000 ± 0.000"
                    
                    row_data.append(fallback_formatted)
                    row_data_raw.append("0.000000 ± 0.000000")
            else:
                # Single run fallback
                try:
                    value = disease_metrics[disease_tcga_code]["average_metrics"][model][metric]
                    row_data_raw.append(value)
                    
                    if 'precision' in metric or metric in ['recall', 'f1_score', 'jaccard']:
                        value = round(value * 100, 1)
                    elif metric == 'overlap_count':
                        value = round(value, 1)
                        
                    row_data.append(value)
                except KeyError:
                    row_data.append(np.nan)
                    row_data_raw.append(np.nan)
                
        df_data.append(row_data)
        df_data_raw.append(row_data_raw)
    
    # Create column headers with overall average as first column and TCGA codes
    if avg_std_metrics and all_runs_disease_data:
        column_headers = ["Overall Average (Mean±Std)"] + [f"{d['tcga_code']} ({d['samples']})" for d in diseases]
        suffix = "_with_std_all_columns"
    else:
        column_headers = ["Overall Average"] + [f"{d['tcga_code']} ({d['samples']})" for d in diseases]
        suffix = ""
    
    # Create DataFrame for display (with rounded values)
    df = pd.DataFrame(df_data, index=pd.MultiIndex.from_tuples(rows), columns=column_headers)
    
    # Create DataFrame with raw values for saving to files (with full precision)
    df_raw = pd.DataFrame(df_data_raw, index=pd.MultiIndex.from_tuples(rows), columns=column_headers)
    
    # Save the table to CSV with full precision (8+ digits)
    os.makedirs('./TargetQA_Results/tables', exist_ok=True)
    
    # Choose filename based on whether we have std data
    if avg_std_metrics and all_runs_disease_data:
        csv_filename = f'./TargetQA_Results/tables/disease_comparison_table_with_std_all_columns.csv'
        excel_filename = f'./TargetQA_Results/tables/disease_comparison_table_with_std_all_columns.xlsx'
    else:
        csv_filename = './TargetQA_Results/tables/disease_comparison_table.csv'
        excel_filename = './TargetQA_Results/tables/disease_comparison_table.xlsx'
    
    df_raw.to_csv(csv_filename)
    
    # Also save an Excel version with full precision
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_raw.to_excel(writer, sheet_name='Full_Precision')
            df.to_excel(writer, sheet_name='Formatted')  # Also include the formatted version
    except Exception as e:
        print(f"Could not save Excel file: {e}")
    
    print(f"Comparison table saved to '{csv_filename}'")
    if avg_std_metrics and all_runs_disease_data:
        print(f"All columns show mean±std format across multiple runs")
    else:
        print(f"Single run format")
    
    # Return the display version with formatted values
    return df


if __name__ == '__main__':
    # Multiple versions of files for averaging
    motasg_gat_files = [
        './TargetQA_Results/motasg_gat_results_20250908_010729.json',
        './TargetQA_Results/motasg_gat_results_20250908_082734.json',
        './TargetQA_Results/motasg_gat_results_20250908_082909.json'
    ]
    
    bmgc_omics_files = [
        './TargetQA_Results/bmgc_omics_results_20250820_171210.json',
        './TargetQA_Results/bmgc_omics_results_20250907_024824.json',
        './TargetQA_Results/bmgc_omics_results_20250907_042129.json'
    ]
    
    bmgc_omicskg_files = [
        './TargetQA_Results/bmgc_omicskg_results_20250820_172236.json',
        './TargetQA_Results/bmgc_omicskg_results_20250907_025702.json',
        './TargetQA_Results/bmgc_omicskg_results_20250907_042114.json'
    ]
    
    plain_omics_files = [
        './TargetQA_Results/plain_omics_results_20250820_145320.json',
        './TargetQA_Results/plain_omics_results_20250907_020929.json',
        './TargetQA_Results/plain_omics_results_20250907_045611.json'
    ]
    
    plain_omicskg_files = [
        './TargetQA_Results/plain_omicskg_results_20250820_155113.json',
        './TargetQA_Results/plain_omicskg_results_20250907_023204.json',
        './TargetQA_Results/plain_omicskg_results_20250907_045114.json'
    ]
    
    qallm_omics_files = [
        './TargetQA_Results/qallm_omics_results_20250820_181011.json',
        './TargetQA_Results/qallm_omics_results_20250907_020220.json',
        './TargetQA_Results/qallm_omics_results_20250907_045550.json'
    ]
    
    qallm_omicskg_files = [
        './TargetQA_Results/qallm_omicskg_results_20250820_180421.json',
        './TargetQA_Results/qallm_omicskg_results_20250907_020838.json',
        './TargetQA_Results/qallm_omicskg_results_20250907_061115.json'
    ]
    
    gretriever_files = [
        './TargetQA_Results/gretriever_results_20250907_000054.json',
        './TargetQA_Results/gretriever_results_20250907_022450.json',
        './TargetQA_Results/gretriever_results_20250907_035755.json'
    ]
    
    galax_files = [
        './TargetQA_Results/galax_results_20250901_030348.json',
        './TargetQA_Results/galax_results_20250901_030752.json',
        './TargetQA_Results/galax_results_20250901_140224.json'
    ]
    
    galax_2nd_add_1st_files = [
        './TargetQA_Results/galax_2nd_step_results_add1st_20250822_221447.json',
        './TargetQA_Results/galax_2nd_step_results_add1st_20250906_024403.json',
        './TargetQA_Results/galax_2nd_step_results_add1st_20250906_030856.json',
        # './TargetQA_Results/galax_2nd_step_results_add1st_20250907_012002.json'
    ]

    # Load the test data
    test_dti_crispr_rna_samples_index_df = pd.read_csv('./data/TargetQA/test_samples.csv')
    test_samples = test_dti_crispr_rna_samples_index_df['depMapID'].tolist()
    
    # Process each model type separately and calculate average ± std
    print("Processing multiple runs for each model...")
    
    # Dictionary to store combined data from all runs
    all_runs_data = []
    
    # Model file groups - all files are now lists for averaging
    model_file_groups = {
        'motasg_gat': motasg_gat_files,
        'bmgc_omics': bmgc_omics_files,
        'bmgc_omicskg': bmgc_omicskg_files,
        'plain_omics': plain_omics_files,
        'plain_omicskg': plain_omicskg_files,
        'qallm_omics': qallm_omics_files,
        'qallm_omicskg': qallm_omicskg_files,
        'gretriever': gretriever_files,
        'galax': galax_files,
        'galax_2nd_add_1st': galax_2nd_add_1st_files
    }
    
    # For each run (assuming all models have same number of runs)
    max_runs = max(len(files) for files in model_file_groups.values())
    
    for run_idx in range(max_runs):
        print(f"\nProcessing run {run_idx + 1}/{max_runs}")
        
        # Build json_paths for this run - no fixed files anymore
        json_paths = []
        
        # Add files from this run index (if available)
        for model_type, files in model_file_groups.items():
            if run_idx < len(files):
                json_paths.append(files[run_idx])
                print(f"  {model_type}: {files[run_idx]}")
            else:
                print(f"  {model_type}: No file for run {run_idx + 1}")
        
        # Process this run
        try:
            combined_data = concatenate_json_files(json_paths, test_samples)
            all_runs_data.append(combined_data)
        except Exception as e:
            print(f"Error processing run {run_idx + 1}: {e}")
            continue
    
    # Calculate average metrics with standard deviation
    print(f"\nCalculating average ± std across {len(all_runs_data)} runs...")
    avg_std_metrics = calculate_average_metrics_with_std(all_runs_data)
    
    # Save the results
    output_path = './TargetQA_Results/combined_average_metrics_with_std.json'
    with open(output_path, 'w') as f:
        json.dump(avg_std_metrics, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Process disease data for each run
    print("\nProcessing disease data for each run...")
    all_runs_disease_data = []
    
    for run_idx, combined_data in enumerate(all_runs_data):
        print(f"Processing disease data for run {run_idx + 1}...")
        disease_data = sep_eval_test_data(combined_data)
        all_runs_disease_data.append(disease_data)
    
    # Generate disease comparison table with mean±std for ALL columns
    print("\nGenerating disease comparison table with mean±std for ALL columns...")
    # Use the last run's disease data for structure, but pass all runs for statistics
    comparison_table_with_std = generate_disease_comparison_table(
        all_runs_disease_data[-1], 
        avg_std_metrics, 
        all_runs_disease_data
    )
    
    if comparison_table_with_std is not None:
        print("\nDisease Comparison Table with Mean±Std for ALL columns Preview:")
        print(comparison_table_with_std.iloc[:10, :5])
    else:
        print("Failed to generate comparison table with std")
    
    # Also generate the single-run version for comparison
    print("\nGenerating single-run disease comparison table...")
    recent_json_paths = [
        motasg_gat_files[-1] if motasg_gat_files else None,
        plain_omics_files[-1] if plain_omics_files else None,
        plain_omicskg_files[-1] if plain_omicskg_files else None,
        bmgc_omics_files[-1] if bmgc_omics_files else None,
        bmgc_omicskg_files[-1] if bmgc_omicskg_files else None,
        qallm_omics_files[-1] if qallm_omics_files else None,
        qallm_omicskg_files[-1] if qallm_omicskg_files else None,
        gretriever_files[-1] if gretriever_files else None,
        galax_files[-1] if galax_files else None,
        galax_2nd_add_1st_files[-1] if galax_2nd_add_1st_files else None
    ]
    
    recent_json_paths = [path for path in recent_json_paths if path is not None]
    combined_data = concatenate_json_files(recent_json_paths, test_samples)
    combined_eval_data(combined_data)
    
    # Single run disease analysis
    disease_data = sep_eval_test_data(combined_data)
    comparison_table_single = generate_disease_comparison_table(disease_data, avg_std_metrics)
    
    if comparison_table_single is not None:
        print("\nSingle-run Disease Comparison Table Preview:")
        print(comparison_table_single.iloc[:10, :5])
