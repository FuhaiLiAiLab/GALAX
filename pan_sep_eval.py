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
        'm2t_bm': {'precision': 0, 'recall': 0, 'f1_score': 0,
                   'overlap_count': 0, 'jaccard': 0, 
                   'precision@5': 0, 'precision@10': 0}, 
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
        # 'llm_2nd_1st': {'precision': 0, 'recall': 0, 'f1_score': 0, 
        #            'overlap_count': 0, 'jaccard': 0, 
        #            'precision@5': 0, 'precision@10': 0},
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
            if 'm2t_target_assignments' in json_path:
                model_type = 'm2t_bm'
            elif 'motasg_gat_results' in json_path:
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
                            # if 'llm_2nd_1st' in sample_data['evaluation_results']:
                            #     combined_data[sample_id]['evaluation_results']['llm_2nd_1st'] = sample_data['evaluation_results']['llm_2nd_1st']
                            # else:
                            #     print(f"⚠️ Warning: No llm_2nd_1st found for {sample_id} in {json_path}")
                            if 'llm_2nd_2nd' in sample_data['evaluation_results']:
                                combined_data[sample_id]['evaluation_results']['llm_2nd_2nd'] = sample_data['evaluation_results']['llm_2nd_2nd']
                            else:
                                print(f"⚠️ Warning: No llm_2nd_2nd found for {sample_id} in {json_path}")
                    elif 'm2t' in model_type:
                        if model_type == 'm2t_bm':
                            if 'm2t_bm' in sample_data['evaluation_results']:
                                combined_data[sample_id]['evaluation_results']['m2t_bm'] = sample_data['evaluation_results']['m2t_bm']
                            else:
                                print(f"⚠️ Warning: No m2t_bm found for {sample_id} in {json_path}")
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
    combined_data_path = './QA_Results/combined_data.json'
    
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
    with open('./QA_Results/combined_average_metrics.json', 'w') as f:
        json.dump(average_metrics, f, indent=2)
    
    return average_metrics


def sep_eval_test_data(json_data):
    """
    Organize metrics by disease and calculate disease-specific averages.
    
    Args:
        json_data (dict): Combined data dictionary with sample_id as key and evaluation_results
        
    Returns:
        dict: Dictionary with disease-specific metrics
    """
    import os
    
    # Load required dataframes
    dti_combined_samples_df = pd.read_csv('./BMG/process_data/dti_combined_samples.csv')
    test_dti_crispr_rna_samples_index_df = pd.read_csv('./BMG/CRISPR-Graph/test_dti_crispr_rna_samples_index.csv')
    
    # Merge the dti_combined_samples_df and test_dti_crispr_rna_samples_index_df on the Sample and depMapID
    test_dti_crispr_rna_samples_index_info_df = test_dti_crispr_rna_samples_index_df.merge(dti_combined_samples_df, left_on='Sample', right_on='depMapID')
    
    # Select columns [depMapID, Name, BMGC_Disease_ID, BMGC_Disease_name]
    test_dti_crispr_rna_samples_index_info_df = test_dti_crispr_rna_samples_index_info_df[['depMapID', 'Name', 'tcga_code', 'BMGC_Disease_ID', 'BMGC_Disease_name']]
    
    # Get the unique diseases using TCGA code
    unique_diseases = test_dti_crispr_rna_samples_index_info_df['tcga_code'].unique()
    print(f"Found {len(unique_diseases)} unique disease TCGA codes")

    # Create dictionaries for disease samples and names
    disease_dict = {}
    disease_names = {}  # Map TCGA codes to disease names

    # Organize samples by disease TCGA code
    for tcga_code in unique_diseases:
        if pd.isna(tcga_code):
            print(f"Warning: Found NA TCGA code, skipping")
            continue
            
        # Get the disease name for this TCGA code
        disease_info = test_dti_crispr_rna_samples_index_info_df[
            test_dti_crispr_rna_samples_index_info_df['tcga_code'] == tcga_code
        ]
        
        if len(disease_info) == 0:
            print(f"Warning: No samples found for TCGA code {tcga_code}")
            continue
            
        # Use the first available disease name for this TCGA code
        disease_name = disease_info['BMGC_Disease_name'].iloc[0]
        
        # Get the depMapID which belongs to this TCGA code
        disease_samples = disease_info['depMapID'].tolist()
        
        disease_dict[tcga_code] = disease_samples
        disease_names[tcga_code] = disease_name
        print(f"TCGA code {tcga_code}: {disease_name} with {len(disease_samples)} samples")

    # Structure to hold disease-organized data and metrics
    disease_organized_data = {
        "disease_info": {},
        "disease_metrics": {}
    }
    
    # Initialize metrics structure for disease metrics
    metrics_template = {
        'm2t_bm': {'precision': 0, 'recall': 0, 'f1_score': 0,
                     'overlap_count': 0, 'jaccard': 0, 
                     'precision@5': 0, 'precision@10': 0},
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
    
    # Add disease info
    for disease_id, samples in disease_dict.items():
        disease_name = disease_names[disease_id]
        print(f"Processing disease {disease_id}: {disease_name} with {len(samples)} samples")
        
        # Create disease information entry
        disease_organized_data["disease_info"][disease_id] = {
            "disease_name": disease_name,
            "sample_count": len(samples),
            "samples": samples
        }
        
        # Create disease metrics entry initialized from template
        disease_organized_data["disease_metrics"][disease_id] = {
            "average_metrics": {model: metrics.copy() for model, metrics in metrics_template.items()},
            "samples_count": 0,
            "disease_name": disease_name
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
                    if disease_id not in disease_organized_data:
                        disease_organized_data[disease_id] = {}
                    
                    disease_organized_data[disease_id][sample_id] = json_data[sample_id]
                    
                    # Accumulate metrics for each model type
                    for model_type in metrics_template.keys():
                        if model_type in sample_results:
                            model_metrics = sample_results[model_type]
                            for metric in metrics_template[model_type]:
                                if metric in model_metrics:
                                    disease_organized_data["disease_metrics"][disease_id]["average_metrics"][model_type][metric] += model_metrics[metric]
        
        # Calculate average for this disease's metrics
        if valid_samples > 0:
            for model_type in metrics_template.keys():
                for metric in metrics_template[model_type]:
                    disease_organized_data["disease_metrics"][disease_id]["average_metrics"][model_type][metric] /= valid_samples
            
            disease_organized_data["disease_metrics"][disease_id]["samples_count"] = valid_samples
        else:
            print(f"Warning: No valid samples found for disease {disease_id}: {disease_name}")
    
    # Create directory if it doesn't exist
    output_dir = './QA_Results/disease_organized'
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


def generate_disease_comparison_table(disease_data):
    """
    Generate a comparison table of model performance across diseases,
    arranged by number of samples per disease (descending).
    Also includes overall average metrics as the first column.
    
    Args:
        disease_data (dict): Disease-organized data from sep_eval_test_data
        
    Returns:
        pd.DataFrame: Table with diseases as columns and models as rows
    """
    import pandas as pd
    import numpy as np
    import os
    
    # Extract disease metrics
    disease_metrics = disease_data.get("disease_metrics", {})
    
    if not disease_metrics:
        print("No disease metrics found in the provided data")
        return None
    
    # Create a list of diseases with sample counts for sorting
    diseases = []
    for disease_id, data in disease_metrics.items():
        samples_count = data.get("samples_count", 0)
        disease_name = data.get("disease_name", "Unknown")
        diseases.append({
            "id": disease_id,
            "name": disease_name,
            "samples": samples_count
        })
    
    # Sort diseases by sample count (descending)
    diseases.sort(key=lambda x: x["samples"], reverse=True)
    disease_ids = [d["id"] for d in diseases]
    
    # Define model types in original order
    model_types = [
        'm2t_bm',
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
    
    # Create empty DataFrame to store raw values for both overall averages and disease-specific metrics
    df_data_raw = []
    # Create empty DataFrame for formatted display values
    df_data = []
    
    # Calculate overall average metrics across all diseases
    overall_metrics = {model: {metric: 0.0 for metric in metrics} for model in model_types}
    total_samples = 0
    
    for disease_id, data in disease_metrics.items():
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
        for disease_id in disease_ids:
            try:
                value = disease_metrics[disease_id]["average_metrics"][model][metric]
                # Store raw value without rounding
                row_data_raw.append(value)
                
                # Format value for display
                if 'precision' in metric or metric in ['recall', 'f1_score', 'jaccard']:
                    value = round(value * 100, 1)  # Convert to percentage and round
                elif metric == 'overlap_count':
                    value = round(value, 1)  # Round to 1 decimal place
                    
                row_data.append(value)
            except KeyError:
                row_data.append(np.nan)
                row_data_raw.append(np.nan)
                
        df_data.append(row_data)
        df_data_raw.append(row_data_raw)
    
    # Create column headers with overall average as first column and TCGA codes instead of disease names
    column_headers = ["Overall Average"] + [f"{d['id']} ({d['samples']})" for d in diseases]
    
    # Create DataFrame for display (with rounded values)
    df = pd.DataFrame(df_data, index=pd.MultiIndex.from_tuples(rows), columns=column_headers)
    
    # Create DataFrame with raw values for saving to files (with full precision)
    df_raw = pd.DataFrame(df_data_raw, index=pd.MultiIndex.from_tuples(rows), columns=column_headers)
    
    # Save the table to CSV with full precision (8+ digits)
    os.makedirs('./QA_Results/tables', exist_ok=True)
    df_raw.to_csv('./QA_Results/tables/disease_comparison_table.csv', float_format='%.8f')
    
    # Also save an Excel version with full precision
    try:
        with pd.ExcelWriter('./QA_Results/tables/disease_comparison_table.xlsx', engine='openpyxl') as writer:
            df_raw.to_excel(writer, sheet_name='Full_Precision')
            df.to_excel(writer, sheet_name='Formatted')  # Also include the formatted version
    except Exception as e:
        print(f"Could not save Excel file: {e}")
    
    print(f"Comparison table saved to './QA_Results/tables/disease_comparison_table.csv' with 8 decimal places")
    print(f"Overall average metrics included as first column")
    
    # Return the display version with formatted values
    return df


if __name__ == '__main__':
    # Load the JSON data from file QA-Results
    m2t_json_path = './QA_Results/m2t_target_assignments.json' # m2t_bm
    motasg_gat_json_path = './QA_Results/motasg_gat_results.json' # gat
    plain_omics_json_path = './QA_Results/plain_omics_results_{}.json' # llm_1st
    plain_omicskg_json_path = './QA_Results/plain_omicskg_results_{}.json' # llm_1st
    bmgc_omics_json_path = './QA_Results/bmgc_omics_results_{}.json' # llm_1st
    bmgc_omicskg_json_path = './QA_Results/bmgc_omicskg_results_{}.json' # llm_1st
    qallm_omics_json_path = './QA_Results/qallm_omics_results_{}.json' # llm_1st
    qallm_omicskg_json_path = './QA_Results/qallm_omicskg_results_{}.json' # llm_1st
    gretriever_json_path = './QA_Results/gretriever_results_{}.json' # llm_1st
    galax_json_path = './QA_Results/galax_results_{}.json' # llm_1st and llm_2nd are in galax_json_path
    galax_2nd_add_1st_json_path = './QA_Results/galax_2nd_step_results_add1st_{}.json' # llm_2nd_1st and llm_2nd_2nd are in galax_2nd_add_1st_json_path

    json_paths = [
        m2t_json_path,
        motasg_gat_json_path,
        plain_omics_json_path,
        plain_omicskg_json_path,
        bmgc_omics_json_path,
        bmgc_omicskg_json_path,
        qallm_omics_json_path,
        qallm_omicskg_json_path,
        gretriever_json_path,
        galax_json_path,
        galax_2nd_add_1st_json_path
    ]

    # Load the test data
    test_dti_crispr_rna_samples_index_df = pd.read_csv('./BMG/CRISPR-Graph/test_dti_crispr_rna_samples_index.csv')
    test_samples = test_dti_crispr_rna_samples_index_df['Sample'].tolist()

    combined_data = concatenate_json_files(json_paths, test_samples)
    combined_eval_data(combined_data)

    # Organize data by disease and calculate disease-specific metrics
    disease_data = sep_eval_test_data(combined_data)
    
    # Generate and display the comparison table
    comparison_table = generate_disease_comparison_table(disease_data)
    print("\nDisease Comparison Table Preview (first few rows and columns):")
    print(comparison_table.iloc[:10, :7])
