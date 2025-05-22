import pandas as pd
import os
import json

def precision_at_k(predicted_list, ground_truth_set, k):
    """
    Calculate precision at k
    
    Args:
        predicted_list (list): Ordered list of predictions
        ground_truth_set (set): Set of ground truth items
        k (int): Number of top predictions to consider
    
    Returns:
        float: Precision at k
    """
    top_k_preds = predicted_list[:k]
    correct = sum(1 for p in top_k_preds if p in ground_truth_set)
    return correct / k if k > 0 else 0

def calculate_metrics(predicted_list, ground_truth_list):
    """
    Calculate evaluation metrics comparing predicted proteins with ground truth.
    
    Args:
        predicted_list (list): List of predicted proteins (HGNC symbols)
        ground_truth_list (list): List of ground truth proteins (HGNC symbols)
        
    Returns:
        tuple: A tuple containing (overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10)
    """
    # Convert lists to sets for intersection/union operations
    predicted_set = set(predicted_list)
    ground_truth_set = set(ground_truth_list)
    
    # Calculate basic metrics
    overlap_set = ground_truth_set.intersection(predicted_set)
    overlap_count = len(overlap_set)
    precision = overlap_count / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = overlap_count / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
    
    # F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Jaccard Similarity
    union_count = len(ground_truth_set.union(predicted_set))
    jaccard = overlap_count / union_count if union_count > 0 else 0
    
    # Precision at K
    precision_at_5 = precision_at_k(predicted_list, ground_truth_set, k=5)
    precision_at_10 = precision_at_k(predicted_list, ground_truth_set, k=10)
    
    return overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10

def motasg_sample_eval():
    # Dictionary to store results
    result_dict = {}
    
    # Read the ground truth data
    ground_truth_path = "./QA_Data/multi_sample_qa_info_k100_bm100_te.json"
    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)

    test_dti_crispr_rna_samples_index_df = pd.read_csv('./BMG/CRISPR-Graph/test_dti_crispr_rna_samples_index.csv')
    test_samples = test_dti_crispr_rna_samples_index_df['Sample'].tolist()
    
    # Process each sample
    for sample_id in test_samples:
        # Construct path to the protein importance file
        importance_file = f'./MOTASG_Analysis/{sample_id}/protein_weighted_importance.csv'
        
        try:
            # Read the protein importance file
            importance_df = pd.read_csv(importance_file)
            
            # Use the specific column names 'HGNC_Symbol' and 'total_weight'
            if 'HGNC_Symbol' in importance_df.columns and 'total_weight' in importance_df.columns:
                top_proteins = importance_df.sort_values('total_weight', ascending=False).head(100)['HGNC_Symbol'].tolist()
            else:
                # Fall back to generic column detection if specific columns not found
                print(f"Warning: Expected columns 'HGNC_Symbol' and 'total_weight' not found in {importance_file}")
                print(f"Available columns: {importance_df.columns.tolist()}")
                
                protein_col = [col for col in importance_df.columns if 'protein' in col.lower() or 'gene' in col.lower() or 'symbol' in col.lower()]
                importance_col = [col for col in importance_df.columns if 'importance' in col.lower() or 'score' in col.lower() or 'weight' in col.lower()]
                
                if protein_col and importance_col:
                    top_proteins = importance_df.sort_values(importance_col[0], ascending=False).head(100)[protein_col[0]].tolist()
                else:
                    print(f"Warning: Could not identify appropriate columns in {importance_file}")
                    continue
            
            # Extract ground truth list for this sample
            try:
                ground_truth_list = ground_truth_data[sample_id]['ground_truth_answer']['top_bm_gene']['hgnc_symbols']
            except KeyError:
                print(f"Warning: Ground truth data not found for sample {sample_id}")
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(top_proteins, ground_truth_list)
            overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10 = metrics
            
            # Add to results dictionary
            result_dict[sample_id] = {
                "motasg_proteins": top_proteins,
                "ground_truth": ground_truth_list,
                "evaluation_results": {
                    "gat": {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "overlap_count": overlap_count,
                        "jaccard": jaccard,
                        "precision@5": precision_at_5,
                        "precision@10": precision_at_10
                    }
                }
            }
            
            # Print evaluation results for this sample
            print(f"\nEvaluation Results for Sample {sample_id}:")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
            print(f"Overlap: {overlap_count}/{len(ground_truth_list)}, Jaccard: {jaccard:.4f}")
            print(f"P@5: {precision_at_5:.4f}, P@10: {precision_at_10:.4f}")
            
        except FileNotFoundError:
            print(f"Warning: Importance file not found for sample {sample_id}: {importance_file}")
            continue
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            continue
    
    # Save results to JSON file
    output_path = "./QA_Results/motasg_gat_results.json"
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"\nEvaluation results saved to {output_path}")
    return result_dict

if __name__ == "__main__":
    motasg_sample_eval()