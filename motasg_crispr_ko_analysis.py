import os
import json
import argparse
import pandas as pd
import numpy as np
import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

# custom modules
from MOTASG_Foundation.utils import tab_printer
from MOTASG_Foundation.model import MOTASG_Foundation, DegreeDecoder, EdgeDecoder, GNNEncoder
from MOTASG_Foundation.mask import MaskEdge

# custom dataloader
from GeoDataLoader.read_geograph import read_analysis_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

from MOTASG_Foundation.lm_model import TextEncoder
from MOTASG_Foundation.downstream_analysis import MOTASG_KO_Reg, DSGNNEncoder


def precision_at_k(predicted_list, ground_truth_set, k):
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
    # Print lengths
    predicted_len = len(predicted_list)
    ground_truth_len = len(ground_truth_list)
    # Convert lists to sets for intersection/union operations
    predicted_set = set(predicted_list)
    ground_truth_set = set(ground_truth_list)
    # Print final sizes after potential truncation
    print(f"Evaluation using {len(predicted_set)} predicted items vs {len(ground_truth_set)} ground truth items")
    # Calculate basic metrics
    overlap_set = ground_truth_set.intersection(predicted_set)
    overlap_count = len(overlap_set)
    precision = overlap_count / len(predicted_set) if len(predicted_set) > 0 else 0
    # Calculate recall against the ground truth set for fairness
    recall = overlap_count / len(ground_truth_set) if overlap_count > 0 else 0
    # F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Jaccard Similarity against truncated set
    union_count = len(ground_truth_set.union(predicted_set))
    jaccard = overlap_count / union_count if union_count > 0 else 0
    # Precision at K - use ground truth list for coverage
    precision_at_5 = precision_at_k(predicted_list, ground_truth_set, k=5)
    precision_at_10 = precision_at_k(predicted_list, ground_truth_set, k=10)
    return overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10


def calculate_average_gat_metrics(data):
    """
    Calculate average metrics across multiple samples for the GAT model.
    
    Args:
        data (dict): Dictionary where keys are sample IDs and values are dictionaries
                    containing evaluation_results with GAT metrics
    
    Returns:
        dict: Dictionary containing average metrics for the GAT model
    """
    # Initialize counters for GAT metrics
    metrics = {
        'gat': {
            'precision': 0, 
            'recall': 0, 
            'f1_score': 0, 
            'overlap_count': 0, 
            'jaccard': 0, 
            'precision@5': 0, 
            'precision@10': 0
        }
    }
    
    # Initialize total ground truth count for calculating average overlap counts
    total_ground_truth_count = 0
    
    # Count samples with valid metrics
    valid_samples = 0
    
    # Sum up all metrics
    for sample_id, sample_data in data.items():
        if 'evaluation_results' in sample_data and 'gat' in sample_data['evaluation_results']:
            valid_samples += 1
            gat_metrics = sample_data['evaluation_results']['gat']
            
            # Sum all numeric metrics
            for metric in metrics['gat']:
                if metric in gat_metrics:
                    metrics['gat'][metric] += gat_metrics[metric]
            
            # Keep track of ground truth counts for proper overlap_count averaging
            if 'ground_truth' in sample_data:
                total_ground_truth_count += len(sample_data['ground_truth'])
    
    # Calculate averages
    if valid_samples > 0:
        for metric in metrics['gat']:
            metrics['gat'][metric] /= valid_samples
    
    # Calculate average overlap count relative to average ground truth size
    avg_ground_truth_size = total_ground_truth_count / valid_samples if valid_samples > 0 else 0
    
    return {
        'average_metrics': metrics,
        'samples_count': valid_samples,
        'average_ground_truth_size': avg_ground_truth_size
    }


def build_pretrain_model(args, device):
    mask = MaskEdge(p=args.p)

    text_encoder = TextEncoder(args.text_lm_model_path, device)

    graph_encoder = GNNEncoder(args.num_omic_feature, args.encoder_channels, args.hidden_channels,
                        num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                        bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    internal_graph_encoder = GNNEncoder(args.num_omic_feature, args.input_dim, args.input_dim,
                            num_layers=args.internal_encoder_layers, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                            num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    pretrain_model = MOTASG_Foundation(text_input_dim=args.lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    input_dim=args.input_dim, 
                    text_encoder=text_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder,
                    edge_decoder=edge_decoder,
                    degree_decoder=degree_decoder,
                    mask=mask).to(device)
    
    return pretrain_model


def build_model(args, device):
    text_encoder = TextEncoder(args.text_lm_model_path, device)

    internal_graph_encoder = DSGNNEncoder(args.train_fusion_dim, args.train_fusion_ko_dim, args.train_fusion_ko_dim,
                            num_layers=args.train_internal_encoder_layers, dropout=args.train_encoder_dropout,
                            bn=args.bn, layer=args.train_layer, activation=args.encoder_activation)

    graph_encoder = DSGNNEncoder(args.train_fusion_ko_dim, args.train_hidden_dim, args.train_output_dim,
                    num_layers=args.train_encoder_layers, dropout=args.train_encoder_dropout,
                    bn=args.bn, layer=args.train_layer, activation=args.encoder_activation)

    model = MOTASG_KO_Reg(text_input_dim=args.lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    pre_input_dim=args.pre_input_dim,
                    fusion_dim=args.train_fusion_dim,
                    internal_graph_output_dim=args.train_fusion_ko_dim, # internal graph encoder output dim
                    graph_output_dim=args.train_output_dim, # graph encoder output dim
                    linear_input_dim=args.train_linear_input_dim,
                    linear_hidden_dim=args.train_linear_hidden_dim,
                    linear_output_dim=args.train_linear_output_dim,
                    num_entity=args.num_entity,
                    text_encoder=text_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder).to(device)
    return model


def analyze_model(analyze_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, pretrain_model, model, device, args, current_sample_id):
    for batch_idx, data in enumerate(analyze_dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        # Use torch.no_grad() for inference
        with torch.no_grad():
            # Use pretrained model to get the embedding
            batch_size = current_cell_num
            name_emb = pretrain_model.name_linear_transform(name_embeddings).clone()
            name_emb = name_emb.repeat(batch_size, 1)
            desc_emb = pretrain_model.desc_linear_transform(desc_embeddings).clone()
            desc_emb = desc_emb.repeat(batch_size, 1)
            omic_emb = pretrain_model.omic_linear_transform(x).clone()
            merged_emb = torch.cat([name_emb, desc_emb, omic_emb], dim=-1)
            cross_x = pretrain_model.cross_modal_fusion(merged_emb) + x
            z = pretrain_model.internal_encoder(cross_x, internal_edge_index)
            pre_x = pretrain_model.encoder.get_embedding(z, ppi_edge_index, mode='last') # mode='cat'
            # Continue the model
            z, attention_weights_dict = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, name_embeddings, desc_embeddings, current_cell_num, current_sample_id)
    return z, attention_weights_dict


def analyze_att(args, pretrain_model, device):
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    
    print('--- LOADING ALL FILES ... ---')
    xAnalysis = np.load("./BMG/CRISPR-Graph/dti_test_crispr_rna_samples_omics.npy")
    analysis_index_df = pd.read_csv("./BMG/CRISPR-Graph/test_dti_crispr_rna_samples_index.csv")
    # Create a dictionary that maps natural indices (0-based) to Sample names
    index_sample_dict = {i: sample for i, sample in enumerate(analysis_index_df['Sample'])}
    omics_nodes_index_df = pd.read_csv("./BMG/CRISPR-Graph/omics_nodes_index.csv")
    # Create a dictionary that maps Index to Node (Index and Node are the column names in the CSV file)
    index_node_dict = {row['Index']: row['Node'] for _, row in omics_nodes_index_df.iterrows()}

    analysis_num_cell = xAnalysis.shape[0]
    num_entity = xAnalysis.shape[1]
    args.num_entity = num_entity
    num_feature = args.num_omic_feature

    # Load edge_index
    all_edge_index = np.load('./BMG/CRISPR-Graph/edge_index.npy')
    internal_edge_index = np.load('./BMG/CRISPR-Graph/internal_edge_index.npy')
    ppi_edge_index = np.load('./BMG/CRISPR-Graph/ppi_edge_index.npy')
    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()

    # Load model
    model = build_model(args, device)
    model.load_state_dict(torch.load(args.train_save_path))
    
    # load textual embeddings into torch tensor
    name_embeddings = np.load('./BMG/Pretrain_data/x_name_emb.npy').reshape(-1, args.lm_emb_dim)
    name_embeddings = torch.from_numpy(name_embeddings)
    print(f'Name Embeddings Shape: {name_embeddings.shape}')
    desc_embeddings = np.load('./BMG/Pretrain_data/x_desc_emb.npy').reshape(-1, args.lm_emb_dim)
    desc_embeddings = torch.from_numpy(desc_embeddings)
    print(f'Description Embeddings Shape: {desc_embeddings.shape}')
    name_embeddings = name_embeddings.float().to(device)
    desc_embeddings = desc_embeddings.float().to(device)
    analysis_batch_size = args.train_batch_size
    
    # Run analysis model
    model.eval()
    upper_index = 0
    for index in range(0, analysis_num_cell, analysis_batch_size):
        if (index + analysis_batch_size) < analysis_num_cell:
            upper_index = index + analysis_batch_size
        else:
            upper_index = analysis_num_cell
        current_sample_id = index_sample_dict[index]
        geo_datalist = read_analysis_batch(index, upper_index, xAnalysis, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
        analysis_dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args.train_batch_size, args.train_num_workers)
        print('ANALYSIS MODEL SAMPLE ID: ', current_sample_id)
        current_cell_num = upper_index - index  # current batch size
        z, attention_weights_dict = analyze_model(analysis_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, pretrain_model, model, device, args, current_sample_id)
        # Save the attention weights
        # First, create a folder named after the sample ID
        sample_folder = os.path.join(args.analysis_result_folder, current_sample_id)
        os.makedirs(sample_folder, exist_ok=True)
        # Save the attention weights for each layer as a separate csv file with the sample ID and column names ['source', 'target', 'weight']
        sample_att_weight_df_list = []
        for layer, att_weights in attention_weights_dict.items():
            # Create DataFrame from dictionary of tensors by first moving tensors to CPU and converting to numpy
            att_data = {
                'source': att_weights['source'].cpu().numpy(),
                'target': att_weights['target'].cpu().numpy(),
                'weight': att_weights['weights'].cpu().numpy()  # Note: key is 'weights' not 'weight'
            }
            att_weights_df = pd.DataFrame(att_data)
            # Instead of using apply with lambda functions
            att_weights_df['source'] = att_weights_df['source'].map(index_node_dict)
            att_weights_df['target'] = att_weights_df['target'].map(index_node_dict)
            att_weights_df.to_csv(os.path.join(sample_folder, f'attention_weights_layer_{layer}.csv'), index=False)
            sample_att_weight_df_list.append(att_weights_df)
        # Calculate the average attention weights across all layers
        avg_att_weights = pd.concat(sample_att_weight_df_list).groupby(['source', 'target'], as_index=False).mean()
        avg_att_weights.to_csv(os.path.join(sample_folder, f'avg_attention_weights.csv'), index=False)


# Add this function to split the multi-gene entries
def expand_multi_genes(df, symbol_col='HGNC_Symbol', weight_col='total_weight'):
    """
    Expand rows with multiple gene symbols separated by semicolons into separate rows.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the gene symbols
        symbol_col (str): Name of column containing gene symbols
        weight_col (str): Name of column containing weights
        
    Returns:
        pandas.DataFrame: Expanded DataFrame with one row per gene symbol
    """
    expanded_rows = []
    
    for _, row in df.iterrows():
        # Check if the symbol is a string
        if isinstance(row[symbol_col], str):
            # Check if the symbol contains semicolons
            symbols = row[symbol_col].split(';')
            if len(symbols) > 1:
                # Create a new row for each symbol with the same weight
                for symbol in symbols:
                    new_row = row.copy()
                    new_row[symbol_col] = symbol.strip()
                    expanded_rows.append(new_row)
            else:
                # Keep the row as is
                expanded_rows.append(row)
        else:
            # For non-string values (like NaN or float), keep the row as is
            expanded_rows.append(row)
    
    return pd.DataFrame(expanded_rows)

def key_targets_analysis(qa_info_data):
    analysis_index_df = pd.read_csv("./BMG/CRISPR-Graph/test_dti_crispr_rna_samples_index.csv")
    cell_num = analysis_index_df.shape[0]
    # Create a dictionary that maps natural indices (0-based) to Sample names
    index_sample_dict = {i: sample for i, sample in enumerate(analysis_index_df['Sample'])}
    omics_nodes_index_df = pd.read_csv("./BMG/CRISPR-Graph/omics_nodes_index.csv")
    bmgc_protein_df = pd.read_csv("./BMG/BioMedGraphica-Conn/Entity/Protein/BioMedGraphica_Conn_Protein.csv")
    
    # Create timestamp for consistent file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    # Iterate the analysis_index_df with Sample, where analysis_index_df contains columns [Sample,Index,Pre_Index]
    for idx, row in analysis_index_df.iterrows():
        sample_id = row['Sample']
        sample_index = row['Index']
        pre_index = row['Pre_Index']
        ground_truth_proteins = qa_info_data[sample_id]['ground_truth_answer']['top_bm_gene']['hgnc_symbols']
        
        print(f"Processing sample: {sample_id}, Index: {sample_index}, Pre_Index: {pre_index}")
        
        # Load the sample's attention weights from the saved CSV files
        sample_folder = os.path.join(args.analysis_result_folder, sample_id)
        avg_att_weights_path = os.path.join(sample_folder, 'avg_attention_weights.csv')
        avg_att_weights_df = pd.read_csv(avg_att_weights_path)

        # Calculate out-degree weights (sum of weights going out from each source node)
        out_degree_weights = avg_att_weights_df.groupby('source')['weight'].sum().reset_index()
        out_degree_weights.rename(columns={'source': 'node', 'weight': 'out_weight'}, inplace=True)

        # Calculate in-degree weights (sum of weights coming into each target node)
        in_degree_weights = avg_att_weights_df.groupby('target')['weight'].sum().reset_index()
        in_degree_weights.rename(columns={'target': 'node', 'weight': 'in_weight'}, inplace=True)

        # Merge the two dataframes to get total node weights
        node_weights = pd.merge(out_degree_weights, in_degree_weights, on='node', how='outer').fillna(0)

        # Calculate total weighted degree (sum of in and out weights)
        node_weights['total_weight'] = node_weights['out_weight'] + node_weights['in_weight']

        # Merge bmgc_protein_df with node_weights to get the protein names
        protein_node_weights = pd.merge(node_weights, bmgc_protein_df[['BioMedGraphica_Conn_ID', 'HGNC_Symbol']], 
                                        left_on='node', right_on='BioMedGraphica_Conn_ID', how='inner').reset_index(drop=True)
        # Drop columns ['node', 'out_weight', 'in_weight', 'BioMedGraphica_Conn_ID'] and make it in order ['HGNC_Symbol', 'total_weight']
        protein_node_weights = protein_node_weights[['HGNC_Symbol', 'total_weight']]

        # Expand multi-gene entries (values containing semicolons)
        expanded_node_weights = expand_multi_genes(protein_node_weights)

        # Now aggregate the weights by HGNC_Symbol with mean aggregation
        protein_node_agg_weights = expanded_node_weights.groupby('HGNC_Symbol', as_index=False).mean()

        # Sort the dataframe by total_weight in descending order
        protein_node_agg_weights.sort_values(by='total_weight', ascending=False, inplace=True)

        # Print how many genes expanded for tracking
        original_count = len(protein_node_weights)
        expanded_count = len(expanded_node_weights)
        print(f"Expanded gene entries from {original_count} to {expanded_count} rows")

        # Fetch the top 100 proteins
        top_100_proteins = protein_node_agg_weights.head(100)
        top_100_protein_hgnc_list = top_100_proteins['HGNC_Symbol'].tolist()

        # Calculate metrics
        overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10 = calculate_metrics(top_100_protein_hgnc_list, ground_truth_proteins)
        
        # Record as gat_metrics
        gat_metrics = (overlap_count, precision, recall, f1_score, jaccard, precision_at_5, precision_at_10)
        
        # Print evaluation results for GAT
        print("\n********** GAT Model Evaluation Results **********")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Overlap Count: {overlap_count}/{len(ground_truth_proteins)}")
        print(f"Jaccard Similarity: {jaccard:.4f}")
        print(f"Precision at 5: {precision_at_5:.4f}")
        print(f"Precision at 10: {precision_at_10:.4f}")
        
        # Create results dictionary similar to your existing format
        results_dict = {
            "top_proteins": top_100_protein_hgnc_list,
            "ground_truth":  ground_truth_proteins,
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
            },
        }
        
        # Save node weights with scores
        protein_node_agg_weights.to_csv(os.path.join(sample_folder, 'protein_weighted_importance.csv'), index=False)
        
        # Add to overall results
        all_results[sample_id] = results_dict
    
    # Save all results to a single JSON file

    output_path = "./QA_Results/motasg_gat_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"All GAT metrics saved to {output_path}")


def arg_parse():
    parser = argparse.ArgumentParser()

    # pre-training loading parameters
    parser.add_argument('--layer', nargs='?', default='gat', help='GNN layer, (default: gat)')
    parser.add_argument('--encoder_activation', nargs='?', default='leaky_relu', help='Activation function for GNN encoder, (default: leaky_relu)')

    parser.add_argument('--num_omic_feature', type=int, default=1, help='Omic feature size. (default: 1)')
    parser.add_argument('--lm_emb_dim', type=int, default=1, help='Text embedding dimension. (default: 1)')

    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension. (default: 1)')
    parser.add_argument('--encoder_channels', type=int, default=8, help='Channels of GNN encoder layers. (default: 8)')
    parser.add_argument('--hidden_channels', type=int, default=8, help='Channels of hidden representation. (default: 8)')
    parser.add_argument('--decoder_channels', type=int, default=4, help='Channels of decoder layers. (default: 4)')

    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--internal_encoder_layers', type=int, default=4, help='Number of layers for internal encoder. (default: 4)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.2, help='Dropout probability of encoder. (default: 0.2)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')

    parser.add_argument('--p', type=float, default=0.0001, help='Mask ratio or sample ratio for MaskEdge')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--graphclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--save_path', nargs='?', default='./Checkpoints/pretrained_models_gat/pretrained_motasg_foundation.pt', help='save path for model. (default: pretrained_motasg_foundation.pt)')
    parser.add_argument('--device', type=int, default=0)

    # downstream task parameters
    parser.add_argument('--train_sample_ratio', type=float, default=0.00005, help='Sampling ratio of training data. (default: 1.0)') # Sample_ratio
    parser.add_argument('--training_sample_random_seed', type=int, default=2025, help='Random seed for sampling training data. (default: 42)')

    parser.add_argument('--text_lm_model_path', nargs='?', default='dmis-lab/biobert-v1.1', help='Path to the pretrained language model. (default: dmis-lab/biobert-v1.1)')
    parser.add_argument('--train_text', default=False, help='Whether to train the text encoder. (default: False)')
    parser.add_argument('--name', nargs='?', default='CRISPR', help='Name for dataset.')

    parser.add_argument('--train_lr', type=float, default=0.25, help='Learning rate for training. (default: 0.0002)')
    parser.add_argument('--train_lr2', type=float, default=0.15, help='Learning rate for training. (default: 0.0001)')
    parser.add_argument('--train_lr3', type=float, default=0.1, help='Learning rate for training. (default: 0.000075)')
    parser.add_argument('--train_lr4', type=float, default=0.05, help='Learning rate for training. (default: 0.00005)')
    parser.add_argument('--train_lr5', type=float, default=0.025, help='Learning rate for training. (default: 0.00001)')
    parser.add_argument('--train_eps', type=float, default=1e-7, help='Epsilon for Adam optimizer. (default: 1e-7)')
    parser.add_argument('--train_weight_decay', type=float, default=1e-3, help='Weight decay for Adam optimizer. (default: 1e-15)')
    parser.add_argument('--train_encoder_dropout', type=float, default=0.05, help='Dropout probability of encoder. (default: 0.05)')

    parser.add_argument('--num_train_epoch', type=int, default=30, help='Number of training epochs. (default: 5)')
    parser.add_argument('--train_batch_size', type=int, default=1, help='Batch size for training. (default: 16)')
    parser.add_argument('--train_num_workers', type=int, default=0, help='Number of workers to load data.')

    parser.add_argument('--train_layer', nargs='?', default='gat', help='GNN layer, (default: gat)')
    parser.add_argument('--train_internal_encoder_layers', type=int, default=3, help='Number of layers for internal encoder. (default: 3)')
    parser.add_argument('--train_encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 3)')

    parser.add_argument('--pre_input_dim', type=int, default=9, help='Input feature dimension for pretraining. (default: 8)') # should be same as hidden_channels + 1 for ko_mask
    parser.add_argument('--train_fusion_dim', type=int, default=1, help='Fusion feature dimension for training. (default: 1)') # fused_dim, due to using internal_emb + x, should be same as omics_feature_dim (used for fusion/internal_encoder/pre_transformer)
    parser.add_argument('--train_fusion_ko_dim', type=int, default=2, help='Fusion feature dimension for training. (default: 1)') # fused_dim, due to using internal_emb + x, should be same as omics_feature_dim (used for fusion/internal_encoder/pre_transformer)
    parser.add_argument('--train_hidden_dim', type=int, default=8, help='Hidden feature dimension for training. (default: 8)') # convert the num_omic_feature to hidden_dim (dim for graph encoder if used)
    parser.add_argument('--train_output_dim', type=int, default=8, help='Output feature dimension for training. (default: 8)')  

    parser.add_argument('--train_linear_input_dim', type=int, default=8, help='Input feature dimension for training. (default: 16)') 
    # parser.add_argument('--train_linear_input_dim', type=int, default=512, help='Input feature dimension for training. (default: 16)') 
    parser.add_argument('--train_linear_hidden_dim', type=int, default=128, help='Hidden feature dimension for training. (default: 32)')
    parser.add_argument('--train_linear_output_dim', type=int, default=16, help='Output feature dimension for training. (default: 16)')

    parser.add_argument('--analysis_result_folder', nargs='?', default='MOTASG_Analysis', help='Path to save analysis results. (default: MOTASG_Analysis)')
    parser.add_argument('--model_name', nargs='?', default='MOTASG_Reg', help='Model names. (default: MOTASG_Reg)')
    # parser.add_argument('--train_save_path', nargs='?', default='./MOTASG_Results/CRISPR/MOTASG_Reg_gat_gat/epoch_40_3_best/best_train_model.pt', help='Path to save the trained model.')
    parser.add_argument('--train_save_path', nargs='?', default='./MOTASG_Results/CRISPR/MOTASG_Reg_gat_gat/epoch_25_1_best/best_train_model.pt', help='Path to save the trained model.')

    return parser.parse_args()


if __name__ == "__main__":
    # Set arguments and print
    args = arg_parse()
    print(tab_printer(args))
    # Check device
    device = 'cpu' if args.device < 0 else (f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set model_name with adding pretrain based GNN and train based GNN
    args.model_name = args.model_name + '_' + args.layer + '_' + args.train_layer

    # Load pretrain model
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.load_state_dict(torch.load(args.save_path))
    pretrain_model.eval()

    analyze_att(args, pretrain_model, device)

    json_path = "./QA_Data/multi_sample_qa_info_k100_bm100_te.json"
    with open(json_path, "r") as f:
        qa_info_data = json.load(f)
    key_targets_analysis(qa_info_data)

    # # Load the existing JSON file
    # file_path = './MOTASG_Analysis/metrics/gat_metrics_20250502_040107.json'
    # with open(file_path, 'r') as f:
    #     results_data = json.load(f)
    # # Calculate average metrics
    # average_metrics = calculate_average_gat_metrics(results_data)
    # print("Average metrics for testing data:")
    # print(json.dumps(average_metrics, indent=2))
