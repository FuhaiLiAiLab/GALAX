import copy
import os
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import trange

from MOTASG_Foundation.mask import MaskEdge
from MOTASG_Foundation.lm_model import TextEncoder
from MOTASG_Foundation.model import MOTASG_Foundation, DegreeDecoder, EdgeDecoder, GNNEncoder
from MOTASG_Foundation.downstream import MOTASG_Class, DSGNNEncoder

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from explain.explainer import Explainer, XGNNExplainer
from explain.explanation import ExplanationSetSampler

from torch_geometric.utils import to_networkx
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    TransformerConv,
    global_mean_pool
)

from config import arg_parse

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


def pre_embed(pretrain_model, model, selected_sample_x, name_embeddings, desc_embeddings, all_edge_index, internal_edge_index, ppi_edge_index, device):
    selected_sample_x = selected_sample_x.float()
    with torch.no_grad():
        # Use pretrained model to get the embedding
        name_emb = pretrain_model.name_linear_transform(name_embeddings).clone()
        desc_emb = pretrain_model.desc_linear_transform(desc_embeddings).clone()
        omic_emb = pretrain_model.omic_linear_transform(selected_sample_x).clone()
        merged_emb = torch.cat([name_emb, desc_emb, omic_emb], dim=-1)
        cross_x = pretrain_model.cross_modal_fusion(merged_emb) + selected_sample_x
        z = pretrain_model.internal_encoder(cross_x, internal_edge_index)
        pre_x = pretrain_model.encoder(z, ppi_edge_index)
        # Continue using model to get the embedding on the protein graph
        tr_name_emb = model.name_linear_transform(name_embeddings).clone()
        tr_desc_emb = model.desc_linear_transform(desc_embeddings).clone()
        tr_omic_emb = model.omic_linear_transform(selected_sample_x).clone()
        tr_merged_emb = torch.cat([tr_name_emb, tr_desc_emb, tr_omic_emb], dim=-1)
        tr_cross_x = model.cross_modal_fusion(tr_merged_emb)
        tr_z = model.act(model.internal_encoder(tr_cross_x, internal_edge_index)) + selected_sample_x
        tr_z = model.pre_transform(pre_x) + tr_z
    return tr_z


def build_model(args, device):
    text_encoder = TextEncoder(args.text_lm_model_path, device)

    internal_graph_encoder = DSGNNEncoder(args.train_fusion_dim, args.train_fusion_dim, args.train_fusion_dim,
                            num_layers=args.train_internal_encoder_layers, dropout=args.train_encoder_dropout,
                            bn=args.bn, layer=args.train_layer, activation=args.encoder_activation)

    graph_encoder = DSGNNEncoder(args.train_fusion_dim, args.train_hidden_dim, args.train_output_dim,
                    num_layers=args.train_encoder_layers, dropout=args.train_encoder_dropout,
                    bn=args.bn, layer=args.train_layer, activation=args.encoder_activation)

    model = MOTASG_Class(text_input_dim=args.lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    pre_input_dim=args.pre_input_dim,
                    fusion_dim=args.train_fusion_dim,
                    internal_graph_output_dim=args.train_fusion_dim, # internal graph encoder output dim
                    graph_output_dim=args.train_output_dim, # graph encoder output dim
                    linear_input_dim=args.train_linear_input_dim,
                    linear_hidden_dim=args.train_linear_hidden_dim,
                    linear_output_dim=args.train_linear_output_dim,
                    num_entity=args.num_entity,
                    num_class=args.num_class,
                    text_encoder=text_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder).to(device)
    return model


def extract_disease_protein(selected_sample_disease_bmgc_id, dti_edge_index_path, 
                    omics_node_index_df, nodeid_index_dict, index_nodeid_dict):
    """
    Extract protein directly connected to a disease node.
    
    Args:
        sample_disease_bmgc_id_index (int): Index of the disease node
        dti_edge_index_path (str): Path to the DTI edge index file
        omics_node_index_df (pd.DataFrame): DataFrame with node type information
        
    Returns:
        tuple: (disease_protein_index, disease_protein_bmgc_id)
    """
    # Extract the index based on the selected disease BMGC ID
    sample_disease_bmgc_id_index = nodeid_index_dict[selected_sample_disease_bmgc_id]
    
    # Load DTI edge index
    dti_all_edge_index = np.load(dti_edge_index_path)
    
    # Find incoming edges (source nodes that point to the disease)
    incoming_mask = dti_all_edge_index[1, :] == sample_disease_bmgc_id_index
    incoming_source_nodes = dti_all_edge_index[0, incoming_mask]
    
    # Find outgoing edges (target nodes that the disease points to)
    outgoing_mask = dti_all_edge_index[0, :] == sample_disease_bmgc_id_index
    outgoing_target_nodes = dti_all_edge_index[1, outgoing_mask]
    
    # Combine all neighbor nodes (both incoming and outgoing)
    disease_related_nodes = np.concatenate([incoming_source_nodes, outgoing_target_nodes])
    unique_disease_related_nodes = np.unique(disease_related_nodes)
    
    # Get protein node index
    protein_node_index_df = omics_node_index_df[omics_node_index_df['Type'] == 'Protein']
    protein_node_index_list = protein_node_index_df['Index'].tolist()
    
    # Filter to get only protein nodes directly connected to the disease
    disease_protein_index = sorted(
        list(set(unique_disease_related_nodes) & set(protein_node_index_list))
    )
    
    # Map protein index to BMGC id
    disease_protein_bmgc_id = [index_nodeid_dict[i] for i in disease_protein_index]
    
    return disease_protein_index, disease_protein_bmgc_id


def extract_ppi_nodes(disease_protein_index, dti_edge_index_path, omics_node_index_df, index_nodeid_dict):
    """
    Extract protein-protein interaction (PPI) nodes that interact with disease-related protein.
    
    Args:
        disease_protein_index (list): List of protein node index directly related to the disease
        dti_edge_index_path (str): Path to the DTI edge index file
        omics_node_index_df (pd.DataFrame): DataFrame with node type information
        
    Returns:
        tuple: (ppi_nodes_index, ppi_nodes_bmgc_id)
    """

    # Load DTI edge index
    dti_all_edge_index = np.load(dti_edge_index_path)
    
    # Get protein node index
    protein_node_index_df = omics_node_index_df[omics_node_index_df['Type'] == 'Protein']
    protein_node_index_list = protein_node_index_df['Index'].tolist()
    
    # Get all nodes related to the identified protein neighbors (second hop)
    protein_related_nodes = []
    
    # Iterate through each protein neighbor node index
    for protein_node_idx in disease_protein_index:
        # Find incoming edges (nodes that point to this protein)
        protein_incoming_mask = dti_all_edge_index[1, :] == protein_node_idx
        protein_incoming_sources = dti_all_edge_index[0, protein_incoming_mask]
        
        # Find outgoing edges (nodes that this protein points to)
        protein_outgoing_mask = dti_all_edge_index[0, :] == protein_node_idx
        protein_outgoing_targets = dti_all_edge_index[1, protein_outgoing_mask]
        
        # Add these connected nodes to our list
        protein_related_nodes.extend(protein_incoming_sources)
        protein_related_nodes.extend(protein_outgoing_targets)
    
    # Convert to numpy array and get unique nodes
    protein_related_nodes = np.array(protein_related_nodes)
    unique_protein_related_nodes = np.unique(protein_related_nodes)
    
    # Remove any protein nodes themselves from this list to avoid duplication
    unique_protein_related_nodes = np.setdiff1d(
        unique_protein_related_nodes, disease_protein_index
    )
    
    # Filter to only keep protein nodes among the second-hop neighbors
    ppi_nodes_index = sorted(
        list(set(unique_protein_related_nodes) & set(protein_node_index_list))
    )
    
    # Map PPI node index to BMGC id
    ppi_nodes_bmgc_id = [index_nodeid_dict[i] for i in ppi_nodes_index]

    return ppi_nodes_index, ppi_nodes_bmgc_id
    

def remove_isolated_nodes(graph):
    """
    Removes isolated nodes from a graph and remaps node index.
    Keeps all nodes that have at least one connection.
    
    Args:
        graph (torch_geometric.data.Data): Input graph with x, edge_index, and node_id
        
    Returns:
        torch_geometric.data.Data: New graph with isolated nodes removed
    """
    # First, ensure the edge_index tensor has the correct shape
    if not hasattr(graph, 'edge_index') or not torch.is_tensor(graph.edge_index) or graph.edge_index.numel() == 0:
        return Data(x=torch.tensor([]), edge_index=torch.tensor([[],[]]), node_id=[])
    
    # Check if edge_index has the correct shape (needs explicit shape check)
    if len(graph.edge_index.shape) < 2 or graph.edge_index.shape[0] != 2:
        print(f"Warning: Invalid edge_index shape: {graph.edge_index.shape}")
        return Data(x=torch.tensor([]), edge_index=torch.tensor([[],[]]), node_id=[])
    
    # If there are no edges, return empty graph
    if graph.edge_index.shape[1] == 0:
        return Data(x=torch.tensor([]), edge_index=torch.tensor([[],[]]), node_id=[])
    
    # Get all nodes that appear in edge_index (connected nodes)
    connected_nodes = torch.unique(graph.edge_index.flatten()).tolist()
    
    # Create mapping from old index to new index
    old_to_new = {}
    new_node_id = []
    new_feature = []
    
    for new_idx, old_idx in enumerate(connected_nodes):
        old_to_new[old_idx] = new_idx
        new_node_id.append(graph.node_id[old_idx])
        new_feature.append(graph.x[old_idx].clone())  # Use clone for safety
    
    # Create new edge_index with remapped index
    new_edge_index = []
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
        # Both src and dst should be in connected_nodes
        new_edge_index.append([old_to_new[src], old_to_new[dst]])
    
    # Convert to tensor format - ensure proper contiguous memory
    if new_feature:
        new_x = torch.stack(new_feature).contiguous()
    else:
        new_x = torch.tensor([])
        
    if new_edge_index:
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
    else:
        new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=graph.x.device)
    
    # Create new graph
    new_graph = Data(x=new_x, edge_index=new_edge_index, node_id=new_node_id)
    
    return new_graph


def completion_additional_ppi_edges(best_graph, ppi_edges, best_graph_index_origin_index_dict):
    """
    Extract additional PPI edges for the best connected graph without modifying the original edges.
    
    Args:
        best_graph: The original graph
        ppi_edges: List of PPI edges in original index format
        best_graph_index_origin_index_dict: Mapping of new indices to original indices
    
    Returns:
        completed_best_graph: Data object with both original edge_index and additional_edge_index
    """
    # Get current edge index as numpy
    best_graph_edge_index = best_graph.edge_index.cpu().numpy()
    original_num_edges = best_graph_edge_index.shape[1]
    
    # Create reverse mapping from original index to new index in best_graph
    origin_to_new_index = {v: k for k, v in best_graph_index_origin_index_dict.items()}
    
    # Convert existing edge_index to a set of tuples for quick lookup
    existing_edges = set([(best_graph_edge_index[0, i], best_graph_edge_index[1, i]) 
                        for i in range(original_num_edges)])
    
    # Find additional edges from ppi_edges
    new_edges_src = []
    new_edges_dst = []
    
    for edge in ppi_edges:
        src_orig, dst_orig = edge[0], edge[1]
        # Check if both source and destination nodes are in our best connected graph
        if src_orig in origin_to_new_index and dst_orig in origin_to_new_index:
            # Map original indices to new indices in best_graph
            src_new = origin_to_new_index[src_orig]
            dst_new = origin_to_new_index[dst_orig]
            # Add edge if it doesn't already exist
            if (src_new, dst_new) not in existing_edges:
                new_edges_src.append(src_new)
                new_edges_dst.append(dst_new)
    
    # Create numpy array for new edges
    additional_edge_index = np.array([new_edges_src, new_edges_dst]) if new_edges_src else np.empty((2, 0))
    
    # Log what we found
    num_new_edges = additional_edge_index.shape[1]
    print(f"Found {num_new_edges} additional PPI edges (original edges: {original_num_edges})")
    
    # Create a copy of the original graph to avoid modifying it
    completed_best_graph = copy.deepcopy(best_graph)
    
    # Add the additional_edge_index as a new attribute
    completed_best_graph.additional_edge_index = torch.tensor(
        additional_edge_index, 
        dtype=torch.long, 
        device=best_graph.edge_index.device
    )
    
    return completed_best_graph


def masked_softmax(vector, mask):
    """Apply softmax to only selected elements of the vector, as indicated by
    the mask. The output will be a probability distribution where unselected
    elements are 0.

    Args:
        vector (torch.Tensor): Input vector.
        mask (torch.Tensor): Mask indicating which elements to softmax.

    Returns:
        torch.Tensor: Softmaxed vector.
    """
    mask = mask.bool()
    masked_vector = vector.masked_fill(~mask, float('-inf'))
    softmax_result = F.softmax(masked_vector, dim=0)

    return softmax_result


def safe_categorical(probs, dim=-1):
    """Creates a categorical distribution with safeguards against NaN values."""
    epsilon = 1e-8
    # Replace NaNs with small values
    probs = torch.where(torch.isnan(probs), torch.tensor(epsilon).to(probs.device), probs)
    # Ensure positivity
    probs = torch.clamp(probs, min=epsilon)
    # Normalize
    probs = probs / probs.sum(dim=dim, keepdim=True)
    return torch.distributions.Categorical(probs)


class GraphGenerator(torch.nn.Module, ExplanationSetSampler):
    """Graph generator that generates a new graph state from a given graph
    state.

    Inherits:
        torch.nn.Module: Base class for all neural network modules.
        ExplanationSetSampler: Base class for sampling from an explanation set.

    Args:
        candidate_set (dict): Set of candidate nodes for graph generation.
        dropout (float): Dropout rate for regularization.
        initial_node_id (str, optional): Initial node type for graph initialization.
        device (str): Device to run the model on ('cuda' or 'cpu').
        layer_type (str): Type of GNN layer ('gcn', 'gat', 'transformer').
        hidden_channels (list): List of hidden channel dimensions for each layer.
        heads (int): Number of attention heads for GAT and Transformer layers.
    """
    def __init__(self, candidate_set, dropout, initial_node_id=None, device='cuda', 
                 layer_type='gat', hidden_channels=[16, 24, 32], heads=1, alpha=0.2):
        super(GraphGenerator, self).__init__()
        # Initialize basic parameters
        self.candidate_set = candidate_set
        self.initial_node_id = initial_node_id
        self.device = device
        self.dropout = dropout
        self.alpha = alpha
        self.layer_type = layer_type
        
        # Determine feature dimensions
        num_node_feature = len(next(iter(self.candidate_set.values())))
        self.hidden_channels = hidden_channels
        self.output_dim = hidden_channels[-1]
        # Set a leaky ReLU activation function
        self.act = torch.nn.LeakyReLU(alpha)
        
        # Initialize GNN layers based on the specified layer type
        self.gnn_layers = torch.nn.ModuleList()
        
        if layer_type == 'gcn':
            self.gnn_layers.append(GCNConv(num_node_feature, hidden_channels[0]))
            for i in range(1, len(hidden_channels)):
                self.gnn_layers.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))
                
        elif layer_type == 'gat':
            # For GAT, we need to account for multi-head attention by dividing the output channels
            self.gnn_layers.append(GATConv(num_node_feature, hidden_channels[0] // heads, heads=heads))
            for i in range(1, len(hidden_channels)):
                self.gnn_layers.append(GATConv(hidden_channels[i-1], hidden_channels[i] // heads, heads=heads))
                
        elif layer_type == 'transformer':
            self.gnn_layers.append(TransformerConv(num_node_feature, hidden_channels[0] // heads, heads=heads))
            for i in range(1, len(hidden_channels)):
                self.gnn_layers.append(TransformerConv(hidden_channels[i-1], hidden_channels[i] // heads, heads=heads))
                
        else:
            # Default to GCN if an invalid type is specified
            print(f"Warning: Unknown layer type '{layer_type}', defaulting to GCN")
            self.layer_type = 'gcn'
            self.gnn_layers.append(GCNConv(num_node_feature, hidden_channels[0]))
            for i in range(1, len(hidden_channels)):
                self.gnn_layers.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))

        # Decision networks for start and end nodes
        self.mlp_start_node = torch.nn.Sequential(
            torch.nn.Linear(self.output_dim, self.output_dim // 2),
            torch.nn.LeakyReLU(alpha),
            torch.nn.Linear(self.output_dim // 2, self.output_dim // 2),
            torch.nn.LeakyReLU(alpha),
            torch.nn.Linear(self.output_dim // 2, 1),
            torch.nn.ReLU6()
        )
        
        self.mlp_end_node = torch.nn.Sequential(
            torch.nn.Linear(self.output_dim, self.output_dim // 2),
            torch.nn.LeakyReLU(alpha),
            torch.nn.Linear(self.output_dim // 2, self.output_dim // 2),
            torch.nn.LeakyReLU(alpha),
            torch.nn.Linear(self.output_dim // 2, 1),
            torch.nn.ReLU6()
        )

        # Store tensors for candidate set that are moved to the right device
        self.candidate_tensors = {}

        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset all learnable parameters of the model."""
        # Reset GNN layers
        for gnn_layer in self.gnn_layers:
            gnn_layer.reset_parameters()
        
        # Reset MLP layers for both start and end node decisions
        for layer in self.mlp_start_node:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.mlp_end_node:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def extract_candidate_feature(self, id_list, candidate_set):
        """
        Select multiple feature vectors from candidate_set based on a list of BMGC id
        
        Args:
            bmgc_id_list (list): List of BMGC id to select feature for
            candidate_set (dict): Dictionary mapping BMGC id to feature vectors
            device (str): Device to put tensors on
            
        Returns:
            torch.Tensor: Stacked feature vectors for the requested BMGC id
        """
        # Collect feature vectors for each BMGC ID in the list
        selected_feature = []
        
        for bmgc_id in id_list:
            if bmgc_id in candidate_set:
                selected_feature.append(candidate_set[bmgc_id])
            else:
                print(f"Warning: BMGC ID {bmgc_id} not found in candidate set")
        
        # Stack into a single tensor
        if selected_feature:
            return torch.stack(selected_feature).to(self.device)
        else:
            return torch.tensor([]).to(self.device)

    def initialize_graph_state(self, graph_state):
        r"""Initializes the graph state with a single node.

        Args:
            graph_state (torch_geometric.data.Data): The graph state to
            initialize.

        Returns:
            torch_geometric.data.Data: The initialized graph state.
        """
        # Get device from graph_state or use default
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_state = graph_state.to(self.device)

        feature = self.extract_candidate_feature(self.initial_node_id, self.candidate_set).to(self.device)
        edge_index = torch.tensor([], dtype=torch.long).view(2, -1).to(self.device)
        # update graph state
        graph_state.x = feature
        graph_state.edge_index = edge_index
        graph_state.node_id = self.initial_node_id

    def forward(self, graph_state):
        """Generates a new graph state from the given graph state.

        Args:
            graph_state (torch_geometric.data.Data): The graph state to
            generate a new graph state from.

        Returns:
            ((torch.Tensor, torch.Tensor), (torch.Tensor, torch.Tensor)): The
            logits and one hot encodings for the start and end node.
            torch_geometric.data.Data: The new graph state.
        """
        # Make graph_state to device on cuda
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_state = graph_state.to(self.device)
        graph_state = copy.deepcopy(graph_state)

        if graph_state.x.shape[0] == 0:
            self.initialize_graph_state(graph_state)  # initialize graph state if it is empty
        
        # contatenate graph_state feature with candidate_set feature
        node_feature_graph = graph_state.x.detach().clone()
        candidate_feature = torch.stack(list(self.candidate_set.values())).to(self.device)  # Move to same device as graph_state
        node_feature = torch.cat((node_feature_graph, candidate_feature), dim=0).float()
        node_edges = graph_state.edge_index.detach().clone()
        node_feature = node_feature.to(self.device)
        node_edges = node_edges.to(self.device)
        
        # compute node encodings with GNN layers
        node_encodings = node_feature
        for gnn_layer in self.gnn_layers:
            # node_encodings = F.relu6(gcn_layer(node_encodings, node_edges))
            node_encodings = self.act(gnn_layer(node_encodings, node_edges))
            node_encodings = F.dropout(node_encodings, self.dropout, training=self.training)

        # get start node probabilities and mask out candidates
        start_node_logits = self.mlp_start_node(node_encodings)

        candidate_set_mask = torch.ones_like(start_node_logits)
        candidate_set_index = torch.arange(node_feature_graph.shape[0],
                                             node_encodings.shape[0])
        # set candidate set probabilities to 0
        candidate_set_mask[candidate_set_index] = 0
        start_node_probs = masked_softmax(start_node_logits,
                                          candidate_set_mask).squeeze()

        # sample start node
        p_start = safe_categorical(start_node_probs)
        start_node = p_start.sample()

        # get end node probabilities and mask out start node
        end_node_logits = self.mlp_end_node(node_encodings)
        start_node_mask = torch.ones_like(end_node_logits)
        start_node_mask[start_node] = 0
        end_node_probs = masked_softmax(end_node_logits,
                                        start_node_mask).squeeze()

        # sample end node
        end_node = torch.distributions.Categorical(end_node_probs).sample()
        num_nodes_graph = graph_state.x.shape[0]
        if end_node >= num_nodes_graph:
            # add new node feature to graph state
            graph_state.x = torch.cat(
                [graph_state.x, node_feature[end_node].unsqueeze(0).float()],
                dim=0)
            graph_state.node_id.append(
                list(self.candidate_set.keys())[end_node - num_nodes_graph])
            new_edge = torch.tensor([[start_node], [num_nodes_graph]])
        else:
            new_edge = torch.tensor([[start_node], [end_node]])
        new_edge = new_edge.to(self.device)
        graph_state.edge_index = torch.cat((graph_state.edge_index, new_edge), dim=1)

        # one hot encoding of start and end node
        start_node_one_hot = torch.eye(start_node_probs.shape[0])[start_node]
        end_node_one_hot = torch.eye(end_node_probs.shape[0])[end_node]

        return ((start_node_logits.squeeze(), start_node_one_hot),
                (end_node_logits.squeeze(), end_node_one_hot)), graph_state

    def sample(self, num_samples: int, **kwargs):
        """Samples a number of graphs from the generator.

        Args:
            num_samples (int): The number of graphs to sample.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If neither num_nodes nor max_steps is specified.

        Returns:
            List[torch_geometric.data.Data]: The list of sampled graphs.
        """
        # extract num_nodes and max_steps from kwargs or set them to None
        num_nodes = kwargs.get('num_nodes', None)
        max_steps = kwargs.get('max_steps', None)

        # check that either num_nodes or max_steps is not None
        if num_nodes is None and max_steps is None:
            raise ValueError("Either num_nodes or max_steps must be specified")

        # create empty graph state
        empty_graph = Data(x=torch.tensor([]), edge_index=torch.tensor([]),
                           node_id=[])
        current_graph_state = copy.deepcopy(empty_graph)

        # sample graphs
        sampled_graphs = []

        max_steps_reached = False
        num_nodes_reached = False
        self.eval()
        for _ in range(num_samples):
            step = 0
            while not max_steps_reached and not num_nodes_reached:
                G = copy.deepcopy(current_graph_state)
                ((p_start, a_start),
                 (p_end, a_end)), current_graph_state = self.forward(G)
                step += 1
                # check if max_steps is reached
                max_steps_reached = max_steps is not None and step >= max_steps
                # check if num_nodes is reached
                num_nodes_reached = (num_nodes is not None
                                     and current_graph_state.x.shape[0]
                                     > num_nodes)
            # add sampled graph to list
            sampled_graphs.append(G)
            # reset current graph state
            current_graph_state = copy.deepcopy(empty_graph)
            # reset max_steps_reached and num_nodes_reached
            max_steps_reached = False
            num_nodes_reached = False
        return sampled_graphs


class RLGenExplainer(XGNNExplainer):
    """RL-based generator for graph explanations using XGNN.

    Inherits:
        XGNNExplainer: Base class for explanation generation using XGNN method.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        candidate_set (dict): Set of candidate nodes for graph generation.
        validity_args (numpy): Arguments for graph validity check.
        initial_node_id (str, optional): Initial node type for graph
        initialization.
    """
    def __init__(self, epochs, lr, candidate_set, ppi_nodeid_index_dict, validity_args, device='cuda', initial_node_id=None, num_entity=None):
        super(RLGenExplainer, self).__init__(epochs, lr)
        self.candidate_set = candidate_set
        self.graph_generator = GraphGenerator(candidate_set, 0.1, initial_node_id, device)
        if torch.cuda.is_available():
            self.graph_generator = self.graph_generator.to('cuda')
        self.max_steps = 50
        self.lambda_1 = 1
        self.lambda_2 = 1
        self.num_classes = 2
        self.ppi_nodeid_index_dict = ppi_nodeid_index_dict
        self.validity_args = validity_args
        self.device = device
        self.num_entity = num_entity

    def reward_tf(self, pre_trained_gnn, graph_state, num_classes, target_class):
        r"""Computes the reward for the given graph state by evaluating the
        graph with the pre-trained GNN.

        Args:
            pre_trained_gnn (torch.nn.Module): The pre-trained GNN to use for
            computing the reward.
            graph_state (torch_geometric.data.Data): The graph state to compute
            the reward for.
            num_classes (int): The number of classes in the dataset.

        Returns:
            torch.Tensor: The reward for the given graph state.
        """
        # Move graph state to the same device as pre_trained_gnn
        graph_state = graph_state.to(self.device)
        x = graph_state.x
        edge_index = graph_state.edge_index

        with torch.no_grad():
            try:
                z = pre_trained_gnn.act(pre_trained_gnn.encoder(x, edge_index))
                # Option 1
                # num_nodes = z.size(0)
                # batch = torch.zeros(num_nodes, dtype=torch.long, device=z.device)  # All nodes belong to the same graph
                # z = global_mean_pool(z, batch)
                # output = pre_trained_gnn.classifier(z_zero)
                # probabilities = torch.softmax(output, dim=1).squeeze() # Calculate all probabilities with softmax
                # probability_of_target_class = probabilities[target_class] # Extract only the probability for the target class
                # Option 2
                z = pre_trained_gnn.lin_transform(z).squeeze()
                # Create a torch zeros tensor with same entity size of whole graph
                z_zero = torch.zeros([self.num_entity], dtype=torch.float32, device=z.device)
                # Fetch the corrsponding indices from the ppi_nodeid_index_dict for current graph node_id
                z_indices = [self.ppi_nodeid_index_dict[node_id] for node_id in graph_state.node_id]
                z_zero[z_indices] = z
                z_zero = pre_trained_gnn.linear_repr(z_zero)
                output = pre_trained_gnn.classifier(z_zero)
                probabilities = torch.softmax(output, dim=0) # Calculate all probabilities with softmax
                probability_of_target_class = probabilities[target_class] # Extract only the probability for the target class
                return probability_of_target_class - 1 / num_classes
            except Exception as e:
                print(f"Error in reward calculation: {str(e)}")
                return 0.0  # Return neutral reward on error

    def rollout_reward(self, intermediate_graph_state, pre_trained_gnn,
                       target_class, num_classes, num_rollouts=5):
        r"""Computes the rollout reward for the given graph state.

        Args:
            intermediate_graph_state (torch_geometric.data.Data): The
            intermediate graph state to compute the rollout reward for.
            pre_trained_gnn (torch.nn.Module): The pre-trained GNN to use for
            computing the reward.
            target_class (int): The target class to explain.
            num_classes (int): The number of classes in the dataset.
            num_rollouts (int): The number of rollouts to perform.

        Returns:
            float: The average rollout reward for the given graph state.
        """
        # Move intermediate_graph_state to the same device as pre_trained_gnn
        intermediate_graph_state = intermediate_graph_state.to(self.device)

        final_rewards = []
        for _ in range(num_rollouts):
            # make copy of intermediate graph state
            intermediate_graph_state_copy = copy.deepcopy(
                intermediate_graph_state)
            _, final_graph = self.graph_generator(
                intermediate_graph_state_copy)
            # Evaluate the final graph
            reward = self.reward_tf(pre_trained_gnn, final_graph, num_classes, target_class)
            final_rewards.append(reward)

            del intermediate_graph_state_copy

        average_final_reward = sum(final_rewards) / len(final_rewards)
        return average_final_reward

    def evaluate_graph_validity(self, graph_state):
        """Evaluates the validity of the given graph state.

        Args:
            graph_state (torch_geometric.data.Data): The graph state to evaluate.

        Returns:
            tuple: (validity_score, invalid_edges)
                - validity_score: 0 if the graph is valid, -1 otherwise.
                - invalid_edges: List of (src, dst) tuples representing invalid edges
        """
        # import pdb; pdb.set_trace()
        # Get the current edge index from the graph state
        current_edge_index = graph_state.edge_index
        # Create the mapping dictionary from current graph.x (with graph.node_id) to ppi_index 
        # Map 1: Create a mapping dictionary from index in current graph to graph.node_id
        current_index_id_dict = {i: graph_state.node_id[i] for i in range(len(graph_state.node_id))}
        # Map 2: Create one more mapping dictionary from graph.node_id to ppi_index
        current_index_id_ppi_index_map_dict = {i: self.ppi_nodeid_index_dict[graph_state.node_id[i]] for i in range(len(graph_state.node_id))}
        
        # If there are no edges, the graph is valid
        if current_edge_index.shape[1] == 0:
            return 0
        
        # Get the original PPI edge index
        reference_edges = self.validity_args

        # Check if any edge in current_edge_index is not in reference_edges
        invalid_edges = []
        
        for i in range(current_edge_index.shape[1]):
            src = current_edge_index[0, i].item()
            dst = current_edge_index[1, i].item()
            
            # Map index to PPI graph index using candidate_set_index_map_dict
            if src in current_index_id_ppi_index_map_dict and dst in current_index_id_ppi_index_map_dict:
                ppi_src = current_index_id_ppi_index_map_dict[src]
                ppi_dst = current_index_id_ppi_index_map_dict[dst]
                mapped_edge = (ppi_src, ppi_dst)
                
                if mapped_edge not in reference_edges:
                    # Found an invalid edge
                    invalid_edges.append((src, dst))  # Store original index for reference
            else:
                # If either node isn't in the mapping, the edge is invalid
                invalid_edges.append((src, dst))
        
        # Return validity score and list of invalid edges
        validity_score = -1 if invalid_edges else 0
        
        # if invalid_edges:
        #     print(f"Found {len(invalid_edges)} invalid edges:")
        #     for src, dst in invalid_edges:
        #         # Show node id, index, and mapped index if available
        #         src_id = current_index_id_dict.get(src, "unknown")
        #         dst_id = current_index_id_dict.get(dst, "unknown")
                
        #         if src in current_index_id_ppi_index_map_dict and dst in current_index_id_ppi_index_map_dict:
        #             ppi_src = current_index_id_ppi_index_map_dict[src]
        #             ppi_dst = current_index_id_ppi_index_map_dict[dst]
        #             print(f"  {src}({src_id}) - {dst}({dst_id}) (mapped to {ppi_src} - {ppi_dst})")
        #         else:
        #             print(f"  {src}({src_id}) - {dst}({dst_id}) (unmapped)")
        
        return validity_score

    def calculate_reward(self, graph_state, pre_trained_gnn, target_class, num_classes):
        r"""Calculates the reward for the given graph state.

        Args:
            graph_state (torch_geometric.data.Data): The graph state to compute
            the reward for.
            pre_trained_gnn (torch.nn.Module): The pre-trained GNN to use for
            computing the reward.
            target_class (int): The target class to explain.
            num_classes (int): The number of classes in the dataset.

        Returns:
            torch.Tensor: The final reward for the given graph state.
        """
        # Move graph_state to the device of pre_trained_gnn
        graph_state = graph_state.to(self.device)

        # import pdb; pdb.set_trace()
        intermediate_reward = self.reward_tf(pre_trained_gnn, graph_state, num_classes, target_class)
        final_graph_reward = self.rollout_reward(graph_state, pre_trained_gnn, target_class, num_classes)
        # Compute graph validity score (R_tr),
        # based on the specific graph rules of the dataset
        graph_validity_score = self.evaluate_graph_validity(graph_state)
        reward = (intermediate_reward + self.lambda_1 * final_graph_reward + self.lambda_2 * graph_validity_score)
        return reward

    def train_generative_model(self, model_to_explain, for_class):
        """Trains the generative model for the given number of epochs. We us
        RL approach to train the generative model.

        Args:
            model_to_explain (_type_): The model to explain.
            for_class (_type_): The class to explain.

        Returns:
            torch_geometric.data.Data: The trained generative model.
        """
        optimizer = torch.optim.Adam(self.graph_generator.parameters(),
                                     lr=self.lr, betas=(0.9, 0.99))
        losses = []
        for epoch in trange(self.epochs):
            # print out the epoch number
            # print(f"---------- Epoch {epoch + 1}/{self.epochs} ----------")

            # create empty graph state
            empty_graph = Data(x=torch.tensor([]), edge_index=torch.tensor([]), node_id=[])
            current_graph_state = empty_graph

            for step in range(self.max_steps):
                # print out the step number
                # print(f"---------- Step {step + 1}/{self.max_steps} ----------")
                model_to_explain.train()
                optimizer.zero_grad()
                
                current_graph_state = current_graph_state.to('cuda')
                new_graph_state = copy.deepcopy(current_graph_state)
                ((p_start, a_start), (p_end, a_end)), new_graph_state = self.graph_generator(new_graph_state)
                reward = self.calculate_reward(new_graph_state, model_to_explain, for_class, self.num_classes)
                p_start = p_start.to('cuda')
                a_start = a_start.to('cuda')
                a_end = a_end.to('cuda')
                LCE_start = F.cross_entropy(p_start, a_start)
                LCE_end = F.cross_entropy(p_end, a_end)
                loss = -reward * (LCE_start + LCE_end)

                loss.backward()
                optimizer.step()

                if reward > 0:
                    current_graph_state = new_graph_state

            losses.append(loss.item())

        return self.graph_generator


def create_candidate_explainer(initial_node_id_list, candidate_set, model,
                             nodeid_index_dict, ppi_edges, num_entity, device, epochs=10, lr=0.01):
    """
    Creates a candidate set and explainer for graph explanation generation.
    
    Args:
        initial_node_id_list (list): List of initial node IDs for graph generation
        candidate_set (dict): Dictionary mapping BMGC ID to feature vectors
        model: Pre-trained GNN model to be explained
        nodeid_index_dict (dict): Mapping from BMGC ID to node index
        ppi_edges (numpy): PPI edges for graph validity check
        num_entity (int): Number of entities in the dataset
        device: Computation device (CPU or CUDA)
        epochs (int): Number of training epochs (default: 10)
        lr (float): Learning rate (default: 0.01)
        
    Returns:
        Explainer object configured to generate explanations
    """
    
    # Create the explainer
    explainer = Explainer(
        model=model,
        algorithm=RLGenExplainer(
            epochs=epochs, 
            lr=lr, 
            candidate_set=candidate_set, 
            ppi_nodeid_index_dict=nodeid_index_dict,
            validity_args=ppi_edges, 
            device=device,
            initial_node_id=initial_node_id_list,
            num_entity=num_entity
        ).to(device),
        explanation_type='generative',
        model_config=dict(mode='binary_classification', task_level='graph', return_type='probs')
    )
    
    # Move graph generator to the appropriate device
    explainer.algorithm.graph_generator.to(device)
    
    return explainer


def generate_best_graph(explainer, target_class=1, device='cuda', nodeid_index_dict=None,
                      ppi_edges=None, num_samples=5, num_nodes=100, max_steps=50,
                      origin_output_path='./plot', origin_output_com_path='./plot_com',
                      conn_output_path='./plot_conn', conn_output_com_path='./plot_conn_com', mask_plot=False):
    """
    Generates multiple candidate explanation graphs and returns the best one.
    
    Args:
        explainer: Configured explainer object
        target_class: Target class to explain (default: 1)
        device: Computation device (CPU or CUDA)
        num_samples: Number of graphs to sample (default: 5)
        num_nodes: Maximum number of nodes per graph (default: 100)
        max_steps: Maximum steps for graph generation (default: 50)
        origin_output_path: Path to save original graph visualization (default: './plot')
        origin_output_com_path: Path to save original completed graph visualization (default: './plot_origin_com')
        conn_output_path: Path to save connected graph visualization (default: './plot_conn')
        conn_output_com_path: Path to save connected completed graph visualization (default: './plot_conn_com')
        
    Returns:
        The best connected explanation graph
    """
    # Create empty input tensors (required by explainer API)
    x = torch.tensor([]).to(device)
    edge_index = torch.tensor([[], []]).to(device)
    
    # Generate explanation
    explanation = explainer(x, edge_index, for_class=target_class)
    explanation_set = explanation.explanation_set
    
    # Sample multiple graphs
    sampled_graphs = explanation_set.sample(num_samples=num_samples, 
                                           num_nodes=num_nodes, 
                                           max_steps=max_steps)
    
    # Evaluate each graph
    probabilities = []
    model = explainer.model
    
    for sampled_graph in sampled_graphs:
        sampled_graph = sampled_graph.to(device)
        x = sampled_graph.x
        edge_index = sampled_graph.edge_index
        
        with torch.no_grad():
            # Add better handling for empty edge_index
            if edge_index.numel() == 0 or not hasattr(edge_index, 'shape') or len(edge_index.shape) < 2 or edge_index.shape[1] == 0:
                print(f"⚠️ Warning: Skipping graph with empty or invalid edge_index: {edge_index}")
                # import pdb; pdb.set_trace()  # Break here for debugging
                probabilities.append(0)
                continue
                
            z = model.act(model.encoder(x, edge_index))
            num_nodes = z.size(0)
            batch = torch.zeros(num_nodes, dtype=torch.long, device=z.device)
            z = global_mean_pool(z, batch)
            output = model.classifier(z)
            probabilities_vector = torch.softmax(output, dim=1).squeeze()
            probability_of_target = probabilities_vector[target_class]
        
        probabilities.append(probability_of_target)
    
    # Select best graph
    best_graph_index = probabilities.index(max(probabilities))
    best_sampled_graph = sampled_graphs[best_graph_index]
    # Build up the best_sampled_graph (best_sampled_graph.x.shape[0]) and node_id dict
    best_sampled_graph_bmgc_id_list = best_sampled_graph.node_id
    best_sampled_graph_index_origin_index_dict = {}
    for i, node_id in enumerate(best_sampled_graph_bmgc_id_list):
        best_sampled_graph_index_origin_index_dict[i] = nodeid_index_dict[node_id]
    # Make the best_sampled_graph edge completed by adding existed ppi_edges
    completed_best_sampled_graph = completion_additional_ppi_edges(
        best_graph=best_sampled_graph,
        ppi_edges=ppi_edges,
        best_graph_index_origin_index_dict=best_sampled_graph_index_origin_index_dict
    )
    
    # Remove isolated nodes for better visualization
    connected_best_graph = remove_isolated_nodes(best_sampled_graph)
    
    # Build up the best_connected_graph (best_sampled_graph.x.shape[0]) and node_id dict
    best_connected_graph_bmgc_id_list = connected_best_graph.node_id
    best_connected_graph_index_origin_index_dict = {}
    for i, node_id in enumerate(best_connected_graph_bmgc_id_list):
        best_connected_graph_index_origin_index_dict[i] = nodeid_index_dict[node_id]
    # Make the best_connected_graph edge completed by adding existed ppi_edges
    completed_best_connected_graph = completion_additional_ppi_edges(
        best_graph=connected_best_graph,
        ppi_edges=ppi_edges,
        best_graph_index_origin_index_dict=best_connected_graph_index_origin_index_dict
    )

    # Print statistics
    print(f"Original graph: {best_sampled_graph.x.shape[0]} nodes, "
          f"{best_sampled_graph.edge_index.shape[1]} edges")
    print(f"Original completed graph: {completed_best_sampled_graph.x.shape[0]} nodes, "
          f"{completed_best_sampled_graph.edge_index.shape[1]} edges, "
          f"{completed_best_sampled_graph.additional_edge_index.shape[1]} additional edges")
    print(f"Connected graph: {connected_best_graph.x.shape[0]} nodes, "
          f"{connected_best_graph.edge_index.shape[1]} edges")
    print(f"Completed graph: {completed_best_connected_graph.x.shape[0]} nodes, "  
          f"{completed_best_connected_graph.edge_index.shape[1]} edges,"
          f" {completed_best_connected_graph.additional_edge_index.shape[1]} additional edges")
    
    if mask_plot == False:
        # Visualize the original graph
        explanation.visualize_explanation_graph(best_sampled_graph,
                                              path=origin_output_path, 
                                              backend='networkx')
        # Visualize the original completed graph
        explanation.visualize_explanation_completion_graph(completed_best_sampled_graph,
                                                path=origin_output_com_path, 
                                                backend='networkx')
        # Visualize the connected graph
        explanation.visualize_explanation_graph(connected_best_graph, 
                                              path=conn_output_path, 
                                              backend='networkx')
        # Visualize the completed graph
        explanation.visualize_explanation_completion_graph(completed_best_connected_graph, 
                                              path=conn_output_com_path, 
                                              backend='networkx')
    
    return best_sampled_graph, completed_best_sampled_graph, connected_best_graph, completed_best_connected_graph


def kg_data(args, device):
    print('Loading kg data...')
    # Load DTI data
    xAll = np.load('./BMG/DTI_data/dti_bmgc_omics.npy')  # Omics Input
    xAll = torch.from_numpy(xAll).to(device)
    # tailor to pretrain data
    omics_node_index_df = pd.read_csv('./BMG/Pretrain_data/omics_nodes_index.csv')
    omics_node_index_list = omics_node_index_df['Index'].tolist()
    omics_node_index = torch.tensor(omics_node_index_list, dtype=torch.long).to(device)
    xAll_omics = xAll[:, omics_node_index]
    # other data just import from pretrain data
    name_embeddings = np.load('./BMG/Pretrain_data/x_name_emb.npy').reshape(-1, args.lm_emb_dim) # temporary using Pretrain data/ should use DTI
    name_embeddings = torch.from_numpy(name_embeddings).float().to(device)
    desc_embeddings = np.load('./BMG/Pretrain_data/x_desc_emb.npy').reshape(-1, args.lm_emb_dim) # temporary using Pretrain data/ should use DTI
    desc_embeddings = torch.from_numpy(desc_embeddings).float().to(device)
    all_edge_index = np.load('./BMG/Pretrain_data/edge_index.npy')
    internal_edge_index = np.load('./BMG/Pretrain_data/internal_edge_index.npy')
    ppi_edge_index = np.load('./BMG/Pretrain_data/ppi_edge_index.npy')
    all_edge_index = torch.from_numpy(all_edge_index).long().to(device)
    internal_edge_index = torch.from_numpy(internal_edge_index).long().to(device)
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long().to(device)
    # Create edge validation set
    print('Creating edge validation set...')
    edge_array = ppi_edge_index.cpu().numpy().T
    reverse_edge_array = edge_array[:, [1, 0]]
    all_edges_array = np.vstack([edge_array, reverse_edge_array])
    ppi_edges = set(map(tuple, all_edges_array))
    return xAll_omics, omics_node_index_df, name_embeddings, desc_embeddings, all_edge_index, internal_edge_index, ppi_edge_index, ppi_edges


def bmgc_pt_id_to_hgnc(bmgc_id_list, bmgc_protein_df):
    """
    Convert a list of BioMedGraphica IDs to their corresponding HGNC symbols.
    
    Args:
        bmgc_id_list (list): List of BioMedGraphica IDs
        bmgc_protein_df (pd.DataFrame): DataFrame with BioMedGraphica IDs and HGNC symbols
        
    Returns:
        tuple: (
            dict: Dictionary mapping each BioMedGraphica ID to its list of HGNC symbols,
            list: Combined list of all HGNC symbols
        )
    """
    # Ensure bmgc_id_list is actually a list
    if not isinstance(bmgc_id_list, list):
        bmgc_id_list = [bmgc_id_list]
    
    results = {}
    all_hgnc_symbols = []
    
    for bmgc_id in bmgc_id_list:
        # Filter the DataFrame for the given BioMedGraphica ID
        filtered_df = bmgc_protein_df[bmgc_protein_df['BioMedGraphica_Conn_ID'] == bmgc_id]
        
        # Skip if no match found
        if filtered_df.empty:
            results[bmgc_id] = []
            continue
            
        # Get the HGNC symbols
        hgnc_value = filtered_df['HGNC_Symbol'].values[0]
        
        # Skip if HGNC symbol is NaN
        if pd.isna(hgnc_value):
            results[bmgc_id] = []
            continue
        
        # Process valid HGNC symbols
        hgnc_list = list(set(hgnc_value.split(';')))
        hgnc_list = [hgnc.strip() for hgnc in hgnc_list if hgnc.strip() != '']
        
        results[bmgc_id] = hgnc_list
        all_hgnc_symbols.extend(hgnc_list)
    
    # Remove duplicates from the combined list
    all_hgnc_symbols = list(set(all_hgnc_symbols))
    
    return results, all_hgnc_symbols


def hgnc_to_bmgc_pt_id(hgnc_list, bmgc_protein_df):
    """
    Convert a list of HGNC symbols to their corresponding BioMedGraphica IDs.
    
    Args:
        hgnc_list (list): List of HGNC symbols
        bmgc_protein_df (pd.DataFrame): DataFrame with BioMedGraphica IDs and HGNC symbols
        
    Returns:
        tuple: (
            dict: Dictionary mapping each HGNC symbol to its list of BioMedGraphica IDs,
            list: Combined list of all BioMedGraphica IDs
        )
    """
    # Ensure hgnc_list is actually a list
    if not isinstance(hgnc_list, list):
        hgnc_list = [hgnc_list]
    
    results = {}
    all_bmgc_ids = []
    
    for hgnc in hgnc_list:
        # Filter the DataFrame for the given HGNC symbol
        filtered_df = bmgc_protein_df[bmgc_protein_df['HGNC_Symbol'] == hgnc]
        
        # Skip if no match found
        if filtered_df.empty:
            results[hgnc] = []
            continue
        
        # Get the BioMedGraphica IDs
        bmgc_value = filtered_df['BioMedGraphica_Conn_ID'].values[0]
        
        # Skip if BioMedGraphica ID is NaN
        if pd.isna(bmgc_value):
            results[hgnc] = []
            continue
        
        # Process valid BioMedGraphica IDs
        bmgc_list = list(set(bmgc_value.split(';')))
        bmgc_list = [bmgc.strip() for bmgc in bmgc_list if bmgc.strip() != '']
        
        results[hgnc] = bmgc_list
        all_bmgc_ids.extend(bmgc_list)
    
    # Remove duplicates from the combined list
    all_bmgc_ids = list(set(all_bmgc_ids))
    
    return results, all_bmgc_ids


def convert_edges_to_hgnc_symbols(edge_tensor, index_hgnc_dict, index_nodeid_dict):
    """
    Convert edge tensor indices to HGNC symbol pairs.
    
    Parameters:
    -----------
    edge_tensor : torch.Tensor
        A tensor of shape [2, E] where E is the number of edges.
        First row contains source indices, second row contains target indices.
    index_hgnc_dict : dict
        Dictionary mapping node indices to HGNC symbols.
    index_nodeid_dict : dict
        Dictionary mapping node indices to their original node IDs.
        
    Returns:
    --------
    tuple (list of tuples, list of strings)
        - List of (source_hgnc, target_hgnc) pairs representing edges
        - List of text descriptions of relationships in "A -> B" format
    """
    hgnc_edges = []
    edge_text_descriptions = []
    
    # Convert tensor to numpy for easier processing if it's not already
    if torch.is_tensor(edge_tensor):
        edge_array = edge_tensor.cpu().numpy()
    else:
        edge_array = edge_tensor
    
    # Check if edge_array is empty
    if edge_array.size == 0 or edge_array.shape[1] == 0:
        return [], []
    
    # Process each edge
    for e in range(edge_array.shape[1]):
        source_idx = edge_array[0, e]
        target_idx = edge_array[1, e]
        
        # Get HGNC symbols if available in the dictionary, fallback to nodeid if available, then unknown
        source_idx_val = source_idx.item() if torch.is_tensor(source_idx) else source_idx
        target_idx_val = target_idx.item() if torch.is_tensor(target_idx) else target_idx
        
        # First try to get HGNC symbol
        source_hgnc = index_hgnc_dict.get(source_idx_val)
        if source_hgnc is None:
            # If not in HGNC dict, try to get the node ID
            source_hgnc = index_nodeid_dict.get(source_idx_val, f"Unknown-{source_idx_val}")
        
        target_hgnc = index_hgnc_dict.get(target_idx_val)
        if target_hgnc is None:
            # If not in HGNC dict, try to get the node ID
            target_hgnc = index_nodeid_dict.get(target_idx_val, f"Unknown-{target_idx_val}")
        
        # Extract the actual symbol from list format if needed
        if isinstance(source_hgnc, list):
            source_hgnc = source_hgnc[0] if source_hgnc else f"Unknown-{source_idx_val}"
        
        if isinstance(target_hgnc, list):
            target_hgnc = target_hgnc[0] if target_hgnc else f"Unknown-{target_idx_val}"
        
        # Add to list of tuples (as plain strings, not lists)
        hgnc_edges.append((source_hgnc, target_hgnc))
        
        # Create text description in "A -> B" format (without list notation)
        edge_text_description = f"{source_hgnc} -> {target_hgnc}"
        # # Create text description in "(A, B)" format (without list notation)
        # edge_text_description = f"({source_hgnc}, {target_hgnc})"
        edge_text_descriptions.append(edge_text_description)
    
    return hgnc_edges, edge_text_descriptions


def convert_protein_relations_to_edges(protein_relations):
    """
    Convert protein relationship strings like "MT-CO1 -> MT-CO2" to tuple format
    
    Args:
        protein_relations (list): List of strings like "protein1 -> protein2"
        
    Returns:
        list: List of tuples like [(MT-CO1, MT-CO2), (MT-CO1, MT-CO3), ...]
    """
    # Extract all edges as tuples with original protein names
    edges = []
    for relation in protein_relations:
        if "->" in relation:
            source, target = relation.split("->")
            source = source.strip()
            target = target.strip()
            
            # Add edge as tuple
            edges.append((source, target))
    
    return edges

def find_protein_relationships(hgnc_symbols, bmgc_protein_df, bmgc_relation_df):
    """
    Find relationships between a list of proteins based on HGNC symbols.
    
    Parameters:
    -----------
    hgnc_symbols : list
        List of HGNC symbols to find relationships between
    bmgc_protein_df : pandas DataFrame
        DataFrame containing BioMedGraphica_Conn_ID and HGNC_Symbol columns
    bmgc_relation_df : pandas DataFrame
        DataFrame containing BMGC_From_ID and BMGC_To_ID columns
        
    Returns:
    --------
    tuple (pandas DataFrame, list)
        - DataFrame with source_symbol, target_symbol and their relationship
        - List of text descriptions of relationships in "A -> B" format
    """
    # Filter the protein DataFrame to only include the proteins we care about
    filtered_proteins = bmgc_protein_df[bmgc_protein_df['HGNC_Symbol'].isin(hgnc_symbols)]
    
    # Create a mapping from HGNC symbol to BMGC ID
    hgnc_to_bmgc = dict(zip(filtered_proteins['HGNC_Symbol'], filtered_proteins['BioMedGraphica_Conn_ID']))
    bmgc_to_hgnc = dict(zip(filtered_proteins['BioMedGraphica_Conn_ID'], filtered_proteins['HGNC_Symbol']))
    
    # Get all BMGC IDs of our proteins
    bmgc_ids = list(hgnc_to_bmgc.values())
    
    # Filter the relationship DataFrame to only include relationships between our proteins
    protein_relations = bmgc_relation_df[
        bmgc_relation_df['BMGC_From_ID'].isin(bmgc_ids) & 
        bmgc_relation_df['BMGC_To_ID'].isin(bmgc_ids)
    ]
    
    # Map the BMGC IDs back to HGNC symbols
    result_data = []
    text_descriptions = []
    
    for _, row in protein_relations.iterrows():
        source_bmgc = row['BMGC_From_ID']
        target_bmgc = row['BMGC_To_ID']
        
        if source_bmgc in bmgc_to_hgnc and target_bmgc in bmgc_to_hgnc:
            source_symbol = bmgc_to_hgnc[source_bmgc]
            target_symbol = bmgc_to_hgnc[target_bmgc]
            
            # Create text description
            text_description = f"{source_symbol} -> {target_symbol}"
            text_descriptions.append(text_description)
            
            # If relation_type column exists, include it in the description and data
            relation_info = {
                'source_symbol': source_symbol,
                'target_symbol': target_symbol
            }
            
            # Add relation type if it exists in the DataFrame
            if 'relation_type' in bmgc_relation_df.columns:
                relation_type = row['relation_type']
                relation_info['relation_type'] = relation_type
                text_descriptions[-1] = f"{source_symbol} -{relation_type}-> {target_symbol}"
                
            result_data.append(relation_info)
    
    # Create a DataFrame from the results
    result_df = pd.DataFrame(result_data)
    
    return result_df, text_descriptions

def convert_to_original_index(edge_index, index_mapping):
    """
    Converts edge indices in a graph to their original indices using a mapping dictionary.
    
    Args:
        edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges]
        index_mapping (dict): Dictionary mapping local indices to original indices
        
    Returns:
        torch.Tensor: Edge index tensor with converted original indices
    """
    original_edge_index = torch.zeros_like(edge_index)
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        original_src_idx = index_mapping.get(src_idx, src_idx)
        original_dst_idx = index_mapping.get(dst_idx, dst_idx)
        original_edge_index[0, i] = original_src_idx
        original_edge_index[1, i] = original_dst_idx
    return original_edge_index


def visualize_hgnc_edges(
    hgnc_edges,
    hgnc_additional_edges,
    output_path,
    output_com_path,
    node_labels=None
):
    """
    Visualizes a graph with HGNC protein symbols as nodes, showing both primary and additional edges.
    
    Args:
        hgnc_edges (list): List of tuples containing (source_hgnc, target_hgnc) for primary edges
        hgnc_additional_edges (list): List of tuples containing (source_hgnc, target_hgnc) for additional edges
        output_path (str): Path to save the visualization
        output_com_path (str): Path to save the completed visualization
        node_labels (dict, optional): Optional dictionary to override node labels
    
    Returns:
        The graph object for further manipulation if needed
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from math import sqrt
    from matplotlib.lines import Line2D
    
    # Create networkx graph
    g = nx.DiGraph()
    
    # Collect all unique protein names
    all_proteins = set()
    for src, dst in hgnc_edges:
        all_proteins.add(src)
        all_proteins.add(dst)
    
    for src, dst in hgnc_additional_edges:
        all_proteins.add(src)
        all_proteins.add(dst)
    
    # Add nodes to graph with HGNC symbols as labels
    for protein in all_proteins:
        label = protein if node_labels is None else node_labels.get(protein, protein)
        g.add_node(protein, label=label)
    
    # Add original edges (will be red)
    for src, dst in hgnc_edges:
        g.add_edge(src, dst, color='red', weight=1)
    
    # Add additional edges (will be gray)
    for src, dst in hgnc_additional_edges:
        g.add_edge(src, dst, color='gray', weight=0.7)  # Slightly less weight for additional edges
    
    # Draw the graph
    plt.figure(figsize=(14, 10))
    
    # Use a layout that's good for biological networks
    # Try different layouts and choose the best one
    if len(g.nodes()) > 100:
        pos = nx.spring_layout(g, k=0.5, iterations=50, seed=42)  # For larger graphs
    else:
        pos = nx.kamada_kawai_layout(g)  # Often better for smaller biological networks
    
    node_size = 1000
    
    # Draw edges with appropriate colors
    edge_colors = [g[u][v]['color'] for u, v in g.edges()]
    edge_weights = [g[u][v]['weight'] for u, v in g.edges()]
    
    # Create custom arrows for more control over appearance
    ax = plt.gca()
    for u, v, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[v],  # Arrow destination
            xytext=pos[u],  # Arrow source
            arrowprops=dict(
                arrowstyle="->",
                color=data['color'],
                alpha=data['weight'],
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )
    
    # Draw nodes with a slightly larger size for protein labels
    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size, 
                                  node_color='lightblue', edgecolors='black')
    
    # Draw labels with a white background for better readability
    labels = {node: data['label'] for node, data in g.nodes(data=True)}
    text_objects = nx.draw_networkx_labels(g, pos, labels, font_size=9, font_weight='bold')
    
    # Add a white background to the text for better readability
    for _, text in text_objects.items():
        text.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
    
    # Add a legend
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Inferred Signaling Edges'),
        Line2D([0], [0], color='gray', lw=2, label='Additional Supporting Edges')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add title
    plt.title("Protein Signaling Network", fontsize=14)
    
    plt.tight_layout()
    plt.axis('off')
    
    # Save the image
    plt.savefig(output_com_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph visualization saved to {output_com_path}")
    
    return g