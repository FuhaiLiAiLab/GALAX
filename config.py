import torch
import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    ########################################################################################################################################################################
    # ### pre-training loading parameters
    parser.add_argument('--layer', nargs='?', default='gat', help='GNN layer, (default: gcn)')
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

    ########################################################################################################################################################################
    # ### downstream task parameters
    parser.add_argument('--text_lm_model_path', nargs='?', default='dmis-lab/biobert-v1.1', help='Path to the pretrained language model. (default: dmis-lab/biobert-v1.1)')
    parser.add_argument('--train_text', default=False, help='Whether to train the text encoder. (default: False)')
    parser.add_argument('--task', nargs='?', default='class', help='Task for training downstream tasks. (default: class)')
    parser.add_argument('--name', nargs='?', default='DepMap', help='Name for dataset.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for classification. (default: 2)')
    parser.add_argument('--train_weight_decay', type=float, default=1e-15, help='Weight decay for Adam optimizer. (default: 1e-15)')
    parser.add_argument('--train_encoder_dropout', type=float, default=0.1, help='Dropout probability of encoder. (default: 0.1)')
    parser.add_argument('--train_layer', nargs='?', default='gat', help='GNN layer, (default: gcn)')
    parser.add_argument('--train_internal_encoder_layers', type=int, default=3, help='Number of layers for internal encoder. (default: 3)')
    parser.add_argument('--train_encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--pre_input_dim', type=int, default=8, help='Input feature dimension for pretraining. (default: 8)') # should be same as hidden_channels
    parser.add_argument('--train_fusion_dim', type=int, default=1, help='Fusion feature dimension for training. (default: 1)') # fused_dim, due to using internal_emb + x, should be same as omics_feature_dim (used for fusion/internal_encoder/pre_transformer)
    parser.add_argument('--train_hidden_dim', type=int, default=8, help='Hidden feature dimension for training. (default: 8)') # convert the num_omic_feature to hidden_dim (dim for graph encoder if used)
    parser.add_argument('--train_output_dim', type=int, default=8, help='Output feature dimension for training. (default: 8)')
    parser.add_argument('--train_linear_input_dim', type=int, default=8, help='Input feature dimension for training. (default: 16)') # should be same as the train_output_dim
    parser.add_argument('--train_linear_hidden_dim', type=int, default=32, help='Hidden feature dimension for training. (default: 32)')
    parser.add_argument('--train_linear_output_dim', type=int, default=16, help='Output feature dimension for training. (default: 16)')

    ########################################################################################################################################################################
    # ### GALAX task parameters
    parser.add_argument('--graph_foundation_model_path', type=str, default='./checkpoints/graph_foundation/best_combined_model.pt', help='Path to the graph foundation model for GALAX task.')
    parser.add_argument('--output_result_dir', type=str, default='./TargetQA_Results/', help='Output directory for GALAX task.')


    return parser.parse_args()