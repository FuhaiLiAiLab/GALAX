import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

# custom modules
from MOTASG_Foundation.utils import tab_printer
from MOTASG_Foundation.model import MOTASG_Foundation, DegreeDecoder, EdgeDecoder, GNNEncoder
from MOTASG_Foundation.mask import MaskEdge

# custom dataloader
from GeoDataLoader.read_geograph import read_index_komask_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

from MOTASG_Foundation.lm_model import TextEncoder
from MOTASG_Foundation.downstream import MOTASG_Reg, DSGNNEncoder


# Check device and select GPU with most available memory
def get_gpu_with_max_free_memory():
    """
    Returns the GPU device ID with the most available memory.
    Returns 'cpu' if no CUDA devices are available.
    """
    if not torch.cuda.is_available():
        return 'cpu'
    
    try:
        # Try to use nvidia-smi to get memory info
        import subprocess
        import re
        
        # Run nvidia-smi to get memory usage info
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                                        encoding='utf-8')
        # Parse the output to get free memory values
        free_memory = [int(x) for x in result.strip().split('\n')]
        
        # Get the GPU with maximum free memory
        max_free_device_id = free_memory.index(max(free_memory))
        print(f"Selected GPU {max_free_device_id} with {max(free_memory)} MB free memory")
        return f'cuda:{max_free_device_id}'
    
    except (subprocess.CalledProcessError, FileNotFoundError, ImportError) as e:
        # If nvidia-smi fails, try to use torch's built-in function (less accurate)
        print(f"Warning: nvidia-smi failed ({e}). Using first available GPU.")
        try:
            # Get device count and find one with most memory
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return 'cpu'
            elif device_count == 1:
                return 'cuda:0'
            
            # Try to find the GPU with most free memory
            max_free = 0
            max_device = 0
            for device_id in range(device_count):
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                free_mem = torch.cuda.memory_reserved(device_id) - torch.cuda.memory_allocated(device_id)
                if free_mem > max_free:
                    max_free = free_mem
                    max_device = device_id
            
            print(f"Selected GPU {max_device} based on torch memory stats")
            return f'cuda:{max_device}'
        except:
            # Fall back to the first GPU if all else fails
            return 'cuda:0'

def learning_rate_schedule(args, dl_input_num, iteration_num, e1, e2, e3, e4):
    epoch_iteration = int(dl_input_num / args.train_batch_size)
    l1 = (args.train_lr - args.train_lr2) / (e1 * epoch_iteration)
    l2 = (args.train_lr2 - args.train_lr3) / (e2 * epoch_iteration)
    l3 = (args.train_lr3 - args.train_lr4) / (e3 * epoch_iteration)
    l4 = (args.train_lr4 - args.train_lr5) / (e4 * epoch_iteration)
    l5 = args.train_lr5
    if iteration_num <= (e1 * epoch_iteration):
        learning_rate = args.train_lr - iteration_num * l1
    elif iteration_num <= (e1 + e2) * epoch_iteration:
        learning_rate = args.train_lr2 - (iteration_num - e1 * epoch_iteration) * l2
    elif iteration_num <= (e1 + e2 + e3) * epoch_iteration:
        learning_rate = args.train_lr3 - (iteration_num - (e1 + e2) * epoch_iteration) * l3
    elif iteration_num <= (e1 + e2 + e3 + e4) * epoch_iteration:
        learning_rate = args.train_lr4 - (iteration_num - (e1 + e2 + e3) * epoch_iteration) * l4
    else:
        learning_rate = l5
    print('-------LEARNING RATE: ' + str(learning_rate) + '-------' )
    return learning_rate

def filter_edges(edge_index, knockout_indices):
        """
        Efficiently filter out edges that contain knockout nodes.
        
        Args:
            edge_index: Edge index tensor of shape [2, num_edges]
            knockout_indices: Tensor of node indices to knock out
            
        Returns:
            Filtered edge_index with knockout edges removed
        """
        # Create a boolean mask for all nodes (False for knockout nodes)
        max_node_idx = torch.max(edge_index).item() + 1
        node_mask = torch.ones(max_node_idx, dtype=torch.bool, device=edge_index.device)
        node_mask[knockout_indices] = False
        
        # Check if both source and target nodes are not in knockout indices
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]
        edge_mask = node_mask[src_nodes] & node_mask[tgt_nodes]
        
        # Return only the edges where both endpoints are not knockout nodes
        return edge_index[:, edge_mask]

def apply_knockouts(x, ko_mask, internal_edge_index, ppi_edge_index, edge_index, device):
    """
    Applies knockouts to node features and removes edges connected to knockout nodes.
    
    Args:
        x: Tensor of node features [num_nodes, feature_dim]
        ko_mask: Tensor of node indices to knock out
        internal_edge_index: Internal edge index tensor [2, num_edges]
        ppi_edge_index: PPI edge index tensor [2, num_edges]
        edge_index: All edge index tensor [2, num_edges]
        device: Device to place tensors on
        
    Returns:
        x: Node features with knockouts applied
        internal_edge_index: Filtered internal edge index
        ppi_edge_index: Filtered PPI edge index
        edge_index: Filtered all edge index
    """
    print("Knocking out nodes...")
    if ko_mask is None or ko_mask.numel() == 0:
        return x, internal_edge_index, ppi_edge_index, edge_index
    
    # Create a mask for features (1 for all nodes except knockout nodes)
    feature_mask = torch.ones(x.shape[0], 1, device=device)
    
    # Set values at knockout indices to 0
    feature_mask[ko_mask] = 0
    
    # Apply mask to node features
    x = x * feature_mask
    
    # Filter all edge indices
    internal_edge_index_filtered = filter_edges(internal_edge_index, ko_mask)
    ppi_edge_index_filtered = filter_edges(ppi_edge_index, ko_mask)
    edge_index_filtered = filter_edges(edge_index, ko_mask)

    # Log the changes (optional)
    num_internal_removed = internal_edge_index.shape[1] - internal_edge_index_filtered.shape[1]
    num_ppi_removed = ppi_edge_index.shape[1] - ppi_edge_index_filtered.shape[1]
    num_all_removed = edge_index.shape[1] - edge_index_filtered.shape[1]
    
    print(f"Applied knockout to {ko_mask.shape[0]} nodes")
    print(f"Removed edges - internal: {num_internal_removed}, ppi: {num_ppi_removed}, all: {num_all_removed}")
    
    return x, internal_edge_index_filtered, ppi_edge_index_filtered, edge_index_filtered


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

    internal_graph_encoder = DSGNNEncoder(args.train_fusion_dim, args.train_fusion_dim, args.train_fusion_dim,
                            num_layers=args.train_internal_encoder_layers, dropout=args.train_encoder_dropout,
                            bn=args.bn, layer=args.train_layer, activation=args.encoder_activation)

    graph_encoder = DSGNNEncoder(args.train_fusion_dim, args.train_hidden_dim, args.train_output_dim,
                    num_layers=args.train_encoder_layers, dropout=args.train_encoder_dropout,
                    bn=args.bn, layer=args.train_layer, activation=args.encoder_activation)

    model = MOTASG_Reg(text_input_dim=args.lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    pre_input_dim=args.pre_input_dim,
                    fusion_dim=args.train_fusion_dim,
                    internal_graph_output_dim=args.train_fusion_dim, # internal graph encoder output dim
                    graph_output_dim=args.train_output_dim, # graph encoder output dim
                    linear_input_dim=args.train_linear_input_dim,
                    linear_hidden_dim=args.train_linear_hidden_dim,
                    linear_output_dim=args.train_linear_output_dim,
                    num_entity=args.num_entity,
                    text_encoder=text_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder).to(device)
    return model


def write_best_model_info(path, max_test_pearson_id, epoch_loss_list, epoch_pearson_list, test_loss_list, test_pearson_list):
    best_model_info = (
        f'\n-------------BEST TEST PEARSON MODEL ID INFO: {max_test_pearson_id} -------------\n'
        '--- TRAIN ---\n'
        f'BEST MODEL TRAIN LOSS: {epoch_loss_list[max_test_pearson_id - 1]}\n'
        f'BEST MODEL TRAIN PEARSON: {epoch_pearson_list[max_test_pearson_id - 1]}\n'
        '--- TEST ---\n'
        f'BEST MODEL TEST LOSS: {test_loss_list[max_test_pearson_id - 1]}\n'
        f'BEST MODEL TEST PEARSON: {test_pearson_list[max_test_pearson_id - 1]}\n'
    )
    with open(os.path.join(path, 'best_model_info.txt'), 'w') as file:
        file.write(best_model_info)
    
    # Save all metrics to CSV files
    # Training metrics
    train_metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(epoch_loss_list) + 1)),
        'loss': epoch_loss_list,
        'pearson': epoch_pearson_list
    })
    train_metrics_df.to_csv(os.path.join(path, 'training_metrics.csv'), index=False)
    
    # Testing metrics
    test_metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(test_loss_list) + 1)),
        'loss': test_loss_list,
        'pearson': test_pearson_list
    })
    test_metrics_df.to_csv(os.path.join(path, 'testing_metrics.csv'), index=False)


def train_model(train_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, pretrain_model, model, device, optimizer, args):
    batch_loss = 0
    for batch_idx, data in enumerate(train_dataset_loader):
        optimizer.zero_grad()
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        ko_mask = Variable(data.ko_mask, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)

        # import pdb; pdb.set_trace()
        # Apply knockout to nodes and edges
        x, internal_edge_index, ppi_edge_index, edge_index = apply_knockouts(x, ko_mask, internal_edge_index, ppi_edge_index, edge_index, device)

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
        # import pdb; pdb.set_trace()
        # Continue training the model
        output = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, name_embeddings, desc_embeddings, current_cell_num, ko_mask)
        output = output.squeeze(-1)
        loss = model.loss(output, label)
        loss.backward()
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', output)
        print('Loss: ', loss.item())
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        # # check pretrain model parameters
        # state_dict = pretrain_model.internal_encoder.state_dict()
        # print(state_dict['convs.1.lin.weight'])
        # print(model.embedding.weight.data)
    torch.cuda.empty_cache()
    return model, batch_loss, output


def test_model(test_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, pretrain_model, model, device, args):
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_loss = 0
    for batch_idx, data in enumerate(test_dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        ko_mask = Variable(data.ko_mask, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)

        # Apply knockout to nodes and edges
        x, internal_edge_index, ppi_edge_index, edge_index = apply_knockouts(x, ko_mask, internal_edge_index, ppi_edge_index, edge_index, device)
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
            output = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, name_embeddings, desc_embeddings, current_cell_num, ko_mask)
            output = output.squeeze(-1)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', output)
        print('Loss: ', loss.item())

    return model, batch_loss, output


def stratified_sample_data(data, sample_ratio=1.0, random_seed=42):
    """
    Sample data from a regression dataset with continuous scores.
    
    Args:
        data: numpy array with complex structure where data[i,0] is ID, data[i,2] is score
        sample_ratio: percentage of data to sample (default: 1.0, which keeps all data)
        random_seed: random seed for reproducibility
        
    Returns:
        sampled_data: numpy array with sampled data
    """
    if sample_ratio >= 1.0:
        return data
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    original_sample_size = len(data)
    
    # With continuous scores, we can simply random sample the data
    # rather than stratified sampling (which requires categorical variables)
    n_samples = max(1, int(original_sample_size * sample_ratio))
    sampled_indices = np.random.choice(original_sample_size, n_samples, replace=False)
    sampled_data = data[sampled_indices]
    
    # For analysis, we can report score distribution information before and after sampling
    before_scores = np.array([row[2] for row in data])  # Assuming scores are in the 3rd position [2]
    after_scores = np.array([row[2] for row in sampled_data])
    
    print(f"Score statistics before sampling:")
    print(f"  Mean: {before_scores.mean():.4f}")
    print(f"  Median: {np.median(before_scores):.4f}")
    print(f"  Min: {before_scores.min():.4f}")
    print(f"  Max: {before_scores.max():.4f}")
    print(f"  Std: {before_scores.std():.4f}")
    
    print(f"Score statistics after sampling:")
    print(f"  Mean: {after_scores.mean():.4f}")
    print(f"  Median: {np.median(after_scores):.4f}")
    print(f"  Min: {after_scores.min():.4f}")
    print(f"  Max: {after_scores.max():.4f}")
    print(f"  Std: {after_scores.std():.4f}")
    
    print(f'Original data shape: {original_sample_size} samples')
    print(f'Sampled data shape: {sampled_data.shape[0]} samples')
    
    return sampled_data


def train(args, pretrain_model, device):
    ### Load data
    print('--- LOADING TRAINING FILES ... ---')
    xAll = np.load('./BMG/CRISPR-Graph/dti_crispr_rna_samples_omics.npy')
    yTr = np.load('./BMG/CRISPR-Graph/train_crispr_score.npy', allow_pickle=True)
    yTe = np.load('./BMG/CRISPR-Graph/test_crispr_score.npy', allow_pickle=True)

    # import pdb; pdb.set_trace()

    # Sampling ratio of yTr for training with random seed for selecting the index
    sample_ratio = args.train_sample_ratio if hasattr(args, 'train_sample_ratio') else 1.0
    random_seed = args.training_sample_random_seed if hasattr(args, 'random_seed') else 42

    if sample_ratio < 1.0:
        print(f'Stratified sampling {sample_ratio * 100}% of training data with random seed {random_seed}')
        # Use the new stratified sampling function
        yTr = stratified_sample_data(yTr, sample_ratio, random_seed)
        yTe = stratified_sample_data(yTe, sample_ratio, random_seed)

    # Use the first column as the indices and second column as the labels
    yTr_index = yTr[:, 0].astype(np.int32).reshape(-1, 1)
    yTr_ko_mask = yTr[:, 1]
    yTr_label = yTr[:, 2].astype(np.float32).reshape(-1, 1)
    yTe_index = yTe[:, 0].astype(np.int32).reshape(-1, 1)
    yTe_ko_mask = yTe[:, 1]
    yTe_label = yTe[:, 2].astype(np.float32).reshape(-1, 1)
    yAll = np.concatenate((yTr_label, yTe_label), axis=0)
    yAll_index = np.concatenate((yTr_index, yTe_index), axis=0)
    yAll_ko_mask = np.concatenate((yTr_ko_mask, yTe_ko_mask), axis=0)

    # Load edge_index
    all_edge_index = np.load('./BMG/CRISPR-Graph/edge_index.npy')
    internal_edge_index = np.load('./BMG/CRISPR-Graph/internal_edge_index.npy')
    ppi_edge_index = np.load('./BMG/CRISPR-Graph/ppi_edge_index.npy')
    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()
    # Load textual embeddings
    if args.train_text:
        # Use language model to embed the name and description
        s_name_df = pd.read_csv('./BMG/Pretrain_data/bmgc_omics_name.csv')
        s_desc_df = pd.read_csv('./BMG/Pretrain_data/bmgc_omics_desc.csv')
        name_sentence_list = s_name_df['Names_and_IDs'].tolist()
        name_sentence_list = [str(name) for name in name_sentence_list]
        desc_sentence_list = s_desc_df['Description'].tolist()
        desc_sentence_list = [str(desc) for desc in desc_sentence_list]
        text_encoder = pretrain_model.text_encoder
        text_encoder.load_model()
        name_embeddings = text_encoder.generate_embeddings(name_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.lm_emb_dim)
        print(f'Name Embeddings Shape: {name_embeddings.shape}')
        text_encoder.save_embeddings(name_embeddings, './BMG/Pretrain_data/x_name_emb.npy')
        desc_embeddings = text_encoder.generate_embeddings(desc_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.lm_emb_dim)
        print(f'Description Embeddings Shape: {desc_embeddings.shape}')
        text_encoder.save_embeddings(desc_embeddings, './BMG/Pretrain_data/x_desc_emb.npy')
    else:
        name_embeddings = np.load('./BMG/Pretrain_data/x_name_emb.npy').reshape(-1, args.lm_emb_dim)
        name_embeddings = torch.from_numpy(name_embeddings)
        print(f'Name Embeddings Shape: {name_embeddings.shape}')
        desc_embeddings = np.load('./BMG/Pretrain_data/x_desc_emb.npy').reshape(-1, args.lm_emb_dim)
        desc_embeddings = torch.from_numpy(desc_embeddings)
        print(f'Description Embeddings Shape: {desc_embeddings.shape}')
    # load textual embeddings into torch tensor
    name_embeddings = name_embeddings.float().to(device)
    desc_embeddings = desc_embeddings.float().to(device)

    # import pdb; pdb.set_trace()

    ### Build Pretrain and Train Model
    pretrain_model = build_pretrain_model(args, device)
    num_feature = args.num_omic_feature
    args.num_entity = xAll.shape[1]
    # Train the model depends on the task
    model = build_model(args, device)
    model.train()
    model.reset_parameters()

    num_entity = xAll.shape[1]
    train_num_cell = yTr_label.shape[0]
    epoch_num = args.num_train_epoch
    train_batch_size = args.train_batch_size

    # Add iteration counter
    iteration_num = 0
    dl_input_num = train_num_cell
    
    # Add learning rate schedule parameters
    e1, e2, e3, e4 = 1, 1, 3, 5  # Example values, adjust as needed

    epoch_loss_list = []
    epoch_pearson_list = []
    test_loss_list = []
    test_pearson_list = []

    max_train_pearson = 0
    max_test_pearson = 0
    max_test_pearson_id = 0

    # Clean result previous epoch_i_pred files
    folder_name = 'epoch_' + str(epoch_num)
    path = './' + args.train_result_folder  + '/' + args.name + '/' + args.model_name + '/%s' % (folder_name)
    unit = 1
    # Ensure the parent directories exist
    os.makedirs('./' + args.train_result_folder  + '/' + args.name + '/' + args.model_name, exist_ok=True)
    while os.path.exists(path):
        path = './' + args.train_result_folder  + '/' + args.name + '/' + args.model_name + '/%s_%d' % (folder_name, unit)
        unit += 1
    os.mkdir(path)

    # Save training arguments to a YAML file
    try:
        import yaml
        # Convert args to dictionary
        args_dict = vars(args)
        # Add some additional info
        args_dict['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        args_dict['device_used'] = str(device)
        
        # Write to YAML file
        with open(os.path.join(path, 'training_config.yaml'), 'w') as f:
            yaml.dump(args_dict, f, default_flow_style=False)
        print(f"Training configuration saved to {os.path.join(path, 'training_config.yaml')}")
    except Exception as e:
        print(f"Warning: Failed to save configuration to YAML file: {e}")
        # Fallback to text file if YAML fails
        with open(os.path.join(path, 'training_config.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        print(f"Training configuration saved to {os.path.join(path, 'training_config.txt')}")

    for i in range(1, epoch_num + 1):
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        epoch_ypred = np.zeros((1, 1))
        upper_index = 0
        batch_loss_list = []
        for index in range(0, train_num_cell, train_batch_size):
            if (index + train_batch_size) < train_num_cell:
                upper_index = index + train_batch_size
            else:
                upper_index = train_num_cell
            # Update learning rate based on current iteration
            updated_lr = learning_rate_schedule(args, dl_input_num, iteration_num, e1, e2, e3, e4)
            iteration_num += 1

            geo_train_datalist = read_index_komask_batch(index, upper_index, xAll, yTr_index, yTr_ko_mask, yTr_label, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)

            train_dataset_loader = GeoGraphLoader.load_graph(geo_train_datalist, args.train_batch_size, args.train_num_workers)
            current_cell_num = upper_index - index # current batch size
            optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=updated_lr, eps=args.train_eps, weight_decay=args.train_weight_decay)
            model, batch_loss, batch_ypred = train_model(train_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, pretrain_model, model, device, optimizer, args)
            print('BATCH LOSS: ', batch_loss)
            batch_loss_list.append(batch_loss)
            # PRESERVE PREDICTION OF BATCH TRAINING DATA
            batch_ypred = batch_ypred.detach().cpu().numpy().reshape(-1, 1)
            epoch_ypred = np.vstack((epoch_ypred, batch_ypred))
        epoch_loss = np.mean(batch_loss_list)
        print('TRAIN EPOCH ' + str(i) + ' LOSS: ', epoch_loss)
        epoch_loss_list.append(epoch_loss)
        epoch_ypred = np.delete(epoch_ypred, 0, axis = 0)
        # print('ITERATION NUMBER UNTIL NOW: ' + str(iteration_num))
        # Preserve corr for every epoch
        score_lists = list(yTr_label)
        score_list = [item for elem in score_lists for item in elem]
        epoch_ypred_lists = list(epoch_ypred)
        epoch_ypred_list = [item for elem in epoch_ypred_lists for item in elem]
        train_dict = {'label': score_list, 'prediction': epoch_ypred_list}
        tmp_training_input_df = pd.DataFrame(train_dict)
        # Calculating pearson correlation based on [tmp_training_input_df]
        epoch_pearson = tmp_training_input_df.corr(method='pearson')
        epoch_pearson = epoch_pearson['prediction'][0]
        epoch_pearson_list.append(epoch_pearson)
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index=False, header=True)
        print('EPOCH ' + str(i) + ' TRAINING PEARSON: ', epoch_pearson)
        print('\n-------------EPOCH TRAINING PEARSON LIST: -------------')
        print(epoch_pearson_list)
        print('\n-------------EPOCH TRAINING LOSS LIST: -------------')
        print(epoch_loss_list)

        # Test model on test dataset
        test_pearson, test_loss, tmp_test_input_df = test(args, pretrain_model, model, xAll, yTe_index, yTe_ko_mask, yTe_label, all_edge_index, internal_edge_index, ppi_edge_index, device, i)
        test_pearson = test_pearson['prediction'][0]
        test_pearson_list.append(test_pearson)
        test_loss_list.append(test_loss)
        tmp_test_input_df.to_csv(path + '/TestPred' + str(i) + '.txt', index=False, header=True)
        print('\n-------------EPOCH TEST PEARSON LIST: -------------')
        print(test_pearson_list)
        print('\n-------------EPOCH TEST MSE LOSS LIST: -------------')
        print(test_loss_list)
        # # Save each epoch model
        # torch.save(model.state_dict(), path + '/epoch_model_'+ str(i) +'.pt')
        # SAVE BEST TEST MODEL
        if test_pearson >= max_test_pearson and epoch_pearson >= max_train_pearson:
            print('Saving best model...')
            max_train_pearson = epoch_pearson 
            max_test_pearson = test_pearson
            max_test_pearson_id = i
            # torch.save(model.state_dict(), path + '/best_train_model'+ str(i) +'.pt')
            torch.save(model.state_dict(), path + '/best_train_model.pt')
            tmp_training_input_df.to_csv(path + '/BestTrainingPred.txt', index=False, header=True)
            tmp_test_input_df.to_csv(path + '/BestTestPred.txt', index=False, header=True)
            write_best_model_info(path, max_test_pearson_id, epoch_loss_list, epoch_pearson_list, test_loss_list, test_pearson_list)
        print('\n-------------BEST TEST PEARSON MODEL ID INFO:' + str(max_test_pearson_id) + '-------------')
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', epoch_loss_list[max_test_pearson_id - 1])
        print('BEST MODEL TRAIN PEARSON: ', epoch_pearson_list[max_test_pearson_id - 1])
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss_list[max_test_pearson_id - 1])
        print('BEST MODEL TEST PEARSON: ', test_pearson_list[max_test_pearson_id - 1])



def test(args, pretrain_model, model, xAll, yTe_index, yTe_ko_mask, yTe_label, all_edge_index, internal_edge_index, ppi_edge_index, device, i):
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    
    print('--- LOADING TESTING FILES ... ---')
    print('xAll: ', xAll.shape)
    print('yTe_label: ', yTe_label.shape)
    test_num_cell = yTe_label.shape[0]
    num_entity = xAll.shape[1]
    num_feature = args.num_omic_feature
    
    # load textual embeddings into torch tensor
    name_embeddings = np.load('./BMG/Pretrain_data/x_name_emb.npy').reshape(-1, args.lm_emb_dim)
    name_embeddings = torch.from_numpy(name_embeddings)
    print(f'Name Embeddings Shape: {name_embeddings.shape}')
    desc_embeddings = np.load('./BMG/Pretrain_data/x_desc_emb.npy').reshape(-1, args.lm_emb_dim)
    desc_embeddings = torch.from_numpy(desc_embeddings)
    print(f'Description Embeddings Shape: {desc_embeddings.shape}')
    name_embeddings = name_embeddings.float().to(device)
    desc_embeddings = desc_embeddings.float().to(device)
    test_batch_size = args.train_batch_size
    
    # Run test model
    model.eval()
    all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, test_num_cell, test_batch_size):
        if (index + test_batch_size) < test_num_cell:
            upper_index = index + test_batch_size
        else:
            upper_index = test_num_cell
        geo_datalist = read_index_komask_batch(index, upper_index, xAll, yTe_index, yTe_ko_mask, yTe_label, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
        test_dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args.train_batch_size, args.train_num_workers)
        print('TEST MODEL...')
        current_cell_num = upper_index - index # current batch size
        model, batch_loss, batch_ypred = test_model(test_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, pretrain_model, model, device, args)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        # PRESERVE PREDICTION OF BATCH TEST DATA
        batch_ypred = batch_ypred.detach().cpu().numpy().reshape(-1, 1)
        all_ypred = np.vstack((all_ypred, batch_ypred))
    test_loss = np.mean(batch_loss_list)
    print('EPOCH ' + str(i) + ' TEST LOSS: ', test_loss)
    # Preserve pearson for every epoch
    all_ypred = np.delete(all_ypred, 0, axis=0)
    all_ypred_lists = list(all_ypred)
    all_ypred_list = [item for elem in all_ypred_lists for item in elem]
    score_lists = list(yTe_label)
    score_list = [item for elem in score_lists for item in elem]
    test_dict = {'label': score_list, 'prediction': all_ypred_list}
    # import pdb; pdb.set_trace()
    tmp_test_input_df = pd.DataFrame(test_dict)
    # Calculating pearson correlation based on [tmp_test_input_df]
    test_pearson = tmp_test_input_df.corr(method='pearson')
    print('EPOCH ' + str(i) + ' TEST PEARSON: ', test_pearson['prediction'][0])
    return test_pearson, test_loss, tmp_test_input_df


def arg_parse():
    parser = argparse.ArgumentParser()

    # pre-training loading parameters
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

    parser.add_argument('--save_path', nargs='?', default='./Checkpoints/pretrained_models_gat/pretrained_motasg_foundation.pt', help='save path for model. (default: pretrained_motasg_foundation.pt)')
    parser.add_argument('--device', type=int, default=0)

    # downstream task parameters
    parser.add_argument('--train_sample_ratio', type=float, default=0.00001, help='Sampling ratio of training data. (default: 1.0)')
    parser.add_argument('--training_sample_random_seed', type=int, default=2025, help='Random seed for sampling training data. (default: 42)')

    parser.add_argument('--text_lm_model_path', nargs='?', default='dmis-lab/biobert-v1.1', help='Path to the pretrained language model. (default: dmis-lab/biobert-v1.1)')
    parser.add_argument('--train_text', default=False, help='Whether to train the text encoder. (default: False)')
    parser.add_argument('--name', nargs='?', default='CRISPR', help='Name for dataset.')

    parser.add_argument('--train_lr', type=float, default=0.0002, help='Learning rate for training. (default: 0.0002)')
    parser.add_argument('--train_lr2', type=float, default=0.0001, help='Learning rate for training. (default: 0.0001)')
    parser.add_argument('--train_lr3', type=float, default=0.000075, help='Learning rate for training. (default: 0.000075)')
    parser.add_argument('--train_lr4', type=float, default=0.00005, help='Learning rate for training. (default: 0.00005)')
    parser.add_argument('--train_lr5', type=float, default=0.00001, help='Learning rate for training. (default: 0.00001)')
    parser.add_argument('--train_eps', type=float, default=1e-7, help='Epsilon for Adam optimizer. (default: 1e-7)')
    parser.add_argument('--train_weight_decay', type=float, default=1e-15, help='Weight decay for Adam optimizer. (default: 1e-15)')
    parser.add_argument('--train_encoder_dropout', type=float, default=0.1, help='Dropout probability of encoder. (default: 0.1)')

    parser.add_argument('--num_train_epoch', type=int, default=15, help='Number of training epochs. (default: 50)')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Batch size for training. (default: 16)')
    parser.add_argument('--train_num_workers', type=int, default=0, help='Number of workers to load data.')

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

    parser.add_argument('--train_result_folder', nargs='?', default='MOTASG_Results', help='Path to save training results. (default: MOTASG_Results)')
    parser.add_argument('--model_name', nargs='?', default='MOTASG_Reg', help='Model names. (default: MOTASG_Reg)')

    return parser.parse_args()




if __name__ == "__main__":
    # Set arguments and print
    args = arg_parse()
    print(tab_printer(args))
    # Check device and select the one with most available memory
    if args.device < 0:
        device = 'cpu'
    else:
        # Auto-select GPU with most free memory
        device = get_gpu_with_max_free_memory() if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set model_name with adding pretrain based GNN and train based GNN
    args.model_name = args.model_name + '_' + args.layer + '_' + args.train_layer

    # Load pretrain model
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.load_state_dict(torch.load(args.save_path))
    pretrain_model.eval()
    train(args, pretrain_model, device)

    # ### Train the model
    # train_num = 5
    # for number in range(1, train_num + 1):
    #     # Train the model
    #     train(args, pretrain_model, device)
    
    ### Test the model
    # print('--- LOADING TEST FILES ... ---')
    # xAll = np.load('./BMG/Pretrain_data/pretrain_bmgc_omics.npy')
    # yTe = np.load('./BMG/Pretrain_data/balanced_test.npy')
    # print(xAll.shape, yTe.shape)
    # # Use the first column as the indices and second column as the labels
    # yTe_index = yTe[:, 0].astype(np.int32).reshape(-1, 1)
    # yTe_label = yTe[:, 1].astype(np.int32).reshape(-1, 1)
    # model = build_model(args, device)
    # model.load_state_dict(torch.load('./Results/DepMap/MOTASG-Class/epoch_10_best/epoch_model_3.pt'))
    # model.eval()
    # # Load edge_index
    # all_edge_index = np.load('./BMG/Pretrain_data/edge_index.npy')
    # internal_edge_index = np.load('./BMG/Pretrain_data/internal_edge_index.npy')
    # ppi_edge_index = np.load('./BMG/Pretrain_data/ppi_edge_index.npy')
    # all_edge_index = torch.from_numpy(all_edge_index).long()
    # internal_edge_index = torch.from_numpy(internal_edge_index).long()
    # ppi_edge_index = torch.from_numpy(ppi_edge_index).long()
    # test(args, pretrain_model, model, xAll, yTe_index, yTe_label, all_edge_index, internal_edge_index, ppi_edge_index, device, i=3)