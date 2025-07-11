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
from GeoDataLoader.read_geograph import read_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from MOTASG_Foundation.lm_model import TextEncoder
from MOTASG_Foundation.downstream import MOTASG_Class, DSGNNEncoder


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


def write_best_model_info(path, max_test_acc_id, epoch_loss_list, epoch_acc_list, epoch_f1_list, test_loss_list, test_acc_list, test_f1_list):
    best_model_info = (
        f'\n-------------BEST TEST ACCURACY MODEL ID INFO: {max_test_acc_id} -------------\n'
        '--- TRAIN ---\n'
        f'BEST MODEL TRAIN LOSS: {epoch_loss_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TRAIN ACCURACY: {epoch_acc_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TRAIN F1: {epoch_f1_list[max_test_acc_id - 1]}\n'
        '--- TEST ---\n'
        f'BEST MODEL TEST LOSS: {test_loss_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TEST ACCURACY: {test_acc_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TEST F1: {test_f1_list[max_test_acc_id - 1]}\n'
    )
    with open(os.path.join(path, 'best_model_info.txt'), 'w') as file:
        file.write(best_model_info)
    
    # Save all metrics to CSV files
    # Training metrics
    train_metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(epoch_loss_list) + 1)),
        'loss': epoch_loss_list,
        'accuracy': epoch_acc_list,
        'f1_score': epoch_f1_list
    })
    train_metrics_df.to_csv(os.path.join(path, 'training_metrics.csv'), index=False)
    
    # Testing metrics
    test_metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(test_loss_list) + 1)),
        'loss': test_loss_list,
        'accuracy': test_acc_list,
        'f1_score': test_f1_list
    })
    test_metrics_df.to_csv(os.path.join(path, 'testing_metrics.csv'), index=False)


def train_model(train_dataset_loader, current_cell_num, num_entity, protein_node_index, name_embeddings, desc_embeddings, pretrain_model, model, device, optimizer, args):
    batch_loss = 0
    for batch_idx, data in enumerate(train_dataset_loader):
        optimizer.zero_grad()
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)

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
        # Continue training the model
        output, ypred = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, protein_node_index, name_embeddings, desc_embeddings, current_cell_num)
        loss = model.loss(output, label)
        loss.backward()
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', ypred)
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        # # check pretrain model parameters
        # state_dict = pretrain_model.internal_encoder.state_dict()
        # print(state_dict['convs.1.lin.weight'])
        # print(model.embedding.weight.data)
    torch.cuda.empty_cache()
    return model, batch_loss, batch_acc, ypred


def test_model(test_dataset_loader, current_cell_num, num_entity, protein_node_index, name_embeddings, desc_embeddings, pretrain_model, model, device, args):
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_loss = 0
    all_ypred = np.zeros((1, 1))
    for batch_idx, data in enumerate(test_dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
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
            output, ypred = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, protein_node_index, name_embeddings, desc_embeddings, current_cell_num)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', ypred)
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        all_ypred = np.vstack((all_ypred, ypred.cpu().numpy().reshape(-1, 1)))
        all_ypred = np.delete(all_ypred, 0, axis=0)
    return model, batch_loss, batch_acc, all_ypred


def train(args, pretrain_model, device):
    ### Load data
    print('--- LOADING TRAINING FILES ... ---')
    xTr = np.load(os.path.join(args.data_path, 'pretrain_status_train_feature.npy'))
    xTe = np.load(os.path.join(args.data_path, 'pretrain_status_test_feature.npy'))
    yTr = np.load(os.path.join(args.data_path, 'pretrain_status_train_label.npy'))
    yTe = np.load(os.path.join(args.data_path, 'pretrain_status_test_label.npy'))
    # Load the dictionary for node_index
    node_index_df = pd.read_csv(os.path.join(args.data_path, 'nodes_index.csv'))
    # Fetch the protein node in the column Type == Protein
    protein_node_index_df = node_index_df[node_index_df['Type'] == 'Protein']
    protein_node_index_list = protein_node_index_df['Index'].tolist()
    # Convert protein_node_index_list to torch tensor
    protein_node_index = torch.tensor(protein_node_index_list, dtype=torch.long).to(device)

    # # Map yTr to 0-(number of unique values-1)
    # unique_values = np.unique(yTr)
    # print("Number of classes: ", len(unique_values))
    # args.num_class = len(unique_values)
    # value_to_index = {value: index for index, value in enumerate(unique_values)}
    # yTr = np.vectorize(value_to_index.get)(yTr)
    # yTr = yTr.reshape(-1, 1)  # Ensure yTr is a 2D array
    # print(xTr.shape, yTr.shape)
    # unique_values = np.unique(yTe)
    # print("Number of classes: ", len(unique_values))
    # args.num_class = len(unique_values)
    # value_to_index = {value: index for index, value in enumerate(unique_values)}
    # yTe = np.vectorize(value_to_index.get)(yTe)
    # yTe = yTe.reshape(-1, 1)  # Ensure yTe is a 2D array
    # print(xTe.shape, yTe.shape)

    # Map yTr to 0-(number of unique values-1)
    unique_values = np.unique(yTr)
    print("Number of classes: ", len(unique_values))
    args.num_class = len(unique_values)
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    yTr = np.vectorize(value_to_index.get)(yTr)
    yTr = yTr.reshape(-1, 1)  # Ensure yTr is a 2D array
    print("Before upsampling - xTr shape:", xTr.shape, "yTr shape:", yTr.shape)

    # Check class distribution before upsampling
    unique_train, counts_train = np.unique(yTr, return_counts=True)
    print("Training class distribution before upsampling:", dict(zip(unique_train, counts_train)))

    # Upsample to balance classes 0 and 1
    class_0_indices = np.where(yTr.flatten() == 0)[0]
    class_1_indices = np.where(yTr.flatten() == 1)[0]

    print(f"Original class distribution - Class 0: {len(class_0_indices)}, Class 1: {len(class_1_indices)}")

    # Determine which class is minority and which is majority
    if len(class_0_indices) < len(class_1_indices):
        minority_class = 0
        minority_indices = class_0_indices
        majority_count = len(class_1_indices)
        print(f"Upsampling class 0: {len(class_0_indices)} -> {majority_count} samples")
    elif len(class_1_indices) < len(class_0_indices):
        minority_class = 1
        minority_indices = class_1_indices
        majority_count = len(class_0_indices)
        print(f"Upsampling class 1: {len(class_1_indices)} -> {majority_count} samples")
    else:
        minority_class = None
        print("Classes are already balanced, no upsampling needed")

    # Perform upsampling if needed
    if minority_class is not None:
        # Calculate how many additional samples we need for the minority class
        n_additional = majority_count - len(minority_indices)
        # Randomly sample with replacement from existing minority class samples
        additional_indices = np.random.choice(minority_indices, size=n_additional, replace=True)
        # Combine original indices with additional indices
        upsampled_indices = np.concatenate([np.arange(len(yTr)), additional_indices])
        # Shuffle the upsampled indices to randomize the order
        np.random.shuffle(upsampled_indices)
        # Apply upsampling to both features and labels
        xTr = xTr[upsampled_indices]
        yTr = yTr[upsampled_indices]
        print("After upsampling - xTr shape:", xTr.shape, "yTr shape:", yTr.shape)
        
        # Check class distribution after upsampling
        unique_train_after, counts_train_after = np.unique(yTr, return_counts=True)
        print("Training class distribution after upsampling:", dict(zip(unique_train_after, counts_train_after)))

    # Process test labels (MOVED OUTSIDE THE IF BLOCK)
    unique_values = np.unique(yTe)
    print("Number of classes in test: ", len(unique_values))
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    yTe = np.vectorize(value_to_index.get)(yTe)
    yTe = yTe.reshape(-1, 1)  # Ensure yTe is a 2D array
    print("Test data - xTe shape:", xTe.shape, "yTe shape:", yTe.shape)

    # Check test class distribution
    unique_test, counts_test = np.unique(yTe, return_counts=True)
    print("Test class distribution:", dict(zip(unique_test, counts_test)))

    # Load edge_index
    all_edge_index = np.load(os.path.join(args.data_path, 'edge_index.npy'))
    internal_edge_index = np.load(os.path.join(args.data_path, 'internal_edge_index.npy'))
    ppi_edge_index = np.load(os.path.join(args.data_path, 'ppi_edge_index.npy'))
    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()
    # Load textual embeddings
    if args.train_text:
        # Use language model to embed the name and description
        s_name_df = pd.read_csv(os.path.join(args.data_path, 'bmgc_omics_name.csv'))
        s_desc_df = pd.read_csv(os.path.join(args.data_path, 'bmgc_omics_desc.csv'))
        name_sentence_list = s_name_df['Names_and_IDs'].tolist()
        name_sentence_list = [str(name) for name in name_sentence_list]
        desc_sentence_list = s_desc_df['Description'].tolist()
        desc_sentence_list = [str(desc) for desc in desc_sentence_list]
        text_encoder = pretrain_model.text_encoder
        text_encoder.load_model()
        name_embeddings = text_encoder.generate_embeddings(name_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.lm_emb_dim)
        print(f'Name Embeddings Shape: {name_embeddings.shape}')
        text_encoder.save_embeddings(name_embeddings, os.path.join(args.data_path, 'x_name_emb.npy'))
        desc_embeddings = text_encoder.generate_embeddings(desc_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.lm_emb_dim)
        print(f'Description Embeddings Shape: {desc_embeddings.shape}')
        text_encoder.save_embeddings(desc_embeddings, os.path.join(args.data_path, 'x_desc_emb.npy'))
    else:
        name_embeddings = np.load(os.path.join(args.data_path, 'x_name_emb.npy')).reshape(-1, args.lm_emb_dim)
        name_embeddings = torch.from_numpy(name_embeddings)
        print(f'Name Embeddings Shape: {name_embeddings.shape}')
        desc_embeddings = np.load(os.path.join(args.data_path, 'x_desc_emb.npy')).reshape(-1, args.lm_emb_dim)
        desc_embeddings = torch.from_numpy(desc_embeddings)
        print(f'Description Embeddings Shape: {desc_embeddings.shape}')
    # load textual embeddings into torch tensor
    name_embeddings = name_embeddings.float().to(device)
    desc_embeddings = desc_embeddings.float().to(device)

    ### Build Pretrain and Train Model
    pretrain_model = build_pretrain_model(args, device)
    num_feature = args.num_omic_feature
    args.num_entity = xTr.shape[1]
    # Train the model depends on the task
    model = build_model(args, device)
    model.train()
    model.reset_parameters()

    num_entity = xTr.shape[1]
    train_num_cell = yTr.shape[0]
    epoch_num = args.num_train_epoch
    train_batch_size = args.train_batch_size

    # Add iteration counter
    iteration_num = 0
    dl_input_num = train_num_cell
    
    # Add learning rate schedule parameters
    e1, e2, e3, e4 = 3, 3, 2, 2  # Example values, adjust as needed

    epoch_loss_list = []
    epoch_f1_list = []
    epoch_acc_list = []
    test_loss_list = []
    test_f1_list = []
    test_acc_list = []

    max_train_acc = 0
    max_train_f1 = 0
    max_test_acc = 0
    max_test_f1 = 0
    max_test_acc_id = 0

    # Initialize best_metrics before the training loop (add this before the for loop)
    best_metrics = {}

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
            geo_train_datalist = read_batch(index, upper_index, xTr, yTr, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
            train_dataset_loader = GeoGraphLoader.load_graph(geo_train_datalist, args.train_batch_size, args.train_num_workers)
            current_cell_num = upper_index - index # current batch size
            optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=updated_lr, eps=args.train_eps, weight_decay=args.train_weight_decay)
            model, batch_loss, batch_acc, batch_ypred = train_model(train_dataset_loader, current_cell_num, num_entity, protein_node_index, name_embeddings, desc_embeddings, pretrain_model, model, device, optimizer, args)
            print('BATCH LOSS: ', batch_loss)
            print('BATCH ACCURACY: ', batch_acc)
            batch_loss_list.append(batch_loss)
            # PRESERVE PREDICTION OF BATCH TRAINING DATA
            batch_ypred = (Variable(batch_ypred).data).cpu().numpy().reshape(-1, 1)
            epoch_ypred = np.vstack((epoch_ypred, batch_ypred))
        epoch_loss = np.mean(batch_loss_list)
        print('TRAIN EPOCH ' + str(i) + ' LOSS: ', epoch_loss)
        epoch_loss_list.append(epoch_loss)
        epoch_ypred = np.delete(epoch_ypred, 0, axis = 0)
        # print('ITERATION NUMBER UNTIL NOW: ' + str(iteration_num))
        # Preserve acc corr for every epoch
        score_lists = list(yTr)
        score_list = [item for elem in score_lists for item in elem]
        epoch_ypred_lists = list(epoch_ypred)
        epoch_ypred_list = [item for elem in epoch_ypred_lists for item in elem]
        train_dict = {'label': score_list, 'prediction': epoch_ypred_list}
        tmp_training_input_df = pd.DataFrame(train_dict)
        # Calculating metrics
        accuracy = accuracy_score(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index=False, header=True)
        epoch_acc_list.append(accuracy)
        f1 = f1_score(tmp_training_input_df['label'], tmp_training_input_df['prediction'], average='binary')
        epoch_f1_list.append(f1)
        conf_matrix = confusion_matrix(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        tn, fp, fn, tp = conf_matrix.ravel()
        print('EPOCH ' + str(i) + ' TRAINING ACCURACY: ', accuracy)
        print('EPOCH ' + str(i) + ' TRAINING F1: ', f1)
        print('EPOCH ' + str(i) + ' TRAINING CONFUSION MATRIX: ', conf_matrix)
        print('EPOCH ' + str(i) + ' TRAINING TN: ', tn)
        print('EPOCH ' + str(i) + ' TRAINING FP: ', fp)
        print('EPOCH ' + str(i) + ' TRAINING FN: ', fn)
        print('EPOCH ' + str(i) + ' TRAINING TP: ', tp)


        print('\n-------------EPOCH TRAINING ACCURACY LIST: -------------')
        print(epoch_acc_list)
        print('\n-------------EPOCH TRAINING F1 LIST: -------------')
        print(epoch_f1_list)
        print('\n-------------EPOCH TRAINING LOSS LIST: -------------')
        print(epoch_loss_list)

        # # # Test model on test dataset
        test_acc, test_f1, test_loss, tmp_test_input_df = test(args, pretrain_model, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, device, i)
        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)
        test_loss_list.append(test_loss)
        tmp_test_input_df.to_csv(path + '/TestPred' + str(i) + '.txt', index=False, header=True)
        print('\n-------------EPOCH TEST ACCURACY LIST: -------------')
        print(test_acc_list)
        print('\n-------------EPOCH TEST F1 LIST: -------------')
        print(test_f1_list)
        print('\n-------------EPOCH TEST MSE LOSS LIST: -------------')
        print(test_loss_list)
        # # Save each epoch model
        # torch.save(model.state_dict(), path + '/epoch_model_'+ str(i) +'.pt')
        # SAVE BEST MODEL using improved strategy
        # Calculate balanced criteria
        train_acc, train_f1, train_loss = accuracy, f1, epoch_loss
        test_acc_val, test_f1_val, test_loss_val = test_acc, test_f1, test_loss
        
        # Avoid overfitting: don't save if training accuracy too much higher than test
        overfitting_threshold = 0.2  # Max 20% gap between train and test accuracy
        overfitting_detected = train_acc - test_acc_val > overfitting_threshold
        
        # Don't save models with very poor training performance
        poor_training = train_acc < 0.7
        
        # Composite score: weighted average of test metrics
        current_score = 0.6 * test_f1_val + 0.4 * test_acc_val
        best_score = best_metrics.get('composite_score', 0.0)
        
        if not overfitting_detected and not poor_training and current_score > best_score:
            print('Saving best model with improved composite score...')
            max_train_acc = accuracy 
            max_test_acc = test_acc
            max_train_f1 = f1
            max_test_f1 = test_f1
            max_test_acc_id = i
            
            # Update best metrics
            best_metrics['composite_score'] = current_score
            best_metrics['epoch'] = i
            best_metrics['train_acc'] = train_acc
            best_metrics['test_acc'] = test_acc_val
            best_metrics['train_f1'] = train_f1
            best_metrics['test_f1'] = test_f1_val
            
            # Save both pretrain_model and model together
            combined_state_dict = {
                'pretrain_model': pretrain_model.state_dict(),
                'downstream_model': model.state_dict(),
                'epoch': i,
                'max_train_acc': max_train_acc,
                'max_test_acc': max_test_acc,
                'max_train_f1': max_train_f1,
                'max_test_f1': max_test_f1,
                'composite_score': current_score,
                'args': vars(args)  # Save hyperparameters for reproducibility
            }
            torch.save(combined_state_dict, path + '/best_combined_model.pt')
            
            # Keep the original individual model saving for backward compatibility
            torch.save(model.state_dict(), path + '/best_train_model.pt')
            
            tmp_training_input_df.to_csv(path + '/BestTrainingPred.txt', index=False, header=True)
            tmp_test_input_df.to_csv(path + '/BestTestPred.txt', index=False, header=True)
            write_best_model_info(path, max_test_acc_id, epoch_loss_list, epoch_acc_list, epoch_f1_list, test_loss_list, test_acc_list, test_f1_list)
            
            print(f"New best model saved at epoch {i} with composite score: {current_score:.4f}")
            print(f"  - Train Acc: {train_acc:.3f}, Test Acc: {test_acc_val:.3f}")
            print(f"  - Train F1: {train_f1:.3f}, Test F1: {test_f1_val:.3f}")
        
        elif overfitting_detected:
            print(f"Epoch {i}: Potential overfitting detected (train_acc: {train_acc:.3f}, test_acc: {test_acc_val:.3f})")
        elif poor_training:
            print(f"Epoch {i}: Training accuracy too low ({train_acc:.3f}), skipping save")
        else:
            print(f"Epoch {i}: Current composite score ({current_score:.4f}) not better than best ({best_score:.4f})")
        
        print('\n-------------BEST TEST ACCURACY MODEL ID INFO:' + str(max_test_acc_id) + '-------------')
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', epoch_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TRAIN ACCURACY: ', epoch_acc_list[max_test_acc_id - 1])
        print('BEST MODEL TRAIN F1: ', epoch_f1_list[max_test_acc_id - 1])
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TEST ACCURACY: ', test_acc_list[max_test_acc_id - 1])
        print('BEST MODEL TEST F1: ', test_f1_list[max_test_acc_id - 1])



def test(args, pretrain_model, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, device, i):
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
    print('xTe: ', xTe.shape)
    print('yTe: ', yTe.shape)
    test_num_cell = yTe.shape[0]
    num_entity = xTe.shape[1]
    num_feature = args.num_omic_feature
    # Load the dictionary for node_index
    node_index_df = pd.read_csv(os.path.join(args.data_path, 'nodes_index.csv'))
    # Fetch the protein node in the column Type == Protein
    protein_node_index_df = node_index_df[node_index_df['Type'] == 'Protein']
    protein_node_index_list = protein_node_index_df['Index'].tolist()
    # Convert protein_node_index_list to torch tensor
    protein_node_index = torch.tensor(protein_node_index_list, dtype=torch.long).to(device)
    
    # load textual embeddings into torch tensor
    name_embeddings = np.load(os.path.join(args.data_path, 'x_name_emb.npy')).reshape(-1, args.lm_emb_dim)
    name_embeddings = torch.from_numpy(name_embeddings)
    print(f'Name Embeddings Shape: {name_embeddings.shape}')
    desc_embeddings = np.load(os.path.join(args.data_path, 'x_desc_emb.npy')).reshape(-1, args.lm_emb_dim)
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
        geo_datalist = read_batch(index, upper_index, xTe, yTe, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
        test_dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args.train_batch_size, args.train_num_workers)
        print('TEST MODEL...')
        current_cell_num = upper_index - index # current batch size
        model, batch_loss, batch_acc, batch_ypred = test_model(test_dataset_loader, current_cell_num, num_entity, protein_node_index, name_embeddings, desc_embeddings, pretrain_model, model, device, args)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        print('BATCH ACCURACY: ', batch_acc)
        # PRESERVE PREDICTION OF BATCH TEST DATA
        batch_ypred = batch_ypred.reshape(-1, 1)
        all_ypred = np.vstack((all_ypred, batch_ypred))
    test_loss = np.mean(batch_loss_list)
    print('EPOCH ' + str(i) + ' TEST LOSS: ', test_loss)
    # Preserve accuracy for every epoch
    all_ypred = np.delete(all_ypred, 0, axis=0)
    all_ypred_lists = list(all_ypred)
    all_ypred_list = [item for elem in all_ypred_lists for item in elem]
    score_lists = list(yTe)
    score_list = [item for elem in score_lists for item in elem]
    test_dict = {'label': score_list, 'prediction': all_ypred_list}
    tmp_test_input_df = pd.DataFrame(test_dict)
    # Calculating metrics
    accuracy = accuracy_score(tmp_test_input_df['label'], tmp_test_input_df['prediction'])
    f1 = f1_score(tmp_test_input_df['label'], tmp_test_input_df['prediction'], average='binary')
    conf_matrix = confusion_matrix(tmp_test_input_df['label'], tmp_test_input_df['prediction'])
    tn, fp, fn, tp = conf_matrix.ravel()
    print('EPOCH ' + str(i) + ' TEST ACCURACY: ', accuracy)
    print('EPOCH ' + str(i) + ' TEST F1: ', f1)
    print('EPOCH ' + str(i) + ' TEST CONFUSION MATRIX: ', conf_matrix)
    print('EPOCH ' + str(i) + ' TEST TN: ', tn)
    print('EPOCH ' + str(i) + ' TEST FP: ', fp)
    print('EPOCH ' + str(i) + ' TEST FN: ', fn)
    print('EPOCH ' + str(i) + ' TEST TP: ', tp)
    test_acc = accuracy
    test_f1 = f1
    return test_acc, test_f1, test_loss, tmp_test_input_df


def load_combined_model(checkpoint_path, args, device):
    """
    Load both pretrain_model and downstream model from a combined checkpoint.
    
    Args:
        checkpoint_path (str): Path to the combined model checkpoint
        args: Arguments object with model parameters
        device: Device to load models on
    
    Returns:
        tuple: (pretrain_model, downstream_model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build models
    pretrain_model = build_pretrain_model(args, device)
    downstream_model = build_model(args, device)
    
    # Load state dictionaries
    pretrain_model.load_state_dict(checkpoint['pretrain_model'])
    downstream_model.load_state_dict(checkpoint['downstream_model'])
    
    # Set to evaluation mode
    pretrain_model.eval()
    downstream_model.eval()
    
    # Extract additional info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', None),
        'max_train_acc': checkpoint.get('max_train_acc', None),
        'max_test_acc': checkpoint.get('max_test_acc', None),
        'max_train_f1': checkpoint.get('max_train_f1', None),
        'max_test_f1': checkpoint.get('max_test_f1', None),
        'saved_args': checkpoint.get('args', None)
    }
    
    return pretrain_model, downstream_model, checkpoint_info


# Example usage for loading the combined model
def load_and_use_combined_model(checkpoint_path, args, device):
    """Example of how to load and use the combined model"""
    pretrain_model, downstream_model, info = load_combined_model(checkpoint_path, args, device)
    
    print(f"Loaded model from epoch {info['epoch']}")
    print(f"Best test F1: {info['max_test_f1']}")
    print(f"Best test accuracy: {info['max_test_acc']}")
    
    return pretrain_model, downstream_model


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

    parser.add_argument('--save_path', nargs='?', default='./checkpoints/pretrained_models_gat/pretrained_plain_foundation.pt', help='save path for model. (default: pretrained_plain_foundation.pt)')
    parser.add_argument('--device', type=int, default=0)

    # downstream task parameters
    parser.add_argument('--data_path', nargs='?', default='./data/pretrain_status_data', help='Path to the pretrain status data. (default: ./data/pretrain_status_data)')
    parser.add_argument('--text_lm_model_path', nargs='?', default='dmis-lab/biobert-v1.1', help='Path to the pretrained language model. (default: dmis-lab/biobert-v1.1)')
    parser.add_argument('--train_text', default=False, help='Whether to train the text encoder. (default: False)')
    parser.add_argument('--task', nargs='?', default='class', help='Task for training downstream tasks. (default: class)')
    parser.add_argument('--name', nargs='?', default='DepMap', help='Name for dataset.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for classification. (default: 2)')

    parser.add_argument('--train_lr', type=float, default=0.001, help='Learning rate for training. (default: 0.001)')
    parser.add_argument('--train_lr2', type=float, default=0.00075, help='Learning rate for training. (default: 0.00075)')
    parser.add_argument('--train_lr3', type=float, default=0.0005, help='Learning rate for training. (default: 0.0005)')
    parser.add_argument('--train_lr4', type=float, default=0.00025, help='Learning rate for training. (default: 0.00025)')
    parser.add_argument('--train_lr5', type=float, default=0.0001, help='Learning rate for training. (default: 0.0001)')
    parser.add_argument('--train_eps', type=float, default=1e-7, help='Epsilon for Adam optimizer. (default: 1e-7)')
    parser.add_argument('--train_weight_decay', type=float, default=1e-15, help='Weight decay for Adam optimizer. (default: 1e-15)')
    parser.add_argument('--train_encoder_dropout', type=float, default=0.1, help='Dropout probability of encoder. (default: 0.1)')

    parser.add_argument('--num_train_epoch', type=int, default=20, help='Number of training epochs. (default: 20)')
    parser.add_argument('--train_batch_size', type=int, default=2, help='Batch size for training. (default: 2)')
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
    parser.add_argument('--model_name', nargs='?', default='MOTASG_Class', help='Model names. (default: MOTASG_Class)')

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

    # # Set model_name with adding pretrain based GNN and train based GNN
    # args.model_name = args.model_name + '_' + args.layer + '_' + args.train_layer

    # # Load pretrain model
    # pretrain_model = build_pretrain_model(args, device)
    # pretrain_model.load_state_dict(torch.load(args.save_path))
    # pretrain_model.eval()
    # # train(args, pretrain_model, device)

    # ### Train the model
    # print('--- LOADING TRAIN FILES ... ---')
    # train_num = 5
    # for number in range(1, train_num + 1):
    #     # Train the model
    #     train(args, pretrain_model, device)
    
    # ### Test the model using combined checkpoint
    # print('--- LOADING TEST FILES ... ---')
    # xTe = np.load(os.path.join(args.data_path, 'pretrain_status_test_feature.npy'))
    # yTe = np.load(os.path.join(args.data_path, 'pretrain_status_test_label.npy'))
    # yTe = yTe.reshape(-1, 1)  # Ensure yTe is a 2D array
    # print(xTe.shape, yTe.shape)
    # args.num_entity = xTe.shape[1]
    # # Load both pretrain_model and downstream model from combined checkpoint
    # combined_checkpoint_path = './MOTASG_Results/DepMap/MOTASG_Class_gat_gat/epoch_50_1/best_combined_model.pt'
    # pretrain_model, model, checkpoint_info = load_combined_model(combined_checkpoint_path, args, device)
    # print(f"Loaded combined model from epoch {checkpoint_info['epoch']}")
    # print(f"Model test F1: {checkpoint_info['max_test_f1']}")
    # print(f"Model test accuracy: {checkpoint_info['max_test_acc']}")
    # # Load edge_index
    # all_edge_index = np.load(os.path.join(args.data_path, 'edge_index.npy'))
    # internal_edge_index = np.load(os.path.join(args.data_path, 'internal_edge_index.npy'))
    # ppi_edge_index = np.load(os.path.join(args.data_path, 'ppi_edge_index.npy'))
    # all_edge_index = torch.from_numpy(all_edge_index).long()
    # internal_edge_index = torch.from_numpy(internal_edge_index).long()
    # ppi_edge_index = torch.from_numpy(ppi_edge_index).long()
    # test(args, pretrain_model, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, device, i=0)