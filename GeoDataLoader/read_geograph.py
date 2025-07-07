import torch
import numpy as np
import pandas as pd
import networkx as nx

from numpy import inf
from torch_geometric.data import Data

class ReadGeoGraph():
    def __init__(self):
        pass

    def read_feature(self, num_graph, num_feature, num_node, xBatch):
        # FORM [graph_feature_list]
        xBatch = xBatch.reshape(num_graph, num_node, num_feature)
        graph_feature_list = []
        for i in range(num_graph):
            graph_feature_list.append(xBatch[i, :, :])
        return graph_feature_list

    def read_label(self, yBatch):
        yBatch_list = [label[0] for label in list(yBatch)]
        graph_label_list = yBatch_list
        return graph_label_list

    def form_geo_pretrain_datalist(self, num_graph, graph_feature_list, all_edge_index, internal_edge_index, ppi_edge_index):
        geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            # CONVERT [numpy] TO [torch]
            graph_feature = torch.from_numpy(graph_feature).float()
            geo_data = Data(x=graph_feature, edge_index=ppi_edge_index, internal_edge_index=internal_edge_index, all_edge_index=all_edge_index)
            geo_datalist.append(geo_data)
        return geo_datalist

    def form_geo_datalist(self, num_graph, graph_feature_list, graph_label_list, all_edge_index, internal_edge_index, ppi_edge_index):
        geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            graph_label = graph_label_list[i]
            # CONVERT [numpy] TO [torch]
            graph_feature = torch.from_numpy(graph_feature).float()
            graph_label = torch.from_numpy(np.array([graph_label])).float()
            geo_data = Data(x=graph_feature, edge_index=ppi_edge_index, internal_edge_index=internal_edge_index, all_edge_index=all_edge_index, label=graph_label)
            geo_datalist.append(geo_data)
        return geo_datalist
    
    def form_geo_datalist_with_komask(self, num_graph, graph_feature_list, graph_label_list, all_edge_index, internal_edge_index, ppi_edge_index, ko_mask_list_array):
        geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            graph_label = graph_label_list[i]
            ko_mask = ko_mask_list_array[i]
            
            # CONVERT [numpy] TO [torch]
            graph_feature = torch.from_numpy(graph_feature).float()
            graph_label = torch.from_numpy(np.array([graph_label])).float()
            
            # Convert ko_mask to tensor (handle different possible input types)
            if isinstance(ko_mask, list):
                ko_mask = torch.tensor(ko_mask)
            elif isinstance(ko_mask, np.ndarray):
                ko_mask = torch.from_numpy(ko_mask)
            
            geo_data = Data(x=graph_feature, edge_index=ppi_edge_index, internal_edge_index=internal_edge_index, all_edge_index=all_edge_index, label=graph_label, ko_mask=ko_mask)
            geo_datalist.append(geo_data)
        return geo_datalist

    def form_geo_analysis_datalist(self, num_graph, graph_feature_list, all_edge_index, internal_edge_index, ppi_edge_index):
        geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            # CONVERT [numpy] TO [torch]
            graph_feature = torch.from_numpy(graph_feature).float()
            geo_data = Data(x=graph_feature, edge_index=ppi_edge_index, internal_edge_index=internal_edge_index, all_edge_index=all_edge_index)
            geo_datalist.append(geo_data)
        return geo_datalist


def read_pretrain_batch(index, upper_index, x_input, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index):
    # FORMING BATCH FILES
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = x_input[index : upper_index, :]
    print(xBatch.shape)
    # PREPARE LOADING LISTS OF [features, labels, drugs, edge_index]
    print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_node, xBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    geo_datalist = ReadGeoGraph().form_geo_pretrain_datalist(num_graph, graph_feature_list, all_edge_index, internal_edge_index, ppi_edge_index)
    return geo_datalist


def read_batch(index, upper_index, x_input, y_input, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index):
    # FORMING BATCH FILES
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = x_input[index : upper_index, :]
    yBatch = y_input[index : upper_index, :]
    print(xBatch.shape)
    print(yBatch.shape)
    # PREPARE LOADING LISTS OF [features, labels, drugs, edge_index]
    print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_node, xBatch)
    # print('READING BATCH LABELS ...')
    graph_label_list = ReadGeoGraph().read_label(yBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    geo_datalist = ReadGeoGraph().form_geo_datalist(num_graph, graph_feature_list, graph_label_list, all_edge_index, internal_edge_index, ppi_edge_index)
    return geo_datalist


def read_index_batch(index, upper_index, x_input, y_input_index, y_input_label, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index):
    # FORMING BATCH FILES
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    yBatch_index = y_input_index[index : upper_index, :].flatten()
    xBatch = x_input[yBatch_index, :]
    yBatch = y_input_label[index : upper_index, :]
    print(xBatch.shape)
    print(yBatch.shape)
    # PREPARE LOADING LISTS OF [features, labels, drugs, edge_index]
    print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_node, xBatch)
    # print('READING BATCH LABELS ...')
    graph_label_list = ReadGeoGraph().read_label(yBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    geo_datalist = ReadGeoGraph().form_geo_datalist(num_graph, graph_feature_list, graph_label_list, all_edge_index, internal_edge_index, ppi_edge_index)
    return geo_datalist


def read_index_komask_batch(index, upper_index, x_input, y_input_index, yTr_ko_mask, y_input_label, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index):
    # FORMING BATCH FILES
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    yBatch_index = y_input_index[index : upper_index, :].flatten()
    xBatch = x_input[yBatch_index, :] # x input need to be fetched by y index
    yBatch = y_input_label[index : upper_index, :] # y label is stable with index from 0-end
    ko_mask_list_array = yTr_ko_mask[index : upper_index] # yTr_ko_mask is stable with index 0-end
    print(xBatch.shape)
    print(yBatch.shape)
    print(ko_mask_list_array.shape)
    # PREPARE LOADING LISTS OF [features, labels, drugs, edge_index]
    print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_node, xBatch)
    # print('READING BATCH LABELS ...')
    graph_label_list = ReadGeoGraph().read_label(yBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    geo_datalist = ReadGeoGraph().form_geo_datalist_with_komask(num_graph, graph_feature_list, graph_label_list, all_edge_index, internal_edge_index, ppi_edge_index, ko_mask_list_array)
    return geo_datalist


def read_analysis_batch(index, upper_index, x_input, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index):
    # FORMING BATCH FILES
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = x_input[index : upper_index, :]
    print(xBatch.shape)
    # PREPARE LOADING LISTS OF [features, labels, drugs, edge_index]
    print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_node, xBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    geo_datalist = ReadGeoGraph().form_geo_analysis_datalist(num_graph, graph_feature_list, all_edge_index, internal_edge_index, ppi_edge_index)
    return geo_datalist