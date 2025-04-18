import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


def pre_processing(df_graph, 
                   df_features, 
                   col_features,
                   col_target='severity'):

    # encoder target
    encoder = LabelEncoder()
    # create node
    df_features['node'] = np.arange(0, df_features.shape[0])

    # select graph with same node features
    nodes_of_features = list(df_features.leg_pos.unique())
    df_graph_subsample = df_graph.query(" pos1 in @nodes_of_features and pos2 in @nodes_of_features ")
    
    # sync nodes
    df_graph_subsample['src'] = pd.NA
    df_graph_subsample['dst'] = pd.NA
    for i in tqdm(range(df_features.shape[0])):
        node_emb, node = df_features[['leg_pos', 'node']].values[i]
        df_graph_subsample['src'][df_graph_subsample.query(f" pos1 == '{node_emb}' ").index] = node
        df_graph_subsample['dst'][df_graph_subsample.query(f" pos2 == '{node_emb}' ").index] = node

    # subsample graph
    df_graph_subsample = df_graph_subsample.astype({'src': int, 'dst': int})
    # 
    # define x features and target
    #col_features = ['relSESA','consurf_old']
    # 
    df_features[col_target] = encoder.fit_transform(df_features[col_target].values).astype(float)
    #
    pos = df_features.leg_pos.values
    x = torch.tensor(df_features[col_features].values,  dtype=torch.float)
    y = torch.tensor(df_features[col_target].values, dtype=torch.long)
    # index of graph
    edge_index = torch.tensor(df_graph_subsample[['src', 'dst']].values, dtype=torch.long)
    
    #df_graph_subsample[col_target] = encoder.fit_transform(df_graph_subsample[col_target].values).astype(float)
    #edge_labels = torch.tensor(df_graph_subsample[col_target].values, dtype=torch.long)
    
    # weigths edges
    edge_weigths = torch.tensor(df_graph_subsample[['weight']].values, dtype=torch.float)

    return x, y, edge_index, edge_weigths, pos