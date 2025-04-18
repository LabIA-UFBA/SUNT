# basic
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# pre processing
from sklearn import preprocessing as pre
# NN
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss
# val and plot
from torchmetrics.regression import R2Score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from loguru import logger as log
from val import calculate_metrics
# plot
import matplotlib.pyplot as plt
from data import load_dataset
from torch_geometric_temporal.signal import temporal_signal_split
from models import GCN
#
# Config
#
SEED = 1345
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)
plt.style.use('seaborn-whitegrid')
pd.set_option('display.float_format', '{:.16f}'.format)


def train(model, train_dataset, optimizer,  number_epochs=100, device='cpu'):
    
    training_loss = []
    for epoch in tqdm(range(number_epochs)):
        model.train()
        cost = 0
        for time, snapshot in enumerate(train_dataset): # faz o treino em cada bath temporal
            snapshot.to(device)
            
            y_hat = model(snapshot.x, 
                        snapshot.edge_index, 
                        snapshot.edge_attr)
                        
            cost = cost + torch.mean((y_hat-snapshot.y)**2)

        cost = cost / (time+1)
        training_loss.append(cost.cpu().data.numpy())

        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return model, training_loss


def test(model, test_dataset, nodes, device='cpu'):
    model.eval()
    cost, time = 0, 0
    dfs_pred = []

    for time, snapshot in tqdm(enumerate(test_dataset)):
        snapshot.to(device)
            
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        y_true = snapshot.y.cpu().data.numpy()
        y_pred = y_hat.cpu().data.numpy()
        # strore values forecasting
        df_prediction = pd.DataFrame()
        df_prediction['time'] = test_dataset.target_time[time]
        for node in nodes:
            df_prediction[f'y_pred_{node}'] = y_pred[node,:]
        dfs_pred.append(df_prediction)
                    
    cost = cost / (time+1)
    cost = cost.item()
    log.info("MSE test: {:.4f}".format(cost))

    df_forecasting = pd.concat(dfs_pred, ignore_index=True)

    return df_forecasting


def save_forecasting(df_nodes_loader, df_forecasting, model_name):
    
    df_nodes_loader_join = df_nodes_loader.merge(df_forecasting, 
                                        on='time', 
                                        how='left')

    df_test_val = df_nodes_loader_join.query(" partition == 'test' ")
    fpath = f'../../outputs/forecasting/{model_name}_forecasting.csv'
    df_test_val.to_csv(fpath, index=False)
    log.info(f"Saved: {fpath}")

    return df_test_val


def calculate_metrics_test(df_test_val, nodes, mode_name, save=True):
    # forecasting each node test
    dfs_results = []
    for node in nodes:
        res = calculate_metrics(df_test_val[f"carregamento_node_{node}"], df_test_val[f"y_pred_{node}"])
        res['node'] = node
        dfs_results.append(res)

    df_result = pd.DataFrame(dfs_results)
    log.info(f"Mean forecasting metrics for {mode_name}: ")
    log.info(f'{df_result.mean()}')
    log.info("")
    if save:
        fpath = f'../../outputs/forecasting/{mode_name}_metrics.csv'
        df_result.to_csv(fpath, index=False)
        log.info(f"Saved: {fpath}")
    
    return df_result


if __name__ == '__main__':
    #
    # Data
    # 
    train_dataset = load_dataset('../../data/graph_designer/train_test/dataset_train.pkl')
    test_dataset = load_dataset('../../data/graph_designer/train_test/dataset_test.pkl')
    df_nodes = pd.read_csv('../../data/graph_designer/train_test/df_nodes_selected.csv')
    nodes = list(df_nodes.tensor_idx.values)
    df_nodes_loader = pd.read_csv('../../data/graph_designer/train_test/df_nodes_selected_loader.csv')
    df_nodes_loader['time'] = pd.to_datetime(df_nodes_loader['time'], format='%Y-%m-%d %H:%M:%S')
    epochs=500
    hidden_layer = 128
    LR = 0.001
    model_name = 'gcn'

    log.add(f'../../outputs/{model_name}.log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # define model
    model =  GCN(in_channels=36,
                hidden_channels=hidden_layer,
                out_channels=12).to(device)

    log.info(f"model: {model}")    

    optimizer = optim.Adam(model.parameters(), lr=LR)   

    # train
    log.info(f"Train {model_name} ...")
    log.info(f"Number of epochs .....: {epochs}")
    log.info(f"Hidden Layer ... .....: {hidden_layer}")
    log.info(f"Learning rate ... ....: {LR}")
    model, training_loss = train(model, train_dataset, optimizer,  number_epochs=epochs, device=device)

    fpath_model =  f'../../outputs/weights/gcn.pth'
    torch.save(model.state_dict(), fpath_model)
    log.info(f"Saved weigths: {fpath_model}")
    
    # test
    log.info(f"Testing {model_name} ...")
    df_forecasting = test(model, test_dataset, nodes, device)

    # save forecasting
    log.info(f"Sync forecasting to ground true...")
    df_test_val = save_forecasting(df_nodes_loader, df_forecasting, model_name)

    # calculate metrics forecasting
    log.info(f"Calculate metrics...")
    df_result = calculate_metrics_test(df_test_val, nodes, model_name)


    

      