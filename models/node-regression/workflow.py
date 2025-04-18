# basic
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# pre processing
# NN
import torch
# val and plot
from loguru import logger as log
from val import calculate_metrics

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

pd.set_option('display.float_format', '{:.16f}'.format)


def train(model, train_dataset, optimizer,  number_epochs=100, device='cpu', type_forward='all'):
    
    training_loss = []
    for epoch in tqdm(range(number_epochs)):
        model.train()
        cost = 0
        h, c = None, None
        for time, snapshot in enumerate(train_dataset): # faz o treino em cada bath temporal
            snapshot.to(device)

            if type_forward == 'not_edge_attr':
                y_hat = model(snapshot.x, 
                              snapshot.edge_index)
            elif type_forward == 'gconv_lstm':
                y_hat, h, c = model(snapshot.x, 
                                    snapshot.edge_index, 
                                    snapshot.edge_attr, h, c)
            elif type_forward == 'tgcn':
                y_hat, h = model(snapshot.x, 
                                 snapshot.edge_index, 
                                 snapshot.edge_attr, h)    
            else:
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


def test(model, test_dataset, nodes, device='cpu', type_forward='all'):
    model.eval()
    cost, time = 0, 0
    dfs_pred = []
    h, c = None, None

    for time, snapshot in tqdm(enumerate(test_dataset)):
        snapshot.to(device)
            
        #y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        if type_forward == 'not_edge_attr':
            y_hat = model(snapshot.x, 
                          snapshot.edge_index)
        elif type_forward == 'gconv_lstm':
            y_hat, h, c = model(snapshot.x, 
                                snapshot.edge_index, 
                                snapshot.edge_attr, h, c)
        elif type_forward == 'tgcn':
            y_hat, h = model(snapshot.x, 
                             snapshot.edge_index, 
                             snapshot.edge_attr, h)    
        else:
            y_hat = model(snapshot.x, 
                          snapshot.edge_index, 
                          snapshot.edge_attr)
        
        # calculate cost
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        snapshot.y.cpu().data.numpy()
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
    fpath = f'../../outputs/regression/forecasting/{model_name}_forecasting.csv'
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
        fpath = f'../../outputs/regression/forecasting/{mode_name}_metrics.csv'
        df_result.to_csv(fpath, index=False)
        log.info(f"Saved: {fpath}")
    
    return df_result

    

      