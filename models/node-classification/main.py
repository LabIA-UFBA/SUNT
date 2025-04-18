# basic
import os
import numpy as np
import pandas as pd
# pre processing
# NN
import torch
import torch.optim as optim
# val and plot
from loguru import logger as log
# data
from workflow import train, test
from workflow import calculate_metrics_test
from workflow import save_inference
# models
import models as mm
#
# Config - set all seed
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

def run_model(model, 
              data,
              model_name, 
              epochs=500, 
              LR = 0.01, 
              device='cpu', 
              tf='all'):

    log.info(f"Device: {device}")
    log.info(f"model: {model}")    

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)   

    # train
    log.info(f"Train {model_name} ...")
    log.info(f"Number of epochs .....: {epochs}")
    log.info(f"Learning rate ... ....: {LR}")
    model, training_loss = train(model, 
                                 data, 
                                 optimizer, 
                                 criterion, 
                                 epochs, 
                                 device=device, 
                                 type_forward=tf)

    log.info(f"Train - loss in firts epoch ...: {training_loss[0]:.5f}")
    log.info(f"Train - loss in last epoch.....: {training_loss[-1]:.5f}")
    log.info(f"Train - loss stride ...........: {training_loss[0]-training_loss[-1]:.5f}")
    log.info(f"Train - loss variation ........: {np.std(training_loss):.5f}")

    fpath_model =  f'../../outputs/classification/node/rebuttal/weights/{model_name}.pth'
    torch.save(model.state_dict(), fpath_model)
    log.info(f"Saved weigths: {fpath_model}")

    # save model loss
    df_loss = pd.DataFrame({'Epochs': np.arange(1, epochs+1),
                            'loss': training_loss})
    fpath_loss =  f'../../outputs/classification/node/rebuttal/train_loss/loss_{model_name}.csv'
    df_loss.to_csv(fpath_loss, index=False)
    log.info(f"Saved train loss: {fpath_loss}")

    # test
    log.info(f"Testing {model_name} ...")
    y_pred, y_true, prob = test(model, data, device, tf)

    print(f"prob shape: {prob.flatten().shape}")

    # save forecasting
    log.info("Save to ground true and forecasting...")
    fpath_pred = '../../outputs/classification/node/rebuttal/inferences/'
    df_test = save_inference(y_true.flatten(), y_pred, prob.flatten(), model_name, fpath_pred)

    # calculate metrics forecasting
    log.info("Calculate metrics...")
    fpath_metrics = '../../outputs/classification/node/rebuttal/metrics/'
    
    return calculate_metrics_test(df_test, model_name, fpath_metrics, multi=False)


def run_gcn(data, sufix=''):
    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    mn = f'gcn{sufix}'
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")
    model =  mm.GCN(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer).to(device)

    return run_model(model, data, mn, epcs, device=device)


def run_cheb(data, sufix=''):
    # define model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    k=1
    mn = f'cheb{sufix}'
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")
    log.info(f"Filter size  .........: {k}")
    model =  mm.ChebNet(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer, 
                                  filter_size=k).to(device)

    return run_model(model, data, mn, epcs, device=device)


def run_sage(data, sufix=''):
    # define model
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    mn = f'sage{sufix}'
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")
    model =  mm.GraphSAGENet(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer).to(device)

    return run_model(model, data, mn, epcs, device=device, tf='not_edge_attr')


def run_gat(data, sufix=''):
    # define model
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    heads=1
    mn = f'gat{sufix}'
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")
    log.info(f"Num Heads  ...........: {heads}")
    model =  mm.GAT(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer, 
                                  heads=heads).to(device)

    return run_model(model, data, mn, epcs, device=device)


def run_ssg_conv(data, sufix=''):
    # define model
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    heads=1
    mn = f'ssg_conv{sufix}'
    #log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.SSGConvNN(in_channels=input_layer,
                          hidden_channels=hidden_layer,
                          out_channels=out_layer).to(device)

    return run_model(model, data, mn, epcs, device=device)


def run_AntiSymmetricConv(data, sufix=''):
    # define model
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    heads=1
    mn = f'AntiSymmetricConv{sufix}'
    #log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.AntiSymmetricConvGNN(in_channels=input_layer,
                                     out_channels=out_layer).to(device)

    return run_model(model, data, mn, epcs, device=device, tf='not_edge_attr')


def run_EGConvGNN(data, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    mn = f'EGConvGNN{sufix}'
    #log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.EGConvGNN(in_channels=input_layer,
                          hidden_channels=hidden_layer,
                          out_channels=out_layer).to(device)

    return run_model(model, data, mn, epcs, device=device, tf='not_edge_attr')


# 
def run_LEConvGNN(data, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    
    mn = f'LEConvGNN{sufix}'
    #log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.LEConvGNN(in_channels=input_layer,
                          hidden_channels=hidden_layer,
                          out_channels=out_layer).to(device)

    return run_model(model, data, mn, epcs, device=device)


def run_SuperGATConvNN(data, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    mn = f'SuperGATConv{sufix}'
    #log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.SuperGATConvNN(in_channels=input_layer,
                               hidden_channels=hidden_layer,
                               out_channels=out_layer).to(device)

    return run_model(model, data, mn, epcs, device=device, tf='not_edge_attr')


def run_PANConvNN(data, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = data.num_features
    hidden_layer = 64
    out_layer    = 1#data.y.shape[1]
    mn = f'PANConv{sufix}'
    #log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    log.add(f'../../outputs/classification/node/rebuttal/{mn}.log')
    
    epcs=100
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.PANConvNN(in_channels=input_layer,
                          hidden_channels=hidden_layer,
                          out_channels=out_layer).to(device)

    return run_model(model, data, mn, epcs, device=device, tf='not_edge_attr')


def cross_valition():

    data_dict = []
    for fold_idx in np.arange(1, 11):

        data = torch.load(f'../../data/graph_designer/train_test_node_classification_days/data_{fold_idx}.pt')

        #
        # Run models
        #
        log.info('-' * 30)
        log.info("Run models ....")
        log.info('-' * 30)
        #
        # GNNs
        #
        scores_gat   = run_gat(data, f"_{fold_idx}")
        scores_cheb  = run_cheb(data, f"_{fold_idx}")
        scores_gcn   = run_gcn(data, f"_{fold_idx}")
        scores_sage  = run_sage(data, f"_{fold_idx}")
        scores_ssg   = run_ssg_conv(data, f"_{fold_idx}")
        scores_asc   = run_AntiSymmetricConv(data, f"_{fold_idx}")
        scores_egc   = run_EGConvGNN(data, f"_{fold_idx}")
        scores_lec   = run_LEConvGNN(data, f"_{fold_idx}")
        scores_sgat  = run_SuperGATConvNN(data, f"_{fold_idx}")
        scores_pan   = run_PANConvNN(data, f"_{fold_idx}")
        
        
        scores_gcn['fold']   = fold_idx
        scores_gat['fold']   = fold_idx
        scores_cheb['fold']  = fold_idx 
        scores_sage['fold']  = fold_idx
        scores_ssg['fold']   = fold_idx
        scores_asc['fold']   = fold_idx
        scores_egc['fold']   = fold_idx
        scores_lec['fold']   = fold_idx
        scores_sgat['fold']  = fold_idx
        scores_pan['fold']   = fold_idx
        
            
        data_dict.append(scores_gat)
        data_dict.append(scores_cheb)
        data_dict.append(scores_gcn)
        data_dict.append(scores_sage)
        data_dict.append(scores_ssg)
        data_dict.append(scores_asc)
        data_dict.append(scores_egc)
        data_dict.append(scores_lec)
        data_dict.append(scores_sgat)
        data_dict.append(scores_pan)
        
    
    df_result = pd.DataFrame(data_dict)
    #fpath_metrcis = '../../outputs/classification/node/rebuttal/metrics/cross_validation.csv'
    fpath_metrcis = '../../outputs/classification/node/rebuttal/metrics/cross_validation.csv'
    df_result['model_name'] = df_result['model'].apply(lambda x: x.split('_')[0])
    df_result.to_csv(fpath_metrcis, index=False)
    log.info(f"Save cross validation metrics in : {fpath_metrcis} ")

    df_mean = df_result.pivot_table(index='model_name', 
                                    values=['accuracy', 'mcc', 'precision', 'recall', 'f1'],
                                    aggfunc='mean').reset_index()

    log.info("Mean scores of cross validation: ")
    log.info(df_mean)


    log.info('Done')
    log.info('-' * 30)


if __name__ == '__main__':
    #
    # Data
    # 
    # cross valdation 
    cross_valition()




    

      