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
from data import load_dataset
from workflow import train, test
from workflow import calculate_metrics_test, save_forecasting
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
              train_dataset, 
              test_dataset, 
              df_nodes, 
              nodes, 
              df_nodes_loader, 
              model_name, 
              epochs=500, 
              LR = 0.001, 
              device='cpu', 
              tf='all'):


    log.info(f"Device: {device}")
    log.info(f"model: {model}")    

    optimizer = optim.Adam(model.parameters(), lr=LR)   

    # train
    log.info(f"Train {model_name} ...")
    log.info(f"Number of epochs .....: {epochs}")
    log.info(f"Learning rate ... ....: {LR}")
    model, training_loss = train(model, 
                                 train_dataset, 
                                 optimizer,  
                                 number_epochs=epochs, 
                                 device=device, 
                                 type_forward=tf)

    log.info(f"Train - loss in firts epoch ...: {training_loss[0]:.5f}")
    log.info(f"Train - loss in last epoch.....: {training_loss[-1]:.5f}")
    log.info(f"Train - loss stride ...........: {training_loss[0]-training_loss[-1]:.5f}")
    log.info(f"Train - loss variation ........: {np.std(training_loss):.5f}")

    fpath_model =  f'../../outputs/regression/weights/{model_name}.pth'
    torch.save(model.state_dict(), fpath_model)
    log.info(f"Saved weigths: {fpath_model}")

    # save model loss
    df_loss = pd.DataFrame({'Epochs': np.arange(1, epochs+1),
                            'loss': training_loss})
    fpath_loss =  f'../../outputs/regression/train_loss/loss_{model_name}.csv'
    df_loss.to_csv(fpath_loss, index=False)
    log.info(f"Saved train loss: {fpath_loss}")

    
    # test
    log.info(f"Testing {model_name} ...")
    df_forecasting = test(model, 
                          test_dataset, 
                          nodes, 
                          device, 
                          tf)

    # save forecasting
    log.info("Sync forecasting to ground true...")
    df_test_val = save_forecasting(df_nodes_loader, df_forecasting, model_name)

    # calculate metrics forecasting
    log.info("Calculate metrics...")
    calculate_metrics_test(df_test_val, nodes, model_name)


def run_cheb(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    K = 3
    mn = 'cheb'
    log.add(f'../../outputs/{mn}.log')
    epcs=500
    
    log.info(f"Hidden Layer ... .....: {hidden_layer}")
    log.info(f" Filters K cheb: {K}")
    model =  mm.ChebNet(in_channels=36, 
                     hidden_channels=hidden_layer, 
                     out_channels=12, 
                     filter_size=K).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device)


def run_gcn(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    mn = 'gcn'
    log.add(f'../../outputs/{mn}.log')
    epcs=500
    
    log.info(f"Hidden Layer ... .....: {hidden_layer}")
    model =  mm.GCN(in_channels=36,
                hidden_channels=hidden_layer,
                out_channels=12).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device)


def run_sage(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    mn = 'sage'
    log.add(f'../../outputs/{mn}.log')
    epcs=500
    
    log.info(f"Hidden Layer ... .....: {hidden_layer}")
    model =  mm.GraphSAGENet(in_channels=36,
                          hidden_channels=hidden_layer,
                          out_channels=12).to(device)

    run_model(model, 
              train_dataset, 
              test_dataset, 
              df_nodes, 
              nodes, 
              df_nodes_loader, 
              mn, 
              epcs, 
              device=device, 
              tf='not_edge_attr')


def run_gat(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    HEADS = 1
    mn = 'gat'
    log.add(f'../../outputs/{mn}.log')
    epcs=500
    
    log.info(f"Number of heads: {HEADS}")
    model =  mm.GAT(in_channels=36, 
                     hidden_channels=hidden_layer, 
                     out_channels=12, 
                     heads=HEADS).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device)


def run_gru(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    mn = 'gru'
    log.add(f'../../outputs/regression/{mn}.log')
    epcs=3000
    
    model =  mm.GRU(in_channels=36, 
                     hidden_channels=hidden_layer, 
                     out_channels=12).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device, tf='not_edge_attr')


def run_lstm(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    mn = 'lstm'
    log.add(f'../../outputs/regression/{mn}.log')
    epcs=3000
    
    model =  mm.LSTM(in_channels=36, 
                     hidden_channels=hidden_layer, 
                     out_channels=12).to(device)

    run_model(model, 
              train_dataset, 
              test_dataset, 
              df_nodes, 
              nodes, 
              df_nodes_loader, 
              mn, 
              epcs, 
              device=device, 
              tf='not_edge_attr')


def run_gconv_gru(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    K = 1
    mn = 'gconv_gru'
    log.add(f'../../outputs/{mn}.log')
    epcs=500
    
    log.info(f"Number of k: {K}")
    model =  mm.RecurrentGConvGRU(in_channels=36, 
                                  hidden_channels=hidden_layer, 
                                  out_channels=12, 
                                  k=K).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device)


def run_gconv_lstm(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    K = 1
    mn = 'gconv_lstm'
    log.add(f'../../outputs/regression/{mn}.log')
    epcs=3000
    
    log.info(f"Number of k: {K}")
    model =  mm.RecurrentGConvLSTM(in_channels=36, 
                                  hidden_channels=hidden_layer, 
                                  out_channels=12, 
                                  k=K).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device, tf='gconv_lstm')


def run_tgcn(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    mn = 'tgcn'
    log.add(f'../../outputs/regression/{mn}.log')
    epcs=3000
    model =  mm.RecurrentTGCN(in_channels=36, 
                              hidden_channels=hidden_layer, 
                              out_channels=12).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device, tf='tgcn')


def run_dcrnn(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    K = 1
    mn = 'dcrnn'
    log.add(f'../../outputs/regression/{mn}.log')
    epcs=3000
    
    log.info(f"Number of k: {K}")
    model =  mm.RecurrentDCRNN(in_channels=36, 
                               hidden_channels=hidden_layer, 
                               out_channels=12, 
                               k=K).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device)


def run_a3tgcn(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader):
    # define model
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    hidden_layer = 128
    p = 1
    mn = 'a3tgcn'
    log.add(f'../../outputs/regression/{mn}.log')
    epcs=3000
    
    log.info(f"Number of Perios: {p}")
    model =  mm.RecurrentA3TGCN(in_channels=1, 
                               hidden_channels=hidden_layer, 
                               out_channels=12, 
                               periods=p).to(device)

    run_model(model, train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader, mn, epcs, device=device)


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

    #
    # Run models
    #

    log.info('-' * 30)
    log.info("Run models ....")
    log.info('-' * 30)
    #
    # GCN
    #
    #run_cheb(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)
    #run_gcn(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)
    #run_sage(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)
    
    #
    # DNN temporal
    #
    #run_gru(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)
    #run_lstm(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)

    #
    # GCN with attention
    # 
    #run_gat(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)

    #
    # GCN temporal
    #
    #run_gconv_gru(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)
    #run_gconv_lstm(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)

    #
    # GCN temporal spatial
    #
    #run_tgcn(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)
    #run_dcrnn(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)
    run_a3tgcn(train_dataset, test_dataset, df_nodes, nodes, df_nodes_loader)
    
    log.info('Done')
    log.info('-' * 30)




    

      