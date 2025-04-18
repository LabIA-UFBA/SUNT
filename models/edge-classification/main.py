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
#from workflow import train, test
from workflow import train_bin, test_bin
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

# run_model(model, train_dataset, test_dataset, mn, epcs, device=device)
def run_model(model, 
              train_dataset, 
              test_dataset, 
              model_name, 
              epochs=500, 
              LR = 0.01, 
              device='cpu', 
              tf='all'):


    log.info(f"Device: {device}")
    log.info(f"model: {model}")    

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)   

    # train
    log.info(f"Train {model_name} ...")
    log.info(f"Number of epochs .....: {epochs}")
    log.info(f"Learning rate ... ....: {LR}")
    model, training_loss = train_bin(model, 
                                 train_dataset, 
                                 optimizer, 
                                 criterion, 
                                 epochs, 
                                 device=device, 
                                 type_forward=tf)

    log.info(f"Train - loss in firts epoch ...: {training_loss[0]:.5f}")
    log.info(f"Train - loss in last epoch.....: {training_loss[-1]:.5f}")
    log.info(f"Train - loss stride ...........: {training_loss[0]-training_loss[-1]:.5f}")
    log.info(f"Train - loss variation ........: {np.std(training_loss):.5f}")

    fpath_model =  f'../../outputs/classification/edges/rebuttal/weights/{model_name}.pth'
    torch.save(model.state_dict(), fpath_model)
    log.info(f"Saved weigths: {fpath_model}")

    # save model loss
    df_loss = pd.DataFrame({'Epochs': np.arange(1, epochs+1),
                            'loss': training_loss})
    fpath_loss =  f'../../outputs/classification/edges/rebuttal/train_loss/loss_{model_name}.csv'
    df_loss.to_csv(fpath_loss, index=False)
    log.info(f"Saved train loss: {fpath_loss}")

    
    # test
    log.info(f"Testing {model_name} ...")
    y_pred, y_true, prob = test_bin(model, 
                              test_dataset, 
                              device, 
                              tf)

    # save forecasting
    log.info("Save to ground true and forecasting...")
    fpath_pred = '../../outputs/classification/edges/rebuttal/inferences/'
    df_test = save_inference(y_true, y_pred, prob, model_name, fpath_pred)

    # calculate metrics forecasting
    log.info("Calculate metrics...")
    fpath_metrics = '../../outputs/classification/edges/rebuttal/metrics/'
    
    
    return calculate_metrics_test(df_test, model_name, fpath_metrics)


def run_gcn(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    out_layer    = train_dataset.edge_label.shape[1]
    mn = f'gcn{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")
    model =  mm.GCNEdgeClassifier(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device)


def run_cheb(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    out_layer    = train_dataset.edge_label.shape[1]
    k=3
    mn = f'cheb{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")
    log.info(f"Filter size  .........: {k}")
    model =  mm.ChebEdgeClassifier(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer, 
                                  filter_size=k).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device)


def run_sage(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    out_layer    = train_dataset.edge_label.shape[1]
    mn = f'sage{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")
    model =  mm.SAGEEdgeClassifier(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device, tf='not_edge_attr')


def run_gat(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    #out_layer    = 2
    out_layer    = train_dataset.edge_label.shape[1]
    heads=3
    mn = f'gat{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")
    log.info(f"Num Heads  ...........: {heads}")
    model =  mm.GATEdgeClassifier(in_channels=input_layer,
                                  hidden_channels=hidden_layer,
                                  out_channels=out_layer, 
                                  heads=heads).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device)


def run_LEConvGNN(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    #out_layer    = 2
    out_layer    = train_dataset.edge_label.shape[1]
    mn = f'LEConvGNN{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.LEConvGNNEdgeClassifier(in_channels=input_layer,
                                        hidden_channels=hidden_layer,
                                        out_channels=out_layer).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device)



def run_SSGConv(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    #out_layer    = 2
    out_layer    = train_dataset.edge_label.shape[1]
    mn = f'SSGConv{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.SSGConvEdgeClassifier(in_channels=input_layer,
                                        hidden_channels=hidden_layer,
                                        out_channels=out_layer).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device)


def run_EGConv(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    #out_layer    = 2
    out_layer    = train_dataset.edge_label.shape[1]
    mn = f'EGConv{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.EGConvEdgeClassifier(in_channels=input_layer,
                                        hidden_channels=hidden_layer,
                                        out_channels=out_layer).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device, tf='not_edge_attr')


def run_AntiSymmetricConv(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    #out_layer    = 2
    out_layer    = train_dataset.edge_label.shape[1]
    mn = f'AntiSymmetricConv{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.AntiSymmetricConvEdgeClassifier(in_channels=input_layer,
                                                hidden_channels=hidden_layer,
                                                out_channels=out_layer).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device, tf='not_edge_attr')


def run_SuperGATConvEdgeClassifier(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    #out_layer    = 2
    out_layer    = train_dataset.edge_label.shape[1]
    mn = f'SuperGATConv{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.SuperGATConvEdgeClassifier(in_channels=input_layer,
                                           hidden_channels=hidden_layer,
                                           out_channels=out_layer).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device, tf='not_edge_attr')



def run_PANConvEdgeClassifier(train_dataset, test_dataset, sufix=''):
    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_layer  = train_dataset.num_features
    hidden_layer = 64
    #out_layer    = 2
    out_layer    = train_dataset.edge_label.shape[1]
    mn = f'PANConv{sufix}'
    log.add(f'../../outputs/classification/edges/rebuttal/{mn}.log')
    epcs=50
    
    log.info(f"input Layer ..........: {input_layer}")
    log.info(f"Hidden Layer .........: {hidden_layer}")
    log.info(f"Output Layer .........: {out_layer}")

    model =  mm.PANConvEdgeClassifier(in_channels=input_layer,
                                      hidden_channels=hidden_layer,
                                      out_channels=out_layer).to(device)

    return run_model(model, train_dataset, test_dataset, mn, epcs, device=device, tf='not_edge_attr')




def cross_valition():

    data = []
    for fold_idx in np.arange(1, 11):

        train_dataset = torch.load(f'../../data/graph_designer/train_test_edge_classification_days/train_data_{fold_idx}.pt')
        test_dataset  = torch.load(f'../../data/graph_designer/train_test_edge_classification_days/test_data_{fold_idx}.pt')

        #
        # Run models
        #

        log.info('-' * 30)
        log.info("Run models ....")
        log.info('-' * 30)
        #
        # GNNs
        #
        
        scores_pan   = run_PANConvEdgeClassifier(train_dataset, test_dataset, f"_{fold_idx}")
        scores_sgat  = run_SuperGATConvEdgeClassifier(train_dataset, test_dataset, f"_{fold_idx}")
        scores_gat   = run_gat(train_dataset, test_dataset, f"_{fold_idx}")
        scores_cheb  = run_cheb(train_dataset, test_dataset, f"_{fold_idx}")
        scores_gcn   = run_gcn(train_dataset, test_dataset, f"_{fold_idx}")
        scores_sage  = run_sage(train_dataset, test_dataset, f"_{fold_idx}")
        scores_lec   = run_LEConvGNN(train_dataset, test_dataset, f"_{fold_idx}")
        scores_ssg   = run_SSGConv(train_dataset, test_dataset, f"_{fold_idx}")
        scores_egc   = run_EGConv(train_dataset, test_dataset, f"_{fold_idx}")
        scores_asc   = run_AntiSymmetricConv(train_dataset, test_dataset, f"_{fold_idx}")
        
        scores_gat['fold']  = fold_idx
        scores_cheb['fold'] = fold_idx
        scores_gcn['fold']  = fold_idx
        scores_sage['fold'] = fold_idx
        scores_lec['fold']  = fold_idx
        scores_ssg['fold']  = fold_idx
        scores_egc['fold']  = fold_idx
        scores_asc['fold']  = fold_idx
        scores_sgat['fold'] = fold_idx
        scores_pan['fold']  = fold_idx
            
        data.append(scores_gat)
        data.append(scores_cheb)
        data.append(scores_gcn)
        data.append(scores_sage)
        data.append(scores_lec)
        data.append(scores_ssg)
        data.append(scores_egc)
        data.append(scores_asc)
        data.append(scores_sgat)
        data.append(scores_pan)
        
        
    df_result = pd.DataFrame(data)
    #fpath_metrcis = '../../outputs/classification/edges/rebuttal/metrics/cross_validation.csv'
    fpath_metrcis = '../../outputs/classification/edges/rebuttal/metrics/cross_validation.csv'
    
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




    

      