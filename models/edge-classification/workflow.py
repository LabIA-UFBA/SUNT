import torch
import pandas as pd
from loguru import logger as log
from val import calculate_classification_metrics

def train(model, data, optimizer, criterion, nb_epochs, device='cpu', type_forward='all'):
    loss_scores = []
    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()
        if type_forward == 'not_edge_attr':
            out = model(data.x.to(device), 
                        data.edge_index.to(device))
        else:
            out = model(data.x.to(device), 
                        data.edge_index.to(device), 
                        data.edge_attr.to(device))

        loss = criterion(out, data.edge_label.to(device))
        loss.backward()
        optimizer.step()
        loss_scores.append(loss.item())

    return model, loss_scores 


def train_bin(model, data, optimizer, criterion, nb_epochs, device='cpu', type_forward='all'):
    m = torch.nn.Sigmoid()
    loss_scores = []
    for epoch in range(nb_epochs):
        model.train()
        optimizer.zero_grad()

        
        if type_forward == 'not_edge_attr':
            out = model(data.x.to(device), 
                        data.edge_index.to(device))
        else:
            out = model(data.x.to(device), 
                        data.edge_index.to(device), 
                        data.edge_attr.to(device))
            
        x_sig = m(out.view(-1))
        loss = criterion(x_sig, data.edge_label.to(device).flatten().float())
        #loss = criterion(out, data.edge_label.to(device))
        loss.backward()
        optimizer.step()
        loss_scores.append(loss.item())



    return model, loss_scores 


def test_bin(model, data, device='cpu', type_forward='all', tresh=0.511):
    m = torch.nn.Sigmoid()
    model.eval()
    with torch.no_grad():

        if type_forward == 'not_edge_attr':
            out = model(data.x.to(device), 
                        data.edge_index.to(device))
        else:
            out = model(data.x.to(device), 
                        data.edge_index.to(device), 
                        data.edge_attr.to(device))


        x_sig = m(out.view(-1))
        pred = [0 if x < tresh else 1 for x in x_sig]
        true = data.edge_label.to(device).cpu().numpy().flatten()#.argmax(dim=1)

        #pred = out.argmax(dim=1)
        #true = data.edge_label.to(device).argmax(dim=1)
        #correct = (pred == true).sum()
        #acc = correct / data.edge_label.size(0)
        return pred, true, x_sig.cpu().numpy()


def test(model, data, device='cpu', type_forward='all'):
    model.eval()
    with torch.no_grad():

        if type_forward == 'not_edge_attr':
            out = model(data.x.to(device), 
                        data.edge_index.to(device))
        else:
            out = model(data.x.to(device), 
                        data.edge_index.to(device), 
                        data.edge_attr.to(device))

        pred = out.argmax(dim=1)
        true = data.edge_label.to(device).argmax(dim=1)
        correct = (pred == true).sum()
        acc = correct / data.edge_label.size(0)
        return pred.cpu().numpy(), true.cpu().numpy(), acc


def save_inference(y_true, y_pred, prob, model_name, fpath):

    df_pred = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'prob': prob})
    
    fpath = f'{fpath}{model_name}_predictions.csv'
    df_pred.to_csv(fpath, index=False)
    log.info(f"Saved: {fpath}")

    return df_pred


def calculate_metrics_test(df_test, mode_name, fpath, save=True):
    #unpack
    y_true = df_test.y_true.to_numpy()
    y_pred = df_test.y_pred.to_numpy()

    res = calculate_classification_metrics(y_true, y_pred, mode_name)

    return res

    # # Create a DataFrame with the results
    # df_result = pd.DataFrame(res)
    
    # log.info(f"Mean forecasting metrics for {mode_name}: ")
    # log.info(f'{df_result.mean()}')
    # log.info("")
    # if save:
    #     fpath = f'{fpath}{mode_name}_metrics.csv'
    #     df_result.to_csv(fpath, index=False)
    #     log.info(f"Saved: {fpath}.")
    
    # return df_result
