import torch
import pandas as pd
from loguru import logger as log
from val import calculate_classification_metrics
from collections import Counter

def train(model, data, optimizer, criterion, nb_epochs, device='cpu', type_forward='all'):
    loss_scores = []
    m = torch.nn.Sigmoid()
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

        #loss = criterion(out, data.edge_label.to(device))
        loss = criterion(m(out[data.train_mask]), data.y[data.train_mask].to(device)) 
        loss.backward()
        optimizer.step()
        loss_scores.append(loss.item())

    return model, loss_scores 


def test(model, data, device='cpu', type_forward='all', tresh=0.5):
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
        
        prob = m(out[data.test_mask]).cpu().numpy()
        pred = [0 if x < tresh else 1 for x in prob]
        true = data.y[data.test_mask].to(device)#.argmax(dim=1)

    
        return pred, true.cpu().numpy(), prob


def save_inference(y_true, y_pred, y_prob, model_name, fpath):

    df_pred = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'prob': y_prob})
    
    fpath = f'{fpath}{model_name}_predictions.csv'
    df_pred.to_csv(fpath, index=False)
    log.info(f"Saved: {fpath}")

    return df_pred


def calculate_metrics_test(df_test, mode_name, fpath, save=True, multi=False):
    #unpack
    y_true = df_test.y_true.to_numpy()
    y_pred = df_test.y_pred.to_numpy()

    print(f"Counter true: {Counter(y_true)}")
    print(f"Counter pred: {Counter(y_pred)}")

    res = calculate_classification_metrics(y_true, y_pred, mode_name, multi=multi)

    return res

