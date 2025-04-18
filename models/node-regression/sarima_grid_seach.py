import ast
import time
import pickle
import warnings
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

from datetime import timedelta 
from data import load_dataset
from loguru import logger as log
 
warnings.simplefilter('ignore')

# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            pass
    # check for an interesting result
#    if result is not None:
#        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] is not None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = range(2)
    d_params = range(2)
    q_params = range(5)
    t_params = ['n','c','t','ct']
    P_params = range(5)
    D_params = range(2)
    Q_params = range(2)
    #m_params = range(2)
    m_params = [123, 240]
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models


def run_grid_seach_sarima(data, n_test, fpath_out):
    mn = 'sarima_best_params_gs_stats'
    log.add(f'../../outputs/{mn}.log')
    #data = data['y'].values
    log.info(f"shape data: {data.shape}")
    # data split
    #n_test = 10
    # model configs
    cfg_list = sarima_configs()
    log.info(f'Configurations: {len(cfg_list)} ')
    # grid search
    scores = grid_search(data['y'].values, cfg_list, n_test)
    log.info('done')
    # list top 3 configs
    for cfg, error in scores[:1]:
        log.info(f" best params: {cfg}, error: {error} ")
        #cfg = list(cfg)
        cfg = ast.literal_eval(cfg)
        best_params = {
            'order': cfg[0],               # p, d, q parameters
            'seasonal_order': cfg[1],   # P, D, Q, s parameters
            'trend': cfg[2]                      # Trend parameter
        }

    log.info("Grid seach Sarima Done")
    log.info(f"Best Params: {best_params}")

    log.info('Save best params')
    # Save the best parameters to a file
    with open(fpath_out, 'wb') as f:
        pickle.dump(best_params, f)



if __name__ == '__main__':

    ini_time = time.time()
    
    # define dataset
    train_dataset = load_dataset('../../data/graph_designer/train_test/dataset_train.pkl')
    test_dataset = load_dataset('../../data/graph_designer/train_test/dataset_test.pkl')
    df_nodes = pd.read_csv('../../data/graph_designer/train_test/df_nodes_selected.csv')
    nodes = list(df_nodes.tensor_idx.values)
    df_nodes_loader = pd.read_csv('../../data/graph_designer/train_test/df_nodes_selected_loader.csv')
    df_nodes_loader['time'] = pd.to_datetime(df_nodes_loader['time'], format='%Y-%m-%d %H:%M:%S')
    
    node = nodes[0]

    fpath_model =  f'../../outputs/weights/sarima_stats_{node}.pth'
    # select train ts
    df_train_kats = df_nodes_loader.query(" partition == 'train' ").copy()
    ts = df_train_kats[[f'carregamento_node_{node}']]
    ts = ts.rename({f'carregamento_node_{node}': 'y'}, axis=1)

    total = ts.shape[0]
    test_size = int(total * 0.1)
    train_size = total - test_size

    log.info(f"Time series size .........: {total}.")
    log.info(f"Train size ...............: {train_size}")
    log.info(f"Test size ................: {test_size}")
    
    run_grid_seach_sarima(ts, test_size, fpath_model)

    end = time.time()
    time_cons = end - ini_time
    time_cons = str(timedelta(seconds=time_cons)).split('.')[0]
    log.info(f"Process node in {time_cons}.")
