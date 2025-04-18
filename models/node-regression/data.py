import pickle

def load_dataset(fpath):
    # Load the StaticGraphTemporalSignal object from the file
    with open(fpath, 'rb') as f:
        loaded_temporal_signal = pickle.load(f)
    return loaded_temporal_signal
    
