import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit

class CPreprocessing(object):
    def __init__(self, path):
        self.__path = path
        self.__file = None

    def import_file(self, path=None):
        if path:
            self.__path = path
         
        self.__file = pd.read_csv(self.__path)
    
    def clear_nan(self):
        self.__file = self.__file.dropna()

    def select_vars(self, features, target, lag):
        if lag != 0:
            return np.array(self.__file[features])[:lag], np.array(self.__file[target].shift(periods=lag, fill_value='NaN')).reshape(-1,1)[:lag] 

        elif lag==0:
            return np.array(self.__file[features]), np.array(self.__file[target].shift(periods=lag, fill_value='NaN')).reshape(-1,1)
    
    
    def split_datasetX(self, features, target, test_size=0.1, shuffle=False, random_state=42):
        return train_test_split(features, target,test_size=test_size, shuffle=shuffle, random_state=random_state)
    
    def split_dataset(self, features, target, max_train_size=0.7):
        tscv = TimeSeriesSplit()
        TimeSeriesSplit(max_train_size=max_train_size)
        for train_index, test_index in tscv.split(features):
            print("TRAIN:", train_index, "TEST:", test_index)

            x_treino, x_teste = features[train_index], features[test_index]
            y_treino, y_teste = target[train_index], target[test_index]
        
        return x_treino, x_teste, y_treino, y_teste
    
    def df(self):
        return self.__file