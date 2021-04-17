from sklearn.metrics import r2_score
import numpy as np

class CMetrics(object):

    @staticmethod
    def mean_absolute_percentage_error(X, y):
        return np.mean(np.abs((np.array(X) - np.array(y)) /np.array(X))) * 100
    
    @staticmethod
    def R2(X, y):
        return r2_score(X, y)