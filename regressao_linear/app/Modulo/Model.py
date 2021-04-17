from sklearn.linear_model import LinearRegression
from .Preprocessing import CPreprocessing
from .Metrics import CMetrics
from .Graphics import CGraphics

class CModelLinear(CPreprocessing, CMetrics, CGraphics):
    def __init__(self, path):
        self.__model = LinearRegression()
        super(CModelLinear, self).__init__(path)
    
    def fit(self, X, y):
        self.__model.fit(X, y)
    
    def predict(self, X):
        return self.__model.predict(X)
    
    ###