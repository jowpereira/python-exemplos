import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class CGraphics(object):

    @staticmethod
    def poli2(X, y, xlabel, ylabel):
        plt.scatter(X, y)
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)
        a, b, c= np.polyfit(X, y, 2)
        plt.plot(X, a*pow(X, 2)+X*b+c, color='r')
        plt.show()
    
    @staticmethod
    def poli1(X, y, *args):
        plt.scatter(X, y)
        plt.ylabel(args[0])
        plt.xlabel(args[1])
        a, b = np.polyfit(X, y, 1)
        plt.plot(X, a*X+b, color='r')
        plt.show()
    
    @staticmethod
    def corr(df):
        fig, ax = plt.subplots(figsize=(19, 15)) 
        mask = np.zeros_like(df.corr())
        mask[np.triu_indices_from(mask)] = 1
        sns.heatmap(df.corr(), mask= mask, ax= ax, annot= True).set_title("Correlation Matrix")
        plt.show()
    
    @staticmethod
    def plot(X, y, *args):
        plt.plot(X, label = args[0])
        plt.plot(y, label = args[1])
        plt.legend()
        plt.show()