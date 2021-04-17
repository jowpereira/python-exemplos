from Modulo import CModelLinear

import numpy as np

if __name__ == "__main__":
    
    model = CModelLinear(r'Base\VALE3.SA.csv')
    model.import_file()

    model.clear_nan()

    model.corr(model.df())

    feature, target = model.select_vars(['High', 'Low', 'Close'], ['Open'], -1)

    x_treino, x_teste, y_treino, y_teste = model.split_dataset(feature, target, 0.7)

    model.fit(x_treino, y_treino)

    pred = model.predict(x_teste)
    #
    print(model.mean_absolute_percentage_error(y_teste, pred))
    print(model.R2(y_teste, pred))
    #
    model.plot(pred, y_teste, 'previsto', 'real')