import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import pmdarima as pm
import MetaTrader5 as mt5
from pandas.plotting import lag_plot
from pmdarima.arima import ndiffs

if __name__ == "__main__":
    
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
    stockdata = pd.DataFrame()
    rates = mt5.copy_rates_from_pos("USDJPY", mt5.TIMEFRAME_M3, 0, 300)
    mt5.shutdown()
    
    df = pd.DataFrame(rates)
    df['time']=pd.to_datetime(df['time'], unit='s')
    #df = df.set_index(['time'])
    print(df.head(5))

    train_len = int(df.shape[0] * 0.7)
    train_data, test_data = df[:train_len], df[train_len:]

    y_train = train_data['open'].values
    y_test = test_data['open'].values

    print(f"{train_len} train samples")
    print(f"{df.shape[0] - train_len} test samples")


    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    plt.title('MSFT Autocorrelation plot')

    # The axis coordinates for the plots
    ax_idcs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1)
    ]

    for lag, ax_coords in enumerate(ax_idcs, 1):
        ax_row, ax_col = ax_coords
        axis = axes[ax_row][ax_col]
        lag_plot(df['open'], lag=lag, ax=axis)
        axis.set_title(f"Lag={lag}")
        
    plt.show()



    kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)

    print(f"Estimated differencing term: {n_diffs}")

    auto = pm.auto_arima(y_train, d=n_diffs, seasonal=False, stepwise=True,
                     suppress_warnings=True, error_action="ignore", max_p=6,
                     max_order=None, trace=True)

    print(auto.order)

    from sklearn.metrics import mean_squared_error
    from pmdarima.metrics import smape

    model = auto

    def forecast_one_step():
        fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
        return (
            fc.tolist()[0],
            np.asarray(conf_int).tolist()[0])

    forecasts = []
    confidence_intervals = []

    for new_ob in y_test:
        fc, conf = forecast_one_step()
        forecasts.append(fc)
        confidence_intervals.append(conf)
        
        # Updates the existing model with a small number of MLE steps
        model.update(new_ob)
        
    print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
    print(f"SMAPE: {smape(y_test, forecasts)}")


    # --------------------- Actual vs. Predicted with confidence intervals ----------------
    plt.plot(y_train, color='blue', marker='.', label='Training Data')
    plt.plot(test_data.index, forecasts, color='green', marker='.', label='Predicted Price')
    plt.plot(test_data.index, y_test, color='red', marker='.', label='Actual Price')
    plt.title('USDJPY Prices Predictions & Confidence Intervals')
    plt.xlabel('Dates')
    plt.ylabel('Prices')

    conf_int = np.asarray(confidence_intervals)
    plt.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], alpha=0.9, color='orange', label="Confidence Intervals")

    plt.legend()
    plt.show()