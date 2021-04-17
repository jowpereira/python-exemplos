#%matplotlib notebook
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time

import pandas as pd
import MetaTrader5 as mt5

class LiveChartEnv:
    def __init__(self, tick_data, time_frame, 
                 candle_window, wait = 0.01):
        self.tick_data = tick_data
        self.time_frame = time_frame
        self.candle_window = candle_window
        self.wait = wait
        print('Class initialized succesfully')
    
    def initialize_chart(self):
        self.fig = plt.figure(figsize=(8,5))
        self.ax = plt.subplot2grid((1,1), (0,0))
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()
        
    def update_chart(self, candle_data):
            candle_counter = range(len(candle_data["open"]))
            ohlc = []
            for candle in candle_counter:
                append_me = candle_counter[candle], \
                            candle_data["open"][candle], \
                            candle_data["high"][candle],  \
                            candle_data["low"][candle], \
                            candle_data["close"][candle]
                ohlc.append(append_me)
            self.ax.clear() # - Clear the chart
            candlestick_ohlc(self.ax, ohlc, width=0.4, 
                             colorup='#075105', 
                             colordown='#AF141A')
            for label in self.ax.xaxis.get_ticklabels():
                label.set_rotation(45)
            self.ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
            self.ax.grid(True)
            plt.grid(False)
            plt.xlabel('Candle count')
            plt.ylabel('Price')
            plt.title('Candlestick chart simulation')
            plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, 
                                top=0.90, wspace=0.2, hspace=0)
            self.fig.canvas.draw() # - Draw on the chart
    
    def convert_ticks_to_ohlc(df, df_column, timeframe):
        data_ohlc = df[df_column].resample(timeframe).ohlc()
        return data_ohlc
        
    def candlestick_simulation(self):
        candlestick_data = convert_ticks_to_ohlc(self.tick_data, 
                                                 "ask", 
                                                 self.time_frame)
        all_candles = len(candlestick_data)
        self.initialize_chart()
        for candle in range((all_candles - self.candle_window)):
            candles_to_show = candlestick_data[candle:(candle+self.candle_window)]
            self.update_chart(candles_to_show)
            time.sleep(self.wait)    #sleep


if not mt5.initialize():
	print("initialize() failed")
	mt5.shutdown()
rates = mt5.copy_ticks("ITSA4", mt5.TIMEFRAME_M15, 0, 1000)
mt5.shutdown()
series = pd.DataFrame(rates)
series['time']=pd.to_datetime(series['time'], unit='s')
#series = series.set_index(['time'])
series = series.sort_values('time')

df = series

tick_data = pd.read_csv("EURUSD-2019_01_01-2019_02_01.csv",
                        index_col=["time"], 
                        usecols=["time", "ask", "bid"],
                        parse_dates=["time"])


candlestick_chart = LiveChartEnv(tick_data, "1min", 30)

candlestick_chart.candlestick_simulation()