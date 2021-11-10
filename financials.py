import numpy as np
import pandas as pd


def plot_chart(symbol, start, end, df):
    raw = df  
    raw = raw[symbol].to_frame().dropna()
    raw = raw.loc[start : end]
    raw.rename(columns={symbol: "price"}, inplace = True)
    raw["returns"] = np.log(raw / raw.shift(1))
    data = raw

    title = "{} | Start = {} | End = {}".format(symbol , start, end)
    data['price'].plot(title=title,  figsize=(10,6))
    
    
def performance_metrics(symbol, start, end, df):
    raw = df  
    raw = raw[symbol].to_frame().dropna()
    raw = raw.loc[start : end]
    raw.rename(columns={symbol: "price"}, inplace = True)
    raw["returns"] = np.log(raw / raw.shift(1))
    data = raw

      