"""   MEAN REVERSION STRATEGY  """

#%%
#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
plt.style.use("dark_background")
import streamlit as st



#Defining CLass for Contrarian Strategy

#%%
class MeanRevBacktester():
    
    ''' Class for the vectorized backtesting of Bollinger Bands-based trading strategies.'''
    def __init__(self,symbol, SMA, dev, start, end, tc, data):
        
        '''
        Attributes
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA: int
            moving window in bars (e.g. days) for SMA
        dev: int
            distance for Lower/Upper Bands in Standard Deviation units
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.SMA = SMA
        self.dev = dev
        self.start  = start 
        self.end = end
        self.tc = tc
        self.results = None
        self.data = data
        self.get_data()
        self.prepare_data()
        
#%%
    
    def  __repr__(self):   
        
        '''Getting represantation function of my class '''
        
        rep = 'MeanRevBacktester(symbol = {}, SMA = {}, dev = {}, start = {}, end = {}, tc = {})'
        return rep.format(self.symbol, self.SMA, self.dev, self.start, self.end, self.tc, self.data)
    
# #%%           
    def get_data(self):
        
        '''Importing data from csv file'''
        
        # raw = pd.read_csv('data.csv', parse_dates = {'Datetime': ['Date', 'Time']}, index_col = 'Datetime')
        
        raw = self.data  
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start : self.end]
        raw.rename(columns={self.symbol: "price"}, inplace = True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
           
    def chart(self):
        title = "{} | Start = {} | End = {}".format(self.symbol , self.start, self.end)
        self.data['price'].plot(title=title, figsize= (12,8), )       

#%%        
    def prepare_data(self):
        
        '''Prepares the data for strategy backtesting ''' 
        
        data = self.data.copy()
        data['SMA'] = data['price'].rolling(self.SMA).mean()
        data['Lower'] = data['SMA'] - data['price'].rolling(self.SMA).std() * self.dev
        data['Upper'] = data['SMA'] + data['price'].rolling(self.SMA).std() * self.dev
        self.data = data
        
    def set_parameters(self, SMA = None, dev = None):
        ''' Updates parameters (SMA, dev) and the prepared dataset.
        '''
        if SMA is not None:
            self.SMA = SMA
            self.data["SMA"] = self.data["price"].rolling(self.SMA).mean()
            self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(self.SMA).std() * self.dev
            
        if dev is not None:
            self.dev = dev
            self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(self.SMA).std() * self.dev
        
    def test_strategy(self):
        
        ''' Backtests the Bollinger Bands-based trading strategy.'''
        
        data = self.data.copy().dropna()
        data["distance"] = data.price - data.SMA
        data["position"] = np.where(data.price < data.Lower, 1, np.nan)   #oversold - LONG!
        data['position'] = np.where(data.price > data.Upper, -1, data.position)   #overbought - SHORT!
        data['position'] = np.where(data.distance * data.distance.shift(1) < 0, 0, data.position) #price crossing SMA - stay neutral
        data['position'] = data.position.ffill().fillna(0)  #if none of above is True hold neutral position
        data['strategy'] = data.position.shift(1) * data["returns"]
        data.dropna(inplace = True)
    
        #determine number of trades in each bar, each trade = .5 spread
        data['trades'] = data.position.diff().fillna(0).abs()
        
        #substracting trading cost from gross return
        data.strategy = data.strategy - data.trades * self.tc
        
        #computing cumulative returns
        data['buy&hold'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        
        #computing performance
        perf = data.cstrategy.iloc[-1]                 #abs performance 
        outperf = perf - data['buy&hold'].iloc[-1]     #performance in relation to buy and hold
        #print(f' Strategy Performance: {round(perf, 3)} | Buy&Hold Performance : {round(outperf, 3)}')
        return round(perf, 3), round(outperf, 3)
    
    
    
    
    
    def plot_results(self):
        
        '''Plots the performance of the trading strategy and compares to "buy and hold".'''
        
        if self.results is None:
            print("Hey let's run test_strategy() first!" )
        else:
            
            title = "{} | SMA = {} | dev = {} | TC = {} ".format(self.symbol ,self.SMA,self.dev, self.tc)
            self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize= (12,8))
#%%            
    def optimize_parameters(self, SMA_range, dev_range):
        ''' Finds the optimal strategy (global maximum) given the Bollinger Bands parameter ranges.

        Parameters
        ----------
        SMA_range, dev_range: tuple
            tuples of the form (start, end, step size)
        '''
        
        combinations = list(product(range(*SMA_range), range(*dev_range)))
        
        # test all combinations
        results = []
        for comb in combinations:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])
        
        best_perf = np.max(results) # best performance
        opt = combinations[np.argmax(results)] # optimal parameters
        
        # run/set the optimal strategy
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()
                   
        # create a df with many results
        many_results =  pd.DataFrame(data = combinations, columns = ["SMA", "dev"])
        many_results["performance"] = results
        self.results_overview = many_results.nlargest(5,'performance')
                            
        return opt, best_perf
     
        
"""   SMA CROSSOVER STRATEGY  """         
  
class SMABacktester():
    ''' Class for the vectorized backtesting of SMA-based trading strategies.
    '''
    
    def __init__(self, symbol, SMA_S, SMA_L, start, end, data):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        start: str
            start date for data import
        end: str
            end date for data import
        '''
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.results = None
        self.data =data
        self.get_data()
        self.prepare_data()
        
    def __repr__(self):
        return "SMABacktester(symbol = {}, SMA_S = {}, SMA_L = {}, start = {}, end = {})".format(self.symbol, self.SMA_S, self.SMA_L, self.start, self.end, self.data)
        
    def get_data(self):
        ''' Imports the data from forex_pairs.csv (source can be changed).
        '''
        raw = self.data
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end].copy()
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
        
    def prepare_data(self):
        '''Prepares the data for strategy backtesting (strategy-specific).
        '''
        data = self.data.copy()
        data["SMA_S"] = data["price"].rolling(self.SMA_S).mean()
        data["SMA_L"] = data["price"].rolling(self.SMA_L).mean()
        self.data = data
        
    def set_parameters(self, SMA_S = None, SMA_L = None):
        ''' Updates SMA parameters and the prepared dataset.
        '''
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()    
            
    def test_strategy(self):
        ''' Backtests the SMA-based trading strategy.'''
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["buy&hold"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["buy&hold"].iloc[-1] # out-/underperformance of strategy
        return round(perf, 3), round(outperf, 3)
            
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | SMA_S = {} | SMA_L = {}".format(self.symbol, self.SMA_S, self.SMA_L)
            self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize=(12, 8))  
                        
    def optimize_parameters(self, SMA_S_range, SMA_L_range):
        ''' Finds the optimal strategy (global maximum) given the SMA parameter ranges.

        Parameters
        ----------
        SMA_S_range, SMA_L_range: tuple
            tuples of the form (start, end, step size)
        '''
        
        combinations = list(product(range(*SMA_S_range), range(*SMA_L_range)))
        
        # test all combinations
        results = []
        for comb in combinations:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])
        
        best_perf = np.max(results) # best performance
        opt = combinations[np.argmax(results)] # optimal parameters
        
        # run/set the optimal strategy
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()
                   
        # create a df with many results
        many_results =  pd.DataFrame(data = combinations, columns = ["SMA_S", "SMA_L"])
        many_results["performance"] = results
        self.results_overview = many_results.nlargest(5,'performance')
                            
        return opt, best_perf            
             











    
    
    
    
    
    
            
             

