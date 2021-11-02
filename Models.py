"""   BACK TO MEAN  """

#%%
#importing libraries
from sklearn.linear_model import LogisticRegression
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
        #data['position'] = np.where(data.distance * data.distance.shift(1) < 0, 0, data.position) #price crossing SMA - stay neutral
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
        
        perf_percent = (perf - 1)* 100
        outperf_percent = outperf * 100
        
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["PERF %", "OUTPERF %"])
        self.results_overview = many_results.nlargest(1,"PERF %")
        return round(perf, 4), round(outperf, 4)
               
    
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
     
        
"""   SMA CROSSOVER   """         
  
class SMABacktester():
    ''' Class for the vectorized backtesting of SMA-based trading strategies.
    '''
    
    def __init__(self, symbol, SMA_F, SMA_S, start, end, tc, data):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA_F: int
            moving window in bars (e.g. days) for faster SMA
        SMA_S: int
            moving window in bars (e.g. days) for slower SMA
        start: str
            start date for data import
        end: str
            end date for data import
        '''
        self.symbol = symbol
        self.SMA_F = SMA_F
        self.SMA_S = SMA_S
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.data = data
        self.get_data()
        self.prepare_data()
        
    def __repr__(self):
        return "SMABacktester(symbol = {}, SMA_F = {}, SMA_S = {}, start = {}, end = {})".format(self.symbol, self.SMA_F, self.SMA_S, self.start, self.end, self.tc, self.data)
        
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
        data["SMA_F"] = data["price"].rolling(self.SMA_F).mean()
        data["SMA_S"] = data["price"].rolling(self.SMA_S).mean()
        self.data = data
        
    def set_parameters(self, SMA_F = None, SMA_S = None):
        ''' Updates SMA parameters and the prepared dataset.
        '''
        if SMA_F is not None:
            self.SMA_F = SMA_F
            self.data["SMA_F"] = self.data["price"].rolling(self.SMA_F).mean()
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()    
            
    def test_strategy(self):
        ''' Backtests the SMA-based trading strategy.'''
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_F"] > data["SMA_S"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        
        #determine number of trades in each bar, each trade = .5 spread
        data['trades'] = data.position.diff().fillna(0).abs()
        
        #substracting trading cost from gross return
        data.strategy = data.strategy - data.trades * self.tc
        
        data["buy&hold"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["buy&hold"].iloc[-1] # out-/underperformance of strategy
        
        perf_percent = (perf - 1)* 100
        outperf_percent = outperf * 100
        
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["PERF %", "OUTPERF %"])
        self.results_overview = many_results.nlargest(1,"PERF %")
        return round(perf, 4), round(outperf, 4)
            
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | SMA_F = {} | SMA_S = {}".format(self.symbol, self.SMA_F, self.SMA_S)
            self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize=(12, 8))  
                        
    def optimize_parameters(self, SMA_F_range, SMA_S_range):
        ''' Finds the optimal strategy (global maximum) given the SMA parameter ranges.

        Parameters
        ----------
        SMA_F_range, SMA_S_range: tuple
            tuples of the form (start, end, step size)
        '''
        
        combinations = list(product(range(*SMA_F_range), range(*SMA_S_range)))
        
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
        many_results =  pd.DataFrame(data = combinations, columns = ["SMA_F", "SMA_S"])
        many_results["PERF"] = results 
        self.results_overview = many_results.nlargest(5,"PERF")
                            
        return opt, best_perf            
             

"""  LOGISTIC REGRESSION """     

class MLBacktester():
    ''' Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).
    '''

    def __init__(self, symbol, start, end, tc, data, train_ratio):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.train_ratio = train_ratio
        self.data = data
        self.model = LogisticRegression(C = 1e6, max_iter = 100000, multi_class = "ovr")
        self.results = None
        self.get_data()
    
    def __repr__(self):
        rep = "MLBacktester(symbol = {}, start = {}, end = {}, tc = {}, train_ratio = {})"
        return rep.format(self.symbol, self.start, self.end, self.tc, self.data)
                             
    def get_data(self):
        ''' Imports the data from five_minute_pairs.csv (source can be changed).
        '''
        raw = self.data
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
                             
    def split_data(self, start, end):
        ''' Splits the data into training set & test set.
        '''
        data = self.data.loc[start:end].copy()
        return data
    
    def prepare_features(self, start, end):
        ''' Prepares the feature columns for training set and test set.
        '''
        self.data_subset = self.split_data(start, end)
        self.feature_columns = []
        for lag in range(1, self.features + 1):
            col = "lag{}".format(lag)
            self.data_subset[col] = self.data_subset["returns"].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)
        
    def fit_model(self, start, end):
        ''' Fitting the ML Model
        '''
        self.prepare_features(start, end)
        self.model.fit(self.data_subset[self.feature_columns], np.sign(self.data_subset["returns"]))
        
    def test_strategy(self, train_ratio = 0.7, features = 5):
        ''' 
        Backtests the ML-based strategy.
        
        Parameters
        ----------
        train_ratio: float (between 0 and 1.0 excl.)
            Splitting the dataset into training set (train_ratio) and test set (1 - train_ratio).
        features: int
            number of features serving as model features.
        '''
        self.features = features
                  
        # determining datetime for start, end and split (for training an testing period)
        full_data = self.data.copy()
        split_index = int(len(full_data) * train_ratio)
        split_date = full_data.index[split_index-1]
        train_start = full_data.index[0]
        test_end = full_data.index[-1]
        
        # fit the model on the training set
        self.fit_model(train_start, split_date)
        
        # prepare the test set
        self.prepare_features(split_date, test_end)
                  
        # make predictions on the test set
        predict = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset["pred"] = predict
        
        # calculate Strategy Returns
        self.data_subset["strategy"] = self.data_subset["pred"] * self.data_subset["returns"]
        
        # determine the number of trades in each bar
        self.data_subset["trades"] = self.data_subset["pred"].diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        self.data_subset.strategy = self.data_subset.strategy - self.data_subset.trades * self.tc
        
        # calculate cumulative returns for strategy & buy and hold
        self.data_subset["buy&hold"] = self.data_subset["returns"].cumsum().apply(np.exp)
        self.data_subset["cstrategy"] = self.data_subset['strategy'].cumsum().apply(np.exp)
        self.results = self.data_subset
        
        perf = self.results["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - self.results["buy&hold"].iloc[-1] # out-/underperformance of strategy
        
        perf_percent = (perf - 1)* 100
        outperf_percent = outperf * 100
        
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["PERF %", "OUTPERF %"])
        self.results_overview = many_results.nlargest(1,"PERF %")
        return round(perf, 4), round(outperf, 4)
        
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "Logistic Regression: {} | TC = {}".format(self.symbol, self.tc)
            self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize=(12, 8))
    
    def optimize_features(self, features):
        for feature in range(1, features):
            return (feature, self.test_strategy(self.train_ratio, features = features))
                








    
    
    
    
    
    
            
             

