"""   BACK TO MEAN  """


#importing libraries
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
# import keras
pd.set_option('display.float_format', lambda x: '%.2f' % x)
from DNNModel import *
from scipy.optimize import brute

plt.style.use("dark_background")



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
        

    
    def  __repr__(self):   
        
        '''Getting represantation function of my class '''
        
        rep = 'MeanRevBacktester(symbol = {}, SMA = {}, dev = {}, start = {}, end = {}, tc = {})'
        return rep.format(self.symbol, self.SMA, self.dev, self.start, self.end, self.tc)
    
         
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
        
        
        perf_percent = (perf - 1) * 100
        outperf_percent = outperf * 100
        
        
        #Calcuating Sharpe for the Model
        
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["Model Perf%", "Model Out/Under Perf%"])
        
        
        self.results_overview = many_results.nlargest(1, "Model Perf%")
        
        return round(perf, 4), round(outperf, 4)
               
    def sharpe_ratios(self):
        
        
        #calculating Sharpe for the Model
        td_year = self.results.strategy.count() / ((self.results.strategy.index[-1] - self.results.strategy.index[0]).days / 365)
        Msharpe = self.results.strategy.mean() / self.results.strategy.std() * np.sqrt(td_year)
        BHsharpe = self.results.returns.mean() / self.results.returns.std() * np.sqrt(td_year)
        
        Sharpe_ratios = pd.DataFrame(data = [[Msharpe,BHsharpe]], columns= ['Model Sharpe', 'Buy & Hold Sharpe '])
        
        return Sharpe_ratios
    
    
    def plot_results(self):
        
        '''Plots the performance of the trading strategy and compares to "buy and hold".'''
        
        if self.results is None:
            print("Hey let's run test_strategy() first!" )
        else:
            
            title = "{} | SMA = {} | dev = {} | TC = {} ".format(self.symbol ,self.SMA,self.dev, self.tc)
            self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize= (12,8))
          
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
        many_results["Model Perf%"] = results
        many_results["Model Perf%"] = (many_results["Model Perf%"] -1) * 100
        
        #Calcuating Sharpe for the Model
        td_year = self.results.strategy.count() / ((self.results.strategy.index[-1] - self.results.strategy.index[0]).days / 365)
        many_results["Model Sharpe"] = self.results.strategy.mean() / self.results.strategy.std() * np.sqrt(td_year)
        

        self.results_overview = many_results.nlargest(1,"Model Perf%").round(2)
                            
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
        return "SMABacktester(symbol = {}, SMA_F = {}, SMA_S = {}, start = {}, end = {})".format(self.symbol, self.SMA_F, self.SMA_S, self.start, self.end, self.tc)
        
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
        
        
        #determine number of trades in each bar, each trade = .7 spread
        data['trades'] = data.position.diff().fillna(0).abs()
        
        #substracting trading cost from gross return
        data.strategy = data.strategy - data.trades * self.tc
        
        data["buy&hold"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["buy&hold"].iloc[-1] # out-/underperformance of strategy
        
        perf_percent = (perf - 1) * 100
        outperf_percent = outperf * 100
        
        #Calcuating Sharpe for the Model
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["Model Perf%", "Model Out/Under Perf%"])
        td_year = data.strategy.count() / ((data.strategy.index[-1] - data.strategy.index[0]).days / 365)
        many_results["Model Sharpe"] = data.strategy.mean() / data.strategy.std() * np.sqrt(td_year)
        many_results["Buy&Hold Sharpe"] = data.returns.mean() / data.returns.std() * np.sqrt(td_year)
        
        #Calcuating DD for the Model
        # data["strategy_cummax"] = data["cstrategy"].cummax()
        # data["buyhold_cummax"] = data['buy&hold'].cummax()
        # many_results["Model Max DD%"] = -(data["cstrategy"] - data["strategy_cummax"]) / data["strategy_cummax"]
        # many_results["B&H Max DD%"] = -(data["buy&hold"] - data["buyhold_cummax"]) / data["buyhold_cummax"]
        
        self.results_overview = many_results
        
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
        
        many_results["Model Perf%"] = results
        many_results["Model Perf%"] = (many_results["Model Perf%"] -1) * 100
        
        
        td_year = self.results.strategy.count() / ((self.results.strategy.index[-1] - self.results.strategy.index[0]).days / 365)
        many_results["Model Sharpe"] = self.results.strategy.mean() / self.results.strategy.std() * np.sqrt(td_year)
        
        
        
        self.results_overview = many_results.nlargest(1,"Model Perf%").round(2)
                            
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
        #self.model = RandomForestClassifier(n_estimators=100)
        #self.model = RandomForestRegressor(n_estimators = 100, random_state = 0)
        self.model = LogisticRegression(C = 1e6, max_iter = 100000, multi_class = "ovr")
        self.results = None
        self.get_data()
    
    def __repr__(self):
        rep = "MLBacktester(symbol = {}, start = {}, end = {}, tc = {}, train_ratio = {})"
        return rep.format(self.symbol, self.start, self.end, self.tc, self.train_ratio)
                             
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
    
    # def prepare_features(self, start, end, lags=2, window=50):
    #     ''' Prepares the feature columns for training set and test set.
    #     '''
        
    #     self.data_subset = self.split_data(start, end)
    #     self.data_subset["RSI"] = talib.RSI(self.data_subset["price"], timeperiod=14)
    #     self.data_subset["dir"] = np.where(self.data_subset["returns"] > 0, 1, 0)
    #     self.data_subset["sma"] = self.data_subset["price"].rolling(window).mean() - self.data_subset["price"].rolling(150).mean()
    #     self.data_subset["boll"] = (self.data_subset["price"] - self.data_subset["price"].rolling(window).mean()) / self.data_subset["price"].rolling(50).std()
    #     self.data_subset["min"] = self.data_subset["price"].rolling(window).min() / self.data_subset["price"] - 1
    #     self.data_subset["max"] = self.data_subset["price"].rolling(window).max() / self.data_subset["price"] - 1
    #     self.data_subset["mom"] = self.data_subset["returns"].rolling(3).mean()
    #     self.data_subset["vol"] = self.data_subset["returns"].rolling(window).std()
        
        # self.feature_columns = []
        # new_features = ["RSI", "sma","boll"] #,"min", "max", "mom", "vol"]
        # for f in new_features:
        #     for lag in range(1, lags + 1):
        #         col = "{}lag{}".format(f, lag)
        #         self.data_subset[col] = self.data_subset[f].shift(lag)
        #         self.feature_columns.append(col)
        # self.data_subset.dropna(inplace=True)
    
    def prepare_features(self, start, end):
        ''' Prepares the feature columns for training set and test set.
        '''
        self.data_subset = self.split_data(start, end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1):
            col = "lag{}".format(lag)
            self.data_subset[col] = self.data_subset["returns"].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)
                
    def fit_model(self, start, end):
        ''' Fitting the ML Model
        '''
        self.prepare_features(start, end)
        self.model.fit(self.data_subset[self.feature_columns], np.sign(self.data_subset["returns"]))
        
    def test_strategy(self, train_ratio = 0.8, lags = 5):
        ''' 
        Backtests the ML-based strategy.
        
        Parameters
        ----------
        train_ratio: float (between 0 and 1.0 excl.)
            Splitting the dataset into training set (train_ratio) and test set (1 - train_ratio).
        lags: int
            number of lags serving as model lags.
        '''
        self.lags = lags
                  
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
        
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["Model Perf%", "Model Out/Under Perf%"])
        td_year = self.data_subset.strategy.count() / ((self.data_subset.strategy.index[-1] - self.data_subset.strategy.index[0]).days / 365)
        many_results["Strategy Sharpe"] = self.data_subset.strategy.mean() / self.data_subset.strategy.std() * np.sqrt(td_year)
        many_results["Buy&Hold Sharpe"] = self.data_subset.returns.mean() / self.data_subset.returns.std() * np.sqrt(td_year)
        
        self.results_overview = many_results
        return round(perf, 4), round(outperf, 4)
        
    def plot_results(self, features):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "Logistic Regression: {} | TC = {} | LAGS = {}".format(self.symbol, self.tc, features)
            self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize=(12, 8))
    
    def optimize_features(self, lags):
        results = []
        for feature in range(1, lags):
            results.append(self.test_strategy(train_ratio = 0.8, lags = feature))
        best_perf = np.max(results)
        perf_percent = round((best_perf - 1) * 100, 2)
        opt = results.index(max(results))
        many_results =  pd.DataFrame([[perf_percent, opt]], columns = ["Model Perf%", "Lags"])
        
        #Calcuating Sharpe for the Model
        td_year = self.data_subset.strategy.count() / ((self.data_subset.strategy.index[-1] - self.data_subset.strategy.index[0]).days / 365)
        many_results["Model Sharpe"] = self.data_subset.strategy.mean() / self.data_subset.strategy.std() * np.sqrt(td_year)
        
        
        
        self.results_overview = many_results.nlargest(1,"Model Perf%").round(2)
        self.opt = opt 
        return opt, perf_percent     
    
    
    def plot_results_opt(self):
        title = "Logistic Regression: {} | TC = {} | LAGS = {}".format(self.symbol, self.tc, self.opt)
        self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize=(12, 8))     
            
       

"""  RSI MODEL """ 

class RSIBacktester():

    """Class for vectorised backtesing of RSI Models
    
    Atributes  
    ===========
    symbol : str
         ticker-market to be backtested
    periods: int
        time window to calculate MA(movien average)
    rsi_upper: int
        indicates verbought markets
    rsi_lower: int
        indicates oversold markets
    start: str
        start date
    end: str
        end date
    tc: float
        transaction cost per trade
    
    Methods
    ========
    get_data:
        getsand prepares data for backtesting
    
    set_parameters:
        setts new parameters for backtesting
            
    test_strategy:
        runs bactesting algo
        
    plot_results:
        plots performance of backtesting in comparison to buy & hold
    
    update_and_run        
        updates RSI parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for three RSI parameters
    """    
        
    def __init__(self, symbol, periods, rsi_upper, rsi_lower, start, end, tc, data):
        self.symbol = symbol
        self.periods = periods
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        self.start = start
        self.end = end
        self.tc = tc
        self.data = data
        self.results = None
        self.get_data()
        self.set_parameters()
        self.test_strategy()
        self.plot_results()
        
    def __repr__(self):
        return "RSIBacktester(symbol = {}, periods = {}, rsi_upper = {}, rsi_lower = {} , start = {}, end = {}, tc = {}".format(self.symbol, self.periods, self.rsi_upper, self.rsi_lower, self.start, self.end, self.tc)
    
    def get_data(self):
        """
        gets and prepares data for backtesting"""
        
        raw = self.data
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        
        raw["U"] = np.where(raw.price.diff() > 0, raw.price.diff(), 0)
        raw["D"] = np.where(raw.price.diff() < 0, -raw.price.diff(), 0)
        raw["MA_U"] = raw.U.rolling(self.periods).mean()
        raw["MA_D"] = raw.D.rolling(self.periods).mean()
        raw["RSI"] = raw.MA_U / (raw.MA_U + raw.MA_D) * 100
        self.data = raw
        
    def set_parameters(self, periods = None, rsi_upper = None, rsi_lower = None):
        """
        sets new parameters for backtesting"""
        
        if periods is not None:
            self.periods = periods
            self.data["MA_U"] = self.data.U.rolling(self.periods).mean()
            self.data["MA_D"] = self.data.D.rolling(self.periods).mean()
            self.data["RSI"] = self.data.MA_U / (self.data.MA_U + self.data.MA_D) * 100
        if rsi_upper is not None:
            self.rsi_upper = rsi_upper
        if rsi_lower is not None:
            self.rsi_lower = rsi_lower
            
    def test_strategy(self):
        """
        run backtesting algo"""
        data = self.data.copy().dropna()
        data["position"] = np.where(data.RSI > self.rsi_upper, -1, np.nan)
        data["position"] = np.where(data.RSI < self.rsi_lower, 1, data.position)
        data.position = data.position.fillna(0)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        data["buy&hold"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        perf = data.cstrategy.iloc[-1]
        outperf =  perf - data["buy&hold"].iloc[-1]
        
        perf_percent = (perf - 1) * 100
        outperf_percent = outperf * 100
        
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["Model Perf%", "Model Out/Under Perf %"])
        td_year = data.strategy.count() / ((data.index[-1] - data.index[0]).days / 365)
        many_results["Model Sharpe"] = data.strategy.mean() / data.strategy.std() * np.sqrt(td_year)
        many_results["Buy&hold Sharpe"] = data.returns.mean() / data.returns.std() * np.sqrt(td_year)
        
        self.results_overview = many_results
        
        return round(perf, 4), round(outperf, 4)
            
        
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
    compared to buy and hold.
    '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | RSI ({}, {}, {}) | TC = {}".format(self.symbol, self.periods, self.rsi_upper, self.rsi_lower, self.tc)
            self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize=(12, 8))  
                
    
    def update_and_run(self, RSI):
        ''' Updates RSI parameters and returns the negative absolute performance (for minimization algorithm).
      
        Parameters
        ==========
        RSI: tuple 
            RSI parameter tuple
        '''
      
        self.set_parameters(int(RSI[0]), int(RSI[1]), int(RSI[2]))
        return -self.test_strategy()[0]
    
    
    
    def optimize_parameters(self, periods_range, rsi_upper_range, rsi_lower_range):
        ''' Finds global maximum given the RSI parameter ranges.

        Parameters
        ==========
        periods_range, rsi_upper_range, rsi_lower_range : tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (periods_range, rsi_upper_range, rsi_lower_range), finish=None)
        
        many_results_opt = pd.DataFrame(data = [opt], columns = ["RSI Time Period", "RSI_upper", "RSI_lower"])
        
        
        
        
        self.results_opt = many_results_opt
        
        return opt, -self.update_and_run(opt)
    
                  

"""" DNN PHANTOM"""


  
class DNNBacktester():
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
       
        self.results = None
        self.get_data()
    
    def __repr__(self):
        rep = "DNNBacktester(symbol = {}, start = {}, end = {}, tc = {}, train_ratio = {})"
        return rep.format(self.symbol, self.start, self.end, self.tc, self.train_ratio)
                             
    def get_data(self):
        ''' Imports the data from five_minute_pairs.csv (source can be changed).
        '''
        raw = self.data
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
                             
    def test_strategy(self, lags):
        '''   
        Backtests the ML-based strategy.
        
        Parameters
        ----------
        train_ratio: float (between 0 and 1.0 excl.)
            Splitting the dataset into training set (train_ratio) and test set (1 - train_ratio).
        lags: int
            number of lags serving as model lags.
        '''
        
        window = 20
        self.data = self.data.copy()
        self.data['dir'] = np.where(self.data['returns'] > 0 , 1 , 0)
        self.data['sma'] = self.data['price'].rolling(window).mean() 
        self.data['boll'] = (self.data['price'] - self.data['price'].rolling(window).mean()) / self.data['price'].rolling(window).std()
        self.data['min'] = self.data['price'].rolling(window).min() / self.data['price'] -1
        self.data['max'] = self.data['price'].rolling(window).max() / self.data['price'] -1
        self.data['mom'] = self.data['returns'].rolling(3).mean()
        self.data['vol'] =self.data['returns'].rolling(window).std()
        self.data.dropna(inplace= True)
        
        
        self.feature_columns = []
        new_features = [ "dir","sma", "boll", "min", "max", "mom", "vol"]
        
        for f in new_features:
            for lag in range(1, lags + 1):
                col = "{}lag{}".format(f, lag)
                self.data[col] = self.data[f].shift(lag)
                self.feature_columns.append(col)
        self.data.dropna(inplace=True)
    
        self.lags = lags
              
        #split data
        split = int(len(self.data) * self.train_ratio)
        train = self.data.iloc[:split].copy()
        test = self.data.iloc[split:].copy()
        
        # standardization
        mu,std = train.mean(), train.std() 
        train_scaled = (train - mu) / std  #need to standarised with train set arameters!
        test_scaled = (test- mu) / std
        
        # fit the model on the training set
        set_seeds(100)
        model = create_model(hl = 2, hu = 10, dropout = True, input_dim = len(self.feature_columns))
        model.fit(x = train_scaled[self.feature_columns], y = train["dir"], epochs = 50, verbose = False,
                validation_split = 0.2, shuffle = False, class_weight = cw(train))
                  
        # make predictions on the test set
        test["pred"] = model.predict(test_scaled[self.feature_columns])
        
        # calculate Strategy Returns
        test["strategy"] = test["pred"] * test["returns"]
        test['position'] =np.where(test.pred < 0.47, -1, np.nan)   #simply short when probability is less than 0.47
        test['position'] =np.where(test.pred > 0.53, 1, test.position)   #simply buy when probability is more than 0.53
        
        #time-varying position
        test.index = pd.to_datetime(test.index)
        test.index = test.index.tz_localize('UTC')
        test['NYTime'] = test.index.tz_convert('AMerica/New_York')
        test['hour'] = test.NYTime.dt.hour
        test['position'] = np.where(~test.hour.between(2,12),0, test.position)    # neutral in non busy hours
        test['position'] = test.position.ffill()     # in all other cases hold position
        
        # determine the number of trades in each bar
        test["trades"] = test["pred"].diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        test.strategy = test.strategy - test.trades * self.tc
        
        # calculate cumulative returns for strategy & buy and hold
        test["buy&hold"] = test["returns"].cumsum().apply(np.exp)
        test["cstrategy"] = test['strategy'].cumsum().apply(np.exp)
        self.results = test
        
        perf = self.results["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - self.results["buy&hold"].iloc[-1] # out-/underperformance of strategy
        
        perf_percent = (perf - 1)* 100
        outperf_percent = outperf * 100
        
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["Model Perf%", "Model Out/Under Perf %"])
        td_year = test.strategy.count() / ((test.strategy.index[-1] - test.strategy.index[0]).days / 365)
        many_results["Model Sharpe"] = test.strategy.mean() / test.strategy.std() * np.sqrt(td_year)
        many_results["Buy&Hold Sharpe"] = test.returns.mean() / test.returns.std() * np.sqrt(td_year)
        
        self.results_overview = many_results.nlargest(1,"Model Perf%")
        
        
        return round(perf, 4), round(outperf, 4)
        
    def plot_results(self, lags):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "DNN_PHANTOM: {} | TC = {} | LAGS = {}".format(self.symbol, self.tc, lags)
            self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize=(12, 8))
    
    def optimize_features(self, lags):
        results = []
        for feature in range(1, lags):
            results.append(self.test_strategy(train_ratio = 0.8, lags = feature))
        best_perf = np.max(results)
        perf_percent = round((best_perf - 1) * 100, 2)
        opt = results.index(max(results))
        many_results =  pd.DataFrame([[perf_percent, opt]], columns = ["PERF%", "LAGS"])
        
        self.results_overview = many_results
        self.opt = opt 
        return opt, perf_percent     
    
    
    def plot_results_opt(self):
        title = "Logistic Regression: {} | TC = {} | LAGS = {}".format(self.symbol, self.tc, self.opt)
        self.results[["buy&hold", "cstrategy"]].plot(title=title, figsize=(12, 8))  
        
    
    
    
    
    
    
            
             


