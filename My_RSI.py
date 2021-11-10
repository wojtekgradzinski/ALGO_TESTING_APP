import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")




class RSIbacktester():

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
        raw = self.symbol.to_frame().dropna()
        raw = raw.rename(columns = {"symbol": "price"}, inplace = True)
        raw["returns"] = np.log(raw/raw.shift(1))
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
        runs backtesting algo"""
        
        data =  self.data.copy().dropna()
        data["position"] = np.where(data.RSI > self.rsi_upper, -1, np.nan)
        data["position"] = np.where(data.RSI < self.rsi_lower, 1, data.position) 
        data.position = data.position.fillna(0) 
        data["strategy"] = data.position.shift(1) * data.returns
        data.dropna(inplace = True)
        
        #determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        #substract transaction cost from returns
        data.strategy = data.strategy - data.trades * self.tc
        
        data["buy&hold"] = data.returns.cumsum().apply(np.exp)
        data["strategy"] = data.strategy.cumsum().apply(np.exp)
        self.results = data
        
        perf = data.strategy.iloc[-1]
        outperf =  perf - data["buy&hold"].iloc[-1]
        
        perf_percent = (perf - 1) * 100
        outperf_percent = outperf * 100
        
        many_results =  pd.DataFrame(data = [[perf_percent, outperf_percent]], columns = ["PERF %", "OUTPERF %"])
        self.results_overview = many_results.nlargest(1,"PERF %")
        return round(perf, 4), round(outperf, 4)
            
        
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
    compared to buy and hold.
    '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | RSI ({}, {}, {}) | TC = {}".format(self.symbol, self.periods, self.rsi_upper, self.rsi_lower, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))  
                
    
    def update_and_run(self, RSI):
        ''' Updates RSI parameters and returns the negative absolute performance (for minimization algorithm).
      
        Parameters
        ==========
        RSI: tuple 
            RSI parameter tuple
        '''
      
        self.set_parameters(int(RSI[0]), int(RSI[1]), int(RSI[2]))
        return -self.test_strategy()
    
    
    def optimize_parameters(self, periods_range, rsi_upper_range, rsi_lower_range):
        ''' Finds global maximum given the RSI parameter ranges.

        Parameters
        ==========
        periods_range, rsi_upper_range, rsi_lower_range : tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (periods_range, rsi_upper_range, rsi_lower_range), finish=None)
        return opt, -self.update_and_run(opt)
        
      
      
      
      
    