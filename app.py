#importing libraries
# from PIL import Image
# import requests
# from io import BytesIO
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import numpy as np
import streamlit as st
from datetime import date
import Models as M
import financials
import base64 
import matplotlib.pyplot 




#Insertig picture
st.markdown("<h1 style='text-align: center; color: rgba(161,201,255); text-shadow: 0 0 20px #6a93a7; '>VOYTECH ALGO TESTER</h1>", unsafe_allow_html=True)



# url = 'https://www.inteldig.com/wp-content/uploads/2021/03/machine.jpeg'
# response = requests.get(url)
# img = Image.open(BytesIO(response. content))
# st.image(img, use_column_width=True)

file_ = open("market.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="gif">',
    unsafe_allow_html=True,
)



#Uploding data
uploaded_file = st.file_uploader("CHOOSE YOUR FILE")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates = {'Datetime': ['Date', 'Time']}, index_col = 'Datetime')
    st.write(df)
    st.sidebar.header("CHOOSE YOUR MARKET")  
    symbol = st.sidebar.selectbox(label="FOREX & CRYPTO & CFDs", options=["AUDJPY", "AUDUSD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPUSD", "NZDUSD", 
                "USDCAD","USDCHF","USDJPY","SILVER","GOLD","NAS","DAX","CAC","FTSE", "DOW","BTC","LTC","ETH"])    
    
     
    #SELECT data period
    
    start = st.sidebar.date_input(
    label="Start Date",
    value=date(2010, 3, 1),
    min_value=date(2010, 3, 1),
    max_value=date.today())

    end = st.sidebar.date_input(
    label="End Date",
    value=date.today(),
    min_value=date(2010, 3, 1),
    max_value=date.today())
    
    #Plot data & define testers
    
    st.set_option('deprecation.showPyplotGlobalUse', False)  
    st.pyplot(financials.plot_chart(symbol, start, end, df))
    
    
      
    #Choose strategy and backtest!
    tc = float(st.sidebar.text_input("Transaction Cost", 0.00007))
    st.sidebar.header("MODEL BACKTESTER") 
    backtest_init = st.sidebar.checkbox('Run Backtester')
    backtest_options = ["BACK TO MEAN","SMA CROSSOVER", "LOG REGRESSION", "RSI MODEL", "DNN_PHANTOM"] 
    backtest_options = st.sidebar.selectbox("Select model", backtest_options)
    
    
    if backtest_options == "BACK TO MEAN":
        SMA = int(st.sidebar.text_input("Moving Average (SMA)",150))
        dev = int(st.sidebar.text_input('Standard Deviation',2))
        back_to_mean = M.MeanRevBacktester(symbol, SMA, dev, start, end, tc, df)
        if backtest_init == True:  
            back_to_mean.test_strategy()
            st.write(back_to_mean.results_overview)
            st.write(back_to_mean.sharpe_ratios())
            st.pyplot(back_to_mean.plot_results())
        
    if backtest_options == "SMA CROSSOVER":
        SMA_FAST = int(st.sidebar.text_input("Moving Average (FAST)",50))
        SMA_SLOW = int(st.sidebar.text_input("Moving Average (SLOW)",150))
        SMA_crossover = M.SMABacktester(symbol, SMA_FAST, SMA_SLOW, start, end, tc, df)
        if backtest_init == True:     
            SMA_crossover.test_strategy()
            st.write(SMA_crossover.results_overview)
            st.pyplot(SMA_crossover.plot_results())
            
    if backtest_options == "LOG REGRESSION":
        train_ratio =   float(st.sidebar.text_input('Train Data',0.8))
        lags =   int(st.sidebar.text_input('Lags',5)) 
        ML = M.MLBacktester(symbol, start, end, tc, df, train_ratio)
        if backtest_init == True: 
            ML.test_strategy(train_ratio, lags)
            st.write(ML.results_overview)
            st.pyplot(ML.plot_results(lags))
     
    if backtest_options == "RSI MODEL":
        RSI_upper = int(st.sidebar.text_input("RSI Upper",80))
        RSI_lower = int(st.sidebar.text_input("RSI Lower",20))
        periods = int(st.sidebar.text_input("RSI Time Period",20))
        RSI_MODEL = M.RSIBacktester(symbol, periods, RSI_upper, RSI_lower, start, end, tc, df)
        if backtest_init == True:    
            RSI_MODEL.test_strategy()
            st.write(RSI_MODEL.results_overview)   
            st.pyplot(RSI_MODEL.plot_results())
    
    if backtest_options == "DNN_PHANTOM":
        train_ratio =   float(st.sidebar.text_input('Train Data',0.8))
        lags =   int(st.sidebar.text_input('Lags',5)) 
        DNN_PHANTOM = M.DNNBacktester(symbol, start, end, tc, df, train_ratio)
        if backtest_init == True:     
            DNN_PHANTOM.test_strategy(lags)
            st.write(DNN_PHANTOM.results_overview)
            st.pyplot(DNN_PHANTOM.plot_results(lags))                   
               

    #Strategy optimization 
    st.sidebar.header("MODEL OPTIMIZER") 
    optimization = st.sidebar.checkbox('Run Optimizer')
    
    if backtest_options == "BACK TO MEAN":
        SMA_s = int(st.sidebar.text_input("SMA Strat",2))
        SMA_e = int(st.sidebar.text_input("SMA End",20))
        dev_s = int(st.sidebar.text_input('Standard Deviation Start',2))
        dev_e = int(st.sidebar.text_input('Standard Deviation End',10))
        if optimization == True:    
            back_to_mean.optimize_parameters((SMA_s, SMA_e ),(dev_s, dev_e))
            st.write(back_to_mean.results_overview)
            st.pyplot(back_to_mean.plot_results()) 
        
        
    if backtest_options == "SMA CROSSOVER":
        SMA_fast_s = int(st.sidebar.text_input('SMA Fast  Start',30))
        SMA_fast_e = int(st.sidebar.text_input('SMA Fast End',50))
        SMA_slow_s = int(st.sidebar.text_input('SMA Slow  Start',120))
        SMA_fast_e = int(st.sidebar.text_input('SMA Slow  End',150))
        if optimization == True:
            SMA_crossover.optimize_parameters((SMA_fast_s,SMA_fast_e),(SMA_slow_s,SMA_fast_e))
            st.write(SMA_crossover.results_overview)
            st.pyplot(SMA_crossover.plot_results()) 
            
    if backtest_options == "LOG REGRESSION":
        lags =   int(st.sidebar.text_input('Opt Lags',10))
        if optimization == True:
           ML.optimize_features(lags)
           st.write(ML.results_overview)
           st.pyplot(ML.plot_results_opt()) 
    
    if backtest_options == "RSI MODEL":
        RSI_upper_s = int(st.sidebar.text_input('RSI Upper Start',70))
        RSI_upper_e = int(st.sidebar.text_input('RSI Upper Start',80))
        RSI_lower_s = int(st.sidebar.text_input('RSI Lower Start',20))
        RSI_lower_e = int(st.sidebar.text_input('RSI Lower End',30))
        Time_s = int(st.sidebar.text_input('RSI Time Period Start',5))
        Time_e = int(st.sidebar.text_input('RSI Time period End',20))
        if optimization == True:
            RSI_MODEL.optimize_parameters((Time_s,Time_e,1),(RSI_upper_s,RSI_upper_e,1),(RSI_lower_s,RSI_lower_e,1))
            st.write(RSI_MODEL.results_overview)
            st.write(RSI_MODEL.results_opt)
            st.pyplot(RSI_MODEL.plot_results())       
            
     