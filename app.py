#importing libraries

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from datetime import date
import Strategies as S
import matplotlib.pyplot  
#from yahoofinancials import YahooFinancials


#Insertig picture
st.markdown("<h1 style='text-align: center; color: rgba(25, 119, 241, 0.788); text-shadow: 0 0 20px #6a93a7; '>VOYTECH ALGO TESTER</h1>", unsafe_allow_html=True)

url = 'https://www.inteldig.com/wp-content/uploads/2021/03/machine.jpeg'
response = requests.get(url)
img = Image.open(BytesIO(response. content))
st.image(img, use_column_width=True)


#Uploding data
uploaded_file = st.file_uploader("CHOOSE YOUR FILE")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates = {'Datetime': ['Date', 'Time']}, index_col = 'Datetime')
    st.write(df)
    
    st.sidebar.header("BACKTEST IDEAS") 
    backtest_init = st.sidebar.checkbox('Run Backtest')
    
    symbol = st.sidebar.selectbox(label="FOREX & Crypto & CFDs", options=["AUDJPY", "AUDUSD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPUSD", "NZDUSD", 
                "USDCAD","USDCHF","USDJPY","SILVER","GOLD","NAS","DAX","CAC","FTSE", "DOW","BTC","LTC","ETH"])    
    
    SMA = int(st.sidebar.text_input("Moving Average (SMA)",150))
    dev = int(st.sidebar.text_input('Standard Deviation',2))
    tc = float(st.sidebar.text_input("Transaction cost", 0.0007))      
    SMA_FAST = int(st.sidebar.text_input("Moving Average (FAST)",50))
    SMA_SLOW = int(st.sidebar.text_input("Moving Average (SLOW)",150))

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
    back_to_mean = S.MeanRevBacktester(symbol, SMA, dev, start, end, tc, df)
    
    st.pyplot(back_to_mean.chart())
    
    SMA_crossover = S.SMABacktester(symbol, SMA_FAST, SMA_SLOW, start, end, df)
    
    #Choose strategy and backtest!
    
    backtest_options = ["BACK TO MEAN","SMA CROSSOVER"] 
    backtest_options = st.sidebar.selectbox("Strategies", backtest_options)
    
    if backtest_init == True:
        
        if backtest_options == "BACK TO MEAN":
            st.write(back_to_mean.test_strategy())
            st.pyplot(back_to_mean.plot_results())
        
        if backtest_options == "SMA CROSSOVER":
            st.write(SMA_crossover.test_strategy())
            st.pyplot(SMA_crossover.plot_results())    

    #Strategy optimization 
    
    optimization = st.sidebar.checkbox('Optimize Strategy')
    
    SMA_s = int(st.sidebar.text_input("Moving Average Strat",2))
    SMA_e = int(st.sidebar.text_input("Moving Average End",20))
    dev_s = int(st.sidebar.text_input('Standard Deviation Start',2))
    dev_e = int(st.sidebar.text_input('Standard Deviation End',10))
    
    SMA_fast_s = int(st.sidebar.text_input('SMA Fast  Start',30))
    SMA_fast_e = int(st.sidebar.text_input('SMA Fast End',50))
    SMA_slow_s = int(st.sidebar.text_input('SMA Slow  Start',120))
    SMA_fast_e = int(st.sidebar.text_input('SMA Slow  End',150))
    
    if optimization == True:
        
        if backtest_options == "BACK TO MEAN":
            st.write(back_to_mean.optimize_parameters((SMA_s, SMA_e ),(dev_s, dev_e )))
            st.write(back_to_mean.results_overview)
            st.pyplot(back_to_mean.plot_results()) 
        
        
        if backtest_options == "SMA CROSSOVER":
            st.write(SMA_crossover.optimize_parameters((SMA_fast_s,SMA_fast_e),(SMA_slow_s,SMA_fast_e )))
            st.write(SMA_crossover.results_overview)
            st.pyplot(SMA_crossover.plot_results()) 