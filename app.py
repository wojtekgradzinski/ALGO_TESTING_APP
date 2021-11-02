#importing libraries

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from datetime import date
import Models as M
import matplotlib.pyplot 
import financials
import base64 


#Insertig picture
st.markdown("<h1 style='text-align: center; color: rgba(161,201,255); text-shadow: 0 0 20px #6a93a7; '>VOYTECH ALGO TESTER</h1>", unsafe_allow_html=True)

# video = 'animation.mp4'

# st.video(video, start_time=0)

# st.markdown("![Alt Text](https://www.filepicker.io/api/file/P2BLDAx0Qxq1nrvhS1GO)")

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
    symbol = st.sidebar.selectbox(label="FOREX & Crypto & CFDs", options=["AUDJPY", "AUDUSD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPUSD", "NZDUSD", 
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
    tc = float(st.sidebar.text_input("Transaction cost", 0.0001))
    st.sidebar.header("MODEL BACKTESTER") 
    backtest_init = st.sidebar.checkbox('Run Backtester')
    backtest_options = ["BACK TO MEAN","SMA CROSSOVER", "LOGISTIC REGRESSION"] 
    backtest_options = st.sidebar.selectbox("Select model", backtest_options)
    
    
    if backtest_options == "BACK TO MEAN":
        SMA = int(st.sidebar.text_input("Moving Average (SMA)",150))
        dev = int(st.sidebar.text_input('Standard Deviation',2))
        back_to_mean = M.MeanRevBacktester(symbol, SMA, dev, start, end, tc, df)
        if backtest_init == True:  
            back_to_mean.test_strategy()
            st.write(back_to_mean.results_overview)
            st.pyplot(back_to_mean.plot_results())
        
    if backtest_options == "SMA CROSSOVER":
        SMA_FAST = int(st.sidebar.text_input("Moving Average (FAST)",50))
        SMA_SLOW = int(st.sidebar.text_input("Moving Average (SLOW)",150))
        SMA_crossover = M.SMABacktester(symbol, SMA_FAST, SMA_SLOW, start, end, tc, df)
        if backtest_init == True:     
            SMA_crossover.test_strategy()
            st.write(SMA_crossover.results_overview)
            st.pyplot(SMA_crossover.plot_results())
            
    if backtest_options == "LOGISTIC REGRESSION":
        train_ratio =   float(st.sidebar.text_input('Train data',0.8))
        features =   int(st.sidebar.text_input('Number of features',5)) 
        Logistic_reg = M.MLBacktester(symbol, start, end, tc, df, train_ratio)
        Logistic_reg.test_strategy(train_ratio, features)
        st.write(Logistic_reg.results_overview)
        st.pyplot(Logistic_reg.plot_results()) 
               

    #Strategy optimization 
    st.sidebar.header("MODEL OPTIMIZER") 
    optimization = st.sidebar.checkbox('Run Optimizer')
    
    if backtest_options == "BACK TO MEAN":
        SMA_s = int(st.sidebar.text_input("SMA Strat",2))
        SMA_e = int(st.sidebar.text_input("SMA End",20))
        dev_s = int(st.sidebar.text_input('Standard Deviation Start',2))
        dev_e = int(st.sidebar.text_input('Standard Deviation End',10))
        if optimization == True:    
            back_to_mean.optimize_parameters((SMA_s, SMA_e ),(dev_s, dev_e ))
            st.write(back_to_mean.results_overview)
            st.pyplot(back_to_mean.plot_results()) 
        
        
    if backtest_options == "SMA CROSSOVER":
        SMA_fast_s = int(st.sidebar.text_input('SMA Fast  Start',30))
        SMA_fast_e = int(st.sidebar.text_input('SMA Fast End',50))
        SMA_slow_s = int(st.sidebar.text_input('SMA Slow  Start',120))
        SMA_fast_e = int(st.sidebar.text_input('SMA Slow  End',150))
        if optimization == True:
            SMA_crossover.optimize_parameters((SMA_fast_s,SMA_fast_e),(SMA_slow_s,SMA_fast_e ))
            st.write(SMA_crossover.results_overview)
            st.pyplot(SMA_crossover.plot_results()) 
            
    if backtest_options == "LOGISTIC REGRESSION":
        features_opt =   int(st.sidebar.text_input('Opt features',10))
        if optimization == True:
            for feature in range(1, features_opt):
                st.write(print(feature, Logistic_reg.test_strategy(train_ratio , features_opt))) 
                    