import time

import pandas as pd
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("raw_partner_headlines.csv")
df = df.drop(columns=["url", "publisher", "Unnamed: 0"])
df.dropna(inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
#Market in usa opens between 9:30-16:00
df['adjusted_datetime'] = df['date'].apply(lambda x: x + pd.DateOffset(days=1) if x.hour >= 16 else x)
counts = df['stock'].value_counts().to_dict()
delta_GOOGL = {key: abs(counts['GOOGL'] - count) for key, count in counts.items() if key != 'GOOGL'}
delta_NIO = {key: abs(counts['NIO'] - count) for key, count in counts.items() if key != 'NIO'}
top_50_GOOGL = [key for key, count in sorted(delta_GOOGL.items(), key=lambda item: item[1])[:50]]
top_50_NIO = [key for key, count in sorted(delta_NIO.items(), key=lambda item: item[1])[:50]]

top_100_keys = top_50_NIO + top_50_GOOGL
def merge_start(row):
    specific_date = row['adjusted_datetime'].date().strftime('%Y-%m-%d')
    data = yf.download (row['stock'], start=specific_date, end=pd.to_datetime (specific_date) + pd.DateOffset (days=1),
                     interval='1d', progress=False)
    try:
        return data.loc[specific_date]['Open']
    except:#There is no value
        return None
def merge_end(row):
    specific_date = row['adjusted_datetime'].date().strftime('%Y-%m-%d')
    data = yf.download (row['stock'], start=specific_date, end=pd.to_datetime (specific_date) + pd.DateOffset (days=1),
                     interval='1d', progress=False)
    try:
        return data.loc[specific_date]['Close']
    except:
        return None
#Creating df for each stock containing the stock headline and the opening price + closing price
#for better understanding run the code
stocks = ["NVDA", "AAPL"]
stocks.extend(top_100_keys)
df_stock = []
time1= time.time()
df_stock_temp = df
df_stock_temp = df_stock_temp.loc[df_stock_temp['stock'].isin(stocks)]
df_stock_temp['opening_price'] = df_stock_temp.apply(lambda row: merge_start(row), axis = 1)
df_stock_temp['closing_price'] = df_stock_temp.apply(lambda row: merge_end(row), axis = 1)
df_stock_temp.dropna(inplace=True)
print(time.time() - time1)
print(df_stock_temp)
