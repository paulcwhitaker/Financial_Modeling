import pandas as pd
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.stats.descriptivestats import describe 

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

def get_stock(ticker):
    zf=zipfile.ZipFile(r'C:\Users\derfe\stock_project\stocks.zip')
    stock=[]
    with pd.read_csv(zf.open(r'Stocks/'+ticker+'.us.txt'),chunksize=10000) as reader:
        for chunk in reader:
            stock.append(chunk)
        stock=pd.concat(stock,ignore_index=True)
    return stock
    

def norm(stock,series):
    return stock[series]/stock[series][0]


def returns(series):
    returns=pd.Series(index=range(len(series)))
    returns[0]=0
    for i in range(1,len(series)):
        returns[i]=(series[i]-series[i-1])/series[i-1]*100
    return returns
      

def get_moments(series):
    percents=returns(series)
    return describe(percents,stats=['nobs','mean','std','skew','kurtosis'])

def yearly_stats(stock,series):
    stats=[]
    years=stock['Date'].apply(lambda date:date[0:4])
    years=years.drop_duplicates().reset_index(drop=True)
    
    for year in years:
        temp=stock.loc[stock['Date'].str.contains(str(year))].reset_index(drop=True)
        temp=get_moments(temp[series])
        stats.append(temp.T)
    stats=pd.concat(stats,ignore_index=True)
    stats=stats.set_index(years)
    
    return stats
