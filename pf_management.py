# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 14:26:06 2025

@author: derfe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import scipy.optimize as sco
from playground import get_stock

stock_list=['aapl','fb','jnj','nflx']

#aapl = get_stock('aapl').set_index('Date')

#load stockdata from ticker list
table=pd.DataFrame()
for stock in stock_list:
    vals = get_stock(stock).rename(columns={'Close':str(stock)}).set_index('Date')
    table=pd.concat([table,vals],axis=1)

returns=table.pct_change()
'''
plt.figure(figsize=(14,7))
for c in returns.columns.values:
    plt.plot(returns.index, returns[c],lw=3,alpha=0.8,label=c)
plt.legend(loc='upper left',fontsize=12)
plt.ylabel('%return')
'''


'''
portfolio volatility  sigma_pf = sqrt((weight_1*sigma_1)^2 +w_2*sigma_2)^@ + 2w1w2Cov_1,2)
    or sqrt(W_T *COV *W)

percentage returns is weighted average of daily returns of individual stocks
return=SUM k->n w_k*a_k where a_k is avg return of the stock    
'''

def portfolio_annualised_performance(weights,mean_returns,cov_matrix):
    returns = np.sum(mean_returns*weights)*252
    std = np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))*np.sqrt(252)
    return std, returns

''' expressing risk adjusted returns; ratio describes how much excess return we can make for the extra risk of a 
more volatile stock
=(expected return-riskfreerate)/portfolio stdev
use US T-bill with rate 1.78% as rfr
sharpe ratio works best when little to no correlation between the two investments (existing pf and potential pf)

Eff front is boundary on MPT graph along which we have minimum volatility for same level of return

'''
#maximize sharpe ratio
def neg_sharpe_ratio(weights, mean_returns, cov_matrix,risk_free_rate):
    p_var,p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) /p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns,cov_matrix, risk_free_rate)
    constraints = ({'type':'eq','fun':lambda x:np.sum(x)-1})
    bound=(0.0,1.0) #for the weights
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,],args=args,
                          method = 'SLSQP',bounds=bounds,constraints = constraints)
    
    return result
#minimize volatility    
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args =(mean_returns,cov_matrix)
    constraints = ({'type':'eq','fun':lambda x:np.sum(x)-1})
    bound=(0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,],args=args,
                          method = 'SLSQP',bounds=bounds,constraints = constraints)
    return result

#print(mean_returns)
#takes target return value, returns val with min vol.
def efficient_return(mean_returns, cov_matrix,target):
    num_assets = len(mean_returns) #3 in this case
    args = (mean_returns, cov_matrix)
    
    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]
    constraints = ({'type':'eq','fun':lambda x: portfolio_return(x) -target},# return should be equal to target
                   {'type':'eq','fun':lambda x: np.sum(x)-1}) #weights shold be eqaul to 1
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds = bounds,constraints=constraints)
    return result

#takes i range of return, gives efficient frontier portfolio for each return value aka one with min volatil
def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients=[]
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

#print(efficient_frontier(mean_returns, cov_matrix, [0.3,0.2]))

def display_ef_with_selected(mean_returns,cov_matrix,risk_free_rate):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol = min_variance(mean_returns,cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x, index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns *252
    
    print('-'*80)
    print('max sharpe ratio portfolio allocation\n')
    print('annulized return:', round(rp,2))
    print('annualised volatility:',round(sdp,2))
    print('\n')
    print(max_sharpe_allocation)
    print('-'*80)
    print('mminimum volatility pf allocation\n')
    print('annulized return:', round(rp_min,2))
    print('annualised volatility:',round(sdp_min,2))
    print('\n')
    print(min_vol_allocation)
    print('-'*80)
    print('individual stock returns and volatility\n')
    for i, txt in enumerate(table.columns):
        print(txt,':','annualised return',round(an_rt[i],2),', annualised vol:',round(an_vol[i],2))
    print('-'*80)
    
    fig,ax = plt.subplots(figsize=(10,7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)
    
    for i,txt in enumerate(table.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]),xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Max sharpe ratio')
    ax.scatter(sdp_min,rp_min,marker='*', color='g',s=500,label='Min volatility')
    
    target = np.linspace(rp_min,0.38,50)
    efficient_portfolios = efficient_frontier(mean_returns,cov_matrix,target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.',color='black',label='efficient frontier')#pfun is volatility
    plt.title('Calculated Pf Optimization based on efficient Frontier')
    plt.xlabel('annualized volatility')
    plt.ylabel('annualized returns')
    plt.legend(labelspacing=0.8)



mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios=25000
risk_free_rate = 0.0424

display_ef_with_selected(mean_returns,cov_matrix,risk_free_rate)




