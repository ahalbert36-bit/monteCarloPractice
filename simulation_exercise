#Monte Carlo Simulation - Part 1

#monte carlo - random sampling (same inputs doesn't always lead to same outputs), more simulations to converge on more exact result

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #plotting
import datetime as dt #date and time
import yfinance as yf #module to get stock data from Yahoo Finance


#print(yf.download('AAPL', start='2025-01-01', end='2025-1-31', interval='1h')) #test to see if yfinance is working, prints AAPL stock data from Jan 1, 2023 to Dec 31, 2023

#importing data
def get_data(stocks, start, end): #function that takes in a list of stocks, start date, and end date
    stock_data = yf.download(stocks, start=start, end=end, auto_adjust=False) #gets df of all valuesfrom Yahoo Finance info
    if isinstance(stock_data.columns, pd.MultiIndex): #does df have multiple columns (stocks and value types))
        stock_data = stock_data['Close'] #if yes, show close columns only
    else:
        stock_data = stock_data[['Close']] if 'Close' in stock_data.columns else stock_data #converts to df with only close values if there is only one stock in the list
    returns = stock_data.pct_change() #prints daily returns for each stock (as percentage) for each day - df format
    mean_returns = returns.mean() #mean value at daily returns
    cov_matrix = returns.cov() #gets covariance matrix of daily returns (in matrix form)
    return mean_returns, cov_matrix #for each stock, returns mean daily change and its covariance matrix

stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN'] #list of stocks to analyze
end_date = dt.datetime.now() #end date for data - now
start_date = end_date - dt.timedelta(days=300) #end data minus 300 days (delta = days) to get start date

mean_returns, cov_matrix = get_data(stocks, start_date, end_date) #runs function and assigns outputs these variables

weights = np.random.random(len(mean_returns)) #length = number of stocks (number of mean return values), generates random float numbers btw 0 and 1 for each stock
weights /= np.sum(weights) #divides weight by sum of weights to normalize them (so they add up to 1), 1*4

#Monte Carlo Simulation

mc_sims = 500 #number of simulations to run
num_days = 50 #number of days

mean_m = np.full(shape=(num_days, len(weights)), fill_value=mean_returns) #creates matrix of mean_returns duplicated for number of rows
mean_m = mean_m.T #each col has mean_return values, repeated for each day/column (4 rows (stocks), num_days cols (days))

portfolio_sims = np.full(shape=(num_days, mc_sims), fill_value=0.0) #num_days (rows) * mc_sims (cols), starts blank, each row is a day, each column is a simulation

initial_portfolio = 1000 #initial portfolio value

for m in range(0, mc_sims): #for each simulation
    #cholesky decomposition to determine lower triangular matrix of covariance matrix, simple for monte carlo simulation, allows for correlated random variables
    z = np.random.normal(size=(num_days, len(mean_returns))) #generates matrix of normally distributed random numbers, num_days rows * 4 cols (days * stocks)
    l = np.linalg.cholesky(cov_matrix) #cholesky decomposition of covariance matrix (lower triangle matrix), size = #stocks * #stocks
    daily_returns = mean_m + np.inner(l, z) #stocks x days, existing mean returns + triangle matrix*random numbers
    #inner (sum product) requires number of cols to match (no transpose like regular matrix multiplication), leading to #rowsA * #rowsB for dimensions
    #covar matrix values are randomly distributed in magnitude (relative to mean return)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_portfolio #weights * returns (+1 for percentage format), multiplies by 
    #weights * daily return % (+1 for percentage format) so weight is same for each individual stocks returns, gets % return multiplied by weight invested in each stock
    # 
    #print(weights)
    #print(daily_returns) #row = stock, col = day
    #print(np.inner(weights,daily_returns.T)+1) #total return for the day
    #print(portfolio_sims) #row = day (cumulatively multiplies previous return with new return), col = simulation
    #+1 to get new value for each day

plt.plot(portfolio_sims) #plots all simulations
plt.ylabel('Portfolio Value ($)') #y-axis label
plt.xlabel('Days') #x-axis label
plt.title('Monte Carlo Simulation: Portfolio Value Over Time') #title of plot
#plt.show() #shows plot

#Review Concepts: Covariance matrix, Cholesky decomposition process


#Part 2 ----------------------------

def mcValueRisk(returns, alpha=5): #value at risk, portfolio final results are represented by "returns"
    """ 
    Input: pandas series of returns
    Output: percentile on return distribution to given confidence level (alpha)
    """
    
    if isinstance(returns, pd.Series): #matrix of % changes and the pandas data series
        return np.percentile(returns, alpha) #array of returns, returns the value at the given percentile (alpha)
    else:
        raise TypeError("Expected pandas data series")


def mcConditionalValueRisk(returns, alpha=5): #conditional value at risk
    """ 
    Input: pandas series of returns
    Output: Conditional value at risk (CVaR) at given confidence level (alpha), also known as expected shortfall (average of losses beyond the VaR threshold)
    """
    
    if isinstance(returns, pd.Series): #matrix of % changes and the pandas data series
        below_var = returns <= mcValueRisk(returns, alpha=alpha) #below limit will be returns <= value at the given percentile (alpha), determined by previous function
        return returns[below_var].mean() #takes every value where below_var is True and returns mean of values
    else:
        raise TypeError("Expected pandas data series")


portfolioResults = pd.Series(portfolio_sims[-1, :], name='Portfolio Results') #portfolio results for last day
#pd.Series is to ensure the data is a pd series
#array of end values of each simulation (to plug into functions)

VaR = initial_portfolio - mcValueRisk(portfolioResults, alpha=5) #runs function with portfolio results, subtracts difference from initial value
CondVaR = initial_portfolio - mcConditionalValueRisk(portfolioResults, alpha=5)  

print('Value at Risk (VaR) ${}'.format(round(VaR, 2))) #prints value at risk, formatted to 2 decimal places
print('Conditional Value at Risk (CVaR) ${}'.format(round(CondVaR, 2))) #prints conditional value at risk, formatted to 2 decimal places

#Conditional Value at Risk (CVaR) =  expected loss given that the loss has exceeded the Value at Risk (VaR) threshold (bottom 5%)
