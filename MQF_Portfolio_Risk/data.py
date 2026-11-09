import datetime as dt

import pandas as pd
import yfinance as yf


def get_data(stocks, start, end, trade_days=200):
    """Download stock data, compute daily returns, and summarize risk."""
    stock_data = yf.download(stocks, start=start, end=end, auto_adjust=False) #download stock data from Yahoo Finance

    if isinstance(stock_data.columns, pd.MultiIndex): #if stock data has multiple colummns
        stock_data = stock_data['Close'] #select column labeled 'Close'
    else: #if stock data has 1 column
        if 'Close' in stock_data.columns: #if the 1 column is labeled 'Close'
            stock_data = stock_data[['Close']] #select the column labeled 'Close'
        else: #if the 1 column is not labeled 'Close'
            raise ValueError("The downloaded data does not contain a 'Close' column.") #raise an error
        
    #stock data is now only closing values for each day in 200-day span

    returns = stock_data.pct_change().tail(trade_days) #compute percentage change in closing value for each day in 200-day span
    mean_returns = returns.mean() #mean of daily returns for the stock
    cov_matrix = returns.cov() #covariance matrix of daily returns for the stock
    return mean_returns, cov_matrix, returns


def get_historical_portfolio_returns(stocks, weights, years_back=5):
    """Download historical price data and compute portfolio returns."""
    end_date = pd.Timestamp.today().normalize() #today's date
    start_date = end_date - pd.DateOffset(years=years_back) #date 5 years ago

    prices = yf.download(stocks,start=start_date,end=end_date,auto_adjust=True,)['Close'] #closing prices for each stock on each day in 5-year span

    returns = prices.pct_change().dropna() #percentage change in closing value for each stock and for each day in 5-year span
    portfolio_returns = returns @ weights #multiplies the returns of each stock by its weight , sums returns for each day to get total portfolio return for each day in 5-year span
    return returns, portfolio_returns


def build_window_dates(days_back=300):
    """Return a start and end date for recent market data."""
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days_back)
    return start_date, end_date
